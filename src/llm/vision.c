/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision.c - the vision encoder of Qwen3.5 and Qwen3-VL (vision.h), as their
 * Hugging Face processor and model compute it:
 *   - the picture resized (Pillow's bicubic, image_resize.c) to the size
 *     smart_resize gives: multiples of patch x merge (32), its pixels within
 *     the model's bounds, the aspect kept; scaled to [0, 1], normalized;
 *   - cut into 16x16 patches taken 2x2 block by 2x2 block (the order the
 *     merger joins them in), each patch given as two equal frames;
 *   - the patch embedding (a 3D convolution: one matrix), plus the learned
 *     positions of a 48x48 grid resampled bilinearly (corners aligned) to
 *     the picture's grid;
 *   - blocks of LayerNorm, attention over every patch with a rotary
 *     embedding of the patch's row (first quarter of each half of a head)
 *     and column, LayerNorm and a tanh-GELU MLP, each with its residual;
 *   - the merger: LayerNorm of each patch, four neighbours joined, a linear
 *     layer, exact GELU, a linear layer to the language model's width.
 * Weights are bfloat16 as the checkpoint has them, everything else float.
 */
#include "vision_impl.h"

#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bf16.h"
#include "vmath.h"
#include "common/image.h"
#include "common/pool.h"
#include "jns.h"

const void *janas_vision_tensor(struct janas_vision *v, const char *name,
                                uint32_t type, uint64_t rows, uint64_t cols)
{
    for (uint64_t i = 0; i < v->j.h.n_tensor; i++) {
        const struct janas_jns_tensor *t = &v->j.tensors[i];
        if (strncmp(t->name, name, sizeof t->name))
            continue;
        uint64_t n = t->n_dims == 2 ? t->dims[0] * t->dims[1] : t->dims[0];
        uint64_t esz = type == 30 ? 2 : 4;
        if (t->type != type || n != rows * cols || t->bytes != n * esz ||
            (t->n_dims == 2 && t->dims[0] != cols))
            return NULL;
        return v->data + (t->offset - v->j.h.resident_offset);
    }
    return NULL;
}

void janas_vision_close(struct janas_vision *v)
{
    if (!v)
        return;
    for (uint32_t l = 0; l < MAX_BLOCKS; l++) {
        free(v->l[l].t_qkv);
        free(v->l[l].t_out);
        free(v->l[l].t_up);
        free(v->l[l].t_down);
    }
    free(v->t_patch);
    free(v->t_fc1);
    free(v->t_fc2);
    janas_vg_close(v);
    free(v->data);
    janas_jns_close(&v->j);
    free(v);
}

/* the resident tensors read into memory: NULL, or why not */
static const char *read_data(struct janas_vision *v)
{
    struct janas_jns *j = &v->j;
    v->data = malloc(j->h.resident_bytes ? j->h.resident_bytes : 1);
    if (!v->data)
        return "out of memory";
    size_t got = 0;
    while (got < j->h.resident_bytes) {
        ssize_t k = pread(j->fd, v->data + got, j->h.resident_bytes - got,
                          (off_t)(j->h.resident_offset + got));
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    return got == j->h.resident_bytes ? NULL
                                      : "cannot read the encoder's weights";
}

struct janas_vision *janas_vision_open(const char *path, char *err,
                                       size_t err_len)
{
    struct janas_vision *v = calloc(1, sizeof(*v));
    if (!v) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    if (janas_jns_open(path, &v->j, err, err_len) != 0) {
        free(v);
        return NULL;
    }
    const char *why = NULL;
    v->kind = !strncmp(v->j.h.arch, "qwen3vl_vision", sizeof v->j.h.arch)
                  ? VISION_QWEN
              : !strncmp(v->j.h.arch, "gemma4_vision", sizeof v->j.h.arch)
                  ? VISION_GEMMA
                  : 0;
    if (!v->kind)
        why = "not a vision encoder this runs";
    struct janas_jns *j = &v->j;
    if (!why && v->kind == VISION_GEMMA) { /* vision_gemma.c */
        why = read_data(v);
        if (!why)
            why = janas_vg_open(v);
        goto opened;
    }
    if (!why &&
        (janas_jns_meta_u32(j, "vision.block_count", &v->n_block) ||
         janas_jns_meta_u32(j, "vision.embedding_length", &v->dim) ||
         janas_jns_meta_u32(j, "vision.feed_forward_length", &v->ff) ||
         janas_jns_meta_u32(j, "vision.attention.head_count", &v->heads) ||
         janas_jns_meta_u32(j, "vision.patch_size", &v->patch) ||
         janas_jns_meta_u32(j, "vision.temporal_patch_size", &v->tpatch) ||
         janas_jns_meta_u32(j, "vision.spatial_merge_size", &v->merge) ||
         janas_jns_meta_u32(j, "vision.position_count", &v->npos) ||
         janas_jns_meta_u32(j, "vision.projection_dim", &v->proj) ||
         janas_jns_meta_u32(j, "vision.min_pixels", &v->min_pixels) ||
         janas_jns_meta_u32(j, "vision.max_pixels", &v->max_pixels) ||
         janas_jns_meta_f32(j, "vision.attention.layer_norm_epsilon",
                            &v->eps) ||
         janas_jns_meta_f32(j, "vision.rope.freq_base", &v->theta)))
        why = "the encoder's sizes are missing";
    uint32_t ti = 0, ts = 0, te = 0;
    if (!why && (janas_jns_meta_u32(j, "vision.token.image", &ti) ||
                 janas_jns_meta_u32(j, "vision.token.start", &ts) ||
                 janas_jns_meta_u32(j, "vision.token.end", &te)))
        why = "the image tokens are missing";
    v->tok_image = (int32_t)ti;
    v->tok_start = (int32_t)ts;
    v->tok_end = (int32_t)te;
    /* the processor's mean and deviation, per channel */
    for (int k = 0; !why && k < 3; k++) {
        char key[40];
        snprintf(key, sizeof key, "vision.image_mean.%d", k);
        if (janas_jns_meta_f32(j, key, &v->mean[k]))
            why = "the processor's mean is missing";
        snprintf(key, sizeof key, "vision.image_std.%d", k);
        if (janas_jns_meta_f32(j, key, &v->stdv[k]) || v->stdv[k] == 0)
            why = "the processor's deviation is missing";
    }
    uint32_t side = 0;
    while ((side + 1) * (side + 1) <= v->npos)
        side++;
    if (!why && (v->n_block < 1 || v->n_block > MAX_BLOCKS || !v->heads ||
                 v->dim % v->heads || (v->dim / v->heads) % 4 ||
                 v->dim / v->heads > 512 || v->merge != 2 ||
                 side * side != v->npos || v->patch < 2))
        why = "the encoder's sizes are out of range";
    if (!why)
        why = read_data(v);
    uint64_t pd = 3ull * v->tpatch * v->patch * v->patch;
    uint64_t md = (uint64_t)v->dim * v->merge * v->merge;
    if (!why) {
        v->patch_w = janas_vision_tensor(v, "v.patch_embd", 30, v->dim, pd);
        v->patch_b = janas_vision_tensor(v, "v.patch_embd.bias", 0, 1, v->dim);
        v->pos = janas_vision_tensor(v, "v.pos_embd", 30, v->npos, v->dim);
        v->m_ln_w = janas_vision_tensor(v, "v.merger.ln.weight", 0, 1, v->dim);
        v->m_ln_b = janas_vision_tensor(v, "v.merger.ln.bias", 0, 1, v->dim);
        v->m_fc1 = janas_vision_tensor(v, "v.merger.fc1", 30, md, md);
        v->m_fc1_b = janas_vision_tensor(v, "v.merger.fc1.bias", 0, 1, md);
        v->m_fc2 = janas_vision_tensor(v, "v.merger.fc2", 30, v->proj, md);
        v->m_fc2_b = janas_vision_tensor(v, "v.merger.fc2.bias", 0, 1, v->proj);
        int ok = v->patch_w && v->patch_b && v->pos && v->m_ln_w && v->m_ln_b &&
                 v->m_fc1 && v->m_fc1_b && v->m_fc2 && v->m_fc2_b;
        for (uint32_t l = 0; ok && l < v->n_block; l++) {
            struct vlayer *L = &v->l[l];
            char n[64];
#define T(field, part, type, r, c)                                             \
    snprintf(n, sizeof n, "v.blk.%u." part, l);                                \
    L->field = janas_vision_tensor(v, n, type, r, c);                          \
    ok = ok && L->field
            T(ln1_w, "ln1.weight", 0, 1, v->dim);
            T(ln1_b, "ln1.bias", 0, 1, v->dim);
            T(ln2_w, "ln2.weight", 0, 1, v->dim);
            T(ln2_b, "ln2.bias", 0, 1, v->dim);
            T(qkv, "attn_qkv", 30, 3 * v->dim, v->dim);
            T(qkv_b, "attn_qkv.bias", 0, 1, 3 * v->dim);
            T(out, "attn_out", 30, v->dim, v->dim);
            T(out_b, "attn_out.bias", 0, 1, v->dim);
            T(up, "ffn_up", 30, v->ff, v->dim);
            T(up_b, "ffn_up.bias", 0, 1, v->ff);
            T(down, "ffn_down", 30, v->dim, v->ff);
            T(down_b, "ffn_down.bias", 0, 1, v->dim);
#undef T
        }
        if (!ok)
            why = "a tensor of the encoder is missing or of another shape";
    }
    if (!why) { /* the matrices transposed once, for the products */
        v->t_patch = vk_transposed(v->patch_w, v->dim, pd);
        v->t_fc1 = vk_transposed(v->m_fc1, md, md);
        v->t_fc2 = vk_transposed(v->m_fc2, v->proj, md);
        int ok = v->t_patch && v->t_fc1 && v->t_fc2;
        for (uint32_t l = 0; ok && l < v->n_block; l++) {
            struct vlayer *L = &v->l[l];
            L->t_qkv = vk_transposed(L->qkv, 3 * v->dim, v->dim);
            L->t_out = vk_transposed(L->out, v->dim, v->dim);
            L->t_up = vk_transposed(L->up, v->ff, v->dim);
            L->t_down = vk_transposed(L->down, v->dim, v->ff);
            ok = L->t_qkv && L->t_out && L->t_up && L->t_down;
        }
        if (!ok)
            why = "out of memory";
    }
opened:
    if (why) {
        snprintf(err, err_len, "%s: %s", path, why);
        janas_vision_close(v);
        return NULL;
    }
    return v;
}

uint32_t janas_vision_dim(const struct janas_vision *v)
{
    return v->proj;
}

int janas_vision_both_ways(const struct janas_vision *v)
{
    return v->both_ways;
}

void janas_vision_tokens(const struct janas_vision *v, int32_t *image,
                         int32_t *start, int32_t *end)
{
    *image = v->tok_image;
    *start = v->tok_start;
    *end = v->tok_end;
}

/* Python's round(): halves to the even neighbour */
static double round_even(double x)
{
    double r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2) != 0)
        r -= 1;
    return r;
}

void janas_vision_size(const struct janas_vision *v, int w, int h,
                       int max_tokens, int *rw, int *rh, int *nx, int *ny)
{
    if (v->kind == VISION_GEMMA) {
        janas_vg_size(v, w, h, max_tokens, rw, rh, nx, ny);
        return;
    }
    double f = (double)v->patch * v->merge;
    double maxp = v->max_pixels, minp = v->min_pixels;
    if (max_tokens > 0 && maxp > (double)max_tokens * f * f)
        maxp = (double)max_tokens * f * f;
    double hb = round_even(h / f) * f, wb = round_even(w / f) * f;
    if (hb * wb > maxp) {
        double beta = sqrt((double)h * w / maxp);
        hb = fmax(f, floor(h / beta / f) * f);
        wb = fmax(f, floor(w / beta / f) * f);
    } else if (hb * wb < minp) {
        double beta = sqrt(minp / ((double)h * w));
        hb = ceil(h * beta / f) * f;
        wb = ceil(w * beta / f) * f;
    }
    if (hb < f)
        hb = f;
    if (wb < f)
        wb = f;
    *rw = (int)wb;
    *rh = (int)hb;
    *nx = (int)(wb / f);
    *ny = (int)(hb / f);
}

/* ---- the computation ---- */

int janas_vision_encode(struct janas_vision *v, const struct janas_image *im,
                        int max_tokens, struct janas_pool *pool, float **out,
                        int *nx, int *ny, char *err, size_t err_len)
{
    if (v->kind == VISION_GEMMA)
        return janas_vg_encode(v, im, max_tokens, pool, out, nx, ny, err,
                               err_len);
    *out = NULL;
    int rw, rh;
    janas_vision_size(v, im->w, im->h, max_tokens, &rw, &rh, nx, ny);
    struct janas_image rs;
    const struct janas_image *src = im;
    if (rw != im->w || rh != im->h) {
        if (janas_image_resize(im, rw, rh, &rs) != 0) {
            snprintf(err, err_len, "out of memory");
            return -1;
        }
        src = &rs;
    }
    uint32_t P = v->patch, M = v->merge, D = v->dim;
    uint32_t gw = (uint32_t)rw / P, gh = (uint32_t)rh / P;
    uint32_t n = gw * gh, pd = 3 * v->tpatch * P * P;
    uint32_t hd = D / v->heads;
    float *pix = malloc(sizeof(float) * (size_t)n * pd);
    float *x = malloc(sizeof(float) * (size_t)n * D);
    float *t = malloc(sizeof(float) * (size_t)n * D);
    float *qkv = malloc(sizeof(float) * (size_t)n * 3 * D);
    float *ffb =
        malloc(sizeof(float) * (size_t)n * (v->ff > 3 * D ? v->ff : 3 * D));
    int *prow = malloc(sizeof(int) * n), *pcol = malloc(sizeof(int) * n);
    float *res = NULL, *cs = NULL, *kt = NULL;
    struct janas_pool *own = NULL; /* one thread, when none is given */
    int rc = -1;
    if (!pool && !(pool = own = janas_pool_create(1, NULL))) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    kt = malloc(sizeof(float) * (size_t)n * D * 2); /* keys, then values */
    if (!pix || !x || !t || !qkv || !ffb || !prow || !pcol || !kt) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    /* patches, 2x2 block by block; each patch [channel][frame][y][x] */
    uint32_t idx = 0;
    for (uint32_t by = 0; by < gh / M; by++)
        for (uint32_t bx = 0; bx < gw / M; bx++)
            for (uint32_t my = 0; my < M; my++)
                for (uint32_t mx = 0; mx < M; mx++, idx++) {
                    uint32_t r = by * M + my, c = bx * M + mx;
                    prow[idx] = (int)r;
                    pcol[idx] = (int)c;
                    float *p = pix + (size_t)idx * pd;
                    for (uint32_t ch = 0; ch < 3; ch++)
                        for (uint32_t f = 0; f < v->tpatch; f++)
                            for (uint32_t yy = 0; yy < P; yy++)
                                for (uint32_t xx = 0; xx < P; xx++) {
                                    uint8_t s = src->rgb[((size_t)(r * P + yy) *
                                                              (size_t)rw +
                                                          (c * P + xx)) *
                                                             3 +
                                                         ch];
                                    *p++ = (s / 255.0f - v->mean[ch]) /
                                           v->stdv[ch];
                                }
                }
    vk_matmul(pool, v->t_patch, v->patch_b, pix, x, D, pd, n, 0);
    /* the learned positions, resampled to the grid (corners aligned) */
    uint32_t side = 0;
    while ((side + 1) * (side + 1) <= v->npos)
        side++;
    for (uint32_t i = 0; i < n; i++) {
        double sr = gh > 1 ? (double)prow[i] * (side - 1) / (gh - 1) : 0;
        double sc = gw > 1 ? (double)pcol[i] * (side - 1) / (gw - 1) : 0;
        int r0 = (int)floor(sr), c0 = (int)floor(sc);
        int r1 = r0 + 1 < (int)side ? r0 + 1 : (int)side - 1;
        int c1 = c0 + 1 < (int)side ? c0 + 1 : (int)side - 1;
        float wr = (float)(sr - r0), wc = (float)(sc - c0);
        float w00 = (1 - wr) * (1 - wc), w01 = (1 - wr) * wc;
        float w10 = wr * (1 - wc), w11 = wr * wc;
        const uint16_t *e00 = v->pos + ((size_t)r0 * side + (size_t)c0) * D;
        const uint16_t *e01 = v->pos + ((size_t)r0 * side + (size_t)c1) * D;
        const uint16_t *e10 = v->pos + ((size_t)r1 * side + (size_t)c0) * D;
        const uint16_t *e11 = v->pos + ((size_t)r1 * side + (size_t)c1) * D;
        float *xi = x + (size_t)i * D;
        for (uint32_t k = 0; k < D; k++)
            xi[k] += w00 * vk_bf(e00[k]) + w01 * vk_bf(e01[k]) +
                     w10 * vk_bf(e10[k]) + w11 * vk_bf(e11[k]);
    }
    /* the rotary angles: a quarter of a head for the row, a quarter for the
       column, the half repeated */
    uint32_t q4 = hd / 4;
    cs = malloc(sizeof(float) * (size_t)n * hd * 2);
    if (!cs) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    for (uint32_t i = 0; i < n; i++) {
        float *c = cs + (size_t)i * hd * 2, *s = c + hd;
        for (uint32_t k = 0; k < q4; k++) {
            float inv = 1.0f / powf(v->theta, (float)(2 * k) / (float)(hd / 2));
            float ar = (float)prow[i] * inv, ac = (float)pcol[i] * inv;
            c[k] = c[k + hd / 2] = cosf(ar);
            s[k] = s[k + hd / 2] = sinf(ar);
            c[q4 + k] = c[q4 + k + hd / 2] = cosf(ac);
            s[q4 + k] = s[q4 + k + hd / 2] = sinf(ac);
        }
    }
    for (uint32_t l = 0; l < v->n_block; l++) {
        const struct vlayer *L = &v->l[l];
        for (uint32_t i = 0; i < n; i++)
            vk_layer_norm(t + (size_t)i * D, x + (size_t)i * D, L->ln1_w,
                          L->ln1_b, D, v->eps);
        vk_matmul(pool, L->t_qkv, L->qkv_b, t, qkv, 3 * D, D, n, 0);
        /* rotate q and k: x cos + rotate_half(x) sin, per head */
        for (uint32_t i = 0; i < n; i++) {
            const float *c = cs + (size_t)i * hd * 2, *s = c + hd;
            for (uint32_t part = 0; part < 2; part++)
                for (uint32_t h = 0; h < v->heads; h++) {
                    float *q = qkv + (size_t)i * 3 * D + part * D + h * hd;
                    float tmp[512];
                    for (uint32_t k = 0; k < hd; k++) {
                        float rot = k < hd / 2 ? -q[k + hd / 2] : q[k - hd / 2];
                        tmp[k] = q[k] * c[k] + rot * s[k];
                    }
                    memcpy(q, tmp, sizeof(float) * hd);
                }
        }
        vk_attention(pool, qkv, kt, t, n, D, v->heads, 1.0f / sqrtf((float)hd));
        vk_matmul(pool, L->t_out, L->out_b, t, ffb, D, D, n, 0);
        for (size_t k = 0; k < (size_t)n * D; k++)
            x[k] += ffb[k];
        for (uint32_t i = 0; i < n; i++)
            vk_layer_norm(t + (size_t)i * D, x + (size_t)i * D, L->ln2_w,
                          L->ln2_b, D, v->eps);
        vk_matmul(pool, L->t_up, L->up_b, t, ffb, v->ff, D, n, 0);
        for (size_t k = 0; k < (size_t)n * v->ff; k++)
            ffb[k] = vk_gelu_tanh(ffb[k]);
        vk_matmul(pool, L->t_down, L->down_b, ffb, t, D, v->ff, n, 0);
        for (size_t k = 0; k < (size_t)n * D; k++)
            x[k] += t[k];
    }
    /* the merger: each patch normalized, four consecutive ones joined */
    uint32_t nm = n / (M * M), md = D * M * M;
    for (uint32_t i = 0; i < n; i++)
        vk_layer_norm(t + (size_t)i * D, x + (size_t)i * D, v->m_ln_w,
                      v->m_ln_b, D, v->eps);
    float *h1 = malloc(sizeof(float) * (size_t)nm * md);
    res = malloc(sizeof(float) * (size_t)nm * v->proj);
    if (!h1 || !res) {
        free(h1);
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    vk_matmul(pool, v->t_fc1, v->m_fc1_b, t, h1, md, md, nm, 0);
    for (size_t k = 0; k < (size_t)nm * md; k++)
        h1[k] = vk_gelu_erf(h1[k]);
    vk_matmul(pool, v->t_fc2, v->m_fc2_b, h1, res, v->proj, md, nm, 0);
    free(h1);
    *out = res;
    res = NULL;
    rc = 0;
done:
    free(res);
    free(cs);
    free(kt);
    if (own)
        janas_pool_destroy(own);
    free(pix);
    free(x);
    free(t);
    free(qkv);
    free(ffb);
    free(prow);
    free(pcol);
    if (src == &rs)
        janas_image_free(&rs);
    return rc;
}
