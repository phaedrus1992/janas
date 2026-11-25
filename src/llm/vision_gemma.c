/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision_gemma.c - Gemma 4's vision encoder (vision_impl.h), as its Hugging
 * Face processor and model compute it:
 *   - the picture resized (Pillow's bicubic) to the largest size, aspect
 *     kept, whose 16x16 patches are at most the soft tokens times 3 x 3, a
 *     multiple of 48 a side; pixels scaled to [-1, 1];
 *   - each patch (row, column, channel) through one matrix, plus the
 *     learned embeddings of its column and of its row;
 *   - blocks of RMS norm, attention (queries and keys RMS-normed with a
 *     scale, values without, a rotary embedding of the column on the first
 *     half of each head and of the row on the second, no scaling of the
 *     scores), RMS norm, residual; RMS norm, a gated tanh-GELU MLP, RMS
 *     norm, residual - every linear clamping its input and its output to
 *     bounds of its own;
 *   - the patches averaged 3 x 3, times sqrt(width), standardized where
 *     the model says, RMS-normed and projected to the language model's
 *     width.
 */
#include "vision_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/image.h"
#include "common/pool.h"

/* the soft tokens a picture may have, as the processor allows them */
static const uint32_t SOFT[] = {70, 140, 280, 560, 1120};

const char *janas_vg_open(struct janas_vision *v)
{
    struct janas_jns *j = &v->j;
    uint32_t ti = 0, ts = 0, te = 0, stdz = 0, unified = 0;
    janas_jns_meta_u32(j, "vision.unified", &unified);
    if (unified) /* the 12B's, without blocks */
        return janas_vgu_open(v);
    if (janas_jns_meta_u32(j, "vision.block_count", &v->n_block) ||
        janas_jns_meta_u32(j, "vision.embedding_length", &v->dim) ||
        janas_jns_meta_u32(j, "vision.feed_forward_length", &v->ff) ||
        janas_jns_meta_u32(j, "vision.attention.head_count", &v->heads) ||
        janas_jns_meta_u32(j, "vision.patch_size", &v->patch) ||
        janas_jns_meta_u32(j, "vision.pooling_kernel_size", &v->pool_k) ||
        janas_jns_meta_u32(j, "vision.position_size", &v->pos_size) ||
        janas_jns_meta_u32(j, "vision.max_soft_tokens", &v->max_soft) ||
        janas_jns_meta_u32(j, "vision.projection_dim", &v->proj) ||
        janas_jns_meta_u32(j, "vision.standardize", &stdz) ||
        janas_jns_meta_f32(j, "vision.attention.layer_norm_epsilon", &v->eps) ||
        janas_jns_meta_f32(j, "vision.rope.freq_base", &v->theta))
        return "the encoder's sizes are missing";
    if (janas_jns_meta_u32(j, "vision.token.image", &ti) ||
        janas_jns_meta_u32(j, "vision.token.start", &ts) ||
        janas_jns_meta_u32(j, "vision.token.end", &te))
        return "the image tokens are missing";
    uint32_t both = 0; /* absent in the files of before */
    janas_jns_meta_u32(j, "vision.bidirectional", &both);
    v->both_ways = both != 0;
    v->tok_image = (int32_t)ti;
    v->tok_start = (int32_t)ts;
    v->tok_end = (int32_t)te;
    if (v->n_block < 1 || v->n_block > MAX_BLOCKS || !v->heads ||
        v->dim % v->heads || (v->dim / v->heads) % 4 ||
        v->dim / v->heads > 512 || v->pool_k < 1 || v->pool_k > 8 ||
        v->patch < 2 || !v->pos_size || !v->max_soft || !v->proj)
        return "the encoder's sizes are out of range";
    uint64_t pd = 3ull * v->patch * v->patch;
    const uint16_t *pw = janas_vision_tensor(v, "v.patch_embd", 30, v->dim, pd);
    v->g_pos =
        janas_vision_tensor(v, "v.pos_embd", 30, 2ull * v->pos_size, v->dim);
    const uint16_t *proj =
        janas_vision_tensor(v, "v.proj", 30, v->proj, v->dim);
    if (stdz) {
        v->std_bias = janas_vision_tensor(v, "v.std_bias", 0, 1, v->dim);
        v->std_scale = janas_vision_tensor(v, "v.std_scale", 0, 1, v->dim);
        if (!v->std_bias || !v->std_scale)
            return "the standardization is missing";
    }
    if (!pw || !v->g_pos || !proj)
        return "a tensor of the encoder is missing or of another shape";
    v->t_g_patch = vk_transposed(pw, v->dim, pd);
    v->t_g_proj = vk_transposed(proj, v->proj, v->dim);
    if (!v->t_g_patch || !v->t_g_proj)
        return "out of memory";
    static const char *const LIN[GL_N] = {"q",    "k",  "v",   "o",
                                          "gate", "up", "down"};
    uint32_t hd = v->dim / v->heads;
    for (uint32_t l = 0; l < v->n_block; l++) {
        struct glayer *L = &v->gl[l];
        char n[64];
#define NORM(field, part, len)                                                 \
    snprintf(n, sizeof n, "v.blk.%u." part, l);                                \
    if (!(L->field = janas_vision_tensor(v, n, 0, 1, len)))                    \
        return "a norm of the encoder is missing";
        NORM(n_in, "in_norm", v->dim);
        NORM(n_post_att, "post_att_norm", v->dim);
        NORM(n_pre_ff, "pre_ff_norm", v->dim);
        NORM(n_post_ff, "post_ff_norm", v->dim);
        NORM(q_norm, "q_norm", hd);
        NORM(k_norm, "k_norm", hd);
#undef NORM
        snprintf(n, sizeof n, "v.blk.%u.clip", l);
        if (!(L->clip = janas_vision_tensor(v, n, 0, 1, GL_N * 4)))
            return "the clamps of the encoder are missing";
        for (int k = 0; k < GL_N; k++) {
            uint64_t rows = k == GL_GATE || k == GL_UP ? v->ff : v->dim;
            uint64_t cols = k == GL_DOWN ? v->ff : v->dim;
            snprintf(n, sizeof n, "v.blk.%u.%s", l, LIN[k]);
            const uint16_t *w = janas_vision_tensor(v, n, 30, rows, cols);
            if (!w)
                return "a matrix of the encoder is missing";
            if (!(L->t[k] = vk_transposed(w, rows, cols)))
                return "out of memory";
        }
    }
    return NULL;
}

void janas_vg_close(struct janas_vision *v)
{
    for (uint32_t l = 0; l < MAX_BLOCKS; l++)
        for (int k = 0; k < GL_N; k++)
            free(v->gl[l].t[k]);
    free(v->t_g_patch);
    free(v->t_g_proj);
    free(v->t_u_dense);
}

void janas_vg_size(const struct janas_vision *v, int w, int h, int max_tokens,
                   int *rw, int *rh, int *nx, int *ny)
{
    uint32_t soft = v->max_soft;
    if (max_tokens > 0 && (uint32_t)max_tokens < soft) {
        soft = SOFT[0];
        for (size_t i = 0; i < sizeof SOFT / sizeof SOFT[0]; i++)
            if (SOFT[i] <= (uint32_t)max_tokens)
                soft = SOFT[i];
    }
    double k = v->pool_k, P = v->patch, side = k * P;
    double max_patches = (double)soft * k * k;
    double target = max_patches * P * P;
    double f = sqrt(target / ((double)w * h));
    int th = (int)(floor(f * h / side) * side);
    int tw = (int)(floor(f * w / side) * side);
    int longest = (int)((uint32_t)max_patches / (uint32_t)(k * k) * side);
    if (th == 0 && tw == 0)
        th = tw = (int)side;
    else if (th == 0) {
        th = (int)side;
        tw = (int)(floor((double)w / h) * side);
        if (tw > longest)
            tw = longest;
    } else if (tw == 0) {
        tw = (int)side;
        th = (int)(floor((double)h / w) * side);
        if (th > longest)
            th = longest;
    }
    *rw = tw;
    *rh = th;
    *nx = (int)(tw / side);
    *ny = (int)(th / side);
}

/* x clamped into y */
static void clamp_into(float *y, const float *x, size_t n, float lo, float hi)
{
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] < lo ? lo : x[i] > hi ? hi : x[i];
}

/* y[n] (ldy apart) = clamp(W clamp(x)) with linear k's bounds; tmp holds
   n x cols floats */
static void clipped(struct janas_pool *pool, const struct glayer *L, int k,
                    const float *x, float *tmp, float *y, uint32_t rows,
                    uint32_t cols, uint32_t n, size_t ldy)
{
    const float *c = L->clip + 4 * k;
    clamp_into(tmp, x, (size_t)n * cols, c[0], c[1]);
    vk_matmul(pool, L->t[k], NULL, tmp, y, rows, cols, n, ldy);
    size_t ld = ldy ? ldy : rows;
    for (uint32_t t = 0; t < n; t++)
        clamp_into(y + t * ld, y + t * ld, rows, c[2], c[3]);
}

/* the rotary embedding of position p on half a head (len channels) */
static void rope_half(float *x, uint32_t len, float p, float theta)
{
    uint32_t q = len / 2;
    for (uint32_t i = 0; i < q; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)len);
        float a = p * inv, cs = cosf(a), sn = sinf(a);
        float x0 = x[i], x1 = x[i + q];
        x[i] = x0 * cs - x1 * sn;
        x[i + q] = x1 * cs + x0 * sn;
    }
}

int janas_vg_encode(struct janas_vision *v, const struct janas_image *im,
                    int max_tokens, struct janas_pool *pool, float **out,
                    int *nx, int *ny, char *err, size_t err_len)
{
    if (v->unified)
        return janas_vgu_encode(v, im, max_tokens, pool, out, nx, ny, err,
                                err_len);
    *out = NULL;
    int rw, rh;
    janas_vg_size(v, im->w, im->h, max_tokens, &rw, &rh, nx, ny);
    struct janas_image rs;
    const struct janas_image *src = im;
    if (rw != im->w || rh != im->h) {
        if (janas_image_resize(im, rw, rh, &rs) != 0) {
            snprintf(err, err_len, "out of memory");
            return -1;
        }
        src = &rs;
    }
    uint32_t P = v->patch, D = v->dim, K = v->pool_k, hd = D / v->heads;
    uint32_t gw = (uint32_t)rw / P, gh = (uint32_t)rh / P, n = gw * gh;
    uint32_t pd = 3 * P * P, wide = v->ff > 3 * D ? v->ff : 3 * D;
    if (pd > wide)
        wide = pd;
    float *pix = malloc(sizeof(float) * (size_t)n * pd);
    float *x = malloc(sizeof(float) * (size_t)n * D);
    float *h = malloc(sizeof(float) * (size_t)n * D);
    float *tmp = malloc(sizeof(float) * (size_t)n * wide);
    float *qkv = malloc(sizeof(float) * (size_t)n * 3 * D);
    float *g = malloc(sizeof(float) * (size_t)n * v->ff);
    float *u = malloc(sizeof(float) * (size_t)n * v->ff);
    float *buf = malloc(sizeof(float) * (size_t)n * D * 2);
    float *res = NULL;
    struct janas_pool *own = NULL;
    int rc = -1;
    if (!pool && !(pool = own = janas_pool_create(1, NULL))) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    if (!pix || !x || !h || !tmp || !qkv || !g || !u || !buf) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    /* patches in rows, each (row, column, channel), pixels in [-1, 1] */
    for (uint32_t r = 0; r < gh; r++)
        for (uint32_t c = 0; c < gw; c++) {
            float *p = pix + ((size_t)r * gw + c) * pd;
            for (uint32_t yy = 0; yy < P; yy++)
                for (uint32_t xx = 0; xx < P; xx++)
                    for (uint32_t ch = 0; ch < 3; ch++)
                        *p++ = src->rgb[((size_t)(r * P + yy) * (size_t)rw +
                                         (c * P + xx)) *
                                            3 +
                                        ch] *
                                   (2.0f / 255.0f) -
                               1.0f;
        }
    vk_matmul(pool, v->t_g_patch, NULL, pix, x, D, pd, n, 0);
    for (uint32_t r = 0; r < gh; r++)
        for (uint32_t c = 0; c < gw; c++) {
            float *xi = x + ((size_t)r * gw + c) * D;
            const uint16_t *ex =
                v->g_pos + (size_t)(c < v->pos_size ? c : 0) * D;
            const uint16_t *ey =
                v->g_pos +
                ((size_t)v->pos_size + (r < v->pos_size ? r : 0)) * D;
            for (uint32_t k = 0; k < D; k++)
                xi[k] += vk_bf(ex[k]) + vk_bf(ey[k]);
        }
    for (uint32_t l = 0; l < v->n_block; l++) {
        const struct glayer *L = &v->gl[l];
        for (uint32_t i = 0; i < n; i++)
            vk_rms_norm(h + (size_t)i * D, x + (size_t)i * D, L->n_in, D,
                        v->eps);
        clipped(pool, L, GL_Q, h, tmp, qkv, D, D, n, 3 * (size_t)D);
        clipped(pool, L, GL_K, h, tmp, qkv + D, D, D, n, 3 * (size_t)D);
        clipped(pool, L, GL_V, h, tmp, qkv + 2 * D, D, D, n, 3 * (size_t)D);
        /* per head: queries and keys normed and rotated, values normed */
        for (uint32_t i = 0; i < n; i++) {
            float col = (float)(i % gw), row = (float)(i / gw);
            float *q = qkv + (size_t)i * 3 * D;
            for (uint32_t hh = 0; hh < v->heads; hh++) {
                float *qh = q + hh * hd, *kh = q + D + hh * hd;
                float *vh = q + 2 * D + hh * hd;
                vk_rms_norm(qh, qh, L->q_norm, hd, v->eps);
                vk_rms_norm(kh, kh, L->k_norm, hd, v->eps);
                vk_rms_norm(vh, vh, NULL, hd, v->eps);
                rope_half(qh, hd / 2, col, v->theta);
                rope_half(qh + hd / 2, hd / 2, row, v->theta);
                rope_half(kh, hd / 2, col, v->theta);
                rope_half(kh + hd / 2, hd / 2, row, v->theta);
            }
        }
        vk_attention(pool, qkv, buf, h, n, D, v->heads, 1.0f);
        clipped(pool, L, GL_O, h, tmp, g, D, D, n, 0);
        for (uint32_t i = 0; i < n; i++) {
            float *o = g + (size_t)i * D;
            vk_rms_norm(o, o, L->n_post_att, D, v->eps);
            for (uint32_t k = 0; k < D; k++)
                x[(size_t)i * D + k] += o[k];
        }
        for (uint32_t i = 0; i < n; i++)
            vk_rms_norm(h + (size_t)i * D, x + (size_t)i * D, L->n_pre_ff, D,
                        v->eps);
        clipped(pool, L, GL_GATE, h, tmp, g, v->ff, D, n, 0);
        clipped(pool, L, GL_UP, h, tmp, u, v->ff, D, n, 0);
        for (size_t k = 0; k < (size_t)n * v->ff; k++)
            g[k] = vk_gelu_tanh(g[k]) * u[k];
        clipped(pool, L, GL_DOWN, g, tmp, h, D, v->ff, n, 0);
        for (uint32_t i = 0; i < n; i++) {
            float *o = h + (size_t)i * D;
            vk_rms_norm(o, o, L->n_post_ff, D, v->eps);
            for (uint32_t k = 0; k < D; k++)
                x[(size_t)i * D + k] += o[k];
        }
    }
    /* the patches averaged K x K, row-major over the pooled grid */
    uint32_t pw2 = gw / K, ph2 = gh / K, m = pw2 * ph2;
    memset(h, 0, sizeof(float) * (size_t)m * D);
    for (uint32_t r = 0; r < gh; r++)
        for (uint32_t c = 0; c < gw; c++) {
            float *o = h + ((size_t)(r / K) * pw2 + c / K) * D;
            const float *xi = x + ((size_t)r * gw + c) * D;
            for (uint32_t k = 0; k < D; k++)
                o[k] += xi[k];
        }
    float scale = sqrtf((float)D) / (float)(K * K);
    for (uint32_t i = 0; i < m; i++) {
        float *o = h + (size_t)i * D;
        for (uint32_t k = 0; k < D; k++) {
            o[k] *= scale;
            if (v->std_bias)
                o[k] = (o[k] - v->std_bias[k]) * v->std_scale[k];
        }
        vk_rms_norm(o, o, NULL, D, v->eps);
    }
    res = malloc(sizeof(float) * (size_t)m * v->proj);
    if (!res) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    vk_matmul(pool, v->t_g_proj, NULL, h, res, v->proj, D, m, 0);
    *nx = (int)pw2;
    *ny = (int)ph2;
    *out = res;
    res = NULL;
    rc = 0;
done:
    free(res);
    free(pix);
    free(x);
    free(h);
    free(tmp);
    free(qkv);
    free(g);
    free(u);
    free(buf);
    if (own)
        janas_pool_destroy(own);
    if (src == &rs)
        janas_image_free(&rs);
    return rc;
}
