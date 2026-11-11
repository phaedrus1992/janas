/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision_gemma_u.c - Gemma 4 12B's vision, an embedder without blocks
 * (vision_impl.h): the picture sized as the other Gemma 4 encoders do it
 * (vision_gemma.c, janas_vg_size), cut into patches of 48x48 pixels (row,
 * column, channel; pixels in [0, 1]), each layer-normed, through a dense
 * layer with a bias, layer-normed, plus the learned embeddings of its
 * column and its row, layer-normed again; then RMS-normed without a scale
 * and projected to the language model's width.
 */
#include "vision_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/image.h"
#include "common/pool.h"

const char *janas_vgu_open(struct janas_vision *v)
{
    struct janas_jns *j = &v->j;
    uint32_t ti = 0, ts = 0, te = 0, both = 0;
    if (janas_jns_meta_u32(j, "vision.embedding_length", &v->dim) ||
        janas_jns_meta_u32(j, "vision.patch_size", &v->patch) ||
        janas_jns_meta_u32(j, "vision.pooling_kernel_size", &v->pool_k) ||
        janas_jns_meta_u32(j, "vision.position_size", &v->pos_size) ||
        janas_jns_meta_u32(j, "vision.max_soft_tokens", &v->max_soft) ||
        janas_jns_meta_u32(j, "vision.projection_dim", &v->proj) ||
        janas_jns_meta_f32(j, "vision.attention.layer_norm_epsilon", &v->eps) ||
        janas_jns_meta_f32(j, "vision.patch_norm_epsilon", &v->u_eps))
        return "the encoder's sizes are missing";
    if (janas_jns_meta_u32(j, "vision.token.image", &ti) ||
        janas_jns_meta_u32(j, "vision.token.start", &ts) ||
        janas_jns_meta_u32(j, "vision.token.end", &te))
        return "the image tokens are missing";
    janas_jns_meta_u32(j, "vision.bidirectional", &both);
    v->unified = 1;
    v->both_ways = both != 0;
    v->tok_image = (int32_t)ti;
    v->tok_start = (int32_t)ts;
    v->tok_end = (int32_t)te;
    uint32_t side = v->patch * v->pool_k;
    if (v->patch < 2 || v->pool_k < 1 || side > 256 || !v->dim ||
        !v->pos_size || !v->max_soft || !v->proj)
        return "the encoder's sizes are out of range";
    uint64_t pd = 3ull * side * side;
    static const char *const LN[3] = {"v.u.ln1", "v.u.ln2", "v.u.ln3"};
    for (int k = 0; k < 3; k++) {
        char n[64];
        uint64_t len = k ? v->dim : pd;
        snprintf(n, sizeof n, "%s.weight", LN[k]);
        v->u_ln[k][0] = janas_vision_tensor(v, n, 0, 1, len);
        snprintf(n, sizeof n, "%s.bias", LN[k]);
        v->u_ln[k][1] = janas_vision_tensor(v, n, 0, 1, len);
        if (!v->u_ln[k][0] || !v->u_ln[k][1])
            return "a norm of the encoder is missing";
    }
    const uint16_t *dense = janas_vision_tensor(v, "v.u.dense", 30, v->dim, pd);
    v->u_dense_b = janas_vision_tensor(v, "v.u.dense.bias", 0, 1, v->dim);
    /* rows of (position, axis): x then y for each position */
    v->g_pos =
        janas_vision_tensor(v, "v.pos_embd", 30, 2ull * v->pos_size, v->dim);
    const uint16_t *proj =
        janas_vision_tensor(v, "v.proj", 30, v->proj, v->dim);
    if (!dense || !v->u_dense_b || !v->g_pos || !proj)
        return "a tensor of the encoder is missing or of another shape";
    v->t_u_dense = vk_transposed(dense, v->dim, pd);
    v->t_g_proj = vk_transposed(proj, v->proj, v->dim);
    if (!v->t_u_dense || !v->t_g_proj)
        return "out of memory";
    return NULL;
}

int janas_vgu_encode(struct janas_vision *v, const struct janas_image *im,
                     int max_tokens, struct janas_pool *pool, float **out,
                     int *nx, int *ny, char *err, size_t err_len)
{
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
    uint32_t S = v->patch * v->pool_k, D = v->dim;
    uint32_t gw = (uint32_t)*nx, gh = (uint32_t)*ny, n = gw * gh;
    uint32_t pd = 3 * S * S;
    float *pix = malloc(sizeof(float) * (size_t)n * pd);
    float *x = malloc(sizeof(float) * (size_t)n * D);
    float *res = NULL;
    struct janas_pool *own = NULL;
    int rc = -1;
    if (!pool && !(pool = own = janas_pool_create(1, NULL))) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    if (!pix || !x) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    /* patches in rows, each (row, column, channel) in [0, 1], normed */
    for (uint32_t r = 0; r < gh; r++)
        for (uint32_t c = 0; c < gw; c++) {
            float *p = pix + ((size_t)r * gw + c) * pd, *p0 = p;
            for (uint32_t yy = 0; yy < S; yy++) {
                const uint8_t *row =
                    src->rgb + ((size_t)(r * S + yy) * (size_t)rw + c * S) * 3;
                for (uint32_t k = 0; k < 3 * S; k++)
                    *p++ = row[k] * (1.0f / 255.0f);
            }
            vk_layer_norm(p0, p0, v->u_ln[0][0], v->u_ln[0][1], pd, v->u_eps);
        }
    vk_matmul(pool, v->t_u_dense, v->u_dense_b, pix, x, D, pd, n, 0);
    for (uint32_t r = 0; r < gh; r++)
        for (uint32_t c = 0; c < gw; c++) {
            float *xi = x + ((size_t)r * gw + c) * D;
            vk_layer_norm(xi, xi, v->u_ln[1][0], v->u_ln[1][1], D, v->u_eps);
            uint32_t px = c < v->pos_size ? c : v->pos_size - 1;
            uint32_t py = r < v->pos_size ? r : v->pos_size - 1;
            const uint16_t *ex = v->g_pos + (size_t)px * 2 * D;
            const uint16_t *ey = v->g_pos + ((size_t)py * 2 + 1) * D;
            for (uint32_t k = 0; k < D; k++)
                xi[k] += vk_bf(ex[k]) + vk_bf(ey[k]);
            vk_layer_norm(xi, xi, v->u_ln[2][0], v->u_ln[2][1], D, v->u_eps);
            vk_rms_norm(xi, xi, NULL, D, v->eps);
        }
    res = malloc(sizeof(float) * (size_t)n * v->proj);
    if (!res) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    vk_matmul(pool, v->t_g_proj, NULL, x, res, v->proj, D, n, 0);
    *out = res;
    res = NULL;
    rc = 0;
done:
    free(res);
    free(pix);
    free(x);
    if (own)
        janas_pool_destroy(own);
    if (src == &rs)
        janas_image_free(&rs);
    return rc;
}
