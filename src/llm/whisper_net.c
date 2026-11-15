/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper_net.c - Whisper's encoder and decoder (whisper.h), as the
 * Hugging Face model computes them.
 *
 * Encoder: two convolutions of width 3 over the window of the spectrogram
 * (the second with stride 2, a_ctx positions out), each followed by GELU
 * (erf), plus sinusoidal positions; blocks of LayerNorm, attention (queries,
 * values and output with a bias, keys without), residual, LayerNorm, MLP
 * fc1 - GELU - fc2, residual; a final LayerNorm. Its output gives every
 * decoder block its keys and values once.
 *
 * Decoder: token embeddings plus learned positions; blocks of causal
 * attention over its own tokens (their keys and values kept), attention
 * over the encoder's output and an MLP, each LayerNorm first, a residual
 * after; a final LayerNorm, the logits against the token embeddings. The
 * products are the vision encoders' tiled kernels (vision_kernels.h) on the
 * weights in half precision.
 */
#include "whisper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"
#include "quant.h"
#include "vision_kernels.h"

static void gelu_rows(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        x[i] = vk_gelu_erf(x[i]);
}

static void ln_rows(float *y, const float *x, const float *w, const float *b,
                    uint32_t rows, uint32_t d)
{
    for (uint32_t r = 0; r < rows; r++)
        vk_layer_norm(y + (size_t)r * d, x + (size_t)r * d, w, b, d, 1e-5f);
}

static void mm(struct janas_whisper *w, const uint16_t *wt, const float *b,
               const float *x, float *y, uint32_t rows, uint32_t cols,
               uint32_t n, size_t ldy)
{
    vk_matmul_f16(w->pool, wt, b, x, y, rows, cols, n, ldy);
}

/* ---- the encoder ---- */

int janas_whisper_encode(struct janas_whisper *w, const float *mel,
                         uint32_t stride, uint32_t off, uint32_t len)
{
    uint32_t d = w->dim, n = w->a_ctx, nin = 2 * n, nm = w->n_mels;
    size_t big = (size_t)n * (3 * d > w->a_ffn ? 3 * d : w->a_ffn);
    float *col = malloc(sizeof(float) * (size_t)nin * 3 * nm);
    float *h1 = malloc(sizeof(float) * (size_t)nin * d);
    float *x = malloc(sizeof(float) * (size_t)n * d);
    float *h = malloc(sizeof(float) * (size_t)n * d);
    float *t = malloc(sizeof(float) * big);
    float *buf = malloc(sizeof(float) * 2 * (size_t)n * d);
    float *o = malloc(sizeof(float) * (size_t)n * d);
    int rc = -1;
    if (!col || !h1 || !x || !h || !t || !buf || !o)
        goto done;
    /* conv1: a row per output position, (channel, tap) its inputs */
    for (uint32_t p = 0; p < nin; p++)
        for (uint32_t c = 0; c < nm; c++)
            for (uint32_t k = 0; k < 3; k++) {
                long s = (long)p + (long)k - 1;
                float v = 0;
                if (s >= 0 && s < (long)nin && (uint32_t)s < len)
                    v = mel[(size_t)c * stride + off + (uint32_t)s];
                col[((size_t)p * nm + c) * 3 + k] = v;
            }
    mm(w, w->conv1, w->conv1_b, col, h1, d, 3 * nm, nin, 0);
    gelu_rows(h1, (size_t)nin * d);
    /* conv2, stride 2: the inputs of position p are 2p - 1 .. 2p + 1 */
    float *col2 = t; /* n x 3 d */
    for (uint32_t p = 0; p < n; p++)
        for (uint32_t c = 0; c < d; c++)
            for (uint32_t k = 0; k < 3; k++) {
                long s = 2 * (long)p + (long)k - 1;
                col2[((size_t)p * d + c) * 3 + k] =
                    s >= 0 && s < (long)nin ? h1[(size_t)s * d + c] : 0.0f;
            }
    mm(w, w->conv2, w->conv2_b, col2, x, d, 3 * d, n, 0);
    gelu_rows(x, (size_t)n * d);
    for (size_t i = 0; i < (size_t)n * d; i++)
        x[i] += w->e_pos[i];
    float scale = 1.0f / sqrtf((float)(d / w->a_heads));
    for (uint32_t l = 0; l < w->a_layers; l++) {
        const struct wlayer *L = &w->enc[l];
        ln_rows(h, x, L->ln_w, L->ln_b, n, d);
        mm(w, L->q, L->q_b, h, t, d, d, n, 3 * (size_t)d);
        mm(w, L->k, NULL, h, t + d, d, d, n, 3 * (size_t)d);
        mm(w, L->v, L->v_b, h, t + 2 * d, d, d, n, 3 * (size_t)d);
        vk_attention(w->pool, t, buf, o, n, d, w->a_heads, scale);
        mm(w, L->o, L->o_b, o, h, d, d, n, 0);
        for (size_t i = 0; i < (size_t)n * d; i++)
            x[i] += h[i];
        ln_rows(h, x, L->mln_w, L->mln_b, n, d);
        mm(w, L->fc1, L->fc1_b, h, t, w->a_ffn, d, n, 0);
        gelu_rows(t, (size_t)n * w->a_ffn);
        mm(w, L->fc2, L->fc2_b, t, h, d, w->a_ffn, n, 0);
        for (size_t i = 0; i < (size_t)n * d; i++)
            x[i] += h[i];
    }
    ln_rows(h, x, w->e_ln_w, w->e_ln_b, n, d);
    /* every decoder block's keys and values over it */
    for (uint32_t l = 0; l < w->t_layers; l++) {
        const struct wlayer *L = &w->dec[l];
        size_t at = (size_t)l * n * d;
        mm(w, L->xk, NULL, h, w->xk + at, d, d, n, 0);
        mm(w, L->xv, L->xv_b, h, w->xv + at, d, d, n, 0);
    }
    rc = 0;
done:
    free(col);
    free(h1);
    free(x);
    free(h);
    free(t);
    free(buf);
    free(o);
    return rc;
}

/* ---- the decoder ---- */

struct dec_att {
    const float *q;     /* n x dim */
    const float *k, *v; /* positions x dim */
    float *out;         /* n x dim */
    uint32_t n, dim, heads, hd;
    uint32_t keys; /* how many keys the last token sees */
    int causal;    /* token i sees keys .. keys - n + i */
    float scale;
    float *const *cap; /* by head: n x keys scores kept, or NULL */
};

static void dec_att_worker(void *arg, int tid, int nt)
{
    const struct dec_att *a = arg;
    float *s = malloc(sizeof(float) * a->keys);
    if (!s)
        return;
    size_t total = (size_t)a->n * a->heads;
    for (size_t t = (size_t)tid; t < total; t += (size_t)nt) {
        uint32_t i = (uint32_t)(t / a->heads), hh = (uint32_t)(t % a->heads);
        uint32_t nk = a->causal ? a->keys - a->n + i + 1 : a->keys;
        const float *q = a->q + (size_t)i * a->dim + (size_t)hh * a->hd;
        float mx = -INFINITY;
        for (uint32_t k = 0; k < nk; k++) {
            const float *key = a->k + (size_t)k * a->dim + (size_t)hh * a->hd;
            float dot = 0;
            for (uint32_t e = 0; e < a->hd; e++)
                dot += q[e] * key[e];
            s[k] = dot * a->scale;
            if (s[k] > mx)
                mx = s[k];
        }
        if (a->cap && a->cap[hh])
            memcpy(a->cap[hh] + (size_t)i * a->keys, s, sizeof(float) * nk);
        double sum = 0;
        for (uint32_t k = 0; k < nk; k++) {
            s[k] = expf(s[k] - mx);
            sum += s[k];
        }
        float inv = (float)(1.0 / sum);
        float *o = a->out + (size_t)i * a->dim + (size_t)hh * a->hd;
        memset(o, 0, sizeof(float) * a->hd);
        for (uint32_t k = 0; k < nk; k++) {
            const float *v = a->v + (size_t)k * a->dim + (size_t)hh * a->hd;
            float p = s[k] * inv;
            for (uint32_t e = 0; e < a->hd; e++)
                o[e] += p * v[e];
        }
    }
    free(s);
}

static void attend(struct janas_whisper *w, const float *q, const float *k,
                   const float *v, float *out, uint32_t n, uint32_t keys,
                   int causal, float *const *cap)
{
    struct dec_att a = {.q = q,
                        .k = k,
                        .v = v,
                        .out = out,
                        .n = n,
                        .dim = w->dim,
                        .heads = w->t_heads,
                        .hd = w->dim / w->t_heads,
                        .keys = keys,
                        .causal = causal,
                        .scale = 1.0f / sqrtf((float)(w->dim / w->t_heads)),
                        .cap = cap};
    janas_pool_run(w->pool, dec_att_worker, &a);
}

int janas_whisper_decode(struct janas_whisper *w, const int32_t *tok,
                         uint32_t n, uint32_t past, float *logits, int all)
{
    uint32_t d = w->dim;
    if (!n || past + n > w->t_ctx)
        return -1;
    float *x = w->x, *h = w->h, *q = w->q, *att = w->att;
    for (uint32_t i = 0; i < n; i++) {
        if (tok[i] < 0 || (uint32_t)tok[i] >= w->vocab)
            return -1;
        const float *pos = w->d_pos + (size_t)(past + i) * d;
        for (uint32_t c = 0; c < d; c++)
            x[(size_t)i * d + c] =
                janas_fp16_to_fp32(w->tok[(size_t)c * w->vocab + tok[i]]) +
                pos[c];
    }
    for (uint32_t l = 0; l < w->t_layers; l++) {
        const struct wlayer *L = &w->dec[l];
        float *kc = w->kc + (size_t)l * w->t_ctx * d;
        float *vc = w->vc + (size_t)l * w->t_ctx * d;
        ln_rows(h, x, L->ln_w, L->ln_b, n, d);
        mm(w, L->q, L->q_b, h, q, d, d, n, 0);
        mm(w, L->k, NULL, h, kc + (size_t)past * d, d, d, n, 0);
        mm(w, L->v, L->v_b, h, vc + (size_t)past * d, d, d, n, 0);
        attend(w, q, kc, vc, att, n, past + n, 1, NULL);
        mm(w, L->o, L->o_b, att, h, d, d, n, 0);
        for (size_t i = 0; i < (size_t)n * d; i++)
            x[i] += h[i];
        ln_rows(h, x, L->xln_w, L->xln_b, n, d);
        mm(w, L->xq, L->xq_b, h, q, d, d, n, 0);
        size_t at = (size_t)l * w->a_ctx * d;
        /* the alignment heads' scores, when asked for */
        float *capv[64] = {0}, *const *cap = NULL;
        if (w->cap) {
            for (uint32_t i = 0; i < w->n_aheads; i++)
                if ((uint32_t)w->aheads[2 * i] == l &&
                    (uint32_t)w->aheads[2 * i + 1] < 64)
                    capv[w->aheads[2 * i + 1]] =
                        w->cap + (size_t)i * n * w->a_ctx;
            cap = capv;
        }
        attend(w, q, w->xk + at, w->xv + at, att, n, w->a_ctx, 0, cap);
        mm(w, L->xo, L->xo_b, att, h, d, d, n, 0);
        for (size_t i = 0; i < (size_t)n * d; i++)
            x[i] += h[i];
        ln_rows(h, x, L->mln_w, L->mln_b, n, d);
        mm(w, L->fc1, L->fc1_b, h, w->ff, w->t_ffn, d, n, 0);
        gelu_rows(w->ff, (size_t)n * w->t_ffn);
        mm(w, L->fc2, L->fc2_b, w->ff, h, d, w->t_ffn, n, 0);
        for (size_t i = 0; i < (size_t)n * d; i++)
            x[i] += h[i];
    }
    uint32_t first = all ? 0 : n - 1;
    ln_rows(h, x + (size_t)first * d, w->d_ln_w, w->d_ln_b, n - first, d);
    mm(w, w->tok, NULL, h, logits, w->vocab, d, n - first, 0);
    return 0;
}
