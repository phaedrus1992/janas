/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_check.c - products split between the GPU and the CPU must equal the
 * CPU's bit for bit: random Q4_K, Q5_K, Q6_K, Q6_K_P (bit planes, at its
 * three levels), Q8_0 and the blocks of 32 or 64 (Q4_0, IQ4_NL, Q4_1, Q5_1,
 * MXFP4, Q5_0, Q2_0) matrices in shared memory, and the GPU's aligned copy
 * of Q6_K (on the GPU alone, against the CPU's Q6_K), several shapes, vector
 * counts and strides; for the blocks of 32 or 64 and Q8_0 also rows that
 * end inside a group of 256 (qwen4exp's 640, its shared experts' down
 * matrices Q8_0); Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
 * IQ1_S and IQ1_M (Qwen3.8-Flash-Next's other files, unsloth's 1- and 2-bit
 * ones). Then bf16 weights on float vectors
 * (janas_gpu_bf16_mv) against the CPU's eight lanes (bf16.h), with x and y
 * where the GPU shares them and through its stages (JANAS_GPU_COPY=1: the
 * weights too, as on a card with memory of its own). Skips (exit 0) when
 * no GPU is usable.
 *
 * Usage: gpu_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/aligned.h"
#include "llm/bf16.h"
#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static uint64_t rng = 88172645463325252ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

int main(void)
{
    /* the exact path: the fast prompt (on by default) rounds otherwise */
    setenv("JANAS_GPU_FAST_PROMPT", "0", 1);
    /* GPU_CHECK_BIG_GB=n: n GiB of touched memory, like the engine's
       expert cache */
    const char *bg = getenv("GPU_CHECK_BIG_GB");
    if (bg) {
        size_t big = (size_t)atoi(bg) << 30;
        uint8_t *b = janas_aligned_alloc(4096, big);
        if (b)
            memset(b, 1, big);
    }
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("gpu_check: no GPU (%s), skipped\n", err);
        return 0;
    }
    enum { MAXR = 4096, MAXC = 4096, MAXV = 64 };
    size_t wbytes = (size_t)MAXR * (MAXC / JANAS_QK) * 144;
    wbytes = (wbytes + 4095) / 4096 * 4096;
    uint8_t *w = janas_aligned_alloc(4096, wbytes);
    for (size_t i = 0; i < wbytes; i++)
        w[i] = (uint8_t)rnd();
    struct janas_block_q4k *wb = (struct janas_block_q4k *)w;
    for (size_t b = 0; b < wbytes / 144; b++) {
        wb[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        wb[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q6_K weights too, with small scales */
    size_t w6bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 210 + 4096;
    w6bytes = (w6bytes + 4095) / 4096 * 4096;
    uint8_t *w6 = janas_aligned_alloc(4096, w6bytes);
    for (size_t i = 0; i < w6bytes; i++)
        w6[i] = (uint8_t)rnd();
    struct janas_block_q6k *w6b = (struct janas_block_q6k *)w6;
    for (size_t b = 0; b < (w6bytes - 4096) / 210; b++)
        w6b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    /* the same Q6_K in the GPU's aligned copy (gpu.h JANAS_Q6KA_BLOCK):
       ql, qh, scales, d at byte 208 */
    size_t nb6 = (w6bytes - 4096) / 210;
    size_t wabytes = (nb6 * JANAS_Q6KA_BLOCK + 4095) / 4096 * 4096;
    uint8_t *wa = janas_aligned_alloc(4096, wabytes);
    memset(wa, 0, wabytes);
    for (size_t b = 0; b < nb6; b++) {
        uint8_t *d = wa + b * JANAS_Q6KA_BLOCK;
        memcpy(d, w6b[b].ql, sizeof(w6b[b].ql));
        memcpy(d + 128, w6b[b].qh, sizeof(w6b[b].qh));
        memcpy(d + 192, w6b[b].scales, sizeof(w6b[b].scales));
        memcpy(d + 208, &w6b[b].d, sizeof(w6b[b].d));
    }
    /* Q6_K_P: the high planes (82-byte blocks), then the two bare planes
       (64 bytes a block), row after row as in an expert slot */
    size_t pbase = (size_t)MAXR * (MAXC / JANAS_QK) * 82 + 4096;
    pbase = (pbase + 4095) / 4096 * 4096;
    size_t pplane = (size_t)MAXR * (MAXC / JANAS_QK) * JANAS_Q6KP_PLANE;
    size_t wpbytes = pbase + 2 * pplane;
    uint8_t *wp = janas_aligned_alloc(4096, wpbytes);
    for (size_t i = 0; i < wpbytes; i++)
        wp[i] = (uint8_t)rnd();
    struct janas_block_q6kp *wpb = (struct janas_block_q6kp *)wp;
    for (size_t b = 0; b < (pbase - 4096) / 82; b++)
        wpb[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    const uint8_t *pl1 = wp + pbase, *pl0 = wp + pbase + pplane;
    /* Q5_K */
    size_t w5bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 176;
    w5bytes = (w5bytes + 4095) / 4096 * 4096;
    uint8_t *w5 = janas_aligned_alloc(4096, w5bytes);
    for (size_t i = 0; i < w5bytes; i++)
        w5[i] = (uint8_t)rnd();
    struct janas_block_q5k *w5b = (struct janas_block_q5k *)w5;
    for (size_t b = 0; b < w5bytes / 176; b++) {
        w5b[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        w5b[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q8_0: blocks of 32, eight to 256 weights */
    size_t w8bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 272;
    w8bytes = (w8bytes + 4095) / 4096 * 4096;
    uint8_t *w8 = janas_aligned_alloc(4096, w8bytes);
    for (size_t i = 0; i < w8bytes; i++)
        w8[i] = (uint8_t)rnd();
    struct janas_block_q8_0 *w8b = (struct janas_block_q8_0 *)w8;
    for (size_t b = 0; b < w8bytes / 34; b++)
        w8b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    /* blocks of 32 with four or five bits: Q4_0 and IQ4_NL (18 bytes: d,
       16 of nibbles), Q4_1 (20: d, m, ...), Q5_1 (24: d, m, qh, ...), MXFP4
       (17: the E8M0 byte, 16 of nibbles), Q5_0 (22: d, qh, ...), and Q2_0's
       blocks of 64 (18: d, two bits a weight; 9 bytes for 32 below); random
       bytes with d (and m) set to small scales, MXFP4's byte to powers of
       two from 2^-10 to 2^-3 (a subnormal one, under 2^-125, a GPU may
       flush) */
    static const int t32[7] = {JANAS_Q4_0, JANAS_IQ4_NL, JANAS_Q4_1,
                               JANAS_Q5_1, JANAS_MXFP4,  JANAS_Q5_0,
                               JANAS_Q2_0};
    /* Q3_K (d at byte 108) and IQ3_S (d at 0): 110-byte blocks */
    uint8_t *w3[2];
    size_t w3bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 110 + 4096;
    w3bytes = (w3bytes + 4095) / 4096 * 4096;
    for (int k = 0; k < 2; k++) {
        w3[k] = janas_aligned_alloc(4096, w3bytes);
        for (size_t i = 0; i < w3bytes; i++)
            w3[k][i] = (uint8_t)rnd();
        for (size_t b = 0; b + 110 <= w3bytes; b += 110) {
            uint16_t d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
            memcpy(w3[k] + b + (k == 0 ? 108 : 0), &d, 2);
        }
    }
    /* IQ4_XS: 136-byte blocks, d, scales_h, scales_l, nibbles */
    size_t wxbytes = (size_t)MAXR * (MAXC / JANAS_QK) * 136;
    wxbytes = (wxbytes + 4095) / 4096 * 4096;
    uint8_t *wx = janas_aligned_alloc(4096, wxbytes);
    for (size_t i = 0; i < wxbytes; i++)
        wx[i] = (uint8_t)rnd();
    for (size_t b = 0; b + 136 <= wxbytes; b += 136) {
        uint16_t d = janas_fp32_to_fp16(0.00005f + (rnd() % 1000) * 1e-7f);
        memcpy(wx + b, &d, 2);
    }
    static const size_t s32[7] = {18, 18, 20, 24, 17, 22, 9};
    uint8_t *w32[7];
    size_t w32bytes[7];
    for (int k = 0; k < 7; k++) {
        w32bytes[k] = (size_t)MAXR * (MAXC / JANAS_QK) * 8 * s32[k];
        w32bytes[k] = (w32bytes[k] + 4095) / 4096 * 4096;
        w32[k] = janas_aligned_alloc(4096, w32bytes[k]);
        for (size_t i = 0; i < w32bytes[k]; i++)
            w32[k][i] = (uint8_t)rnd();
        size_t step = t32[k] == JANAS_Q2_0 ? 18 : s32[k];
        for (size_t b = 0; b + step <= w32bytes[k]; b += step) {
            if (t32[k] == JANAS_MXFP4) {
                w32[k][b] = (uint8_t)(118 + rnd() % 8);
                continue;
            }
            uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
            memcpy(w32[k] + b, &d, 2);
            if (t32[k] == JANAS_Q4_1 || t32[k] == JANAS_Q5_1) {
                uint16_t m =
                    janas_fp32_to_fp16(-0.01f + (rnd() % 1000) * 2e-5f);
                memcpy(w32[k] + b + 2, &m, 2);
            }
        }
    }
    /* the 256-block types of iq2, iq3xxs, iq1 and q2k_matvec.comp: random
       bytes, the scales small - d at byte 0, Q2_K's d and dmin at 80 and
       82, IQ1_M's d in the top nibbles of its four scale words (48-55) */
    static const int tq[7] = {JANAS_Q2_K,  JANAS_IQ2_XXS, JANAS_IQ2_XS,
                              JANAS_IQ2_S, JANAS_IQ3_XXS, JANAS_IQ1_S,
                              JANAS_IQ1_M};
    uint8_t *wq[7];
    size_t wqbytes[7];
    for (int k = 0; k < 7; k++) {
        size_t bs = janas_qtype_block_size(tq[k]);
        wqbytes[k] = (size_t)MAXR * (MAXC / JANAS_QK) * bs + 4096;
        wqbytes[k] = (wqbytes[k] + 4095) / 4096 * 4096;
        wq[k] = janas_aligned_alloc(4096, wqbytes[k]);
        for (size_t i = 0; i < wqbytes[k]; i++)
            wq[k][i] = (uint8_t)rnd();
        for (size_t b = 0; b + bs <= wqbytes[k] - 4096; b += bs) {
            uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
            if (tq[k] == JANAS_Q2_K) {
                uint16_t dm =
                    janas_fp32_to_fp16(0.0002f + (rnd() % 1000) * 1e-6f);
                memcpy(wq[k] + b + 80, &d, 2);
                memcpy(wq[k] + b + 82, &dm, 2);
            } else if (tq[k] == JANAS_IQ1_M) {
                for (int s = 0; s < 4; s++) {
                    uint16_t sc;
                    memcpy(&sc, wq[k] + b + 48 + 2 * s, 2);
                    sc = (uint16_t)((sc & 0x0fff) | ((d >> 4 * s & 15) << 12));
                    memcpy(wq[k] + b + 48 + 2 * s, &sc, 2);
                }
            } else {
                memcpy(wq[k] + b, &d, 2);
            }
        }
    }
    for (int k = 0; k < 7; k++)
        if (janas_gpu_import(g, wq[k], wqbytes[k]) != 0) {
            printf("gpu_check: import failed\n");
            return 1;
        }
    if (janas_gpu_import(g, w, wbytes) != 0 ||
        janas_gpu_import(g, w6, w6bytes) != 0 ||
        janas_gpu_import(g, wa, wabytes) != 0 ||
        janas_gpu_import(g, wp, wpbytes) != 0 ||
        janas_gpu_import(g, w5, w5bytes) != 0 ||
        janas_gpu_import(g, w8, w8bytes) != 0 ||
        janas_gpu_import(g, w32[0], w32bytes[0]) != 0 ||
        janas_gpu_import(g, w32[1], w32bytes[1]) != 0 ||
        janas_gpu_import(g, w32[2], w32bytes[2]) != 0 ||
        janas_gpu_import(g, w32[3], w32bytes[3]) != 0 ||
        janas_gpu_import(g, w32[4], w32bytes[4]) != 0 ||
        janas_gpu_import(g, w32[5], w32bytes[5]) != 0 ||
        janas_gpu_import(g, w32[6], w32bytes[6]) != 0 ||
        janas_gpu_import(g, wx, wxbytes) != 0 ||
        janas_gpu_import(g, w3[0], w3bytes) != 0 ||
        janas_gpu_import(g, w3[1], w3bytes) != 0) {
        printf("gpu_check: import failed\n");
        return 1;
    }
    size_t xblocks = (size_t)MAXV * 2 * (MAXC / JANAS_QK);
    struct janas_block_q8k *x = malloc(xblocks * sizeof(*x));
    float *xf = malloc(MAXC * sizeof(float));
    for (size_t v = 0; v < (size_t)MAXV * 2; v++) {
        for (int i = 0; i < MAXC; i++)
            xf[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        janas_q8k_quantize(xf, x + v * (MAXC / JANAS_QK), MAXC);
    }
    /* GPU_CHECK_ENGINE=1: the engine's pool (12 threads on CPUs 0-11,
       the caller pinned to CPU 0), else 4 unpinned threads */
    int cpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int engine = getenv("GPU_CHECK_ENGINE") != NULL;
    struct janas_pool *pool =
        engine ? janas_pool_create(12, cpus) : janas_pool_create(4, NULL);
    if (engine && getenv("GPU_CHECK_PIN"))
        janas_pin_current_thread(0);
    size_t ysz = (size_t)MAXR * MAXV * 2;
    float *ya = malloc(ysz * sizeof(float)), *yb = malloc(ysz * sizeof(float));
    /* the last three: rows ending inside a group of 256, for the blocks
       of 32 or 64 and Q8_0 only (640: qwen4exp's experts' down
       matrices) */
    const size_t shapes[][2] = {{4096, 2048}, {1000, 256},  {128, 4096},
                                {2048, 512},  {4096, 4096}, {777, 1024},
                                {1500, 768}, /* odd blocks per row */
                                {2560, 640},  {300, 896},   {77, 128}};
    const size_t nvs[] = {1, 2, 3, 5, 8, 9, 16, 17, 33, 64};
    int bad = 0, runs = 0;
    /* 0 Q4_K, 1 Q6_K, 2-4 Q6_K_P with three, two and one planes, 5 Q5_K,
       6 Q8_0, 7-10 Q4_0, IQ4_NL, Q4_1, Q5_1, 11 IQ4_XS,
       12 Q3_K, 13 IQ3_S, 14 the aligned Q6_K, 15 MXFP4, 16 Q5_0, 17 Q2_0,
       18-24 Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, IQ1_M */
    static const char *names[] = {
        "Q4_K",   "Q6_K",   "Q6_K_P/3", "Q6_K_P/2", "Q6_K_P/1",
        "Q5_K",   "Q8_0",   "Q4_0",     "IQ4_NL",   "Q4_1",
        "Q5_1",   "IQ4_XS", "Q3_K",     "IQ3_S",    "Q6_K_A",
        "MXFP4",  "Q5_0",   "Q2_0",     "Q2_K",     "IQ2_XXS",
        "IQ2_XS", "IQ2_S",  "IQ3_XXS",  "IQ1_S",    "IQ1_M"};
    for (int q6 = 0; q6 < 25; q6++)
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
            for (size_t k = 0; k < sizeof(nvs) / sizeof(nvs[0]); k++)
                for (int strided = 0; strided < 2; strided++) {
                    /* the blocks of 32 or 64: their table's index */
                    int b32 = q6 >= 7 && q6 <= 10    ? q6 - 7
                              : q6 == 15             ? 4
                              : q6 == 16 || q6 == 17 ? q6 - 11
                                                     : -1;
                    int iq = q6 >= 18 ? q6 - 18 : -1; /* the 256-block kinds */
                    size_t rows = shapes[s][0], cols = shapes[s][1];
                    if (cols % JANAS_QK && b32 < 0 && q6 != 6)
                        continue; /* a tail: blocks of 32 or 64, Q8_0 */
                    /* the activations of a tail row: its group whole */
                    size_t nb = cols / JANAS_QK, nv = nvs[k],
                           nbx = (cols + JANAS_QK - 1) / JANAS_QK;
                    size_t skip = s % 2; /* a row further in, every other
                                            shape */
                    size_t rb32 =
                        b32 >= 0 ? janas_qtype_row_bytes(t32[b32], cols) : 0;
                    struct janas_matvec_task t = {
                        .type = iq >= 0    ? tq[iq]
                                : b32 >= 0 ? t32[b32]
                                : q6 == 14 ? JANAS_Q6_K
                                : q6 == 13 ? JANAS_IQ3_S
                                : q6 == 12 ? JANAS_Q3_K
                                : q6 == 11 ? JANAS_IQ4_XS
                                : q6 == 6  ? JANAS_Q8_0
                                : q6 == 5  ? JANAS_Q5_K
                                : q6 >= 2  ? JANAS_Q6_K_P
                                : q6 == 1  ? JANAS_Q6_K
                                           : JANAS_Q4_K,
                        .w = iq >= 0
                                 ? wq[iq] + skip *
                                                janas_qtype_block_size(tq[iq]) *
                                                nb
                             : b32 >= 0 ? w32[b32] + skip * rb32
                             : q6 == 14 ? w6 + skip * 210 * nb
                             : q6 >= 12 ? w3[q6 - 12] + skip * 110 * nb
                             : q6 == 11 ? wx + skip * 136 * nb
                             : q6 == 6  ? w8 + skip * janas_qtype_row_bytes(
                                                         JANAS_Q8_0, cols)
                             : q6 == 5 ? w5 + skip * 176 * nb
                             : q6 >= 2 ? wp + skip * 82 * nb
                             : q6 == 1 ? w6 + skip * 210 * nb
                                       : w + skip * 144 * nb,
                        .plane = {q6 == 2 || q6 == 3
                                      ? pl1 + skip * JANAS_Q6KP_PLANE * nb
                                      : NULL,
                                  q6 == 2 ? pl0 + skip * JANAS_Q6KP_PLANE * nb
                                          : NULL},
                        .x = x,
                        .rows = rows,
                        .cols = cols,
                        .n_vec = nv,
                        .x_stride = strided ? 2 * nbx : 0,
                        .y_stride = strided ? rows + 5 : 0};
                    size_t ys = strided ? rows + 5 : rows;
                    memset(ya, 0, ysz * sizeof(float));
                    memset(yb, 0, ysz * sizeof(float));
                    t.y = ya;
                    janas_matvec_q4k_group(pool, &t, 1);
                    t.y = yb;
                    if (q6 == 14) {
                        /* the CPU has no aligned copy: all on the GPU */
                        t.type = JANAS_Q6_K_A;
                        t.w = wa + skip * JANAS_Q6KA_BLOCK * nb;
                    }
                    if (getenv("GPU_ONLY") || q6 == 14) {
                        /* every row on the GPU, nothing on the CPU */
                        int waited;
                        if (janas_gpu_begin(g, &t, &rows, 1) != 0 ||
                            janas_gpu_finish(g, &waited) != 0)
                            printf("gpu call failed\n");
                    } else {
                        /* several calls: the share adapts, results must not */
                        for (int c = 0; c < 4; c++) {
                            janas_gpu_matvec_group(g, pool, &t, 1);
                            /* GPU_CHECK_GAP_US: CPU-only work between calls,
                               as in the engine */
                            const char *gap = getenv("GPU_CHECK_GAP_US");
                            if (gap) {
                                struct timespec a, b;
                                clock_gettime(CLOCK_MONOTONIC, &a);
                                do
                                    clock_gettime(CLOCK_MONOTONIC, &b);
                                while ((b.tv_sec - a.tv_sec) * 1000000 +
                                           (b.tv_nsec - a.tv_nsec) / 1000 <
                                       atoi(gap));
                            }
                        }
                    }
                    runs++;
                    for (size_t v = 0; v < nv && bad < 10; v++)
                        for (size_t r = 0; r < rows; r++)
                            if (memcmp(&ya[v * ys + r], &yb[v * ys + r], 4)) {
                                size_t nbad = 0;
                                for (size_t q = 0; q < rows; q++)
                                    nbad += memcmp(&ya[v * ys + q],
                                                   &yb[v * ys + q], 4) != 0;
                                printf("[%zu wrong rows] ", nbad);
                                printf(
                                    "%s %zux%zu nv %zu strided %d: v %zu row %zu: "
                                    "%g vs %g\n",
                                    names[q6], rows, cols, nv, strided, v, r,
                                    ya[v * ys + r], yb[v * ys + r]);
                                for (size_t u = 0;
                                     getenv("GPU_CHECK_DEBUG") && u < 2 * MAXV;
                                     u++)
                                    if (janas_q4k_dot(
                                            (const struct janas_block_q4k *)
                                                    t.w +
                                                r * nb,
                                            x + u * nb, nb) == yb[v * ys + r])
                                        printf("  = row %zu times vector %zu\n",
                                               r, u);
                                bad++;
                                break;
                            }
                }
    /* bf16 products: Qwen3.8-Flash-Next's mixes (10240 wide, 320 wide), a
       tail of three past the last eight, a row shorter than eight; single
       vectors too (as on a card with memory of its own) */
    int b16_runs = 0, b16_refused = 0;
    /* freed after the GPU, which imported them */
    uint16_t *w16 = NULL;
    float *bx = NULL, *by = NULL;
    if (janas_bf16_lanes()) {
        static const size_t bsh[][2] = {{320, 10240}, {10240, 320},
                                        {2560, 2560}, {512, 2560},
                                        {100, 1003},  {37, 13}};
        static const size_t bnv[] = {1, 2, 3, 4, 5, 8, 17, 63};
        enum { BW = 2560 * 2560, BX = 63 * 10240 };
        size_t xbytes = ((size_t)BX * sizeof(float) + 4095) / 4096 * 4096;
        w16 = janas_aligned_alloc(4096, (size_t)BW * 2);
        bx = janas_aligned_alloc(4096, xbytes);
        by = janas_aligned_alloc(4096, xbytes);
        float *bxs = malloc(xbytes), *bys = malloc(xbytes),
              *bref = malloc(xbytes);
        for (size_t i = 0; i < BW; i++) {
            float f = (float)((int)(rnd() % 20001) - 10000) * 1e-5f;
            uint32_t u;
            memcpy(&u, &f, 4);
            w16[i] = (uint16_t)(u >> 16);
        }
        for (size_t i = 0; i < BX; i++)
            bx[i] = bxs[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        janas_gpu_set_min_n(g, 1); /* single vectors too */
        if (janas_gpu_import(g, w16, (size_t)BW * 2) != 0 ||
            janas_gpu_import(g, bx, xbytes) != 0 ||
            janas_gpu_import(g, by, xbytes) != 0) {
            printf("gpu_check: bf16 import failed\n");
            bad++;
        }
        for (size_t s = 0; !bad && s < sizeof(bsh) / sizeof(*bsh); s++)
            for (size_t k = 0; k < sizeof(bnv) / sizeof(*bnv); k++)
                for (int staged = 0; staged < 2; staged++) {
                    size_t rows = bsh[s][0], cols = bsh[s][1], nv = bnv[k];
                    float scale = staged ? 0.25f : 1.0f;
                    const float *xin = staged ? bxs : bx;
                    float *yout = staged ? bys : by;
                    for (size_t v = 0; v < nv; v++)
                        for (size_t r = 0; r < rows; r++)
                            bref[v * rows + r] =
                                janas_bf16_dot(w16 + r * cols, xin + v * cols,
                                               (uint32_t)cols) *
                                scale;
                    memset(yout, 0, nv * rows * sizeof(float));
                    if (janas_gpu_bf16_mv(g, w16, xin, yout, rows, cols, nv,
                                          scale) != 0) {
                        b16_refused++;
                        continue;
                    }
                    b16_runs++;
                    size_t nbad = 0;
                    for (size_t i = 0; i < nv * rows; i++)
                        nbad += memcmp(&bref[i], &yout[i], 4) != 0;
                    if (nbad && bad < 10) {
                        printf("[%zu wrong of %zu] bf16 %zux%zu nv %zu %s: "
                               "%g vs %g\n",
                               nbad, nv * rows, rows, cols, nv,
                               staged ? "through the stages" : "in place",
                               bref[0], yout[0]);
                        bad++;
                    }
                }
        free(bxs);
        free(bys);
        free(bref);
    }
    printf("gpu_check (%s, subgroups of %u): %d shapes, bf16 %d (%d "
           "refused), %s\n",
           janas_gpu_name(g), janas_gpu_subgroup(g), runs, b16_runs,
           b16_refused, bad || b16_refused ? "FAILED" : "ok");
    janas_gpu_destroy(g);
    janas_aligned_free(w16);
    janas_aligned_free(bx);
    janas_aligned_free(by);
    janas_pool_destroy(pool);
    janas_aligned_free(w);
    janas_aligned_free(w6);
    janas_aligned_free(wa);
    janas_aligned_free(wp);
    janas_aligned_free(w5);
    janas_aligned_free(w8);
    for (int k = 0; k < 7; k++)
        janas_aligned_free(w32[k]);
    for (int k = 0; k < 7; k++)
        janas_aligned_free(wq[k]);
    janas_aligned_free(wx);
    janas_aligned_free(w3[0]);
    janas_aligned_free(w3[1]);
    free(x);
    free(xf);
    free(ya);
    free(yb);
    return bad || b16_refused ? 1 : 0;
}
