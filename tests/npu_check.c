/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_check.c - products on the NPU (common/npu.h) against the CPU in
 * double on the same weights: INT8 (dequantized as the NPU computes) and
 * FP16, a few shapes, two products in flight at once with x shared.
 * Skips (exit 0) when no NPU is usable. Errors relative to the outputs'
 * rms; FP16 arithmetic: some 1e-3 is right, the bound is 1e-2.
 *
 * Usage: npu_check [m n k]   (default: the shapes below)
 * NPU_CHECK_CACHE=<dir> keeps the compiled products there (a second run
 * loads them).
 * JANAS_NPU_LOADER=<libze_loader.so.1> for a loader not on the path.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/npu.h"

static uint64_t rng = 88172645463325252ull;

static float rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float)(rng >> 40) / (float)(1u << 24) - 0.5f;
}

static float h2f(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31,
             mant = h & 0x3ffu, b;
    if (!e) {
        float f = ldexpf((float)mant, -24);
        return sign ? -f : f;
    }
    b = sign | ((e - 15 + 127) << 23) | (mant << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint16_t f2h(float f) /* for the FP16 weights: truncation is fine */
{
    uint32_t b;
    memcpy(&b, &f, 4);
    int e = (int)((b >> 23) & 0xff) - 127 + 15;
    if (e <= 0)
        return (uint16_t)((b >> 16) & 0x8000u);
    return (uint16_t)(((b >> 16) & 0x8000u) | ((uint32_t)e << 10) |
                      ((b >> 13) & 0x3ffu));
}

/* rms-relative worst error of y (m x n) against x wd^T in double */
static double check(const float *x, const float *wd, const float *y, int m,
                    int n, int k)
{
    double worst = 0, ss = 0;
    int rows = m < 32 ? m : 32;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int t = 0; t < k; t++)
                s += (double)x[(size_t)i * k + t] * wd[(size_t)j * k + t];
            worst = fmax(worst, fabs(y[(size_t)i * n + j] - s));
            ss += s * s;
        }
    return worst / sqrt(ss / ((double)rows * n) + 1e-30);
}

/* a feed-forward in one run (janas_npu_ffn_create) against the CPU in
   double on the same weights, dequantized as the NPU does; rms-relative
   worst error of y, or -1 if it did not run */
static double check_ffn(struct janas_npu *u, int m, int d, int ff, int act)
{
    char err[256];
    size_t nw = (size_t)ff * d;
    float *w[3];
    int8_t *q[3];
    uint16_t *sc[3];
    int rows[3] = {ff, ff, d}, cols[3] = {d, d, ff};
    for (int k = 0; k < 3; k++) {
        w[k] = malloc(sizeof(float) * nw), q[k] = malloc(nw);
        sc[k] = malloc(2 * (size_t)rows[k]);
        for (size_t i = 0; i < nw; i++)
            w[k][i] = rnd() * 0.15f;
        janas_npu_quant_rows(w[k], rows[k], cols[k], q[k], sc[k]);
        for (size_t i = 0; i < nw; i++) /* as the NPU: FP16 products */
            w[k][i] = h2f(f2h((float)q[k][i] * h2f(sc[k][i / cols[k]])));
    }
    float *x = janas_npu_alloc(u, sizeof(float) * m * d),
          *y = janas_npu_alloc(u, sizeof(float) * m * d);
    for (size_t i = 0; i < (size_t)m * d; i++)
        x[i] = rnd() * 2;
    struct janas_npu_mm *p =
        janas_npu_ffn_create(u, m, d, ff, act, q[0], sc[0], q[1], sc[1], q[2],
                             sc[2], x, y, err, sizeof err);
    double worst = -1;
    if (!p)
        printf("FAIL feed-forward %dx%dx%d: %s\n", m, d, ff, err);
    else if (janas_npu_mm_run(p) == 0) {
        double ss = 0;
        worst = 0;
        double *h = malloc(sizeof(double) * ff);
        int rows_checked = m < 16 ? m : 16;
        for (int i = 0; i < rows_checked; i++) {
            const float *xi = x + (size_t)i * d;
            for (int f = 0; f < ff; f++) {
                double g = 0, up = 0;
                for (int t = 0; t < d; t++) {
                    g += (double)xi[t] * w[0][(size_t)f * d + t];
                    up += (double)xi[t] * w[1][(size_t)f * d + t];
                }
                double a = act == JANAS_NPU_SILU
                               ? g / (1 + exp(-g))
                               : 0.5 * g *
                                     (1 + tanh(0.7978845608 *
                                               (g + 0.044715 * g * g * g)));
                h[f] = a * up;
            }
            for (int o = 0; o < d; o++) {
                double v = 0;
                for (int f = 0; f < ff; f++)
                    v += h[f] * w[2][(size_t)o * ff + f];
                worst = fmax(worst, fabs(y[(size_t)i * d + o] - v));
                ss += v * v;
            }
        }
        worst /= sqrt(ss / ((double)rows_checked * d) + 1e-30);
        free(h);
    }
    janas_npu_mm_destroy(p);
    janas_npu_free(u, x);
    janas_npu_free(u, y);
    for (int k = 0; k < 3; k++)
        free(w[k]), free(q[k]), free(sc[k]);
    return worst;
}

int main(int argc, char **argv)
{
    char err[256];
    struct janas_npu *u = janas_npu_create(err, sizeof err);
    if (!u) {
        printf("npu_check: skipped (%s)\n", err);
        return 0;
    }
    printf("NPU: %s\n", janas_npu_name(u));
    const char *cache = getenv("NPU_CHECK_CACHE");
    if (cache && janas_npu_set_cache(u, cache))
        printf("cache %s not usable\n", cache);
    int shapes[][3] = {{256, 2048, 2048}, {64, 512, 1536}, {256, 1000, 768}};
    int ns = 3;
    if (argc == 4) {
        shapes[0][0] = atoi(argv[1]), shapes[0][1] = atoi(argv[2]);
        shapes[0][2] = atoi(argv[3]), ns = 1;
    }
    int failures = 0;
    for (int si = 0; si < ns; si++) {
        int m = shapes[si][0], n = shapes[si][1], k = shapes[si][2];
        float *w = malloc(sizeof(float) * n * k),
              *wd = malloc(sizeof(float) * n * k),
              *w2d = malloc(sizeof(float) * n * k);
        int8_t *q = malloc((size_t)n * k);
        uint16_t *sc = malloc(2 * (size_t)n), *wh = malloc(2 * (size_t)n * k);
        for (size_t i = 0; i < (size_t)n * k; i++)
            w[i] = rnd() * 0.2f * (1 + (float)(i / k % 5));
        janas_npu_quant_rows(w, n, k, q, sc);
        for (size_t i = 0; i < (size_t)n * k; i++) {
            /* the NPU: the integer in FP16 times the FP16 scale, in FP16 */
            uint16_t p = f2h((float)q[i] * h2f(sc[i / k]));
            wd[i] = h2f(p);
            wh[i] = f2h(w[(size_t)n * k - 1 - i]);
            w2d[i] = h2f(wh[i]);
        }
        float *x = janas_npu_alloc(u, sizeof(float) * m * k),
              *y = janas_npu_alloc(u, sizeof(float) * m * n),
              *y2 = janas_npu_alloc(u, sizeof(float) * m * n);
        for (size_t i = 0; i < (size_t)m * k; i++)
            x[i] = rnd() * 2;
        struct janas_npu_mm *p =
                                janas_npu_mm_create(u, m, n, k, JANAS_NPU_I8, q,
                                                    sc, x, y, err, sizeof err),
                            *p2 = p ? janas_npu_mm_create(
                                          u, m, n, k, JANAS_NPU_F16, wh, NULL,
                                          x, y2, err, sizeof err)
                                    : NULL;
        if (!p || !p2) {
            printf("FAIL %dx%dx%d: %s\n", m, n, k, err);
            failures++;
        } else if (janas_npu_mm_begin(p) || janas_npu_mm_begin(p2) ||
                   janas_npu_mm_finish(p) || janas_npu_mm_finish(p2)) {
            printf("FAIL %dx%dx%d: run\n", m, n, k);
            failures++;
        } else {
            double e8 = check(x, wd, y, m, n, k),
                   e16 = check(x, w2d, y2, m, n, k);
            int bad = !(e8 < 1e-2) || !(e16 < 1e-2);
            failures += bad;
            printf("%s %dx%dx%d: INT8 %.2e, FP16 %.2e of rms\n",
                   bad ? "FAIL" : "ok  ", m, n, k, e8, e16);
            /* a second run on new x: the bound arguments are re-read */
            for (size_t i = 0; i < (size_t)m * k; i++)
                x[i] = rnd();
            if (janas_npu_mm_run(p)) {
                printf("FAIL rerun\n");
                failures++;
            } else if (!(check(x, wd, y, m, n, k) < 1e-2)) {
                printf("FAIL rerun %dx%dx%d: x not re-read\n", m, n, k);
                failures++;
            }
        }
        janas_npu_mm_destroy(p);
        janas_npu_mm_destroy(p2);
        janas_npu_free(u, x);
        janas_npu_free(u, y);
        janas_npu_free(u, y2);
        free(w), free(wd), free(w2d), free(q), free(sc), free(wh);
    }
    /* feed-forwards in one run: FP16 inside (the intermediate too), so a
       looser bound than the products' */
    for (int act = 0; act < 2 && argc != 4; act++) {
        double e = check_ffn(u, 256, 512, 1536, act);
        int bad = !(e >= 0 && e < 3e-2);
        failures += bad;
        printf("%s feed-forward 256x512x1536 %s: %.2e of rms\n",
               bad ? "FAIL" : "ok  ", act ? "GELU" : "SiLU", e);
    }
    int np, nc;
    double cs;
    janas_npu_stats(u, &np, &nc, &cs);
    printf("%d products (%d from the cache) in %.2f s\n", np, nc, cs);
    janas_npu_destroy(u);
    printf("%s\n", failures ? "npu_check: FAILED" : "npu_check: ok");
    return failures != 0;
}
