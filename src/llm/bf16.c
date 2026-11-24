/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bf16.c - products of bf16 weights with float vectors (bf16.h): Gemma 4's
 * per-layer projection (model_ple.c) and Qwen3.8-Flash-Next's mixes of the
 * streams and n-gram embeddings (model_hc.c).
 */
#include "bf16.h"

#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include "quant.h"
#include "vmath.h"

#if defined(__x86_64__)
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static float
dot_avx2(const uint16_t *w, const float *x, uint32_t n)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(w + i));
        __m256 wf = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
    }
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++)
        r += janas_u2f((uint32_t)w[i] << 16) * x[i];
    return r;
}

/* four tokens against one row, the row converted once: for each token the
   same accumulator and the same order as dot_avx2, so the same bits */
__attribute__((target("avx2,fma"))) static void
dot4_avx2(const uint16_t *w, const float *const x[4], uint32_t n, float out[4])
{
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(w + i));
        __m256 wf = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
#pragma GCC unroll 4
        for (int t = 0; t < 4; t++)
            acc[t] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x[t] + i), acc[t]);
    }
#pragma GCC unroll 4
    for (int t = 0; t < 4; t++) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[t]),
                              _mm256_extractf128_ps(acc[t], 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        float r = _mm_cvtss_f32(s);
        for (uint32_t k = i; k < n; k++)
            r += janas_u2f((uint32_t)w[k] << 16) * x[t][k];
        out[t] = r;
    }
}

/*
 * Rows shorter than this many bytes (a mix's up, 320 numbers) have their
 * weights asked for this far ahead, into L1: in memory those are the next
 * rows', which the hardware prefetcher, following each row, did not fetch
 * in time. 10240 x 320 on 12 threads, from RAM: BF16 104 us a product
 * without, 92 with 2 KB ahead, 89.5 with 4 KB; Q8_0 66 and 57. Long rows
 * (a mix's down, 10240 numbers) are fetched in time already, and with it
 * slow down a little (23 Oct 2026).
 */
#define SHORT_ROW_AHEAD 2048

/* four rows, row elements apart, against one vector: each row with
   dot_avx2's accumulator, order and tail, so the same bits, the four
   chains of fused multiply-adds side by side; pf bytes ahead (0: none) */
__attribute__((target("avx2,fma"), always_inline)) static inline void
dot4r_body(const uint16_t *w, size_t row, const float *x, uint32_t n,
           float out[4], int pf)
{
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
#pragma GCC unroll 4
        for (int r = 0; r < 4; r++) {
            if (pf && (i & 31) == 0) /* a line a row every 32 numbers */
                _mm_prefetch((const char *)(w + r * row + i) + pf, _MM_HINT_T0);
            __m128i h = _mm_loadu_si128(
                (const __m128i *)(const void *)(w + r * row + i));
            __m256 wf = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
            acc[r] = _mm256_fmadd_ps(wf, xv, acc[r]);
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < 4; r++) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[r]),
                              _mm256_extractf128_ps(acc[r], 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        float v = _mm_cvtss_f32(s);
        for (uint32_t k = i; k < n; k++)
            v += janas_u2f((uint32_t)w[r * row + k] << 16) * x[k];
        out[r] = v;
    }
}

__attribute__((target("avx2,fma"))) static void
dot4r_avx2(const uint16_t *w, size_t row, const float *x, uint32_t n,
           float out[4])
{
    dot4r_body(w, row, x, n, out, 0);
}

__attribute__((target("avx2,fma"))) static void
dot4r_ahead_avx2(const uint16_t *w, size_t row, const float *x, uint32_t n,
                 float out[4])
{
    dot4r_body(w, row, x, n, out, SHORT_ROW_AHEAD);
}
#endif

/* ------------------------------------------------- Q8_0 x float */

#define Q8W_BLOCK 34 /* fp16 scale + 32 int8 */

static float q8w_scale(const uint8_t *b)
{
    uint16_t h;
    memcpy(&h, b, 2);
    return janas_fp16_to_fp32(h);
}

/* the eight lanes added as janas_hsum8 does */
static float sum8(const float a[8])
{
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

static float q8w_dot_ref(const uint8_t *w, const float *x, uint32_t n)
{
    float acc[8] = {0};
    for (uint32_t b = 0; b < n / 32; b++, w += Q8W_BLOCK, x += 32) {
        const int8_t *q = (const int8_t *)(w + 2);
        float d = q8w_scale(w), bs[8] = {0};
        for (int g = 0; g < 4; g++)
            for (int k = 0; k < 8; k++)
                bs[k] = fmaf((float)q[8 * g + k], x[8 * g + k], bs[k]);
        for (int k = 0; k < 8; k++)
            acc[k] = fmaf(bs[k], d, acc[k]);
    }
    return sum8(acc);
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static float hsum_avx2(__m256 acc)
{
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* nr rows w[r] against nx vectors x[t] (nr, nx constants where it is
   inlined), out[r * nx + t]: each pair the reference's order; the rows and
   vectors side by side only so that their chains of fused multiply-adds
   overlap. The scale by F16C, exact as janas_fp16_to_fp32. The loops over
   rows and vectors unrolled: rolled, GCC kept the accumulators in memory,
   every fused multiply-add a load and a store, and the kernel ran at a
   quarter of its speed (40 GB/s from L3 on 12 threads, 23 Oct 2026).
   pf: SHORT_ROW_AHEAD's prefetch, or 0. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q8w_kern(const uint8_t *const w[], int nr, const float *const x[], int nx,
         uint32_t n, float *out, int pf)
{
    __m256 acc[4][4];
#pragma GCC unroll 4
    for (int r = 0; r < nr; r++)
#pragma GCC unroll 4
        for (int t = 0; t < nx; t++)
            acc[r][t] = _mm256_setzero_ps();
    for (uint32_t b = 0; b < n / 32; b++) {
        __m256 d[4], bs[4][4];
#pragma GCC unroll 4
        for (int r = 0; r < nr; r++) {
            uint16_t h;
            if (pf && (b & 1) == 0) /* a line a row every 68 bytes */
                _mm_prefetch((const char *)(w[r] + (size_t)b * Q8W_BLOCK) + pf,
                             _MM_HINT_T0);
            memcpy(&h, w[r] + (size_t)b * Q8W_BLOCK, 2);
            d[r] = _mm256_set1_ps(_cvtsh_ss(h));
#pragma GCC unroll 4
            for (int t = 0; t < nx; t++)
                bs[r][t] = _mm256_setzero_ps();
        }
#pragma GCC unroll 4
        for (int g = 0; g < 4; g++) {
            __m256 xv[4];
#pragma GCC unroll 4
            for (int t = 0; t < nx; t++)
                xv[t] = _mm256_loadu_ps(x[t] + 32 * b + 8 * g);
#pragma GCC unroll 4
            for (int r = 0; r < nr; r++) {
                __m128i q8 = _mm_loadl_epi64(
                    (const __m128i *)(const void *)(w[r] +
                                                    (size_t)b * Q8W_BLOCK + 2 +
                                                    8 * g));
                __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
#pragma GCC unroll 4
                for (int t = 0; t < nx; t++)
                    bs[r][t] = _mm256_fmadd_ps(qf, xv[t], bs[r][t]);
            }
        }
#pragma GCC unroll 4
        for (int r = 0; r < nr; r++)
#pragma GCC unroll 4
            for (int t = 0; t < nx; t++)
                acc[r][t] = _mm256_fmadd_ps(bs[r][t], d[r], acc[r][t]);
    }
#pragma GCC unroll 4
    for (int r = 0; r < nr; r++)
#pragma GCC unroll 4
        for (int t = 0; t < nx; t++)
            out[r * nx + t] = hsum_avx2(acc[r][t]);
}

__attribute__((target("avx2,fma,f16c"))) static void
q8w_1x1(const uint8_t *w, const float *x, uint32_t n, float *out)
{
    q8w_kern(&w, 1, &x, 1, n, out, 0);
}

__attribute__((target("avx2,fma,f16c"))) static void
q8w_1x4(const uint8_t *w, const float *const x[4], uint32_t n, float *out)
{
    q8w_kern(&w, 1, x, 4, n, out, 0);
}

__attribute__((target("avx2,fma,f16c"))) static void
q8w_4x1(const uint8_t *const w[4], const float *x, uint32_t n, float *out)
{
    q8w_kern(w, 4, &x, 1, n, out, 0);
}

__attribute__((target("avx2,fma,f16c"))) static void
q8w_4x1_ahead(const uint8_t *const w[4], const float *x, uint32_t n, float *out)
{
    q8w_kern(w, 4, &x, 1, n, out, SHORT_ROW_AHEAD);
}

/* AVX2, FMA and F16C, asked once; atomic, as every thread of a product
   may be the first to ask */
static int q8w_lanes(void)
{
    static atomic_int lanes = -1;
    int l = atomic_load_explicit(&lanes, memory_order_relaxed);
    if (l < 0) {
        l = janas_bf16_lanes() && __builtin_cpu_supports("f16c");
        atomic_store_explicit(&lanes, l, memory_order_relaxed);
    }
    return l;
}
#endif

float janas_q8w_dot(const uint8_t *w, const float *x, uint32_t n)
{
#if defined(__x86_64__)
    if (q8w_lanes()) {
        float o;
        q8w_1x1(w, x, n, &o);
        return o;
    }
#endif
    return q8w_dot_ref(w, x, n);
}

void janas_q8w_dot4(const uint8_t *w, const float *const x[4], uint32_t n,
                    float out[4])
{
#if defined(__x86_64__)
    if (q8w_lanes()) {
        q8w_1x4(w, x, n, out);
        return;
    }
#endif
    for (int t = 0; t < 4; t++)
        out[t] = q8w_dot_ref(w, x[t], n);
}

void janas_q8w_dot4r(const uint8_t *w, size_t row, const float *x, uint32_t n,
                     float out[4])
{
#if defined(__x86_64__)
    if (q8w_lanes()) {
        const uint8_t *ws[4] = {w, w + row, w + 2 * row, w + 3 * row};
        if (row < SHORT_ROW_AHEAD)
            q8w_4x1_ahead(ws, x, n, out);
        else
            q8w_4x1(ws, x, n, out);
        return;
    }
#endif
    for (int r = 0; r < 4; r++)
        out[r] = q8w_dot_ref(w + r * row, x, n);
}

#if defined(__x86_64__)
__attribute__((target("avx2"))) static size_t
round_bf16_avx2(const float *x, float *xb, size_t n)
{
    const __m256i one = _mm256_set1_epi32(1), half = _mm256_set1_epi32(0x7fff),
                  hi = _mm256_set1_epi32((int)0xffff0000u),
                  abs = _mm256_set1_epi32(0x7fffffff),
                  inf = _mm256_set1_epi32(0x7f800000);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i u = _mm256_loadu_si256((const __m256i *)(const void *)(x + i));
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
        __m256i r = _mm256_and_si256(
            _mm256_add_epi32(u, _mm256_add_epi32(half, lsb)), hi);
        __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(u, abs), inf);
        _mm256_storeu_si256((__m256i *)(void *)(xb + i),
                            _mm256_blendv_epi8(r, u, nan));
    }
    return i;
}
#endif

void janas_round_bf16_n(const float *x, float *xb, size_t n)
{
    size_t i = 0;
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2"))
        i = round_bf16_avx2(x, xb, n);
#endif
    for (; i < n; i++) {
        uint32_t u;
        memcpy(&u, x + i, 4);
        if ((u & 0x7fffffffu) <= 0x7f800000u) /* not a NaN */
            u = (u + (0x7fffu + ((u >> 16) & 1u))) & 0xffff0000u;
        memcpy(xb + i, &u, 4);
    }
}

int janas_bf16_lanes(void)
{
#if defined(__x86_64__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

float janas_bf16_dot(const uint16_t *w, const float *x, uint32_t n)
{
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return dot_avx2(w, x, n);
#endif
    float r = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        r += janas_u2f((uint32_t)w[i] << 16) * x[i];
    return r;
}

void janas_bf16_dot4r(const uint16_t *w, size_t row, const float *x, uint32_t n,
                      float out[4])
{
#if defined(__x86_64__)
    if (janas_bf16_lanes()) {
        if (row * sizeof(uint16_t) < SHORT_ROW_AHEAD)
            dot4r_ahead_avx2(w, row, x, n, out);
        else
            dot4r_avx2(w, row, x, n, out);
        return;
    }
#endif
    for (int r = 0; r < 4; r++)
        out[r] = janas_bf16_dot(w + r * row, x, n);
}

void janas_bf16_dot4(const uint16_t *w, const float *const x[4], uint32_t n,
                     float out[4])
{
#if defined(__x86_64__)
    if (janas_bf16_lanes()) {
        dot4_avx2(w, x, n, out);
        return;
    }
#endif
    for (int t = 0; t < 4; t++)
        out[t] = janas_bf16_dot(w, x[t], n);
}
