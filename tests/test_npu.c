/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_npu.c - the NPU layer's parts that need no NPU: the rows' INT8
 * quantization (scales in FP16, q s within half a step of the weight) and
 * the IR writer (the compiler's input laid out as the driver reads it).
 * The NPU itself: npu_check.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/npu.h"
#include "common/npu_ir.h"

static float h2f(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31,
             mant = h & 0x3ffu, b;
    if (!e && !mant)
        b = sign;
    else if (!e) {
        float f = ldexpf((float)mant, -24);
        return sign ? -f : f;
    } else
        b = sign | ((e - 15 + 127) << 23) | (mant << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}

int main(void)
{
    int failures = 0;

    /* quantization */
    enum { N = 7, K = 300 };
    float *w = malloc(sizeof(float) * N * K);
    int8_t *q = malloc(N * K);
    uint16_t s[N];
    unsigned r = 12345;
    for (int i = 0; i < N * K; i++) {
        r = r * 1103515245u + 12345u;
        w[i] = ((float)(r >> 8) / (float)(1u << 24) - 0.5f) * (1 + i / K);
    }
    for (int k = 0; k < K; k++)
        w[3 * K + k] = 0;                          /* a row of zeros */
    w[5 * K + 17] = 1e-6f, w[5 * K + 18] = -3e-5f; /* tiny scale */
    for (int k = 19; k < K; k++)
        w[5 * K + k] = 0;
    for (int k = 0; k < 17; k++)
        w[5 * K + k] = 0;
    janas_npu_quant_rows(w, N, K, q, s);
    for (int i = 0; i < N; i++) {
        float sc = h2f(s[i]), worst = 0;
        for (int k = 0; k < K; k++) {
            float e = fabsf(q[i * K + k] * sc - w[i * K + k]);
            worst = fmaxf(worst, e);
            if (q[i * K + k] == -128)
                failures++, printf("FAIL row %d: -128\n", i);
        }
        float amax = 0;
        for (int k = 0; k < K; k++)
            amax = fmaxf(amax, fabsf(w[i * K + k]));
        /* half a step: the scale rounds up, nothing is clipped */
        float bound = sc * 0.5f * 1.0001f;
        if (sc * 127 < amax)
            failures++, printf("FAIL row %d: scale below the row's max\n", i);
        if (i == 3 && (sc != 0 || worst != 0))
            failures++, printf("FAIL zero row: scale %g\n", sc);
        if (worst > bound)
            failures++, printf("FAIL row %d: error %g > %g\n", i, worst, bound);
    }

    /* the IR writer */
    struct npu_ir g;
    npu_ir_begin(&g, "t");
    int xd[2] = {4, 8}, wd[2] = {3, 8};
    int8_t wq[24] = {0};
    int x = npu_ir_param(&g, "x", "f32", 2, xd);
    int c = npu_ir_const(&g, "w", "i8", 2, wd, wq, sizeof wq);
    int m = npu_ir_matmul_tb(&g, "mm", npu_ir_convert(&g, "x16", x, "f16"),
                             npu_ir_convert(&g, "w16", c, "f16"), NULL);
    npu_ir_result(&g, m);
    if (npu_ir_end(&g) || g.nl != 6 || g.ne != 5 || g.nbin != 24)
        failures++, printf("FAIL IR: %d layers, %d edges\n", g.nl, g.ne);
    if (!strstr(g.xml, "<edge from-layer=\"4\" from-port=\"2\" to-layer=\"5\" "
                       "to-port=\"0\"/>"))
        failures++, printf("FAIL IR: matmul's edge\n");
    int bad[2] = {3, 9};
    struct npu_ir h;
    npu_ir_begin(&h, "u");
    int a = npu_ir_param(&h, "a", "f16", 2, xd),
        b = npu_ir_param(&h, "b", "f16", 2, bad);
    if (npu_ir_matmul_tb(&h, "mm", a, b, NULL) >= 0 || !npu_ir_end(&h))
        failures++, printf("FAIL IR: mismatched k accepted\n");
    npu_ir_free(&h);
    size_t bytes = 0;
    uint8_t *mb = npu_ir_model(&g, 8, 3, &bytes);
    uint64_t sx, sb;
    uint32_t parts;
    uint16_t maj, min;
    memcpy(&maj, mb, 2), memcpy(&min, mb + 2, 2), memcpy(&parts, mb + 4, 4);
    memcpy(&sx, mb + 8, 8);
    memcpy(&sb, mb + 16 + sx, 8);
    if (!mb || maj != 8 || min != 3 || parts != 2 || sx != g.len || sb != 24 ||
        bytes != 24 + sx + 24)
        failures++, printf("FAIL model layout\n");
    free(mb);
    npu_ir_free(&g);
    free(w);
    free(q);
    printf("%s\n", failures ? "test_npu: FAILED" : "test_npu: ok");
    return failures != 0;
}
