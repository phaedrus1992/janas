/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_mxfp4.c - MXFP4 blocks rewritten as Q8_0 (janas_mxfp4_to_q8_0, used
 * by gguf2jns for unsloth's Qwen3-Coder-Next files, issue #18) give every
 * weight exactly: each code's value times the block's power of two, as
 * ggml's dequantize_row_mxfp4 computes it, for powers from 2^15 down to
 * fp16's least, 2^-24 (normal and subnormal scales); a block below that is
 * zeros and counted, a scale beyond fp16 refused.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "llm/quant.h"

static const int8_t kv[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                              0, -1, -2, -3, -4, -6, -8, -12};

/* ggml's value of weight i of an MXFP4 block */
static float ref(const uint8_t *b, int i)
{
    int code = i < 16 ? b[1 + i] & 15 : b[1 + i - 16] >> 4;
    return (float)kv[code] * ldexpf(1.0f, (int)b[0] - 128);
}

int main(void)
{
    /* scales: 2^0, 2^-8, 2^15 (the largest), 2^-14 (the least normal
       fp16), 2^-15 and 2^-24 (subnormal), 2^-25 (below: zeros) */
    static const uint8_t es[] = {128, 120, 143, 114, 113, 104, 103};
    enum { N = sizeof(es) };
    uint8_t in[N * JANAS_MXFP4_BLOCK];
    unsigned seed = 12345;
    for (int b = 0; b < N; b++) {
        uint8_t *x = in + b * JANAS_MXFP4_BLOCK;
        x[0] = es[b];
        for (int j = 0; j < 16; j++) {
            seed = seed * 1103515245u + 12345u;
            x[1 + j] = (uint8_t)(seed >> 16);
        }
    }
    in[1] = 0x70; /* codes 0 and 7 (12) in the first block, whatever came */
    struct janas_block_q8_0 out[N];
    size_t tiny = 0;
    int rc = 0;
    if (janas_mxfp4_to_q8_0(in, out, N, &tiny) != 0) {
        printf("FAIL: valid scales refused\n");
        return 1;
    }
    for (int b = 0; b < N; b++) {
        const uint8_t *x = in + b * JANAS_MXFP4_BLOCK;
        float d = janas_fp16_to_fp32(out[b].d);
        for (int i = 0; i < 32; i++) {
            float want = es[b] < 104 ? 0.0f : ref(x, i);
            float got = (float)out[b].qs[i] * d;
            if (got != want) {
                printf("FAIL: block %d (2^%d) weight %d: %g, not %g\n", b,
                       (int)es[b] - 128, i, got, want);
                rc = 1;
                break;
            }
        }
    }
    if (tiny != 1) {
        printf("FAIL: %zu blocks counted below fp16's scales, not 1\n", tiny);
        rc = 1;
    }
    /* beyond fp16: 2^16, and E8M0's NaN */
    for (int k = 0; k < 2; k++) {
        uint8_t bad[JANAS_MXFP4_BLOCK] = {k ? 0xFF : 144};
        if (janas_mxfp4_to_q8_0(bad, out, 1, &tiny) != -1) {
            printf("FAIL: scale byte %d not refused\n", bad[0]);
            rc = 1;
        }
    }
    if (!rc)
        printf("test_mxfp4: ok\n");
    return rc;
}
