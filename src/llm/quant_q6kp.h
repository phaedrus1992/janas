/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_q6kp.h - where a weight's bits sit in Q6_K's bit planes, for quant.c
 * and quant_q6k.c (private to the engine).
 */
#ifndef JANAS_LLM_QUANT_Q6KP_H
#define JANAS_LLM_QUANT_Q6KP_H

#include "quant.h"

/* Byte of weight v in a two-bit plane, and the shift of its bits. */
static inline int q6kp_index(int v, int *shift)
{
    int n = v / 128, part = (v % 128) / 32, l = v % 32;
    *shift = 2 * part;
    return 32 * n + l;
}

/* Weight v of a block from its planes at a level (3: all three, 2: the base
   and a1, 1: the base alone), on a scale of 0..127 whose 64 is zero: the
   centre of the range the missing bits cover, half a step. */
static inline int q6kp_value2(const struct janas_block_q6kp *b,
                              const uint8_t *a1, const uint8_t *a0, int v,
                              int level)
{
    int sh, i = q6kp_index(v, &sh);
    int q = (b->q[i] >> sh) & 3;
    if (level >= 2)
        q = (q << 2) | ((a1[i] >> sh) & 3);
    if (level >= 3)
        q = (q << 2) | ((a0[i] >> sh) & 3);
    return level == 3 ? 2 * q : level == 2 ? 8 * q + 3 : 32 * q + 15;
}

#endif
