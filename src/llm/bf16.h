/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bf16.h - products of bf16 weights with float vectors, as the CPU computes
 * them: with AVX2 and FMA in eight lanes, each the fused multiply-adds of
 * its own elements (8k + lane) in order, then the eight added as
 * janas_hsum8 does, ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)), and
 * the elements past the last eight multiplied and added one at a time;
 * without them one at a time throughout. The GPU computes the eight lanes'
 * way bit for bit (bf16_matvec.comp, gpu.h janas_gpu_bf16_mv): only where
 * the CPU does too (janas_bf16_lanes).
 */
#ifndef JANAS_LLM_BF16_H
#define JANAS_LLM_BF16_H

#include <stddef.h>
#include <stdint.h>

/* w . x over n elements, w as bf16 bits (the high half of a float) */
float janas_bf16_dot(const uint16_t *w, const float *x, uint32_t n);

/* The same for four vectors against one row, read and converted once: for
   each vector the same bits as janas_bf16_dot. */
void janas_bf16_dot4(const uint16_t *w, const float *const x[4], uint32_t n,
                     float out[4]);
/* four rows, row elements apart from w, against one vector: out[r] the bits
   of janas_bf16_dot(w + r * row, x, n) */
void janas_bf16_dot4r(const uint16_t *w, size_t row, const float *x, uint32_t n,
                      float out[4]);

/* xb[i] = x[i] rounded to bf16 as ggml does (to nearest, ties to even, a
   NaN kept), kept as a float; xb may be x */
void janas_round_bf16_n(const float *x, float *xb, size_t n);

/* 1 where the products take the eight lanes (AVX2 and FMA), 0 where they
   go one element at a time. */
int janas_bf16_lanes(void);

/*
 * Q8_0 weights (blocks of 32: an fp16 scale, 32 int8) with float vectors,
 * the vectors kept as they are, not quantized: Qwen3.8-Flash-Next's mixes
 * rewritten at load (model_hc_quant.c), whose inputs at eight bits cost
 * what their weights at eight bits do not. Eight lanes, lane k the
 * elements 8i + k: in each block the fused multiply-adds of its four, then
 * that times the block's scale fused into the lane's sum; the eight added
 * as above. The same bits with AVX2 or without. n a multiple of 32; w is
 * n / 32 blocks of 34 bytes.
 */
float janas_q8w_dot(const uint8_t *w, const float *x, uint32_t n);
void janas_q8w_dot4(const uint8_t *w, const float *const x[4], uint32_t n,
                    float out[4]);
/* four rows, row bytes apart from w, against one vector: out[r] the bits
   of janas_q8w_dot(w + r * row, x, n) */
void janas_q8w_dot4r(const uint8_t *w, size_t row, const float *x, uint32_t n,
                     float out[4]);

#endif
