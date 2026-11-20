/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision_kernels.h - the float kernels of the encoders (vision_kernels.c):
 * the tiled products, the attention of every token over every other, the
 * norms and activations. The vision encoders and Whisper's share them.
 */
#ifndef JANAS_LLM_VISION_KERNELS_H
#define JANAS_LLM_VISION_KERNELS_H

#include <stddef.h>
#include <stdint.h>

struct janas_pool;
/* y[n] (ldy apart, 0: rows) = w x[n] + b, wt the weights transposed (cols
   x rows, bfloat16), every product on the pool */
void vk_matmul(struct janas_pool *pool, const uint16_t *wt, const float *b,
               const float *x, float *y, uint32_t rows, uint32_t cols,
               uint32_t n, size_t ldy);
/* the same, wt in half precision */
void vk_matmul_f16(struct janas_pool *pool, const uint16_t *wt, const float *b,
                   const float *x, float *y, uint32_t rows, uint32_t cols,
                   uint32_t n, size_t ldy);
/* attention of every patch over every other, qkv n x 3 dim (q, k, v), the
   output n x dim; buf 2 n dim floats of room */
void vk_attention(struct janas_pool *pool, const float *qkv, float *buf,
                  float *out, uint32_t n, uint32_t dim, uint32_t heads,
                  float scale);
uint16_t *vk_transposed(const uint16_t *w, uint64_t rows, uint64_t cols);
void vk_layer_norm(float *y, const float *x, const float *w, const float *b,
                   uint32_t n, float eps);
/* x / sqrt(mean(x^2) + eps), times w (NULL: none) */
void vk_rms_norm(float *y, const float *x, const float *w, uint32_t n,
                 float eps);
float vk_gelu_tanh(float x);
float vk_gelu_erf(float x);
float vk_bf(uint16_t b);

#endif
