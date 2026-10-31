/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision_impl.h - the inside of the vision encoders: the encoder as loaded
 * (vision.c: Qwen3.5 and Qwen3-VL; vision_gemma.c: Gemma 4) and the
 * kernels they share (vision_kernels.c).
 */
#ifndef JANAS_LLM_VISION_IMPL_H
#define JANAS_LLM_VISION_IMPL_H

#include <stddef.h>
#include <stdint.h>

#include "jns.h"
#include "vision.h"
#include "vision_kernels.h"

#define MAX_BLOCKS 64

enum { VISION_QWEN = 1, VISION_GEMMA = 2 };

/* a block of Qwen's ViT */
struct vlayer {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    const uint16_t *qkv, *out, *up, *down;
    const float *qkv_b, *out_b, *up_b, *down_b;
    /* the same matrices transposed, for the products (owned) */
    uint16_t *t_qkv, *t_out, *t_up, *t_down;
};

/* a block of Gemma 4's: its norms, its linears transposed (owned) and the
   bounds each clamps its input and output to (q k v o gate up down, each
   in_min in_max out_min out_max) */
enum { GL_Q, GL_K, GL_V, GL_O, GL_GATE, GL_UP, GL_DOWN, GL_N };
struct glayer {
    const float *n_in, *n_post_att, *n_pre_ff, *n_post_ff, *q_norm, *k_norm;
    const float *clip; /* GL_N x 4 */
    uint16_t *t[GL_N];
};

struct janas_vision {
    struct janas_jns j;
    uint8_t *data; /* the resident tensors */
    int kind;      /* VISION_QWEN, VISION_GEMMA */
    int both_ways; /* the pictures' tokens attend both ways */
    uint32_t n_block, dim, ff, heads, patch, tpatch, merge, npos, proj;
    uint32_t min_pixels, max_pixels;
    float eps, theta, mean[3], stdv[3];
    int32_t tok_image, tok_start, tok_end;
    /* Qwen */
    const uint16_t *patch_w, *pos, *m_fc1, *m_fc2;
    const float *patch_b, *m_ln_w, *m_ln_b, *m_fc1_b, *m_fc2_b;
    struct vlayer l[MAX_BLOCKS];
    uint16_t *t_patch, *t_fc1, *t_fc2; /* transposed, owned */
    /* Gemma 4: the pooling, the positions' table (2 x pos_size rows: x,
       then y), the soft tokens a picture at most, the standardization
       (NULL: none), the projection to the language model (transposed) */
    uint32_t pool_k, pos_size, max_soft;
    const uint16_t *g_pos;
    const float *std_bias, *std_scale;
    struct glayer gl[MAX_BLOCKS];
    uint16_t *t_g_patch, *t_g_proj;
    /* Gemma 4 12B's, without blocks (vision_gemma_u.c): a 48x48 patch
       layer-normed, through a dense layer, normed, plus the positions,
       normed (u_ln: weight and bias of each norm, u_eps theirs) */
    int unified;
    float u_eps;
    const float *u_ln[3][2], *u_dense_b;
    uint16_t *t_u_dense;
};

/* a resident tensor of that name, type (0 float, 30 bfloat16) and shape,
   or NULL (vision.c) */
const void *janas_vision_tensor(struct janas_vision *v, const char *name,
                                uint32_t type, uint64_t rows, uint64_t cols);

/* vision_gemma.c: the rest of the open (NULL, or why not), the size of a
   picture, its encoding, what it owns freed */
const char *janas_vg_open(struct janas_vision *v);
void janas_vg_size(const struct janas_vision *v, int w, int h, int max_tokens,
                   int *rw, int *rh, int *nx, int *ny);
int janas_vg_encode(struct janas_vision *v, const struct janas_image *im,
                    int max_tokens, struct janas_pool *pool, float **out,
                    int *nx, int *ny, char *err, size_t err_len);
void janas_vg_close(struct janas_vision *v);
/* vision_gemma_u.c: the same for Gemma 4 12B's encoder without blocks */
const char *janas_vgu_open(struct janas_vision *v);
int janas_vgu_encode(struct janas_vision *v, const struct janas_image *im,
                     int max_tokens, struct janas_pool *pool, float **out,
                     int *nx, int *ny, char *err, size_t err_len);

#endif
