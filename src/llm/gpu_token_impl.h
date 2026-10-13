/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_impl.h - the insides of the whole token on the GPU (private to
 * gpu_token.c, gpu_token_rec.c and gpu_token_moe.c): the shaders' push
 * constants, the token's buffers and the helpers that record its commands.
 * Only with the GPU in the build.
 */
#ifndef JANAS_LLM_GPU_TOKEN_IMPL_H
#define JANAS_LLM_GPU_TOKEN_IMPL_H

#include "gpu_impl.h"
#include "gpu_token.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include <stddef.h>

/* The push constants of each shader, laid out as its GLSL block. */
struct push_nq {
    uint32_t x[2], add[2], w[2], xo[2], q256[2], q32[2], xn[2];
    uint32_t d;
    float eps;
};
struct push_qk {
    uint32_t qkv[2], qn[2], kn[2], rope[2], kc[2], vc[2], ks[2], vs[2];
    uint32_t n_head, n_kv, hd, n_rot, cap, qs;
    float eps;
    uint32_t pad, pos[2]; /* the position's buffer, 8-byte aligned */
};
struct push_at {
    uint32_t q[2], kc[2], vc[2], ks[2], vs[2], part[2];
    uint32_t n_head, n_kv, hd, cap, nch;
    float scale;
    uint32_t qs, ps;
    uint32_t pos[2]; /* the position's buffer */
};
struct push_am {
    uint32_t part[2], q256[2], q32[2];
    uint32_t hd, nch, ps, gs;
    uint32_t pos[2]; /* 8-byte aligned, as the GLSL block puts it */
    uint32_t gate[2];
};
/* the Gated DeltaNet shaders (gdn_*.comp) */
struct push_gconv {
    uint32_t mixed[2], out[2], w[2], state[2], pos[2], log[2];
    uint32_t cc, d, hs, so, n;
};
struct push_gcst {
    uint32_t mixed[2], state[2], pos[2], log[2];
    uint32_t cc, d, hs, so, n;
};
struct push_gl2 {
    uint32_t x[2], pos[2];
    uint32_t cc;
    float eps;
    uint32_t n;
};
struct push_grec {
    uint32_t co[2], att[2], state[2], pos[2], lb[2], ld[2], bd[2];
    uint32_t cc, nv, nkh, n, hs, so, vmap_mod;
    float scale;
};
struct push_gba { /* gdn_ba.comp */
    uint32_t w[2], x[2], y[2], dt[2], a[2];
    uint32_t rows, cols, n, ys;
};
struct push_gact { /* gdn_act.comp */
    uint32_t ba[2], bd[2], dt[2], a[2];
    uint32_t nv, n;
};
struct push_ggate {
    uint32_t att[2], z[2], w[2], q256[2], q32[2], bd[2], lb[2], ld[2], pos[2];
    uint32_t d, nv;
    float eps;
};
struct push_dm { /* gpu.c's dense_mm.comp */
    uint32_t w[2], x[2], y[2];
    uint32_t rows, cols, nv, ys;
    float scale;
};
/* a mixture's (moe_*.comp, and gpu.c's *_mmid.comp) */
struct push_tk {
    uint32_t r[2], pe[2], pw[2], shw[2];
    uint32_t ne, rs, k, n, sh;
};
struct push_ga {
    uint32_t src[2], dst[2], tok[2];
    uint32_t nb;
};
struct push_cb {
    uint32_t dout[2], pw[2], inv[2], sh[2], shw[2], y[2];
    uint32_t dm, k, has_sh;
};
struct push_id {
    uint32_t t[2];
    uint32_t rows, nb, tiles;
};
struct push_sm {
    uint32_t g[2], u[2], h[2];
    uint32_t n;
};
/* The attention's positions a run (attn_decode.comp C), and the shapes its
   shared memory takes: the query heads of a KV head times hd, and times a
   run. */
#define ATT_RUN 128
#define ATT_MAXQ 2048
#define ATT_MAXS 1024

struct push_ac {
    uint32_t g[2], u[2], x[2];
    uint32_t blocks;
};
_Static_assert(sizeof(struct push_qk) <= JANAS_G_PUSH_BYTES, "push size");
_Static_assert(sizeof(struct push_at) <= JANAS_G_PUSH_BYTES, "push size");
/* the uvec2 members where GLSL puts them: on 8 bytes */
_Static_assert(offsetof(struct push_qk, pos) % 8 == 0, "push layout");
_Static_assert(offsetof(struct push_at, pos) % 8 == 0, "push layout");
_Static_assert(offsetof(struct push_am, pos) % 8 == 0, "push layout");
_Static_assert(sizeof(struct push_grec) <= JANAS_G_PUSH_BYTES, "push size");
_Static_assert(sizeof(struct push_ggate) <= JANAS_G_PUSH_BYTES, "push size");
_Static_assert(sizeof(struct push_cb) <= JANAS_G_PUSH_BYTES, "push size");
/* The most tokens of a block whose beta and alpha gdn_ba.comp makes (more:
   dense_mm.comp's tiles) */
#define GBA_MAX_N 16
#define GBA_RW 4 /* its rows a workgroup (gdn_ba.comp RW) */
/* The most tokens a block takes (the model's own blocks: model.h
   JANAS_LLM_MAX_BLOCK), and the memory the attention's runs may take for
   them. */
#define TOK_MAX_BLOCK 256
#define TOK_PART_BUDGET ((size_t)64 << 20)

/* JANAS_GPU_TOKEN_PROFILE=1: the GPU's own timestamps after each phase of
   the token, summed per phase and printed when the model is closed */
enum {
    PH_NORM,
    PH_QKV,
    PH_ROPE,
    PH_ATTN,
    PH_MERGE,
    PH_WO,
    PH_GATEUP,
    PH_ACT,
    PH_DOWN,
    PH_OUT,
    /* a Gated DeltaNet layer: its products (q|k|v and z), beta and alpha,
       the convolution (with its state and the q and k norms), the
       recurrence, the gate (its output product counts as wo) */
    PH_SSM_PROD,
    PH_SSM_BA,
    PH_SSM_CONV,
    PH_SSM_REC,
    PH_SSM_GATE,
    PH_N
};
#define MAX_TS 1024

struct janas_gpu_tok {
    /* its own command buffers, one per number of tokens n (1 .. nb) and
       logits of the last token, of all or of none ([n][all]: 0, 1, 2),
       each recorded once:
       the position is read from the buffer at o_pos, so every block of n
       tokens submits the same commands */
    uint32_t nb;
    /* [n][all][hidden wanted] */
    VkCommandBuffer cb[TOK_MAX_BLOCK + 1][3][2];
    uint32_t cb_nch[TOK_MAX_BLOCK + 1][3]
                   [2]; /* the runs each was recorded for */
    int prof;           /* timestamps in the commands being recorded (n = 1) */
    VkQueryPool qp;     /* timestamps, if profiling */
    uint32_t n_ts;      /* written this token */
    uint8_t ph_of[MAX_TS]; /* the phase each timestamp closes */
    double ph_ns[PH_N];
    double wall_ns;       /* submission to the fence, as the CPU sees it */
    double copy_ns;       /* the logits copied out */
    uint64_t tokens;      /* with profiling: passes measured */
    uint64_t pass_tokens; /* ... and their tokens */
    VkShaderModule sm_nq, sm_qk, sm_at, sm_am;
    VkPipeline p_nq, p_qk, p_at, p_am;
    /* qwen3.5's recurrent layers (sm_gb: the betas and decays of a short
       block, sm_gx: those of a long one's raw dot products) */
    VkShaderModule sm_gc, sm_gs, sm_gl, sm_gr, sm_gg, sm_gb, sm_gx;
    VkPipeline p_gc, p_gs, p_gl, p_gr, p_gg, p_gb, p_gx;
    size_t *par;      /* per layer: its parameters' offsets in buf (PAR_*) */
    struct stage buf; /* every intermediate, one allocation */
    /* offsets into buf, each nb tokens' worth */
    uint32_t nch;       /* the attention's runs of positions, for the cache */
    size_t part_floats; /* room for the runs of a block, floats */
    uint32_t rec_nch;   /* the runs a token of the block being recorded has */
    size_t o_part;
    size_t o_x, o_x1, o_q256, o_q32, o_qkv, o_ywo, o_gu, o_h, o_yd, o_rope,
        o_logits, o_pos;
    /* qwen3.5: the normalized input (f32), q|k|v and z of the recurrent
       layers, their convolution's output (the replayed tokens' rows
       first), beta and alpha, their raw output, the betas and decays used
       (for the log), the query's gates, the final hidden states */
    size_t o_xnf, o_mix, o_z, o_conv, o_ba, o_att, o_bd, o_gate, o_hid;
    int log_ok; /* the model's log is where the GPU reads and writes */
    /* a mixture: a submission a layer ([n][all][hidden], n_layer + 1 of
       them, each recorded for the runs in seg_nch), its shaders; the
       router's logits (n_expert + 1 a token), the pairs' experts and
       weights, the shared expert's weights and output, the activations in
       the experts' order, their gates and ups, activations and outputs,
       each row's token, each pair's row, the tables of the gate and up
       products and of the down ones (their tiles from uvec4 tiles1 and
       tiles2 on), the two dispatches' sizes */
    VkCommandBuffer *seg_cb[TOK_MAX_BLOCK + 1][3][2];
    uint32_t seg_nch[TOK_MAX_BLOCK + 1][3][2];
    VkShaderModule sm_tk, sm_ga, sm_cb, sm_sm;
    VkPipeline p_tk, p_ga, p_cb, p_sm;
    size_t o_r, o_pe, o_pw, o_shw, o_sh, o_xg, o_egu, o_eh, o_dout, o_gt, o_inv,
        o_tab1, o_tab2, o_ind;
    /* a dense layer's or a shared expert's activation as floats and in
       blocks of 32, where a down matrix is in blocks of 32 */
    size_t o_hf, o_h32;
    uint32_t tiles1, tiles2;
    uint32_t n_seg; /* n_layer + 1 */
    /* the CPU's side: the experts' counts and fill, those chosen, their
       slots */
    uint32_t *cnt, *fill, *ids;
    const uint8_t **slots;
};

/* A recurrent layer's parameters copied into buf: beta then alpha (2 nv x
   dm), the convolution (d_conv x conv_ch), dt, A (nv), the norm (ds); a
   mixture's router as f32 (n_expert rows, then the shared expert's gate) */
enum { PAR_BA, PAR_CONV, PAR_DT, PAR_A, PAR_NORM, PAR_R, PAR_N };

/* The GPU's address of host memory in one of its regions, 0 if none. */
VkDeviceAddress janas_gt_addr_of(const struct janas_gpu *g, const void *p,
                                 size_t bytes);
/* Bytes [off, off + bytes) of the token's buffer out of the CPU's caches
   (flush) or reloaded from memory (!flush), where it is not coherent. */
void janas_gt_sync_range(struct janas_gpu *g, struct janas_gpu_tok *t,
                         size_t off, size_t bytes, int flush);
/* A command buffer for cb, allocated, and the commands of segment seg of a
   block of n tokens recorded into it (gpu_token.c): 0, or -1 with *cb
   NULL. */
int janas_gt_make_cb(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md, VkCommandBuffer *cb,
                     uint32_t n, uint32_t first, int hi, uint32_t nch, int seg);
int janas_gt_submit_wait(struct janas_gpu *g, const VkCommandBuffer *cb);

/* gpu_token_rec.c: the commands, recorded into g->cb */
void janas_gt_stamp(struct janas_gpu *g, struct janas_gpu_tok *t, int ph);
void janas_gt_bind(struct janas_gpu *g, VkPipeline *bound, VkPipeline p);
void janas_gt_push(struct janas_gpu *g, const void *pc, size_t bytes);
void janas_gt_rec_norm_quant(struct janas_gpu *g, struct janas_gpu_tok *t,
                             VkPipeline *bound, VkDeviceAddress x,
                             VkDeviceAddress add, VkDeviceAddress w,
                             VkDeviceAddress xo, VkDeviceAddress xn, uint32_t d,
                             float eps, uint32_t n);
int janas_gt_rec_product(struct janas_gpu *g, struct janas_gpu_tok *t,
                         VkPipeline *bound, const struct janas_matvec_task *w,
                         size_t xoff, VkDeviceAddress y, uint32_t n, size_t ys);
size_t janas_gt_part_tok(const struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md, uint32_t nch);
int janas_gt_rec_gdn(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md,
                     const struct janas_gpu_tok_layer *ly, uint32_t l,
                     VkPipeline *bound, VkDeviceAddress YWO, uint32_t n);
int janas_gt_rec_attn(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md,
                      const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                      uint32_t n);
int janas_gt_rec_ffn(struct janas_gpu *g, struct janas_gpu_tok *t,
                     VkPipeline *bound, const struct janas_matvec_task *gate,
                     const struct janas_matvec_task *up,
                     const struct janas_matvec_task *down, uint32_t n,
                     VkDeviceAddress out);

/* gpu_token_moe.c: a mixture's layers */
int janas_gt_rec_router(struct janas_gpu *g, struct janas_gpu_tok *t,
                        const struct janas_gpu_tok_model *md,
                        const struct janas_gpu_tok_layer *ly, uint32_t l,
                        VkPipeline *bound, uint32_t n);
void janas_gt_rec_experts(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md,
                          const struct janas_gpu_tok_layer *ly,
                          VkPipeline *bound, VkDeviceAddress YD, uint32_t n);
void janas_gt_free_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkCommandBuffer **seg);
int janas_gt_run_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md, uint32_t n,
                          uint32_t first, int ai, int hi, uint32_t nch);

#endif
#endif
