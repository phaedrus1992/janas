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
    uint32_t pos[2];  /* the position's buffer */
    uint32_t list[2]; /* qwen4exp's positions a token (ix_select.comp) */
    uint32_t ls, pad;
};
struct push_am {
    uint32_t part[2], q256[2], q32[2];
    uint32_t hd, nch, ps, gs;
    uint32_t pos[2]; /* 8-byte aligned, as the GLSL block puts it */
    uint32_t gate[2];
    uint32_t list[2]; /* as push_at's */
    uint32_t ls, pad;
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
    uint32_t sig; /* sigmoid(z) in place of SiLU(z) (qwen4exp) */
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
struct push_pl { /* moe_plan.comp; offsets in bytes from b */
    uint32_t at[2], b[2];
    uint32_t pe, gt, inv, t1, t2, ind, xg, egu, eh, dout;
    uint32_t n, k, ne, nbd, nbf, ep, dm, eff, s1, s2, tiles1, tiles2;
};
struct push_id {
    uint32_t t[2];
    uint32_t rows, nb, tiles;
    uint32_t ngroups; /* b32_mmid.comp: the groups of 32 a row */
};
struct push_sm {
    uint32_t g[2], u[2], h[2];
    uint32_t n;
};
/* qwen4exp's mixes and n-gram embeddings (hc_*.comp, ng_*.comp) */
struct push_hn {
    uint32_t res[2], w[2], xn[2], xb[2];
    uint32_t d, hc;
    float eps;
};
struct push_ha {
    uint32_t x[2], xb[2];
    uint32_t n, silu;
};
struct push_hm {
    uint32_t xn[2], g[2], out[2];
    uint32_t d, hc;
};
struct push_hcb {
    uint32_t res[2], out[2], inj[2];
    uint32_t d, hc, init;
};
struct push_ngg {
    uint32_t kq[2], res[2], val[2], nk[2], nq[2], nc[2], gv[2], hist[2], pos[2];
    uint32_t d, hc, cap;
    float eps;
};
struct push_ngc {
    uint32_t res[2], gv[2], w[2], hist[2], pos[2];
    uint32_t hd, cap, k, len;
};
/* qwen4exp's indexer (ix_*.comp, gpu_token_qsa.c) */
struct push_ixp {
    uint32_t q[2], k[2], qn[2], rope[2], raw[2], pos[2];
    uint32_t nh, d, n_rot, qs;
    float eps;
};
struct push_ixl {
    uint32_t raw[2], kn[2], rope[2], pool[2], pos[2];
    uint32_t d, r, n, n_rot, rb;
    float eps;
};
struct push_ixs {
    uint32_t q[2], pool[2], sc[2], pos[2];
    uint32_t nh, d, r, keep, nbs, qs;
};
struct push_ixc {
    uint32_t sc[2], list[2], pos[2];
    uint32_t r, keep, nbs, ls;
};
_Static_assert(sizeof(struct push_ngg) <= JANAS_G_PUSH_BYTES, "push size");
_Static_assert(sizeof(struct push_at) <= JANAS_G_PUSH_BYTES, "push size");
/* The attention's positions a run (attn_decode.comp C), and the shapes its
   shared memory takes: the query heads of a KV head times hd, and times a
   run. */
#define ATT_RUN 128
#define JANAS_GT_POS_BYTES 20 /* the pass's words at o_pos */
#define ATT_MAXQ 2048
#define ATT_MAXS 1024
/* ... larger where a KV head has more than 8 query heads (attn_decode.comp
   MAXQ, MAXS) */
#define ATT_MAXQ_G(g) ((g) * 256u > ATT_MAXQ ? (g) * 256u : ATT_MAXQ)
#define ATT_MAXS_G(g) ((g) * ATT_RUN > ATT_MAXS ? (g) * ATT_RUN : ATT_MAXS)

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
#define TOK_MAX_BLOCK JANAS_GPU_TOK_BLOCK
#define TOK_PART_BUDGET ((size_t)64 << 20)

/* JANAS_GPU_TOKEN_PROFILE=1: the GPU's own timestamps after each phase of
   the token, summed per phase and printed when the model is closed; a
   mixture's segments too, with the time between two of them (the CPU's
   plan, the submission's round trip) as a phase of its own */
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
    /* a mixture's: the router's choice, the experts' weighted sum, the
       time from a segment's last timestamp to the next one's start */
    PH_ROUTER,
    PH_COMBINE,
    PH_BETWEEN,
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
                   [2];    /* the runs each was recorded for */
    int prof;              /* timestamps in the commands being recorded */
    VkQueryPool qp;        /* timestamps, if profiling */
    uint32_t n_ts;         /* written this token */
    uint8_t ph_of[MAX_TS]; /* the phase each timestamp closes */
    double ph_ns[PH_N];
    double wall_ns;       /* submission to the fence, as the CPU sees it */
    double copy_ns;       /* the logits copied out */
    double plan_ns;       /* a mixture's plans, on the CPU */
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
    /* on a card with memory of its own buf is in that memory (its map this
       mirror's, host-visible): janas_gt_sync_range copies the ranges the
       host writes or reads between the two */
    struct stage mirror;
    int local;
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
    /* blocks of this many tokens or fewer (by default all): the experts'
       products whose types have one through the matvec shaders (gpu.c
       pipe_mvid, bit for bit with the CPU's), not *_mmid's tiles
       (JANAS_GPU_MVID=<n>, 0 never) */
    uint32_t mvid_n;
    uint32_t n_seg; /* n_layer + 1 */
    /* the CPU's side: the experts' counts and fill, those chosen, their
       slots */
    uint32_t *cnt, *fill, *ids;
    const uint8_t **slots;
    /* a card whose memory does not hold the experts (gpu_token_moe.c): a
       tier of n_slot slots of slot_bytes in it with the experts used most,
       slot_of[layer * n_expert + e] (or -1), owner (layer * n_expert + e,
       or UINT32_MAX), uses[layer * n_expert + e] counted at each block,
       stamp the plan that last counted on a slot (plan_no: that plan's);
       the experts read into it through up (a stage of UP_SLOTS slots); per
       block the experts of a layer that the GPU computes (gpu ids, their
       tier slot), those that the CPU does, the misses asked of fetch */
#define TIER_PIECES 64
    struct stage tier[TIER_PIECES], up; /* the tier in pieces of per slots */
    uint32_t n_slot, n_free, scan, plan_no, per;
    size_t slot_bytes;
    int32_t *slot_of;
    uint32_t *owner, *uses, *stamp, *on_gpu, *gpu_slot, *on_cpu, *miss, *ra,
        *rb;
    const uint8_t **cpu_slots;
    /* a plan's copies in one submission: the last experts read into the
       tier, the CPU's rows */
    struct janas_g_cp *cps;
    uint64_t tier_hits, tier_loads, cpu_experts;
    /* every expert in the tier (janas_gt_tier_fill, tried once): the plan
       on the GPU, moe_plan.comp after each router, a mixture's pass in one
       submission; at: the experts' addresses in the tier, 3 uvec4 each,
       n_expert a layer */
    int gplan, gplan_tried;
    struct stage at;
    VkShaderModule sm_pl;
    VkPipeline p_pl;
    /* the feed-forwards' widths padded to whole groups of 256: the
       experts' (effp) and the shared expert's or dense one's (ffp) */
    uint32_t effp, ffp;
    /* qwen4exp: the streams (hc dm a token), each normalized and at bf16,
       the mixes' low rank and its bf16, their gates, injects and means;
       the n-gram layer's key products, values, the embeddings at bf16 and
       the gated values; their shaders (gpu_token_hc.c) */
    size_t o_res, o_hxn, o_hxb, o_lo, o_lob, o_hg, o_inj, o_hmean, o_ngk, o_ngv,
        o_ngeb, o_nggv, o_ngemb, o_ngw;
    VkShaderModule sm_hn, sm_ha, sm_hm, sm_hcb, sm_ngg, sm_ngc;
    VkPipeline p_hn, p_ha, p_hm, p_hcb, p_ngg, p_ngc;
    /* qwen4exp's indexer: the block's queries and keys, the cos and sin of
       the tokens' positions and the blocks' (the caller's), the scores of
       every block for each token (nbs a token), the positions each attends
       to (ls words a token), the runs that take them; its shaders
       (gpu_token_qsa.c) */
    size_t o_ixq, o_ixk, o_ixrope, o_ixsc, o_ixlist;
    uint32_t ix_nbs, ix_ls, ix_runs;
    VkShaderModule sm_ixp, sm_ixl, sm_ixs, sm_ixc;
    VkPipeline p_ixp, p_ixl, p_ixs, p_ixc;
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
/* Bytes [off, off + bytes) of the mirror out of the CPU's caches (flush)
   or reloaded from memory (!flush), where it is not coherent: the copies
   between it and buf are in the pass's own commands. */
void janas_gt_mirror_cache(struct janas_gpu *g, struct janas_gpu_tok *t,
                           size_t off, size_t bytes, int flush);
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

/* gpu_token_hc.c: qwen4exp's streams. A mix (k 0 before the token mixer,
   1 before the feed-forward, 2 the output's) of the streams into the
   sublayer's input, quantized in q256 / q32 and as floats in xnf, its
   inject (none for the output's) into o_inj; a combine of the sublayer's
   output (out) into the streams; the n-gram layer; the streams set to the
   embeddings at x. */
void janas_gt_rec_hc_mix(struct janas_gpu *g, struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md,
                         const struct janas_gpu_tok_layer *ly, int k,
                         VkPipeline *bound, VkDeviceAddress xnf, uint32_t n);
void janas_gt_rec_hc_combine(struct janas_gpu *g, struct janas_gpu_tok *t,
                             const struct janas_gpu_tok_model *md,
                             VkPipeline *bound, VkDeviceAddress out, int init,
                             uint32_t n);
int janas_gt_rec_ng(struct janas_gpu *g, struct janas_gpu_tok *t,
                    const struct janas_gpu_tok_model *md,
                    const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                    uint32_t n);
int janas_gt_hc_create(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md);
void janas_gt_hc_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);
/* y (n rows of ys) = scale w x (n rows of cols floats): w bf16, rows x
   cols, in dense_mm.comp's tiles */
void janas_gt_rec_bf16(struct janas_gpu *g, VkPipeline *bound,
                       VkDeviceAddress w, VkDeviceAddress x, VkDeviceAddress y,
                       uint32_t rows, uint32_t cols, uint32_t ys, uint32_t n,
                       float scale);
/* the same for w of dense_mm.comp's kind: 0 f32, 1 bf16, 2 Q8_0; on
   DENSE_MV_MAX vectors or fewer a workgroup a row (dense_mv.comp) */
#define DENSE_MV_MAX 8
void janas_gt_rec_dense(struct janas_gpu *g, VkPipeline *bound, int kind,
                        VkDeviceAddress w, VkDeviceAddress x, VkDeviceAddress y,
                        uint32_t rows, uint32_t cols, uint32_t ys, uint32_t n,
                        float scale);

/* gpu_token_qsa.c: qwen4exp's indexer for an attention layer, on its input
   at bf16 (xb, n rows of dm): the keys and queries, the blocks the block
   completes, the scores and the positions each token attends to, which
   janas_gt_rec_attn's runs then read */
int janas_gt_rec_qsa(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md,
                     const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                     VkDeviceAddress xb, uint32_t n);
int janas_gt_qsa_create(struct janas_gpu *g, struct janas_gpu_tok *t);
void janas_gt_qsa_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);

/* gpu_token_moe.c: a mixture's layers */
int janas_gt_rec_router(struct janas_gpu *g, struct janas_gpu_tok *t,
                        const struct janas_gpu_tok_model *md,
                        const struct janas_gpu_tok_layer *ly, uint32_t l,
                        VkPipeline *bound, uint32_t n);
void janas_gt_rec_experts(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md,
                          const struct janas_gpu_tok_layer *ly,
                          VkPipeline *bound, VkDeviceAddress YD, uint32_t n);
int janas_gt_tier_create(struct janas_gpu *g, struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md);
void janas_gt_tier_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);
/* Every expert read into a tier that holds them all, their addresses
   written for moe_plan.comp, t->gplan set: 0, or -1 (the plan stays the
   CPU's). */
int janas_gt_tier_fill(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md);
/* Layer l's plan on the GPU from its router's choice (t->gplan) */
void janas_gt_rec_plan(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md, uint32_t l,
                       VkPipeline *bound, uint32_t n);
void janas_gt_free_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkCommandBuffer **seg);
int janas_gt_run_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md, uint32_t n,
                          uint32_t first, int ai, int hi, uint32_t nch);

#endif
#endif
