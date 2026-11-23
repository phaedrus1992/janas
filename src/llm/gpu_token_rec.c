/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_rec.c - the commands of the whole token on the GPU, recorded
 * into g->cb (gpu_token_impl.h): norms, products, a layer's attention or
 * Gated DeltaNet, a feed-forward.
 */
#include "gpu_token_impl.h"

#include <math.h>
#include <stdlib.h>

#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

/* A timestamp closing phase ph (with profiling on). */
void janas_gt_stamp(struct janas_gpu *g, struct janas_gpu_tok *t, int ph)
{
    if (!t->qp || !t->prof || t->n_ts >= MAX_TS)
        return;
    g->vkCmdWriteTimestamp(g->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, t->qp,
                           t->n_ts);
    t->ph_of[t->n_ts++] = (uint8_t)ph;
}

void janas_gt_bind(struct janas_gpu *g, VkPipeline *bound, VkPipeline p)
{
    if (*bound != p) {
        g->vkCmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        *bound = p;
    }
}

void janas_gt_push(struct janas_gpu *g, const void *pc, size_t bytes)
{
    g->vkCmdPushConstants(g->cb, g->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          (uint32_t)bytes, pc);
}

/* x (+ add, then stored at xo) normalized by w (0: not), quantized into
   q256 and q32: a workgroup per block of 256 values. */
void janas_gt_rec_norm_quant(struct janas_gpu *g, struct janas_gpu_tok *t,
                             VkPipeline *bound, VkDeviceAddress x,
                             VkDeviceAddress add, VkDeviceAddress w,
                             VkDeviceAddress xo, VkDeviceAddress xn, uint32_t d,
                             float eps, uint32_t n)
{
    janas_gt_rec_norm_quant_s(g, t, bound, x, add, w, xo, xn, d, eps, n, 0.0f);
}

/* ... with x + add times os (0: as it is), Gemma 4's output scale */
void janas_gt_rec_norm_quant_s(struct janas_gpu *g, struct janas_gpu_tok *t,
                               VkPipeline *bound, VkDeviceAddress x,
                               VkDeviceAddress add, VkDeviceAddress w,
                               VkDeviceAddress xo, VkDeviceAddress xn,
                               uint32_t d, float eps, uint32_t n, float os)
{
    janas_gt_rec_norm_quant_p(g, t, bound, x, add, w, xo, xn, d, eps, n, os,
                              0.0f);
}

/* ... and the normalized value times ps before w (Gemma 4's router) */
void janas_gt_rec_norm_quant_p(struct janas_gpu *g, struct janas_gpu_tok *t,
                               VkPipeline *bound, VkDeviceAddress x,
                               VkDeviceAddress add, VkDeviceAddress w,
                               VkDeviceAddress xo, VkDeviceAddress xn,
                               uint32_t d, float eps, uint32_t n, float os,
                               float ps)
{
    struct push_nq pc = {.d = d, .eps = eps, .os = os, .ps = ps};
    janas_g_push_addr(pc.xn, xn);
    janas_g_push_addr(pc.x, x);
    janas_g_push_addr(pc.add, add);
    janas_g_push_addr(pc.w, w);
    janas_g_push_addr(pc.xo, xo);
    janas_g_push_addr(pc.q256, t->buf.addr + t->o_q256);
    janas_g_push_addr(pc.q32, t->buf.addr + t->o_q32);
    janas_gt_bind(g, bound, t->p_nq);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, d / JANAS_QK, n, 1); /* y: the vector */
    janas_g_compute_barrier(g);
}

/* A product of the task's weights with n quantized vectors (from byte xoff
   of the q256 or q32 layout, cols / 256 blocks apart), into y (ys floats
   apart; 0: rows). */
int janas_gt_rec_product(struct janas_gpu *g, struct janas_gpu_tok *t,
                         VkPipeline *bound, const struct janas_matvec_task *w,
                         size_t xoff, VkDeviceAddress y, uint32_t n, size_t ys)
{
    struct janas_matvec_task k = *w;
    k.n_vec = n;
    k.x_stride = 0;
    k.y_stride = 0;
    VkDeviceAddress xa =
        t->buf.addr + xoff + (janas_g_x_in_32(k.type) ? t->o_q32 : t->o_q256);
    return janas_g_record_task(g, bound, &k, k.rows, xa, y, ys ? ys : k.rows);
}

/* A token's runs, floats, at nch runs a head. */
size_t janas_gt_part_tok(const struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md, uint32_t nch)
{
    (void)t;
    return (size_t)md->n_head * nch * (md->hd + 2);
}

/* A Gated DeltaNet layer (qwen3.5) on the block normalized in q256 / q32
   and XNF: its output into YWO, as model_rec.c's janas_m_rec_layer. The
   tokens of the last call to replay first (the position buffer's word 3)
   come from the layer's log, and the block's go to it (word 4): the
   convolution and its norm take a + n rows, so their dispatches have room
   for JANAS_GPU_TOK_MAX_REPLAY more. */
int janas_gt_rec_gdn(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md,
                     const struct janas_gpu_tok_layer *ly, uint32_t l,
                     VkPipeline *bound, VkDeviceAddress YWO, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr, MIX = B + t->o_mix, Z = B + t->o_z,
                    CONV = B + t->o_conv, BA = B + t->o_ba, ATT = B + t->o_att,
                    POS = B + t->o_pos;
    const size_t *pr = t->par + (size_t)l * PAR_N;
    uint32_t cc = md->conv_ch, rows = n + JANAS_GPU_TOK_MAX_REPLAY;
    /* the layer's log: log_rows rows a layer (0 without one: never read
       then, the position buffer replaying and logging nothing) */
    size_t lr = (size_t)md->n_rec * md->log_rows;
    VkDeviceAddress
        LM = t->log_ok ? janas_gt_addr_of(g, md->log_mixed, lr * cc * 4) +
                             (VkDeviceAddress)ly->slot * md->log_rows * cc * 4
                       : 0,
        LB = t->log_ok
                 ? janas_gt_addr_of(g, md->log_beta, lr * md->nv * 4) +
                       (VkDeviceAddress)ly->slot * md->log_rows * md->nv * 4
                 : 0,
        LD = t->log_ok
                 ? janas_gt_addr_of(g, md->log_decay, lr * md->nv * 4) +
                       (VkDeviceAddress)ly->slot * md->log_rows * md->nv * 4
                 : 0,
        BD = B + t->o_bd;
    /* q|k|v (the convolution's channels) and z */
    if (janas_gt_rec_product(g, t, bound, &ly->ssm_qkv, 0, MIX, n, cc) != 0 ||
        janas_gt_rec_product(g, t, bound, &ly->ssm_z, 0, Z, n, md->d_inner) !=
            0)
        return -1;
    janas_gt_stamp(g, t, PH_SSM_PROD);
    /* the betas and decays, per token, from the normalized input at f32:
       a short block a row a workgroup (gdn_ba.comp), a longer one in
       dense_mm.comp's tiles, made betas and decays by gdn_act.comp; into
       BD either way */
    if (n <= GBA_MAX_N) {
        struct push_gba pb = {
            .rows = 2 * md->nv, .cols = md->dm, .n = n, .ys = 2 * md->nv};
        janas_g_push_addr(pb.w, B + pr[PAR_BA]);
        janas_g_push_addr(pb.x, B + t->o_xnf);
        janas_g_push_addr(pb.y, BD);
        janas_g_push_addr(pb.dt, B + pr[PAR_DT]);
        janas_g_push_addr(pb.a, B + pr[PAR_A]);
        janas_gt_bind(g, bound, t->p_gb);
        janas_gt_push(g, &pb, sizeof(pb));
        g->vkCmdDispatch(g->cb, (2 * md->nv + GBA_RW - 1) / GBA_RW, (n + 7) / 8,
                         1);
    } else {
        janas_gt_rec_dense(g, bound, 0, B + pr[PAR_BA], B + t->o_xnf, BA,
                           2 * md->nv, md->dm, 2 * md->nv, n, 1.0f);
        janas_g_compute_barrier(g);
        struct push_gact px = {.nv = md->nv, .n = n};
        janas_g_push_addr(px.ba, BA);
        janas_g_push_addr(px.bd, BD);
        janas_g_push_addr(px.dt, B + pr[PAR_DT]);
        janas_g_push_addr(px.a, B + pr[PAR_A]);
        janas_gt_bind(g, bound, t->p_gx);
        janas_gt_push(g, &px, sizeof(px));
        g->vkCmdDispatch(g->cb, (n * 2 * md->nv + 255) / 256, 1, 1);
    }
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_SSM_BA);
    /* the convolution, then the state it leaves, the q and k heads' norms */
    uint32_t hs_c = md->n_rec * (md->d_conv - 1) * cc,
             so_c = ly->slot * (md->d_conv - 1) * cc;
    VkDeviceAddress CS = janas_gt_addr_of(g, md->conv_state, (size_t)hs_c * 8);
    struct push_gconv pc = {
        .cc = cc, .d = md->d_conv, .hs = hs_c, .so = so_c, .n = n};
    janas_g_push_addr(pc.mixed, MIX);
    janas_g_push_addr(pc.out, CONV);
    janas_g_push_addr(pc.w, B + pr[PAR_CONV]);
    janas_g_push_addr(pc.state, CS);
    janas_g_push_addr(pc.pos, POS);
    janas_g_push_addr(pc.log, LM);
    janas_gt_bind(g, bound, t->p_gc);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, cc / 256, rows, 1);
    janas_g_compute_barrier(g);
    struct push_gcst ps = {
        .cc = cc, .d = md->d_conv, .hs = hs_c, .so = so_c, .n = n};
    janas_g_push_addr(ps.mixed, MIX);
    janas_g_push_addr(ps.state, CS);
    janas_g_push_addr(ps.pos, POS);
    janas_g_push_addr(ps.log, LM);
    janas_gt_bind(g, bound, t->p_gs);
    janas_gt_push(g, &ps, sizeof(ps));
    g->vkCmdDispatch(g->cb, cc / 256, 1, 1);
    struct push_gl2 pl = {.cc = cc, .eps = md->eps, .n = n};
    janas_g_push_addr(pl.x, CONV);
    janas_g_push_addr(pl.pos, POS);
    janas_gt_bind(g, bound, t->p_gl);
    janas_gt_push(g, &pl, sizeof(pl));
    g->vkCmdDispatch(g->cb, 2 * md->nkh, rows, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_SSM_CONV);
    /* the recurrence, then each head's norm, gate and quantization */
    uint32_t hs_s = md->n_rec * md->nv * md->ds * md->ds,
             so_s = ly->slot * md->nv * md->ds * md->ds;
    struct push_grec pg = {.cc = cc,
                           .nv = md->nv,
                           .nkh = md->nkh,
                           .n = n,
                           .hs = hs_s,
                           .so = so_s,
                           .vmap_mod = (uint32_t)md->vmap_mod,
                           .scale = 1.0f / sqrtf((float)md->ds)};
    janas_g_push_addr(pg.co, CONV);
    janas_g_push_addr(pg.att, ATT);
    janas_g_push_addr(pg.state,
                      janas_gt_addr_of(g, md->ssm_state, (size_t)hs_s * 4));
    janas_g_push_addr(pg.pos, POS);
    janas_g_push_addr(pg.lb, LB);
    janas_g_push_addr(pg.ld, LD);
    janas_g_push_addr(pg.bd, BD);
    janas_gt_bind(g, bound, t->p_gr);
    janas_gt_push(g, &pg, sizeof(pg));
    g->vkCmdDispatch(g->cb, md->nv * (md->ds / 16), 1, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_SSM_REC);
    struct push_ggate pq = {
        .d = md->d_inner, .nv = md->nv, .eps = md->eps, .sig = md->hc != 0};
    janas_g_push_addr(pq.att, ATT);
    janas_g_push_addr(pq.z, Z);
    janas_g_push_addr(pq.w, B + pr[PAR_NORM]);
    janas_g_push_addr(pq.q256, B + t->o_q256);
    janas_g_push_addr(pq.q32, B + t->o_q32);
    janas_g_push_addr(pq.bd, BD);
    janas_g_push_addr(pq.lb, LB);
    janas_g_push_addr(pq.ld, LD);
    janas_g_push_addr(pq.pos, POS);
    janas_gt_bind(g, bound, t->p_gg);
    janas_gt_push(g, &pq, sizeof(pq));
    g->vkCmdDispatch(g->cb, md->d_inner / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_SSM_GATE);
    if (janas_gt_rec_product(g, t, bound, &ly->ssm_out, 0, YWO, n, 0) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_WO);
    return 0;
}

/* JANAS_GPU_Q16=1: the attention's q at sixteen bits, two eight-bit halves
   (q = 256 hi + lo), its scores two integer products; the CPU takes q in
   float, the eight-bit q's relative error some 4e-3 against 3e-5; heads
   up to 256 only: 256 hi k summed over a longer head could pass 2^31 */
static int q16_on(uint32_t hd)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("JANAS_GPU_Q16");
        on = e && atoi(e) > 0;
    }
    return on && hd <= 256;
}

/* A layer's attention on the block normalized in q256 / q32: q, k and v,
   their norms and RoPE (the keys and values into the cache), the
   attention in runs of positions, joined, then wo into YWO (Gemma 4: its
   sandwich norm after, into o_post). The layer's own geometry (Gemma 4's
   two kinds), the token's buffers laid out for the largest. */
int janas_gt_rec_attn(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md,
                      const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                      uint32_t n)
{
    uint32_t hd = ly->hd, nkv = ly->n_kv;
    uint32_t qd = md->n_head * hd, kvd = nkv * hd;
    size_t qs = qd + 2 * kvd;
    size_t cache = (size_t)nkv * ly->cap * hd << md->kv16,
           scales = (size_t)nkv * ly->cap * 4;
    VkDeviceAddress B = t->buf.addr, QKV = B + t->o_qkv, YWO = B + t->o_ywo,
                    ROPE = B + (ly->swa ? t->o_rope2 : t->o_rope),
                    POS = B + t->o_pos;
    int v_is_k = ly->wv.type == 0;
    /* a layer reading another's keys and values (Gemma 4's E2B, E4B):
       its queries alone */
    int own_kv = ly->wk.type != 0;
    /* q, k and v of each token side by side, qs floats a token; a
       gated query (qwen3.5) head by head, its gates apart */
    if (md->gated_q) {
        size_t rb =
            (size_t)(md->dm / JANAS_QK) * janas_qtype_block_size(ly->wq.type);
        if (!rb) /* a repacked type (JANAS_GPU_REPACK): no rows to cut */
            return -1;
        for (uint32_t h = 0; h < md->n_head; h++) {
            struct janas_matvec_task q = ly->wq, gt = ly->wq;
            q.rows = gt.rows = hd;
            q.w = (const uint8_t *)ly->wq.w + (size_t)2 * h * hd * rb;
            gt.w = (const uint8_t *)q.w + (size_t)hd * rb;
            if (janas_gt_rec_product(g, t, bound, &q, 0,
                                     QKV + (size_t)h * hd * 4, n, qs) != 0 ||
                janas_gt_rec_product(g, t, bound, &gt, 0,
                                     B + t->o_gate + (size_t)h * hd * 4, n,
                                     qd) != 0)
                return -1;
        }
    } else if (janas_gt_rec_product(g, t, bound, &ly->wq, 0, QKV, n, qs) != 0)
        return -1;
    if (own_kv && (janas_gt_rec_product(g, t, bound, &ly->wk, 0,
                                        QKV + (size_t)qd * 4, n, qs) != 0 ||
                   (!v_is_k && janas_gt_rec_product(
                                   g, t, bound, &ly->wv, 0,
                                   QKV + (size_t)(qd + kvd) * 4, n, qs) != 0)))
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_QKV);
    int q16 = !md->kv16 && q16_on(hd);
    struct push_qk pq = {.n_head = md->n_head,
                         .n_kv = nkv,
                         .hd = hd,
                         .n_rot = ly->n_rot,
                         .cap = ly->cap,
                         .qs = (uint32_t)qs,
                         .eps = md->eps,
                         .flags = (ly->q_norm ? 0u : 1u) |
                                  (ly->k_norm ? 0u : 2u) | (q16 ? 4u : 0u) |
                                  (md->kv16 ? 8u : 0u) | (ly->ring ? 16u : 0u) |
                                  (v_is_k ? 32u : 0u) | (md->vnorm ? 64u : 0u)};
    janas_g_push_addr(pq.pos, POS);
    janas_g_push_addr(pq.qkv, QKV);
    janas_g_push_addr(pq.qn,
                      ly->q_norm ? janas_gt_addr_of(g, ly->q_norm, hd * 4) : 0);
    janas_g_push_addr(pq.kn,
                      ly->k_norm ? janas_gt_addr_of(g, ly->k_norm, hd * 4) : 0);
    janas_g_push_addr(pq.rope, ROPE);
    janas_g_push_addr(pq.kc, janas_gt_addr_of(g, ly->kc, cache));
    janas_g_push_addr(pq.vc, janas_gt_addr_of(g, ly->vc, cache));
    janas_g_push_addr(pq.ks, janas_gt_addr_of(g, ly->ks, scales));
    janas_g_push_addr(pq.vs, janas_gt_addr_of(g, ly->vs, scales));
    janas_gt_bind(g, bound, t->p_qk);
    janas_gt_push(g, &pq, sizeof(pq));
    g->vkCmdDispatch(g->cb, md->n_head + (own_kv ? nkv : 0), n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ROPE);
    /* the attention: runs of positions per KV head (and group of its
       query heads), then joined; token j of the block sees positions 0 ..
       pos + j (or the window's last), the block's own keys and values
       written above */
    uint32_t gt = md->n_head / nkv, ga = janas_gt_att_g(gt, hd);
    int k2 = t->p_at2 && ga == t->at_g[1] && ga != t->at_g[0];
    struct push_at pa = {.n_head = md->n_head,
                         .n_kv = nkv,
                         .hd = hd,
                         .cap = ly->cap,
                         .nch = t->rec_nch,
                         .scale = md->attn_scale,
                         .qs = (uint32_t)qs,
                         .ps = (uint32_t)janas_gt_part_tok(t, md, t->rec_nch),
                         .ls = t->ix_ls,
                         .flags = (q16 ? 1u : 0u) | (md->kv16 ? 2u : 0u) |
                                  (ly->ring ? 4u : 0u),
                         .hg = gt / ga,
                         .window = ly->window};
    janas_g_push_addr(pa.pos, POS);
    /* qwen4exp's sparse attention: the positions janas_gt_rec_qsa chose */
    if (md->ix_heads)
        janas_g_push_addr(pa.list, B + t->o_ixlist);
    janas_g_push_addr(pa.q, QKV);
    janas_g_push_addr(pa.kc, janas_gt_addr_of(g, ly->kc, cache));
    janas_g_push_addr(pa.vc, janas_gt_addr_of(g, ly->vc, cache));
    janas_g_push_addr(pa.ks, janas_gt_addr_of(g, ly->ks, scales));
    janas_g_push_addr(pa.vs, janas_gt_addr_of(g, ly->vs, scales));
    janas_g_push_addr(pa.part, B + t->o_part);
    janas_gt_bind(g, bound, k2 ? t->p_at2 : t->p_at);
    janas_gt_push(g, &pa, sizeof(pa));
    g->vkCmdDispatch(g->cb, nkv * pa.hg * t->rec_nch, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ATTN);
    /* the runs joined and the output quantized for wo, a workgroup a
       block of 256 */
    struct push_am pm = {.hd = hd,
                         .nch = t->rec_nch,
                         .ps = (uint32_t)janas_gt_part_tok(t, md, t->rec_nch),
                         .gs = qd,
                         .ls = t->ix_ls,
                         .window = ly->window};
    janas_g_push_addr(pm.gate, md->gated_q ? B + t->o_gate : 0);
    if (md->ix_heads)
        janas_g_push_addr(pm.list, B + t->o_ixlist);
    janas_g_push_addr(pm.part, B + t->o_part);
    janas_g_push_addr(pm.q256, B + t->o_q256);
    janas_g_push_addr(pm.q32, B + t->o_q32);
    janas_g_push_addr(pm.pos, POS);
    janas_gt_bind(g, bound, t->p_am);
    janas_gt_push(g, &pm, sizeof(pm));
    g->vkCmdDispatch(g->cb, qd / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_MERGE);
    /* the output projection */
    if (janas_gt_rec_product(g, t, bound, &ly->wo, 0, YWO, n, 0) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_WO);
    /* Gemma 4: the output normalized before the residual, into o_post */
    if (ly->post_attn)
        janas_gt_rec_post_norm(g, t, bound, YWO, ly->post_attn, md, n);
    return 0;
}

/* Gemma 4's per-layer branch of layer l (E2B, E4B) on the residual x (n
   tokens): its gate (f32, ple_dim rows), GELU times the layer's own inputs
   (ple_act.comp), the projection back (f32) into y, as janas_m_ple_layer
   before its norm. */
void janas_gt_rec_ple(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md,
                      const struct janas_gpu_tok_layer *ly, uint32_t l,
                      VkPipeline *bound, VkDeviceAddress x, VkDeviceAddress y,
                      uint32_t n)
{
    uint32_t P = md->ple_dim, dm = md->dm;
    VkDeviceAddress B = t->buf.addr;
    janas_gt_rec_dense(g, bound, 0,
                       janas_gt_addr_of(g, ly->ple_gate, (size_t)P * dm * 4), x,
                       B + t->o_pg, P, dm, P, n, 1.0f);
    janas_g_compute_barrier(g);
    struct push_pa pa = {.n = n, .pd = P, .w = md->n_layer * P};
    janas_g_push_addr(pa.g, B + t->o_pg);
    janas_g_push_addr(pa.in, B + t->o_ple + (size_t)l * P * 4);
    janas_g_push_addr(pa.h, B + t->o_ph);
    janas_gt_bind(g, bound, t->p_pa);
    janas_gt_push(g, &pa, sizeof(pa));
    g->vkCmdDispatch(g->cb, (n * P + 255) / 256, 1, 1);
    janas_g_compute_barrier(g);
    janas_gt_rec_dense(g, bound, 0,
                       janas_gt_addr_of(g, ly->ple_proj, (size_t)dm * P * 4),
                       B + t->o_ph, y, dm, P, dm, n, 1.0f);
    janas_g_compute_barrier(g);
}

/* Gemma 4's mixture (26B) up to its router, on the residual x (n tokens)
   whose feed-forward norm is in q256: the dense branch into o_ybr, that
   normalized into o_xbr; the router's input into o_rin; last the experts'
   input in q256 / q32 and XNF (the CPU's experts of a tier read it), as
   model.c's branch1_worker and prep2_worker. */
int janas_gt_rec_gemma_moe(struct janas_gpu *g, struct janas_gpu_tok *t,
                           const struct janas_gpu_tok_model *md,
                           const struct janas_gpu_tok_layer *ly,
                           VkPipeline *bound, VkDeviceAddress x, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t dm = md->dm;
    if (janas_gt_rec_ffn(g, t, bound, &ly->dn_gate, &ly->dn_up, &ly->dn_down, n,
                         B + t->o_ybr) != 0)
        return -1;
    janas_gt_rec_norm_quant(g, t, bound, B + t->o_ybr, 0,
                            janas_gt_addr_of(g, ly->post_norm1, dm * 4), 0,
                            B + t->o_xbr, dm, md->eps, n);
    janas_gt_rec_norm_quant_p(
        g, t, bound, x, 0, janas_gt_addr_of(g, ly->rscale, dm * 4), 0,
        B + t->o_rin, dm, md->eps, n, 0.0f, 1.0f / sqrtf((float)dm));
    janas_gt_rec_norm_quant(g, t, bound, x, 0,
                            janas_gt_addr_of(g, ly->pre_norm2, dm * 4), 0,
                            B + t->o_xnf, dm, md->eps, n);
    return 0;
}

/* ... after its experts' sum in yd: that normalized, the dense branch
   added, the sum normalized by post_ffn into o_post (model.c's
   branch2_worker and sandwich_worker before the residual) */
void janas_gt_rec_gemma_moe_tail(struct janas_gpu *g, struct janas_gpu_tok *t,
                                 const struct janas_gpu_tok_model *md,
                                 const struct janas_gpu_tok_layer *ly,
                                 VkPipeline *bound, VkDeviceAddress yd,
                                 uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t dm = md->dm;
    janas_gt_rec_norm_quant(g, t, bound, yd, 0,
                            janas_gt_addr_of(g, ly->post_norm2, dm * 4), 0,
                            B + t->o_post2, dm, md->eps, n);
    janas_gt_rec_norm_quant(g, t, bound, B + t->o_post2, B + t->o_xbr,
                            janas_gt_addr_of(g, ly->post_ffn, dm * 4), 0,
                            B + t->o_post, dm, md->eps, n);
}

/* Gemma 4's sandwich norm: rms_norm(x) * w (n vectors of dm floats) into
   o_post, through norm_quant.comp (its quantized output not used). */
void janas_gt_rec_post_norm(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkPipeline *bound, VkDeviceAddress x,
                            const float *w,
                            const struct janas_gpu_tok_model *md, uint32_t n)
{
    janas_gt_rec_norm_quant(g, t, bound, x, 0,
                            janas_gt_addr_of(g, w, md->dm * 4), 0,
                            t->buf.addr + t->o_post, md->dm, md->eps, n);
}

/* x (n vectors of d floats) quantized as it is, into the q256 and q32
   layouts at q256 and q32: norm_quant.comp with no norm. */
static void rec_quant(struct janas_gpu *g, struct janas_gpu_tok *t,
                      VkPipeline *bound, VkDeviceAddress x,
                      VkDeviceAddress q256, VkDeviceAddress q32, uint32_t d,
                      uint32_t n)
{
    struct push_nq pc = {.d = d, .det = 1}; /* as the CPU's activation */
    janas_g_push_addr(pc.x, x);
    janas_g_push_addr(pc.q256, q256);
    janas_g_push_addr(pc.q32, q32);
    janas_gt_bind(g, bound, t->p_nq);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, d / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
}

/* A feed-forward on the block normalized in q256: gate and up (the n
   tokens' gates, then their ups), SiLU(gate) up quantized, down into out
   (a dense layer's, or a mixture's shared expert). A down matrix in blocks
   of 32 (a shared expert's Q8_0, a dense layer's IQ4_NL) takes the
   activation as floats quantized into both layouts, the activation's
   shader making only the one of 256. */
int janas_gt_rec_ffn(struct janas_gpu *g, struct janas_gpu_tok *t,
                     VkPipeline *bound, const struct janas_matvec_task *gate,
                     const struct janas_matvec_task *up,
                     const struct janas_matvec_task *down, uint32_t n,
                     VkDeviceAddress out)
{
    VkDeviceAddress B = t->buf.addr, GU = B + t->o_gu, H = B + t->o_h, XA = H;
    /* a width ending inside a group of 256 (qwen4exp's 640): each token's
       rows at whole groups, the rest zero */
    uint32_t fr = (gate->rows + JANAS_QK - 1) / JANAS_QK * JANAS_QK;
    VkDeviceAddress UP = GU + (size_t)n * fr * 4;
    if (janas_gt_rec_product(g, t, bound, gate, 0, GU, n, fr) != 0 ||
        janas_gt_rec_product(g, t, bound, up, 0, UP, n, fr) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_GATEUP);
    if (janas_g_x_in_32(down->type) && getenv("JANAS_GPU_ACT_FLOAT")) {
        /* the GPU's own e^x, then quantized (JANAS_GPU_ACT_FLOAT=1: as
           before 26 Oct 2026) */
        struct push_sm ps = {.n = n * fr, .gelu = (uint32_t)t->gelu};
        janas_g_push_addr(ps.g, GU);
        janas_g_push_addr(ps.u, UP);
        janas_g_push_addr(ps.h, B + t->o_hf);
        janas_gt_bind(g, bound, t->p_sm);
        janas_gt_push(g, &ps, sizeof(ps));
        g->vkCmdDispatch(g->cb, (ps.n + 255) / 256, 1, 1);
        janas_g_compute_barrier(g);
        rec_quant(g, t, bound, B + t->o_hf, H, B + t->o_h32, fr, n);
        XA = B + t->o_h32;
    } else if (janas_g_x_in_32(down->type)) {
        /* the CPU's bits: the deterministic activation quantized straight
           into the blocks of 32 (expert_act.comp's KIND 1, as an expert in
           blocks of 32), the groups past the width never read */
        struct push_ac ac = {.blocks = n * fr / JANAS_QK,
                             .gelu = (uint32_t)t->gelu};
        janas_g_push_addr(ac.g, GU);
        janas_g_push_addr(ac.u, UP);
        janas_g_push_addr(ac.x, B + t->o_h32);
        janas_gt_bind(g, bound, g->pipea32);
        janas_gt_push(g, &ac, sizeof(ac));
        g->vkCmdDispatch(g->cb, n * fr / JANAS_QK, 1, 1);
        janas_g_compute_barrier(g);
        XA = B + t->o_h32;
    } else {
        struct push_ac ac = {.blocks = n * fr / JANAS_QK,
                             .gelu = (uint32_t)t->gelu};
        janas_g_push_addr(ac.g, GU);
        janas_g_push_addr(ac.u, UP);
        janas_g_push_addr(ac.x, H);
        janas_gt_bind(g, bound, g->pipea);
        janas_gt_push(g, &ac, sizeof(ac));
        g->vkCmdDispatch(g->cb, n * fr / JANAS_QK, 1, 1);
        janas_g_compute_barrier(g);
    }
    janas_gt_stamp(g, t, PH_ACT);
    if (janas_g_record_task(g, bound,
                            &(struct janas_matvec_task){
                                .type = down->type,
                                .w = down->w,
                                .plane = {down->plane[0], down->plane[1]},
                                .rows = down->rows,
                                .cols = down->cols,
                                .n_vec = n},
                            down->rows, XA, out, down->rows) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_DOWN);
    return 0;
}

#endif
