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
    struct push_nq pc = {.d = d, .eps = eps};
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
        struct push_dm pd = {.rows = 2 * md->nv,
                             .cols = md->dm,
                             .nv = n,
                             .ys = 2 * md->nv,
                             .scale = 1.0f};
        janas_g_push_addr(pd.w, B + pr[PAR_BA]);
        janas_g_push_addr(pd.x, B + t->o_xnf);
        janas_g_push_addr(pd.y, BA);
        janas_gt_bind(g, bound, g->pipe_dm[0]);
        janas_gt_push(g, &pd, sizeof(pd));
        g->vkCmdDispatch(g->cb, (2 * md->nv + 63) / 64, (n + 63) / 64, 1);
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
    struct push_ggate pq = {.d = md->d_inner, .nv = md->nv, .eps = md->eps};
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

/* A layer's attention on the block normalized in q256 / q32: q, k and v,
   their norms and RoPE (the keys and values into the cache), the
   attention in runs of positions, joined, then wo into YWO. */
int janas_gt_rec_attn(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md,
                      const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                      uint32_t n)
{
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    size_t qs = qd + 2 * kvd;
    size_t cache = (size_t)md->n_kv * md->cap * md->hd;
    VkDeviceAddress B = t->buf.addr, QKV = B + t->o_qkv, YWO = B + t->o_ywo,
                    ROPE = B + t->o_rope, POS = B + t->o_pos;
    /* q, k and v of each token side by side, qs floats a token; a
       gated query (qwen3.5) head by head, its gates apart */
    if (md->gated_q) {
        size_t rb =
            (size_t)(md->dm / JANAS_QK) * janas_qtype_block_size(ly->wq.type);
        for (uint32_t h = 0; h < md->n_head; h++) {
            struct janas_matvec_task q = ly->wq, gt = ly->wq;
            q.rows = gt.rows = md->hd;
            q.w = (const uint8_t *)ly->wq.w + (size_t)2 * h * md->hd * rb;
            gt.w = (const uint8_t *)q.w + (size_t)md->hd * rb;
            if (janas_gt_rec_product(g, t, bound, &q, 0,
                                     QKV + (size_t)h * md->hd * 4, n,
                                     qs) != 0 ||
                janas_gt_rec_product(g, t, bound, &gt, 0,
                                     B + t->o_gate + (size_t)h * md->hd * 4, n,
                                     qd) != 0)
                return -1;
        }
    } else if (janas_gt_rec_product(g, t, bound, &ly->wq, 0, QKV, n, qs) != 0)
        return -1;
    if (janas_gt_rec_product(g, t, bound, &ly->wk, 0, QKV + (size_t)qd * 4, n,
                             qs) != 0 ||
        janas_gt_rec_product(g, t, bound, &ly->wv, 0,
                             QKV + (size_t)(qd + kvd) * 4, n, qs) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_QKV);
    struct push_qk pq = {.n_head = md->n_head,
                         .n_kv = md->n_kv,
                         .hd = md->hd,
                         .n_rot = md->n_rot,
                         .cap = md->cap,
                         .qs = (uint32_t)qs,
                         .eps = md->eps};
    janas_g_push_addr(pq.pos, POS);
    janas_g_push_addr(pq.qkv, QKV);
    janas_g_push_addr(pq.qn, janas_gt_addr_of(g, ly->q_norm, md->hd * 4));
    janas_g_push_addr(pq.kn, janas_gt_addr_of(g, ly->k_norm, md->hd * 4));
    janas_g_push_addr(pq.rope, ROPE);
    janas_g_push_addr(pq.kc, janas_gt_addr_of(g, ly->kc, cache));
    janas_g_push_addr(pq.vc, janas_gt_addr_of(g, ly->vc, cache));
    janas_g_push_addr(pq.ks, janas_gt_addr_of(g, ly->ks, cache / md->hd * 4));
    janas_g_push_addr(pq.vs, janas_gt_addr_of(g, ly->vs, cache / md->hd * 4));
    janas_gt_bind(g, bound, t->p_qk);
    janas_gt_push(g, &pq, sizeof(pq));
    g->vkCmdDispatch(g->cb, md->n_head + md->n_kv, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ROPE);
    /* the attention: runs of positions per KV head, then joined; token
       j of the block sees positions 0 .. pos + j, the block's own keys
       and values written above */
    struct push_at pa = {.n_head = md->n_head,
                         .n_kv = md->n_kv,
                         .hd = md->hd,
                         .cap = md->cap,
                         .nch = t->rec_nch,
                         .scale = md->attn_scale,
                         .qs = (uint32_t)qs,
                         .ps = (uint32_t)janas_gt_part_tok(t, md, t->rec_nch)};
    janas_g_push_addr(pa.pos, POS);
    janas_g_push_addr(pa.q, QKV);
    janas_g_push_addr(pa.kc, janas_gt_addr_of(g, ly->kc, cache));
    janas_g_push_addr(pa.vc, janas_gt_addr_of(g, ly->vc, cache));
    janas_g_push_addr(pa.ks, janas_gt_addr_of(g, ly->ks, cache / md->hd * 4));
    janas_g_push_addr(pa.vs, janas_gt_addr_of(g, ly->vs, cache / md->hd * 4));
    janas_g_push_addr(pa.part, B + t->o_part);
    janas_gt_bind(g, bound, t->p_at);
    janas_gt_push(g, &pa, sizeof(pa));
    g->vkCmdDispatch(g->cb, md->n_kv * t->rec_nch, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ATTN);
    /* the runs joined and the output quantized for wo, a workgroup a
       block of 256 */
    struct push_am pm = {.hd = md->hd,
                         .nch = t->rec_nch,
                         .ps = (uint32_t)janas_gt_part_tok(t, md, t->rec_nch),
                         .gs = qd};
    janas_g_push_addr(pm.gate, md->gated_q ? B + t->o_gate : 0);
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
    return 0;
}

/* x (n vectors of d floats) quantized as it is, into the q256 and q32
   layouts at q256 and q32: norm_quant.comp with no norm. */
static void rec_quant(struct janas_gpu *g, struct janas_gpu_tok *t,
                      VkPipeline *bound, VkDeviceAddress x,
                      VkDeviceAddress q256, VkDeviceAddress q32, uint32_t d,
                      uint32_t n)
{
    struct push_nq pc = {.d = d};
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
    VkDeviceAddress UP = GU + (size_t)n * gate->rows * 4;
    if (janas_gt_rec_product(g, t, bound, gate, 0, GU, n, 0) != 0 ||
        janas_gt_rec_product(g, t, bound, up, 0, UP, n, 0) != 0)
        return -1;
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_GATEUP);
    if (janas_g_x_in_32(down->type)) {
        struct push_sm ps = {.n = n * gate->rows};
        janas_g_push_addr(ps.g, GU);
        janas_g_push_addr(ps.u, UP);
        janas_g_push_addr(ps.h, B + t->o_hf);
        janas_gt_bind(g, bound, t->p_sm);
        janas_gt_push(g, &ps, sizeof(ps));
        g->vkCmdDispatch(g->cb, (ps.n + 255) / 256, 1, 1);
        janas_g_compute_barrier(g);
        rec_quant(g, t, bound, B + t->o_hf, H, B + t->o_h32, gate->rows, n);
        XA = B + t->o_h32;
    } else {
        struct push_ac ac = {.blocks = n * gate->rows / JANAS_QK};
        janas_g_push_addr(ac.g, GU);
        janas_g_push_addr(ac.u, UP);
        janas_g_push_addr(ac.x, H);
        janas_gt_bind(g, bound, g->pipea);
        janas_gt_push(g, &ac, sizeof(ac));
        g->vkCmdDispatch(g->cb, n * gate->rows / JANAS_QK, 1, 1);
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
