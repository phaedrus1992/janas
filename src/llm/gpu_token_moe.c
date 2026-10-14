/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_moe.c - a mixture's layers in the whole token on the GPU
 * (gpu_token_impl.h): its router's choice made on the GPU, sorted on the
 * CPU between two submissions, its experts' products and their sum.
 */
#include "gpu_token_impl.h"

#include <stdlib.h>
#include <string.h>

#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

/* A mixture's submissions for one block shape, freed (n_layer + 1 of
   them, the array too). */
void janas_gt_free_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkCommandBuffer **seg)
{
    if (!*seg)
        return;
    for (uint32_t s = 0; s < t->n_seg; s++)
        if ((*seg)[s])
            g->vkFreeCommandBuffers(g->dev, g->cp, 1, &(*seg)[s]);
    free(*seg);
    *seg = NULL;
}

/* A mixture's layer l up to its router's choice, on the block normalized
   in q256 and XNF: the shared expert into SH, the router's logits, then
   each token's experts and weights (PE, PW, SHW), for the CPU to read. */
int janas_gt_rec_router(struct janas_gpu *g, struct janas_gpu_tok *t,
                        const struct janas_gpu_tok_model *md,
                        const struct janas_gpu_tok_layer *ly, uint32_t l,
                        VkPipeline *bound, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t ne = md->n_expert;
    if (ly->sh_gate_w.type &&
        janas_gt_rec_ffn(g, t, bound, &ly->sh_gate_w, &ly->sh_up, &ly->sh_down,
                         n, B + t->o_sh) != 0)
        return -1;
    struct push_dm pd = {.rows = ne + (ly->sh_gate ? 1 : 0),
                         .cols = md->dm,
                         .nv = n,
                         .ys = ne + 1,
                         .scale = 1.0f};
    janas_g_push_addr(pd.w, B + t->par[(size_t)l * PAR_N + PAR_R]);
    janas_g_push_addr(pd.x, B + t->o_xnf);
    janas_g_push_addr(pd.y, B + t->o_r);
    janas_gt_bind(g, bound, g->pipe_dm[0]);
    janas_gt_push(g, &pd, sizeof(pd));
    g->vkCmdDispatch(g->cb, (pd.rows + 63) / 64, (n + 63) / 64, 1);
    janas_g_compute_barrier(g);
    struct push_tk pk = {
        .ne = ne, .rs = ne + 1, .k = md->k, .n = n, .sh = ly->sh_gate != NULL};
    janas_g_push_addr(pk.r, B + t->o_r);
    janas_g_push_addr(pk.pe, B + t->o_pe);
    janas_g_push_addr(pk.pw, B + t->o_pw);
    janas_g_push_addr(pk.shw, B + t->o_shw);
    janas_gt_bind(g, bound, t->p_tk);
    janas_gt_push(g, &pk, sizeof(pk));
    g->vkCmdDispatch(g->cb, (n + 63) / 64, 1, 1);
    janas_g_compute_barrier(g);
    return 0;
}

/* A mixture's experts on the block whose router chose them (plan wrote
   their tables and sizes): the activations gathered in the experts'
   order, gate and up, the activation, down, then each token's weighted
   sum (and the shared expert) into YD. Experts in blocks of 32 (MXFP4)
   read the activations in that layout: the block's q32, and the
   activation quantized into it (expert_act.comp's KIND 1). */
void janas_gt_rec_experts(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md,
                          const struct janas_gpu_tok_layer *ly,
                          VkPipeline *bound, VkDeviceAddress YD, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t np = n * md->k, nbd = md->dm / JANAS_QK, nbf = md->eff / JANAS_QK;
    int dt = ly->down.type;
    struct push_ga pg = {.nb = nbd};
    janas_g_push_addr(
        pg.src, B + (janas_g_x_in_32(ly->gate.type) ? t->o_q32 : t->o_q256));
    janas_g_push_addr(pg.dst, B + t->o_xg);
    janas_g_push_addr(pg.tok, B + t->o_gt);
    janas_gt_bind(g, bound, t->p_ga);
    janas_gt_push(g, &pg, sizeof(pg));
    g->vkCmdDispatch(g->cb, np, 1, 1);
    janas_g_compute_barrier(g);
    struct push_id pi = {.rows = md->eff, .nb = nbd, .tiles = t->tiles1};
    janas_g_push_addr(pi.t, B + t->o_tab1);
    janas_gt_bind(g, bound,
                  ly->gate.type == JANAS_MXFP4 ? g->pipe_idmx : g->pipe_id4);
    janas_gt_push(g, &pi, sizeof(pi));
    g->vkCmdDispatchIndirect(g->cb, t->buf.buf, t->o_ind);
    janas_g_compute_barrier(g);
    struct push_ac ac = {.blocks = np * nbf};
    janas_g_push_addr(ac.g, B + t->o_egu);
    janas_g_push_addr(ac.u, B + t->o_egu + (size_t)np * md->eff * 4);
    janas_g_push_addr(ac.x, B + t->o_eh);
    janas_gt_bind(g, bound, janas_g_x_in_32(dt) ? g->pipea32 : g->pipea);
    janas_gt_push(g, &ac, sizeof(ac));
    g->vkCmdDispatch(g->cb, np * nbf, 1, 1);
    janas_g_compute_barrier(g);
    pi = (struct push_id){.rows = md->dm, .nb = nbf, .tiles = t->tiles2};
    janas_g_push_addr(pi.t, B + t->o_tab2);
    janas_gt_bind(g, bound,
                  dt == JANAS_Q4_K    ? g->pipe_id4
                  : dt == JANAS_Q6_K  ? g->pipe_id6s
                  : dt == JANAS_MXFP4 ? g->pipe_idmx
                                      : g->pipe_id6);
    janas_gt_push(g, &pi, sizeof(pi));
    g->vkCmdDispatchIndirect(g->cb, t->buf.buf, t->o_ind + 16);
    janas_g_compute_barrier(g);
    struct push_cb pc = {
        .dm = md->dm, .k = md->k, .has_sh = ly->sh_gate_w.type != 0};
    janas_g_push_addr(pc.dout, B + t->o_dout);
    janas_g_push_addr(pc.pw, B + t->o_pw);
    janas_g_push_addr(pc.inv, B + t->o_inv);
    janas_g_push_addr(pc.sh, B + t->o_sh);
    janas_g_push_addr(pc.shw, B + t->o_shw);
    janas_g_push_addr(pc.y, YD);
    janas_gt_bind(g, bound, t->p_cb);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, md->dm / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
}

/* An entry of a *_mmid table: weights, activations, outputs (ys floats
   apart), vectors, and the down matrix's bare planes (or 0). */
static void table_entry(uint32_t *e, VkDeviceAddress w, VkDeviceAddress x,
                        VkDeviceAddress y, uint32_t nv, uint32_t ys,
                        VkDeviceAddress p0, VkDeviceAddress p1)
{
    janas_g_push_addr(e, w);
    janas_g_push_addr(e + 2, x);
    janas_g_push_addr(e + 4, y);
    e[6] = nv;
    e[7] = ys;
    janas_g_push_addr(e + 8, p0);
    janas_g_push_addr(e + 10, p1);
}

/*
 * Between two of a mixture's submissions, on the CPU: layer l's choice (n
 * tokens, k experts each) read back and sorted by expert, as model_ffn.c
 * does - each row's token, each pair's row - the experts' slots asked of
 * the model, the tables of the products and the sizes of their dispatches
 * written. 0, or -1 (an expert unreadable, or out of the GPU's reach).
 */
static int plan(struct janas_gpu *g, struct janas_gpu_tok *t,
                const struct janas_gpu_tok_model *md, uint32_t l, uint32_t n)
{
    const struct janas_gpu_tok_layer *ly = &md->layers[l];
    uint32_t k = md->k, ne = md->n_expert, np = n * k, nd = 0;
    uint32_t nbd = md->dm / JANAS_QK, nbf = md->eff / JANAS_QK;
    janas_gt_sync_range(g, t, t->o_pe, (size_t)np * 4, 0);
    const uint32_t *pe = (const uint32_t *)(t->buf.map + t->o_pe);
    uint32_t *cnt = t->cnt, *gt = (uint32_t *)(t->buf.map + t->o_gt),
             *inv = (uint32_t *)(t->buf.map + t->o_inv);
    memset(cnt, 0, (ne + 1) * sizeof(*cnt));
    for (uint32_t p = 0; p < np; p++) {
        if (pe[p] >= ne)
            return -1;
        cnt[pe[p] + 1]++;
    }
    if (md->routed)
        md->routed(md->ctx, l, n, k, pe);
    for (uint32_t e = 0; e < ne; e++) {
        if (cnt[e + 1])
            t->ids[nd++] = e;
        cnt[e + 1] += cnt[e];
        t->fill[e] = cnt[e];
    }
    for (uint32_t p = 0; p < np; p++) {
        uint32_t r = t->fill[pe[p]]++;
        gt[r] = p / k;
        inv[p] = r;
    }
    if (md->fetch(md->ctx, l, t->ids, nd, t->slots) != 0)
        return -1;
    VkDeviceAddress B = t->buf.addr;
    uint32_t *t1 = (uint32_t *)(t->buf.map + t->o_tab1),
             *t2 = (uint32_t *)(t->buf.map + t->o_tab2);
    uint32_t n1 = 0, n2 = 0; /* tiles */
    size_t gb = (size_t)md->eff * nbd * janas_qtype_block_size(ly->gate.type),
           db =
               (size_t)md->dm * nbf * janas_qtype_block_size(ly->down.type) + 4,
           pb = (size_t)md->dm * nbf * JANAS_Q6KP_PLANE;
    for (uint32_t i = 0; i < nd; i++) {
        uint32_t e = t->ids[i], a = cnt[e], nv = cnt[e + 1] - a;
        const uint8_t *s = t->slots[i];
        VkDeviceAddress wg = janas_gt_addr_of(g, s + ly->off[0], gb),
                        wu = janas_gt_addr_of(g, s + ly->off[1], gb),
                        wd = janas_gt_addr_of(g, s + ly->off[2], db), p0 = 0,
                        p1 = 0;
        /* the planes the cache holds (none, one or both: level 1 to 3) */
        if (ly->down.type == JANAS_Q6_K_P &&
            ((ly->poff[0] &&
              !(p0 = janas_gt_addr_of(g, s + ly->poff[0], pb))) ||
             (ly->poff[1] && !(p1 = janas_gt_addr_of(g, s + ly->poff[1], pb)))))
            return -1;
        if (!wg || !wu || !wd)
            return -1;
        VkDeviceAddress x = B + t->o_xg + (VkDeviceAddress)a * nbd * 320;
        table_entry(t1 + 12 * (2 * i), wg, x,
                    B + t->o_egu + (VkDeviceAddress)a * md->eff * 4, nv,
                    md->eff, 0, 0);
        table_entry(t1 + 12 * (2 * i + 1), wu, x,
                    B + t->o_egu + ((VkDeviceAddress)np + a) * md->eff * 4, nv,
                    md->eff, 0, 0);
        table_entry(t2 + 12 * i, wd,
                    B + t->o_eh + (VkDeviceAddress)a * nbf * 320,
                    B + t->o_dout + (VkDeviceAddress)a * md->dm * 4, nv, md->dm,
                    p0, p1);
        for (uint32_t v = 0; v < nv; v += 16) {
            uint32_t *u = t1 + 4 * (t->tiles1 + n1);
            u[0] = 2 * i, u[1] = v, u[2] = u[3] = 0;
            u[4] = 2 * i + 1, u[5] = v, u[6] = u[7] = 0;
            n1 += 2;
            u = t2 + 4 * (t->tiles2 + n2++);
            u[0] = i, u[1] = v, u[2] = u[3] = 0;
        }
    }
    uint32_t *ind = (uint32_t *)(t->buf.map + t->o_ind);
    ind[0] = (md->eff + 63) / 64, ind[1] = 1, ind[2] = n1;
    ind[4] = (md->dm + 63) / 64, ind[5] = 1, ind[6] = n2;
    janas_gt_sync_range(g, t, t->o_gt, (size_t)np * 4, 1);
    janas_gt_sync_range(g, t, t->o_inv, (size_t)np * 4, 1);
    janas_gt_sync_range(g, t, t->o_tab1, ((size_t)t->tiles1 + n1) * 16, 1);
    janas_gt_sync_range(g, t, t->o_tab2, ((size_t)t->tiles2 + n2) * 16, 1);
    janas_gt_sync_range(g, t, t->o_ind, 32, 1);
    return 0;
}

/* A mixture's block: its n_layer + 1 submissions, each recorded once for
   its shape and runs, the CPU's plan between two of them. 0, or -1. */
int janas_gt_run_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md, uint32_t n,
                          uint32_t first, int ai, int hi, uint32_t nch)
{
    VkCommandBuffer **seg = &t->seg_cb[n][ai][hi];
    if (*seg && t->seg_nch[n][ai][hi] != nch)
        janas_gt_free_segments(g, t, seg); /* recorded for other runs: again */
    if (!*seg) {
        *seg = calloc(t->n_seg, sizeof(**seg));
        if (!*seg)
            return -1;
        t->seg_nch[n][ai][hi] = nch;
    }
    /* a block whose tokens choose some 4 times each expert or more: those
       few that no token takes are under 2% (e^-4) */
    int ahead = md->prefetch && (size_t)n * md->k >= 4 * (size_t)md->n_expert;
    for (uint32_t s = 0; s < t->n_seg; s++) {
        if (s && plan(g, t, md, s - 1, n) != 0)
            return -1;
        /* layer s's experts read while the GPU computes up to its router's
           choice, as the CPU's path reads the next layer's (model_ffn.c
           janas_m_prefetch_block): without, the GPU waited for each
           layer's reads, and on Qwen3.6-35B, whose experts do not all fit,
           the cache found 79% of them instead of 87% and the prompt was
           no faster (10 Oct 2026) */
        if (ahead && s + 1 < t->n_seg)
            md->prefetch(md->ctx, s);
        if (!(*seg)[s] && janas_gt_make_cb(g, t, md, &(*seg)[s], n, first, hi,
                                           nch, (int)s) != 0)
            return -1;
        if (janas_gt_submit_wait(g, &(*seg)[s]) != 0)
            return -1;
    }
    return 0;
}

#endif
