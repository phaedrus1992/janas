/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_qsa.c - Qwen3.8-Flash-Next's sparse attention in the whole token
 * on the GPU (gpu_token_impl.h), as model_qsa.c computes it: in each
 * attention layer the indexer's keys and queries from the layer's input at
 * bf16 (dense_mm.comp), the queries normalized and turned and the raw keys
 * kept (ix_prep.comp), the blocks the run completes pooled (ix_pool.comp),
 * each token's scores of the blocks before it (ix_score.comp) and the
 * positions it attends to (ix_select.comp): every one while its blocks are
 * no more than the budget, else the best blocks and its own block's
 * positions. The attention's runs then read those positions. Sums in the
 * GPU's order: a block whose score ties with another's on the CPU may lose
 * here, as llama.cpp's choice differs from both.
 */
#include "gpu_token_impl.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include "ix_pool.h"   /* janas_spv_ix_pool */
#include "ix_prep.h"   /* janas_spv_ix_prep */
#include "ix_score.h"  /* janas_spv_ix_score */
#include "ix_select.h" /* janas_spv_ix_select */

int janas_gt_rec_qsa(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md,
                     const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                     VkDeviceAddress xb, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr, POS = B + t->o_pos;
    uint32_t dm = md->dm, d = md->ix_dim, nh = md->ix_heads, r = md->ix_ratio;
    VkDeviceAddress RAW = janas_gt_addr_of(g, ly->ix_raw,
                                           (size_t)md->cap * d * 2),
                    POOL = janas_gt_addr_of(g, ly->ix_pool,
                                            (size_t)t->ix_nbs * d * 4),
                    WQ = janas_gt_addr_of(g, ly->ix_q, (size_t)nh * d * dm * 2),
                    WK = janas_gt_addr_of(g, ly->ix_k, (size_t)d * dm * 2),
                    QN = janas_gt_addr_of(g, ly->ix_qn, (size_t)d * 4),
                    KN = janas_gt_addr_of(g, ly->ix_kn, (size_t)d * 4);
    if (!RAW || !POOL || !WQ || !WK || !QN || !KN)
        return -1;
    /* the queries and keys */
    janas_gt_rec_bf16(g, bound, WQ, xb, B + t->o_ixq, nh * d, dm, nh * d, n,
                      1.0f);
    janas_gt_rec_bf16(g, bound, WK, xb, B + t->o_ixk, d, dm, d, n, 1.0f);
    janas_g_compute_barrier(g);
    struct push_ixp pp = {
        .nh = nh, .d = d, .n_rot = md->n_rot, .qs = nh * d, .eps = md->eps};
    janas_g_push_addr(pp.q, B + t->o_ixq);
    janas_g_push_addr(pp.k, B + t->o_ixk);
    janas_g_push_addr(pp.qn, QN);
    janas_g_push_addr(pp.rope, B + t->o_ixrope);
    janas_g_push_addr(pp.raw, RAW);
    janas_g_push_addr(pp.pos, POS);
    janas_gt_bind(g, bound, t->p_ixp);
    janas_gt_push(g, &pp, sizeof(pp));
    g->vkCmdDispatch(g->cb, nh + 1, n, 1);
    janas_g_compute_barrier(g);
    /* the blocks completed: at most n / r + 1 of them from pos / r on */
    struct push_ixl pl = {
        .d = d, .r = r, .n = n, .n_rot = md->n_rot, .rb = n, .eps = md->eps};
    janas_g_push_addr(pl.raw, RAW);
    janas_g_push_addr(pl.kn, KN);
    janas_g_push_addr(pl.rope, B + t->o_ixrope);
    janas_g_push_addr(pl.pool, POOL);
    janas_g_push_addr(pl.pos, POS);
    janas_gt_bind(g, bound, t->p_ixl);
    janas_gt_push(g, &pl, sizeof(pl));
    g->vkCmdDispatch(g->cb, n / r + 2, 1, 1);
    janas_g_compute_barrier(g);
    uint32_t keep = md->ix_top / r;
    struct push_ixs ps = {
        .nh = nh, .d = d, .r = r, .keep = keep, .nbs = t->ix_nbs, .qs = nh * d};
    janas_g_push_addr(ps.q, B + t->o_ixq);
    janas_g_push_addr(ps.pool, POOL);
    janas_g_push_addr(ps.sc, B + t->o_ixsc);
    janas_g_push_addr(ps.pos, POS);
    janas_gt_bind(g, bound, t->p_ixs);
    janas_gt_push(g, &ps, sizeof(ps));
    g->vkCmdDispatch(g->cb, (t->ix_nbs + 255) / 256, n, 1);
    janas_g_compute_barrier(g);
    struct push_ixc pc = {
        .r = r, .keep = keep, .nbs = t->ix_nbs, .ls = t->ix_ls};
    janas_g_push_addr(pc.sc, B + t->o_ixsc);
    janas_g_push_addr(pc.list, B + t->o_ixlist);
    janas_g_push_addr(pc.pos, POS);
    janas_gt_bind(g, bound, t->p_ixc);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, 1, n, 1);
    janas_g_compute_barrier(g);
    return 0;
}

int janas_gt_qsa_create(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    if (janas_g_make_pipeline(g, janas_spv_ix_prep, sizeof(janas_spv_ix_prep),
                              &t->sm_ixp, &t->p_ixp, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_ix_pool, sizeof(janas_spv_ix_pool),
                              &t->sm_ixl, &t->p_ixl, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_ix_score, sizeof(janas_spv_ix_score),
                              &t->sm_ixs, &t->p_ixs, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_ix_select,
                              sizeof(janas_spv_ix_select), &t->sm_ixc,
                              &t->p_ixc, 0) != VK_SUCCESS ||
        !g->pipe_dm[1])
        return -1;
    return 0;
}

void janas_gt_qsa_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    VkPipeline p[4] = {t->p_ixp, t->p_ixl, t->p_ixs, t->p_ixc};
    VkShaderModule s[4] = {t->sm_ixp, t->sm_ixl, t->sm_ixs, t->sm_ixc};
    for (int i = 0; i < 4; i++) {
        if (p[i])
            g->vkDestroyPipeline(g->dev, p[i], NULL);
        if (s[i])
            g->vkDestroyShaderModule(g->dev, s[i], NULL);
    }
}

#endif
