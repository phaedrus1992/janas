/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_hc.c - Qwen3.8-Flash-Next's streams in the whole token on the
 * GPU (gpu_token_impl.h): the mixes that make each sublayer's input from
 * the four streams of the residual, the combines that add its output back,
 * and the n-gram embeddings of its second layer, as model_hc.c computes
 * them. The mixes' products are bf16 (or Q8_0, JANAS_HC_MIX=q8) on inputs
 * rounded to bf16, in dense_mm.comp's tiles; the norms and sums in the GPU's
 * order, NOT bit for bit with the CPU.
 */
#include "gpu_token_impl.h"

#include <string.h>

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include "hc_norm.h" /* janas_spv_hc_norm */
#include "hc_act.h"  /* janas_spv_hc_act */
#include "hc_mean.h" /* janas_spv_hc_mean */
#include "hc_comb.h" /* janas_spv_hc_comb */
#include "ng_gate.h" /* janas_spv_ng_gate */
#include "ng_conv.h" /* janas_spv_ng_conv */

void janas_gt_rec_bf16(struct janas_gpu *g, VkPipeline *bound,
                       VkDeviceAddress w, VkDeviceAddress x, VkDeviceAddress y,
                       uint32_t rows, uint32_t cols, uint32_t ys, uint32_t n,
                       float scale)
{
    janas_gt_rec_dense(g, bound, 1, w, x, y, rows, cols, ys, n, scale);
}

void janas_gt_rec_dense(struct janas_gpu *g, VkPipeline *bound, int kind,
                        VkDeviceAddress w, VkDeviceAddress x, VkDeviceAddress y,
                        uint32_t rows, uint32_t cols, uint32_t ys, uint32_t n,
                        float scale)
{
    struct push_dm pd = {
        .rows = rows, .cols = cols, .nv = n, .ys = ys, .scale = scale};
    janas_g_push_addr(pd.w, w);
    janas_g_push_addr(pd.x, x);
    janas_g_push_addr(pd.y, y);
    janas_gt_bind(g, bound, g->pipe_dm[kind]);
    janas_gt_push(g, &pd, sizeof(pd));
    g->vkCmdDispatch(g->cb, (rows + 63) / 64, (n + 63) / 64, 1);
}

/* x rounded to bf16 into xb (silu 0), or SiLU(x) in place and that into
   xb (silu 1): n floats */
static void rec_act(struct janas_gpu *g, struct janas_gpu_tok *t,
                    VkPipeline *bound, VkDeviceAddress x, VkDeviceAddress xb,
                    uint32_t n, int silu)
{
    struct push_ha pa = {.n = n, .silu = (uint32_t)silu};
    janas_g_push_addr(pa.x, x);
    janas_g_push_addr(pa.xb, xb);
    janas_gt_bind(g, bound, t->p_ha);
    janas_gt_push(g, &pa, sizeof(pa));
    g->vkCmdDispatch(g->cb, (n + 255) / 256, 1, 1);
}

void janas_gt_rec_hc_mix(struct janas_gpu *g, struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md,
                         const struct janas_gpu_tok_layer *ly, int k,
                         VkPipeline *bound, VkDeviceAddress xnf, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t dm = md->dm, hc = md->hc, hd = hc * dm, r = md->hc_rank;
    const float *norm = k < 2 ? ly->hc_norm[k] : md->hco_norm;
    const uint16_t *down = k < 2 ? ly->hc_down[k] : md->hco_down,
                   *up = k < 2 ? ly->hc_up[k] : md->hco_up,
                   *inject = k < 2 ? ly->hc_inject[k] : NULL;
    /* each stream normalized, and at bf16 */
    struct push_hn pn = {.d = dm, .hc = hc, .eps = md->eps};
    janas_g_push_addr(pn.res, B + t->o_res);
    janas_g_push_addr(pn.w, janas_gt_addr_of(g, norm, (size_t)hd * 4));
    janas_g_push_addr(pn.xn, B + t->o_hxn);
    janas_g_push_addr(pn.xb, B + t->o_hxb);
    janas_gt_bind(g, bound, t->p_hn);
    janas_gt_push(g, &pn, sizeof(pn));
    g->vkCmdDispatch(g->cb, hc, n, 1);
    janas_g_compute_barrier(g);
    /* the gates' low rank: down / hc, its SiLU and bf16; up; and inject
       beside them, on the same input. Down and up bf16, or Q8_0 (hc_q8) */
    int kind = md->hc_q8 ? 2 : 1;
    size_t mb = md->hc_q8 ? (size_t)r * hd / 32 * 34 : (size_t)r * hd * 2;
    janas_gt_rec_dense(g, bound, kind, janas_gt_addr_of(g, down, mb),
                       B + t->o_hxb, B + t->o_lo, r, hd, r, n,
                       1.0f / (float)hc);
    if (inject)
        janas_gt_rec_bf16(g, bound,
                          janas_gt_addr_of(g, inject, (size_t)hc * hd * 2),
                          B + t->o_hxb, B + t->o_inj, hc, hd, hc, n, 1.0f);
    janas_g_compute_barrier(g);
    rec_act(g, t, bound, B + t->o_lo, B + t->o_lob, n * r, 1);
    janas_g_compute_barrier(g);
    janas_gt_rec_dense(g, bound, kind, janas_gt_addr_of(g, up, mb),
                       B + t->o_lob, B + t->o_hg, hd, r, hd, n, 1.0f);
    janas_g_compute_barrier(g);
    /* the input: the streams' mean, each times sigmoid(its gate) */
    struct push_hm pm = {.d = dm, .hc = hc};
    janas_g_push_addr(pm.xn, B + t->o_hxn);
    janas_g_push_addr(pm.g, B + t->o_hg);
    janas_g_push_addr(pm.out, B + t->o_hmean);
    janas_gt_bind(g, bound, t->p_hm);
    janas_gt_push(g, &pm, sizeof(pm));
    g->vkCmdDispatch(g->cb, (dm + 255) / 256, n, 1);
    janas_g_compute_barrier(g);
    /* quantized for the products, and as floats */
    janas_gt_rec_norm_quant(g, t, bound, B + t->o_hmean, 0, 0, 0, xnf, dm,
                            md->eps, n);
}

void janas_gt_rec_hc_combine(struct janas_gpu *g, struct janas_gpu_tok *t,
                             const struct janas_gpu_tok_model *md,
                             VkPipeline *bound, VkDeviceAddress out, int init,
                             uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t hd = md->hc * md->dm;
    struct push_hcb pc = {.d = md->dm, .hc = md->hc, .init = (uint32_t)init};
    janas_g_push_addr(pc.res, B + t->o_res);
    janas_g_push_addr(pc.out, out);
    janas_g_push_addr(pc.inj, B + t->o_inj);
    janas_gt_bind(g, bound, t->p_hcb);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, (hd + 255) / 256, n, 1);
    janas_g_compute_barrier(g);
}

int janas_gt_rec_ng(struct janas_gpu *g, struct janas_gpu_tok *t,
                    const struct janas_gpu_tok_model *md,
                    const struct janas_gpu_tok_layer *ly, VkPipeline *bound,
                    uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t dm = md->dm, hc = md->hc, hd = hc * dm;
    VkDeviceAddress HIST =
        janas_gt_addr_of(g, md->ng_hist, (size_t)md->ng_cap * hd * 4);
    if (!HIST)
        return -1;
    /* the embeddings (the caller's rows of the table) quantized for the
       key product, and at bf16 for the value's */
    janas_gt_rec_norm_quant(g, t, bound, B + t->o_ngemb, 0, 0, 0, 0, dm,
                            md->eps, n);
    if (janas_gt_rec_product(g, t, bound, &ly->ng_key, 0, B + t->o_ngk, n,
                             hd) != 0)
        return -1;
    rec_act(g, t, bound, B + t->o_ngemb, B + t->o_ngeb, n * dm, 0);
    janas_g_compute_barrier(g);
    janas_gt_rec_bf16(g, bound,
                      janas_gt_addr_of(g, ly->ng_value, (size_t)dm * dm * 2),
                      B + t->o_ngeb, B + t->o_ngv, dm, dm, dm, n, 1.0f);
    janas_g_compute_barrier(g);
    struct push_ngg pg = {.d = dm, .hc = hc, .cap = md->ng_cap, .eps = md->eps};
    janas_g_push_addr(pg.kq, B + t->o_ngk);
    janas_g_push_addr(pg.res, B + t->o_res);
    janas_g_push_addr(pg.val, B + t->o_ngv);
    janas_g_push_addr(pg.nk, janas_gt_addr_of(g, ly->ng_nk, (size_t)hd * 4));
    janas_g_push_addr(pg.nq, janas_gt_addr_of(g, ly->ng_nq, (size_t)hd * 4));
    janas_g_push_addr(pg.nc, janas_gt_addr_of(g, ly->ng_nc, (size_t)hd * 4));
    janas_g_push_addr(pg.gv, B + t->o_nggv);
    janas_g_push_addr(pg.hist, HIST);
    janas_g_push_addr(pg.pos, B + t->o_pos);
    janas_gt_bind(g, bound, t->p_ngg);
    janas_gt_push(g, &pg, sizeof(pg));
    g->vkCmdDispatch(g->cb, hc, n, 1);
    janas_g_compute_barrier(g);
    struct push_ngc pc = {
        .hd = hd, .cap = md->ng_cap, .k = md->ng_kernel, .len = md->ng_len};
    janas_g_push_addr(pc.res, B + t->o_res);
    janas_g_push_addr(pc.gv, B + t->o_nggv);
    janas_g_push_addr(pc.w, B + t->o_ngw);
    janas_g_push_addr(pc.hist, HIST);
    janas_g_push_addr(pc.pos, B + t->o_pos);
    janas_gt_bind(g, bound, t->p_ngc);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, (hd + 255) / 256, n, 1);
    janas_g_compute_barrier(g);
    return 0;
}

int janas_gt_hc_create(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md)
{
    if (janas_g_make_pipeline(g, janas_spv_hc_norm, sizeof(janas_spv_hc_norm),
                              &t->sm_hn, &t->p_hn, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_hc_act, sizeof(janas_spv_hc_act),
                              &t->sm_ha, &t->p_ha, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_hc_mean, sizeof(janas_spv_hc_mean),
                              &t->sm_hm, &t->p_hm, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_hc_comb, sizeof(janas_spv_hc_comb),
                              &t->sm_hcb, &t->p_hcb, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_ng_gate, sizeof(janas_spv_ng_gate),
                              &t->sm_ngg, &t->p_ngg, 0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_ng_conv, sizeof(janas_spv_ng_conv),
                              &t->sm_ngc, &t->p_ngc, 0) != VK_SUCCESS ||
        !g->pipe_dm[1])
        return -1;
    /* the n-gram layer's conv taps, copied once (the model's own are not
       where the GPU reads) */
    if (md->ng_layer < md->n_layer) {
        size_t bytes = (size_t)md->hc * md->dm * md->ng_kernel * 4;
        memcpy(t->buf.map + t->o_ngw, md->layers[md->ng_layer].ng_conv, bytes);
        janas_gt_sync_range(g, t, t->o_ngw, bytes, 1);
    }
    return 0;
}

void janas_gt_hc_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    VkPipeline p[6] = {t->p_hn, t->p_ha, t->p_hm, t->p_hcb, t->p_ngg, t->p_ngc};
    VkShaderModule s[6] = {t->sm_hn,  t->sm_ha,  t->sm_hm,
                           t->sm_hcb, t->sm_ngg, t->sm_ngc};
    for (int i = 0; i < 6; i++) {
        if (p[i])
            g->vkDestroyPipeline(g->dev, p[i], NULL);
        if (s[i])
            g->vkDestroyShaderModule(g->dev, s[i], NULL);
    }
}

#endif
