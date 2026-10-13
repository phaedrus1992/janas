/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.c - a whole token on the GPU with one submission (see
 * gpu_token.h): its buffers and shaders, the checks, the commands of a
 * block and their submission. The commands' pieces are in gpu_token_rec.c,
 * a mixture's in gpu_token_moe.c (gpu_token_impl.h).
 */
#include "gpu_token.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu_token_impl.h"
#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include "norm_quant.h"  /* janas_spv_norm_quant */
#include "qk_rope.h"     /* janas_spv_qk_rope */
#include "attn_decode.h" /* janas_spv_attn_decode */
#include "attn_merge.h"  /* janas_spv_attn_merge */
#include "gdn_conv.h"    /* janas_spv_gdn_conv */
#include "gdn_cstate.h"  /* janas_spv_gdn_cstate */
#include "gdn_l2.h"      /* janas_spv_gdn_l2 */
#include "gdn_rec.h"     /* janas_spv_gdn_rec */
#include "gdn_gate.h"    /* janas_spv_gdn_gate */
#include "gdn_ba.h"      /* janas_spv_gdn_ba */
#include "gdn_act.h"     /* janas_spv_gdn_act */
#include "moe_topk.h"    /* janas_spv_moe_topk */
#include "moe_gather.h"  /* janas_spv_moe_gather */
#include "moe_combine.h" /* janas_spv_moe_combine */
#include "silu_mul.h"    /* janas_spv_silu_mul */

/* The attention's first step in subgroups of 16 where the size may be
   chosen (JANAS_GPU_ATT_SG=n another): twice the lanes a hardware thread
   of the integrated GPU, its latencies better hidden - 11.7 -> 9.9 ms a
   token at 4000 positions - where the products lose with them. */
static uint32_t att_sg(void)
{
    const char *e = getenv("JANAS_GPU_ATT_SG");
    return e ? (uint32_t)atoi(e) : 16;
}
static const char *const ph_name[PH_N] = {"norm+quant",    "q k v",
                                          "q k norm+rope", "attention",
                                          "attn merge",    "wo",
                                          "gate+up",       "act",
                                          "down",          "output",
                                          "ssm q k v z",   "ssm beta+alpha",
                                          "ssm conv",      "ssm recurrence",
                                          "ssm gate"};
int janas_gpu_shares_host(const struct janas_gpu *g)
{
    return g && !g->broken && !g->discrete && g->host_import &&
           !getenv("JANAS_GPU_COPY");
}

/* The GPU's address of host memory in one of its regions, 0 if none. */
VkDeviceAddress janas_gt_addr_of(const struct janas_gpu *g, const void *p,
                                 size_t bytes)
{
    const struct region *rg = janas_g_find_region(g, p, bytes);
    return rg ? rg->addr + (VkDeviceAddress)((const uint8_t *)p - rg->host) : 0;
}

static size_t take(size_t *at, size_t bytes)
{
    size_t o = *at;
    *at += (bytes + 255) / 256 * 256;
    return o;
}

static int fail(char *err, size_t len, const char *why)
{
    snprintf(err, len, "%s", why);
    return -1;
}

/* Whether a layer's feed-forward is one the GPU computes: a dense one's
   weights where it reads them, a mixture's experts of the types its
   *_mmid shaders take and its shared expert's weights. NULL, or why not. */
static const char *check_ffn(const struct janas_gpu *g,
                             const struct janas_gpu_tok_model *md,
                             const struct janas_gpu_tok_layer *ly)
{
    if (!md->n_expert) {
        const struct janas_matvec_task *f[3] = {&ly->gate, &ly->up, &ly->down};
        for (int i = 0; i < 3; i++)
            if (!janas_gpu_can(g, f[i]))
                return "a feed-forward weight the GPU cannot compute";
        return NULL;
    }
    int dt = ly->down.type;
    if (ly->gate.type != JANAS_Q4_K || ly->up.type != JANAS_Q4_K ||
        !(dt == JANAS_Q4_K || dt == JANAS_Q6_K || dt == JANAS_Q6_K_P) ||
        ly->gate.rows != md->eff || ly->up.rows != md->eff ||
        ly->gate.cols != md->dm || ly->up.cols != md->dm ||
        ly->down.rows != md->dm || ly->down.cols != md->eff)
        return "experts of a type or shape the GPU's shaders do not take";
    if (!ly->router16)
        return "a mixture's layer without its router";
    if (ly->sh_gate_w.type) {
        const struct janas_matvec_task *s[3] = {&ly->sh_gate_w, &ly->sh_up,
                                                &ly->sh_down};
        for (int i = 0; i < 3; i++)
            if (!janas_gpu_can(g, s[i]))
                return "a shared expert's weight the GPU cannot compute";
        if (ly->sh_gate_w.rows != md->ff || ly->sh_up.rows != md->ff ||
            ly->sh_down.cols != md->ff || ly->sh_down.rows != md->dm)
            return "a shared expert of another shape";
    }
    return NULL;
}

/* Whether the model's pieces are all where the GPU can read them. */
static int check_model(const struct janas_gpu *g,
                       const struct janas_gpu_tok_model *md, char *err,
                       size_t len)
{
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    if (md->dm % JANAS_QK || qd % JANAS_QK || md->ff % JANAS_QK ||
        md->dm > 8192 || qd > 8192 || md->hd > 512 || md->hd % 16 ||
        md->n_rot > md->hd || md->n_kv == 0 || md->n_head % md->n_kv ||
        md->n_head / md->n_kv * md->hd > ATT_MAXQ ||
        md->n_head / md->n_kv * ATT_RUN > ATT_MAXS)
        return fail(err, len, "a shape the token's shaders do not take");
    if (md->n_expert && (md->n_expert > 512 || md->k == 0 || md->k > 16 ||
                         md->k > md->n_expert || md->eff % JANAS_QK ||
                         !md->fetch || !g->pipe_id4 || !g->pipe_id6 ||
                         !g->pipe_id6s || !g->pipea || !g->pipe_dm[0]))
        return fail(err, len, "a mixture the token's shaders do not take");
    if (md->n_rec &&
        (md->ds != 128 || md->conv_ch % 256 || md->d_inner % 256 ||
         md->d_inner != md->nv * md->ds || md->nv > 64 || !md->nkh ||
         md->nv % md->nkh || md->d_conv < 2 || md->d_conv > 8 ||
         md->d_inner > md->dm * 4 ||
         md->conv_ch != 2 * md->nkh * md->ds + md->d_inner ||
         !janas_gt_addr_of(g, md->ssm_state,
                           2 * (size_t)md->n_rec * md->nv * md->ds * md->ds *
                               2) ||
         !janas_gt_addr_of(g, md->conv_state,
                           2 * (size_t)md->n_rec * (md->d_conv - 1) *
                               md->conv_ch * 4)))
        return fail(err, len,
                    "recurrent layers the token's shaders do not take");
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        const char *why = check_ffn(g, md, ly);
        if (why) {
            snprintf(err, len, "layer %u: %s", l, why);
            return -1;
        }
        if (ly->rec) {
            const struct janas_matvec_task *r[3] = {&ly->ssm_qkv, &ly->ssm_z,
                                                    &ly->ssm_out};
            for (int i = 0; i < 3; i++)
                if (!janas_gpu_can(g, r[i])) {
                    snprintf(err, len,
                             "layer %u: a weight the GPU cannot compute "
                             "(product %d, type %d)",
                             l, i, r[i]->type);
                    return -1;
                }
            if (!janas_gt_addr_of(g, ly->attn_norm, md->dm * 4) ||
                !janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4))
                return fail(err, len, "a norm out of the GPU's reach");
            continue;
        }
        const struct janas_matvec_task *t[4] = {&ly->wq, &ly->wk, &ly->wv,
                                                &ly->wo};
        for (int i = 0; i < 4; i++)
            if (!janas_gpu_can(g, t[i])) {
                snprintf(err, len,
                         "layer %u: a weight the GPU cannot compute (product "
                         "%d, type %d)",
                         l, i, t[i]->type);
                return -1;
            }
        if (md->gated_q && (ly->wq.type == JANAS_Q6_K_P || ly->wq.plane[0]))
            return fail(err, len, "a gated query in planes");
        size_t cache = (size_t)md->n_kv * md->cap * md->hd;
        if (!janas_gt_addr_of(g, ly->attn_norm, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->q_norm, md->hd * 4) ||
            !janas_gt_addr_of(g, ly->k_norm, md->hd * 4) ||
            !janas_gt_addr_of(g, ly->kc, cache) ||
            !janas_gt_addr_of(g, ly->vc, cache) ||
            !janas_gt_addr_of(g, ly->ks, (size_t)md->n_kv * md->cap * 4) ||
            !janas_gt_addr_of(g, ly->vs, (size_t)md->n_kv * md->cap * 4))
            return fail(err, len,
                        "a norm or the KV cache out of the GPU's "
                        "reach");
    }
    if (!janas_gpu_can(g, &md->output) ||
        !janas_gt_addr_of(g, md->out_norm, md->dm * 4))
        return fail(err, len, "the output head out of the GPU's reach");
    (void)kvd;
    return 0;
}

struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len)
{
    if (!janas_gpu_shares_host(g)) {
        fail(err, err_len, "the GPU does not read the host's memory in place");
        return NULL;
    }
    if (check_model(g, md, err, err_len) != 0)
        return NULL;
    struct janas_gpu_tok *t = calloc(1, sizeof(*t));
    if (!t) {
        fail(err, err_len, "out of memory");
        return NULL;
    }
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    size_t big = md->dm > qd ? md->dm : qd;
    size_t at = 0;
    /* the attention's runs: per token, query head and run, hd sums, m and
       l; as many tokens a block as they leave room for */
    /* laid out for the runs the positions reached so far have, not for
       the whole cache's (janas_gpu_tok_block_at): a prompt's blocks stay
       long at any context - sized for the cache, 16384 positions left
       Qwen3-4B 30 tokens a block, too few for the fast prompt's tiles */
    t->nch = (md->cap + ATT_RUN - 1) / ATT_RUN;
    size_t full = (size_t)md->n_head * t->nch * (md->hd + 2);
    t->part_floats = TOK_PART_BUDGET / 4 > full ? TOK_PART_BUDGET / 4 : full;
    t->nb = TOK_MAX_BLOCK;
    t->n_seg = md->n_layer + 1;
    size_t nb = t->nb;
    t->o_x = take(&at, nb * md->dm * 4);
    t->o_x1 = take(&at, nb * md->dm * 4); /* the residual's other buffer */
    t->o_q256 = take(&at, nb * big / JANAS_QK * 320);
    t->o_q32 = take(&at, nb * big / JANAS_QK * 320);
    t->o_qkv = take(&at, nb * (qd + 2 * kvd) * 4);
    t->o_ywo = take(&at, nb * md->dm * 4);
    t->o_gu = take(&at, nb * md->ff * 2 * 4);
    t->o_h = take(&at, nb * md->ff / JANAS_QK * 320);
    t->o_yd = take(&at, nb * md->dm * 4);
    t->o_rope = take(&at, nb * md->n_rot * 4);
    t->o_logits = take(&at, nb * md->n_vocab * 4);
    t->o_pos = take(&at, 32); /* janas_gpu_tok_run's five words */
    t->o_part = take(&at, t->part_floats * 4);
    t->o_xnf = take(&at, nb * md->dm * 4);
    t->o_hid = take(&at, nb * md->dm * 4);
    if (md->gated_q)
        t->o_gate = take(&at, nb * qd * 4);
    if (md->n_rec) {
        t->o_mix = take(&at, nb * md->conv_ch * 4);
        t->o_z = take(&at, nb * md->d_inner * 4);
        t->o_conv =
            take(&at, (nb + JANAS_GPU_TOK_MAX_REPLAY) * md->conv_ch * 4);
        t->o_ba = take(&at, nb * 2 * md->nv * 4);
        t->o_att = take(&at, nb * md->d_inner * 4);
        t->o_bd = take(&at, nb * 2 * md->nv * 4);
        /* the log where the GPU reads and writes it: then the blocks short
           enough to be logged (a chain of drafts) may come here too */
        size_t lr = (size_t)md->n_rec * md->log_rows;
        t->log_ok = md->log_mixed && md->log_beta && md->log_decay &&
                    md->log_rows >= JANAS_GPU_TOK_MAX_REPLAY &&
                    janas_gt_addr_of(g, md->log_mixed, lr * md->conv_ch * 4) &&
                    janas_gt_addr_of(g, md->log_beta, lr * md->nv * 4) &&
                    janas_gt_addr_of(g, md->log_decay, lr * md->nv * 4);
    }
    uint32_t ne = md->n_expert;
    if (ne) {
        /* the pairs of a block of nb tokens, k each; at most a tile of 16
           of them per expert more than if they were all whole tiles */
        size_t np = nb * md->k, nbd = md->dm / JANAS_QK,
               nbf = md->eff / JANAS_QK;
        t->o_r = take(&at, nb * (ne + 1) * 4);
        t->o_pe = take(&at, np * 4);
        t->o_pw = take(&at, np * 4);
        t->o_shw = take(&at, nb * 4);
        t->o_sh = take(&at, nb * md->dm * 4);
        t->o_xg = take(&at, np * nbd * 320);
        t->o_egu = take(&at, np * md->eff * 2 * 4);
        t->o_eh = take(&at, np * nbf * 320);
        t->o_dout = take(&at, np * md->dm * 4);
        t->o_gt = take(&at, np * 4);
        t->o_inv = take(&at, np * 4);
        t->tiles1 = 3 * 2 * ne;
        t->tiles2 = 3 * ne;
        t->o_tab1 = take(&at, (t->tiles1 + 2 * (np / 16 + ne)) * 16);
        t->o_tab2 = take(&at, (t->tiles2 + np / 16 + ne) * 16);
        t->o_ind = take(&at, 32);
        t->cnt = calloc(ne + 1, sizeof(uint32_t));
        t->fill = calloc(ne, sizeof(uint32_t));
        t->ids = calloc(ne, sizeof(uint32_t));
        t->slots = calloc(ne, sizeof(*t->slots));
    }
    /* a down matrix in blocks of 32 - a shared expert's Q8_0, a dense
       layer's IQ4_NL (unsloth's UD files mix it in) - takes the activation
       as floats, quantized then into both layouts (janas_gt_rec_ffn) */
    int d32 = 0;
    for (uint32_t l = 0; l < md->n_layer; l++)
        d32 |= janas_g_x_in_32(ne ? md->layers[l].sh_down.type
                                  : md->layers[l].down.type);
    if (d32) {
        t->o_hf = take(&at, nb * md->ff * 4);
        t->o_h32 = take(&at, nb * md->ff / JANAS_QK * 320);
    }
    if (md->n_rec || ne) {
        t->par = calloc((size_t)md->n_layer * PAR_N, sizeof(size_t));
        if (!t->par || (ne && (!t->cnt || !t->fill || !t->ids || !t->slots))) {
            fail(err, err_len, "out of memory");
            janas_gpu_tok_destroy(g, t);
            return NULL;
        }
        for (uint32_t l = 0; l < md->n_layer; l++) {
            size_t *pr = t->par + (size_t)l * PAR_N;
            if (ne)
                pr[PAR_R] = take(&at, (size_t)(ne + 1) * md->dm * 4);
            if (!md->layers[l].rec)
                continue;
            pr[PAR_BA] = take(&at, 2 * (size_t)md->nv * md->dm * 4);
            pr[PAR_CONV] = take(&at, (size_t)md->d_conv * md->conv_ch * 4);
            pr[PAR_DT] = take(&at, md->nv * 4);
            pr[PAR_A] = take(&at, md->nv * 4);
            pr[PAR_NORM] = take(&at, md->ds * 4);
        }
    }
    /* cached for the host (the logits are read on the CPU: 2.5 ms a token
       from uncached memory on the development laptop), flushed and
       invalidated by hand where it is not coherent */
    if (janas_g_make_stage(g, &t->buf, at, 1) != 0 ||
        janas_g_make_pipeline(g, janas_spv_norm_quant,
                              sizeof(janas_spv_norm_quant), &t->sm_nq, &t->p_nq,
                              0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_qk_rope, sizeof(janas_spv_qk_rope),
                              &t->sm_qk, &t->p_qk, 0) != VK_SUCCESS ||
        janas_g_make_pipeline_sg(
            g, janas_spv_attn_decode, sizeof(janas_spv_attn_decode), &t->sm_at,
            &t->p_at, md->n_head / md->n_kv, att_sg()) == 0 ||
        janas_g_make_pipeline(g, janas_spv_attn_merge,
                              sizeof(janas_spv_attn_merge), &t->sm_am, &t->p_am,
                              0) != VK_SUCCESS ||
        (md->n_rec &&
         (janas_g_make_pipeline(g, janas_spv_gdn_conv,
                                sizeof(janas_spv_gdn_conv), &t->sm_gc, &t->p_gc,
                                0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_cstate,
                                sizeof(janas_spv_gdn_cstate), &t->sm_gs,
                                &t->p_gs, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_l2, sizeof(janas_spv_gdn_l2),
                                &t->sm_gl, &t->p_gl, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_rec, sizeof(janas_spv_gdn_rec),
                                &t->sm_gr, &t->p_gr, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_gate,
                                sizeof(janas_spv_gdn_gate), &t->sm_gg, &t->p_gg,
                                0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_ba, sizeof(janas_spv_gdn_ba),
                                &t->sm_gb, &t->p_gb, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_act, sizeof(janas_spv_gdn_act),
                                &t->sm_gx, &t->p_gx, 0) != VK_SUCCESS ||
          !g->pipe_dm[0])) ||
        (ne && (janas_g_make_pipeline(g, janas_spv_moe_topk,
                                      sizeof(janas_spv_moe_topk), &t->sm_tk,
                                      &t->p_tk, 0) != VK_SUCCESS ||
                janas_g_make_pipeline(g, janas_spv_moe_gather,
                                      sizeof(janas_spv_moe_gather), &t->sm_ga,
                                      &t->p_ga, 0) != VK_SUCCESS ||
                janas_g_make_pipeline(g, janas_spv_moe_combine,
                                      sizeof(janas_spv_moe_combine), &t->sm_cb,
                                      &t->p_cb, 0) != VK_SUCCESS)) ||
        (d32 && janas_g_make_pipeline(g, janas_spv_silu_mul,
                                      sizeof(janas_spv_silu_mul), &t->sm_sm,
                                      &t->p_sm, 0) != VK_SUCCESS)) {
        fail(err, err_len, "cannot create the token's buffers or shaders");
        janas_gpu_tok_destroy(g, t);
        return NULL;
    }
    /* the recurrent layers' parameters, copied once */
    for (uint32_t l = 0; md->n_rec && l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        if (!ly->rec)
            continue;
        const size_t *pr = t->par + (size_t)l * PAR_N;
        size_t nvd = (size_t)md->nv * md->dm * 4;
        memcpy(t->buf.map + pr[PAR_BA], ly->ssm_beta, nvd);
        memcpy(t->buf.map + pr[PAR_BA] + nvd, ly->ssm_alpha, nvd);
        memcpy(t->buf.map + pr[PAR_CONV], ly->conv_w,
               (size_t)md->d_conv * md->conv_ch * 4);
        memcpy(t->buf.map + pr[PAR_DT], ly->dt, md->nv * 4);
        memcpy(t->buf.map + pr[PAR_A], ly->ssm_a, md->nv * 4);
        memcpy(t->buf.map + pr[PAR_NORM], ly->ssm_norm, md->ds * 4);
        janas_gt_sync_range(g, t, pr[PAR_BA],
                            pr[PAR_NORM] + md->ds * 4 - pr[PAR_BA], 1);
    }
    /* a mixture's routers, as f32 (the CPU's f16 weights exactly), the
       shared expert's gate after them */
    for (uint32_t l = 0; ne && l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        size_t o = t->par[(size_t)l * PAR_N + PAR_R], n = (size_t)ne * md->dm;
        float *r = (float *)(t->buf.map + o);
        for (size_t i = 0; i < n; i++)
            r[i] = janas_fp16_to_fp32(ly->router16[i]);
        if (ly->sh_gate)
            memcpy(r + n, ly->sh_gate, md->dm * 4);
        else
            memset(r + n, 0, md->dm * 4);
        janas_gt_sync_range(g, t, o, (n + md->dm) * 4, 1);
    }
    const char *pe = getenv("JANAS_GPU_TOKEN_PROFILE");
    if (pe && atoi(pe) > 0 && g->ts_period > 0) {
        VkQueryPoolCreateInfo qi = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = MAX_TS};
        if (g->vkCreateQueryPool(g->dev, &qi, NULL, &t->qp) != VK_SUCCESS)
            t->qp = VK_NULL_HANDLE;
    }
    return t;
}

void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    if (!t)
        return;
    if (t->qp) {
        double all = 0;
        for (int i = 0; i < PH_N; i++)
            all += t->ph_ns[i];
        if (t->tokens) {
            fprintf(stderr,
                    "gpu token profile, %llu passes of %.1f tokens, %.2f ms "
                    "a pass (%.2f ms from submission to fence, %.2f ms "
                    "copying the logits):\n",
                    (unsigned long long)t->tokens,
                    (double)t->pass_tokens / t->tokens, all / t->tokens * 1e-6,
                    t->wall_ns / t->tokens * 1e-6,
                    t->copy_ns / t->tokens * 1e-6);
            for (int i = 0; i < PH_N; i++)
                fprintf(stderr, "  %-14s %7.3f ms %5.1f%%\n", ph_name[i],
                        t->ph_ns[i] / t->tokens * 1e-6,
                        all > 0 ? 100.0 * t->ph_ns[i] / all : 0.0);
        }
        g->vkDestroyQueryPool(g->dev, t->qp, NULL);
    }
    for (uint32_t n = 0; n <= TOK_MAX_BLOCK; n++)
        for (int a = 0; a < 3; a++)
            for (int h = 0; h < 2; h++) {
                if (t->cb[n][a][h])
                    g->vkFreeCommandBuffers(g->dev, g->cp, 1, &t->cb[n][a][h]);
                janas_gt_free_segments(g, t, &t->seg_cb[n][a][h]);
            }
    VkPipeline p[15] = {t->p_nq, t->p_qk, t->p_at, t->p_am, t->p_gc,
                        t->p_gs, t->p_gl, t->p_gr, t->p_gg, t->p_tk,
                        t->p_ga, t->p_cb, t->p_sm, t->p_gb, t->p_gx};
    VkShaderModule s[15] = {t->sm_nq, t->sm_qk, t->sm_at, t->sm_am, t->sm_gc,
                            t->sm_gs, t->sm_gl, t->sm_gr, t->sm_gg, t->sm_tk,
                            t->sm_ga, t->sm_cb, t->sm_sm, t->sm_gb, t->sm_gx};
    free(t->par);
    free(t->cnt);
    free(t->fill);
    free(t->ids);
    free(t->slots);
    for (int i = 0; i < 15; i++) {
        if (p[i])
            g->vkDestroyPipeline(g->dev, p[i], NULL);
        if (s[i])
            g->vkDestroyShaderModule(g->dev, s[i], NULL);
    }
    if (t->buf.buf)
        g->vkDestroyBuffer(g->dev, t->buf.buf, NULL);
    if (t->buf.mem)
        g->vkFreeMemory(g->dev, t->buf.mem, NULL);
    free(t);
}

/* Bytes [off, off + bytes) of the token's buffer out of the CPU's caches
   (flush) or reloaded from memory (!flush), where it is not coherent. */
void janas_gt_sync_range(struct janas_gpu *g, struct janas_gpu_tok *t,
                         size_t off, size_t bytes, int flush)
{
    if (t->buf.coherent)
        return;
    VkDeviceSize a = off / g->atom * g->atom;
    VkDeviceSize e = (off + bytes + g->atom - 1) / g->atom * g->atom;
    VkMappedMemoryRange mr = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .memory = t->buf.mem,
                              .offset = a,
                              .size =
                                  e >= t->buf.bytes ? VK_WHOLE_SIZE : e - a};
    if (flush)
        g->vkFlushMappedMemoryRanges(g->dev, 1, &mr);
    else
        g->vkInvalidateMappedMemoryRanges(g->dev, 1, &mr);
}

/* The runs of ATT_RUN positions npos positions take, rounded up to a power
   of two (so that a prompt's blocks record their commands again only a few
   times), at most the cache's. */
static uint32_t runs_for(const struct janas_gpu_tok *t, uint32_t npos)
{
    uint32_t c = (npos + ATT_RUN - 1) / ATT_RUN, r = 1;
    while (r < c)
        r *= 2;
    return r < t->nch ? r : t->nch;
}

/* The commands of a block of n tokens into g->cb (the caller points it at
   the block's buffer): every layer, then the output head for the tokens
   from first on (none if first is n); the position of the first token
   comes from the buffer.
   A single token takes the layers' GPU copies (JANAS_GPU_REPACK), a block
   the weights as they are (the copies' shaders take one vector).
   A mixture's block goes in n_layer + 1 submissions, seg the one recorded
   (-1: a dense model's, all of it): submission s takes layer s - 1's
   experts, then layer s up to its router's choice, the last one the final
   norm and the output head. */
static int record(struct janas_gpu *g, struct janas_gpu_tok *t,
                  const struct janas_gpu_tok_model *md, uint32_t n,
                  uint32_t first, int hid, int seg)
{
    const struct janas_gpu_tok_layer *layers =
        n > 1 && md->blayers ? md->blayers : md->layers;
    const struct janas_matvec_task *output =
        n > 1 && md->blayers ? &md->boutput : &md->output;
    /* the residual: read from X, the sum with a layer's output written to
       XN, the two swapped (every block starts from o_x, the same commands) */
    VkDeviceAddress B = t->buf.addr, X = B + t->o_x, XN = B + t->o_x1, SW,
                    YWO = B + t->o_ywo, YD = B + t->o_yd, XNF = B + t->o_xnf;
    if (g->vkResetCommandBuffer(g->cb, 0) != VK_SUCCESS)
        return -1;
    VkCommandBufferBeginInfo bb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0};
    if (g->vkBeginCommandBuffer(g->cb, &bb) != VK_SUCCESS)
        return -1;
    VkPipeline bound = VK_NULL_HANDLE;
    t->prof = seg < 0; /* a pass in one submission, of any length */
    if (t->qp && t->prof) {
        t->n_ts = 0;
        g->vkCmdResetQueryPool(g->cb, t->qp, 0, MAX_TS);
        janas_gt_stamp(g, t, PH_N); /* the start */
    }
    uint32_t nl = md->n_layer;
    for (uint32_t l = 0; l < nl; l++) {
        const struct janas_gpu_tok_layer *ly = &layers[l];
        int on = seg < 0 || seg == (int)l;
        /* a mixture: the previous layer's experts first, into YD */
        if (on && l && md->n_expert)
            janas_gt_rec_experts(g, t, md, &layers[l - 1], &bound, YD, n);
        /* the previous layer's feed-forward added, then this one's norm */
        if (on) {
            janas_gt_rec_norm_quant(
                g, t, &bound, X, l ? YD : 0,
                janas_gt_addr_of(g, ly->attn_norm, md->dm * 4), l ? XN : 0,
                ly->rec ? XNF : 0, md->dm, md->eps, n);
            janas_gt_stamp(g, t, PH_NORM);
        }
        if (l)
            SW = X, X = XN, XN = SW;
        if (on) {
            if ((ly->rec ? janas_gt_rec_gdn(g, t, md, ly, l, &bound, YWO, n)
                         : janas_gt_rec_attn(g, t, md, ly, &bound, n)) != 0)
                return -1;
            janas_gt_rec_norm_quant(
                g, t, &bound, X, YWO,
                janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4), XN,
                md->n_expert ? XNF : 0, md->dm, md->eps, n);
            janas_gt_stamp(g, t, PH_NORM);
        }
        SW = X, X = XN, XN = SW;
        if (on &&
            (md->n_expert ? janas_gt_rec_router(g, t, md, ly, l, &bound, n)
                          : janas_gt_rec_ffn(g, t, &bound, &ly->gate, &ly->up,
                                             &ly->down, n, YD)) != 0)
            return -1;
    }
    /* the last feed-forward added, the final norm, the output head for
       tokens first .. n - 1 (none: the residual left in X, the next block's
       keys and values already in the cache) */
    if (seg < 0 || seg == (int)nl) {
        if (md->n_expert)
            janas_gt_rec_experts(g, t, md, &layers[nl - 1], &bound, YD, n);
        if (first < n || hid) {
            janas_gt_rec_norm_quant(
                g, t, &bound, X, YD,
                janas_gt_addr_of(g, md->out_norm, md->dm * 4), 0,
                hid ? B + t->o_hid : 0, md->dm, md->eps, n);
        }
        if (first < n) {
            if (janas_gt_rec_product(g, t, &bound, output,
                                     (size_t)first * (md->dm / JANAS_QK) * 320,
                                     B + t->o_logits, n - first, 0) != 0)
                return -1;
        }
        janas_gt_stamp(g, t, PH_OUT);
    }
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                            NULL);
    if (g->vkEndCommandBuffer(g->cb) != VK_SUCCESS)
        return -1;
    return 0;
}

uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t)
{
    return t ? t->nb : 0;
}

uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos)
{
    if (!t)
        return 0;
    size_t n =
        t->part_floats / janas_gt_part_tok(t, md, runs_for(t, pos + t->nb));
    return n < 1 ? 1 : n > t->nb ? t->nb : (uint32_t)n;
}

/* A command buffer for cb, allocated, and the commands of segment seg of a
   block of n tokens recorded into it: 0, or -1 with *cb NULL. */
int janas_gt_make_cb(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md, VkCommandBuffer *cb,
                     uint32_t n, uint32_t first, int hi, uint32_t nch, int seg)
{
    VkCommandBufferAllocateInfo ca = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g->cp,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    if (g->vkAllocateCommandBuffers(g->dev, &ca, cb) != VK_SUCCESS) {
        *cb = VK_NULL_HANDLE;
        return -1;
    }
    VkCommandBuffer keep = g->cb;
    g->cb = *cb; /* the helpers record into g->cb */
    t->rec_nch = nch;
    int rc = record(g, t, md, n, first, hi, seg);
    g->cb = keep;
    if (rc != 0) {
        g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
        *cb = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

int janas_gt_submit_wait(struct janas_gpu *g, const VkCommandBuffer *cb)
{
    VkSubmitInfo sub = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = cb};
    pthread_mutex_lock(&g->qlock);
    VkResult r = g->vkQueueSubmit(g->queue, 1, &sub, g->fence);
    pthread_mutex_unlock(&g->qlock);
    if (r == VK_SUCCESS)
        r = g->vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        g->broken = 1;
        return -1;
    }
    g->vkResetFences(g->dev, 1, &g->fence);
    return 0;
}

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row)
{
    if (!t || g->busy || g->broken || n < 1 || n > t->nb || pos + n > md->cap)
        return -1;
    /* a replay, or a block to log, needs the log on the GPU */
    int logs = log_row != UINT32_MAX;
    if (md->n_rec &&
        (replay > JANAS_GPU_TOK_MAX_REPLAY ||
         ((replay || logs) && !t->log_ok) ||
         (logs && log_row + n > md->log_rows) || (replay && src == dst)))
        return -1;
    if (!md->n_rec && (replay || logs))
        return -1;
    uint32_t nch = runs_for(t, pos + n);
    if ((size_t)n * janas_gt_part_tok(t, md, nch) > t->part_floats)
        return -1;
    uint32_t half = md->n_rot / 2, first = all == 1 ? 0 : all == 2 ? n : n - 1;
    memcpy(t->buf.map + t->o_x, x, (size_t)n * md->dm * sizeof(float));
    for (uint32_t j = 0; j < n; j++) {
        uint8_t *r = t->buf.map + t->o_rope + (size_t)j * md->n_rot * 4;
        memcpy(r, cs + (size_t)j * half, half * sizeof(float));
        memcpy(r + half * sizeof(float), sn + (size_t)j * half,
               half * sizeof(float));
    }
    /* the position, the halves of the recurrent state to read and to
       write, the tokens to replay, the log's row: read by the shaders at
       run time */
    uint32_t pw[5] = {pos, src, dst, replay, log_row};
    memcpy(t->buf.map + t->o_pos, pw, sizeof(pw));
    janas_gt_sync_range(g, t, t->o_x, (size_t)n * md->dm * sizeof(float), 1);
    janas_gt_sync_range(g, t, t->o_rope, (size_t)n * md->n_rot * sizeof(float),
                        1);
    janas_gt_sync_range(g, t, t->o_pos, sizeof(pw), 1);
    int ai = all == 1 && n == 1 ? 0 : all, hi = hidden != NULL;
    struct timespec w0, w1;
    clock_gettime(CLOCK_MONOTONIC, &w0);
    /* a failure once the GPU has started: with a replay the base it
       updates in place is no longer the state the CPU would replay on */
    int failed = replay ? -2 : -1;
    if (md->n_expert) {
        if (janas_gt_run_segments(g, t, md, n, first, ai, hi, nch) != 0)
            return failed;
    } else {
        VkCommandBuffer *cb = &t->cb[n][ai][hi];
        if (*cb && t->cb_nch[n][ai][hi] != nch) {
            /* recorded for other runs: again (a few times in a context) */
            g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
            *cb = VK_NULL_HANDLE;
        }
        if (!*cb) {
            if (janas_gt_make_cb(g, t, md, cb, n, first, hi, nch, -1) != 0)
                return -1;
            t->cb_nch[n][ai][hi] = nch;
        }
        if (janas_gt_submit_wait(g, cb) != 0)
            return failed;
    }
    clock_gettime(CLOCK_MONOTONIC, &w1);
    size_t lb = (size_t)(n - first) * md->n_vocab * 4;
    if (lb) {
        janas_gt_sync_range(g, t, t->o_logits, lb, 0);
        memcpy(logits, t->buf.map + t->o_logits, lb);
    }
    if (hidden) {
        size_t hb = (size_t)n * md->dm * 4;
        janas_gt_sync_range(g, t, t->o_hid, hb, 0);
        memcpy(hidden, t->buf.map + t->o_hid, hb);
    }
    struct timespec w2;
    clock_gettime(CLOCK_MONOTONIC, &w2);
    if (t->qp && t->prof && t->n_ts > 1) {
        uint64_t ts[MAX_TS];
        if (g->vkGetQueryPoolResults(g->dev, t->qp, 0, t->n_ts, sizeof(ts), ts,
                                     sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            for (uint32_t i = 1; i < t->n_ts; i++)
                t->ph_ns[t->ph_of[i]] +=
                    (double)(ts[i] - ts[i - 1]) * g->ts_period;
            t->tokens++;
            t->pass_tokens += n;
            t->wall_ns += (double)(w1.tv_sec - w0.tv_sec) * 1e9 +
                          (double)(w1.tv_nsec - w0.tv_nsec);
            t->copy_ns += (double)(w2.tv_sec - w1.tv_sec) * 1e9 +
                          (double)(w2.tv_nsec - w1.tv_nsec);
        }
    }
    return 0;
}

#else /* no GPU in this build */

struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len)
{
    (void)g;
    (void)md;
    snprintf(err, err_len, "no GPU in this build");
    return NULL;
}

void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    (void)g;
    (void)t;
}

uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t)
{
    (void)t;
    return 0;
}

uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos)
{
    (void)t;
    (void)md;
    (void)pos;
    return 0;
}

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row)
{
    (void)replay;
    (void)log_row;
    (void)n;
    (void)all;
    (void)g;
    (void)t;
    (void)md;
    (void)x;
    (void)cs;
    (void)sn;
    (void)pos;
    (void)logits;
    (void)hidden;
    (void)src;
    (void)dst;
    return -1;
}

int janas_gpu_shares_host(const struct janas_gpu *g)
{
    (void)g;
    return 0;
}

#endif
