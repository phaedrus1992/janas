/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_impl.h - the insides of a model, for the files that make up
 * model.c's work (model.c, model_state.c). Nothing outside src/llm/model*.c
 * includes it: the rest of the engine sees the opaque struct of model.h.
 */
#ifndef JANAS_LLM_MODEL_IMPL_H
#define JANAS_LLM_MODEL_IMPL_H

#include "model.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#include "arch.h"
#include "attention.h"
#include "common/pool.h"
#include "gpu.h"
#include "gpu_token.h"
#include "jns.h"
#include "jmath.h"
#include "quant.h"
#include "tuner.h"

#define MAX_K 64
#define MAX_CONV 8
#define MAX_SSM_DIM 256
#define SSM_ROWS 32 /* state rows per recurrence work item */

struct layer {
    int rec;       /* Gated DeltaNet layer (qwen3next), else attention */
    uint32_t slot; /* index among the layers of its kind: KV or state */
    /* attention geometry: the model's, or per kind of layer (Gemma 4) */
    uint32_t hd, n_kv, n_rot;
    int swa; /* sliding window, its own RoPE table and attention */
    /* wv NULL: V is K (Gemma 4's full layers) */
    const struct janas_jns_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm,
        *k_norm, *ffn_norm, *gate_inp;
    /* sandwich norms of attention's and the FFN's outputs, and the layer's
       output scale (1 if none) */
    const struct janas_jns_tensor *post_attn, *post_ffn;
    float out_scale;
    /* Gemma 4's mixture: the dense feed-forward (the shared expert, weight
       1, no gate) and the experts are two branches, each normalized on the
       way in and out; the router reads its own input, scaled per dimension,
       and each expert's output has a scale of its own */
    const struct janas_jns_tensor *pre_norm2, *post_norm1, *post_norm2;
    const float *router_scale, *exp_scale;
    /* Gemma 4's per-layer embeddings (E2B, E4B): after the FFN, the layer's
       own input of ple_dim numbers per token gates a branch of the
       residual, gelu(gate x) * input, projected back, normalized, added */
    const struct janas_jns_tensor *ple_gate, *ple_proj, *ple_post;
    /* Gated DeltaNet */
    const struct janas_jns_tensor *ssm_in, *ssm_ba, *conv, *dt_bias, *ssm_a,
        *ssm_norm, *ssm_out;
    /* qwen35: q|k|v (conv_ch rows), z (d_inner rows), alpha and beta
       (n_vh rows each, as f32: the file's, or made from its Q8_0) */
    const struct janas_jns_tensor *ssm_qkv, *ssm_z;
    const float *ssm_alpha, *ssm_beta;
    float *alpha_own, *beta_own;
    /* shared expert: gate, up, down as matrices of a slot at offset 0 */
    const uint8_t *sh_w[3];
    struct janas_jns_matrix sh_mx[3];
    float *sh_gate;           /* d_model weights of the sigmoid gate, as f32 */
    const uint16_t *router16; /* router weights, f16 (in place of the f32) */
    uint16_t *router_own;     /* the zero router of a dense layer, if any */
    float *conv_t;            /* conv weights transposed: d_conv x conv_ch */
};

_Static_assert(JANAS_ATTN_MAX_BLOCK == JANAS_LLM_MAX_BLOCK, "block size");

struct janas_llm_model {
    struct janas_jns j;
    uint8_t *resident; /* the resident region, read at load */
    struct janas_pool *compute, *io;
    int own_compute; /* zero when the pool came from the caller */
    /* compute threads: the pool lists one CPU per performance core, their
       other threads, then the efficiency cores; the tuner picks a prefix
       (and whether the GPU helps) for each kind of pass */
    int p_cores, p_threads, all_threads;
    struct janas_tuner tuner;
    char tune_key[768];
    int gpu_use;             /* the current pass may use the GPU */
    struct janas_gpu *gpu;   /* optional, for products over blocks */
    struct janas_m_npu *npu; /* optional (JANAS_NPU=1), model_npu.c */
    int npu_use;             /* the current pass may use the NPU */
    char npu_why[160];       /* no NPU: why (janas_llm_model_npu_why) */
    int resident_gpu;        /* the resident region belongs to the GPU */
    int gpu_off;             /* weights not shared with it: CPU only */
    /* the user keeps prompts off the GPU (--prompt-on cpu, npu): its fast
       prompt, which reads them whatever the tuner chose, does not */
    int prompts_off_gpu;
    /* the user's devices for prompts, drafts, replies (common/devices.h:
       janas_devices_vars' order), as this machine has them; 0 as the
       engine chooses */
    int dev_on[3];
    /* the experts' arena on the GPU (dense models, and mixtures whose
       every expert fits the cache): 0 not yet, 1 there, -1 not possible;
       arena_stale when a slot changed since - the GPU's copy no longer
       true, the experts' products stay on the CPU */
    int arena_gpu, arena_stale;
    uint64_t arena_loads;
    int arena_in_place; /* read by the GPU where it is: never stale */
    char gpu_why[160];  /* no GPU: why (janas_llm_model_gpu_why) */
    struct janas_expert_cache *cache;

    const struct janas_arch *a; /* what this architecture has (arch.h) */
    uint32_t n_layer, d_model, n_head, n_head_kv, head_dim, n_rot, n_expert, k,
        n_vocab, n_ctx, d_ff, n_attn, n_rec;
    uint32_t ff_buf; /* rows of a pair's buffers: d_ff, or a wider shared
                        feed-forward's (Gemma 4's 26B) */
    float eps, rope_base;
    /* Gemma 4: the sliding-window layers' geometry (head size and RoPE in
       head_swa, n_rot_swa, rope_base_swa; n_head_kv, head_dim and n_rot
       are the full layers'), the window, the full layers' RoPE factors
       (theta / factor per pair, NULL: none), the logits' soft cap (0:
       none), the widest head and KV row of any layer */
    uint32_t head_swa, n_rot_swa, n_kv_swa, window, hd_max, kvd_max;
    float rope_base_swa, softcap;
    const float *rope_freqs;
    /* Gemma 4's per-layer embeddings (0: none): ple_dim numbers per layer
       and token, from a table of their own read a row at a time from the
       file (ple_tok, never in memory) and a projection of the embedding
       (ple_model, bf16), normalized by ple_norm; ple holds the block's,
       token after token, layer after layer, ple_g a branch's rows,
       ple_row a row of the table as floats, then as read (room for twice
       its numbers in floats) */
    uint32_t ple_dim;
    const struct janas_jns_tensor *ple_tok, *ple_model, *ple_norm;
    float *ple, *ple_g, *ple_row;
    struct janas_block_q8k *ple_q;
    /* layers from n_kv_layers on keep no keys and values of their own:
       they read those of the last earlier layer of their kind */
    uint32_t n_kv_layers;
    /* the KV cache of slot s (attention layer, then the MTP block's) starts
       at kv_off[s] bytes and its scales at ks_off[s] floats; kv_hd and
       kv_heads its geometry. Uniform unless the layers differ (Gemma 4) */
    size_t *kv_off, *ks_off;
    uint32_t *kv_hd, *kv_heads;
    int kv16; /* keys and values of 16 bits (arch, or JANAS_KV=8|16) */
    /* positions a slot holds per KV head: n_ctx, or for a sliding-window
       layer a ring of fewer (kv_ring), position p in row p mod kv_cap;
       ring_hi: one past the last position written */
    uint32_t *kv_cap;
    uint8_t *kv_ring, *kv_swa; /* and whether its layer is a sliding one */
    uint32_t ring_hi;
    /* Gated DeltaNet geometry: ds = head size (keys and values), n_kh key
       heads, n_vh value heads (n_vh / n_kh per key head), conv over the
       q|k|v channels */
    uint32_t d_conv, ds, n_kh, n_vh, d_inner, conv_ch, qkvz_dim;
    int avx2; /* AVX2 and FMA: the vector kernels of this file */
    int f16c;
    int trace; /* JANAS_TRACE=1: print the state after every layer */
    /* which key head a value head shares: qwen3next repeats each key head
       for its consecutive value heads (h / rf), qwen35moe cycles through
       them (h % n_kh); JANAS_VMAP=div|mod overrides */
    int vmap_mod;
    struct layer *layers;
    const struct janas_jns_tensor *tok_embd, *out_norm, *output;

    /*
     * n_attn x n_head_kv x n_ctx x head_dim: the positions of one KV head
     * are contiguous, so attention streams them. Interleaving the heads of a
     * position (a 2 KiB stride between the rows one head reads) defeated the
     * hardware prefetcher, which does not cross 4 KiB pages: every position
     * cost a DRAM access and attention took 6.5 ms per token instead of ~1.
     * Half precision: attention at long context is bound by reading it.
     */
    int8_t *kcache, *vcache;
    size_t kv_bytes, ks_bytes; /* the caches' and the scales' allocations */
    /* the whole token on the GPU (gpu_token.h, JANAS_GPU_TOKEN=1): 0 not
       tried yet, 1 ready, -1 not for this model; loads: the experts' loads
       it was built with (a change and it is built again) */
    struct janas_gpu_tok *gtok;
    struct janas_gpu_tok_model gtok_md;
    struct janas_gpu_tok_layer *gtok_layers;
    struct janas_gpu_tok_layer *gblk_layers; /* the weights as they are,
                                                 with a GPU copy */
    int gtok_state, gtok_kv;
    uint64_t gtok_loads;
    /* the aligned copy of its Q6_K weights for the GPU (JANAS_GPU_REPACK),
       in the GPU driver's memory, kept for a rebuild */
    uint8_t *gtok_repack;
    size_t gtok_repack_bytes;
    float *kscale, *vscale; /* one per position and KV head */
    /*
     * Recurrent state in two buffers, a base and a tip, so that rolling back
     * costs no memory traffic. Each holds, per DeltaNet layer, the state
     * (n_vh x ds x ds, deltanet.h layout) and the last d_conv - 1 inputs of
     * every channel ((d_conv - 1) x conv_ch). The base holds positions
     * 0 .. base_pos - 1; the tip the log_n positions after them, run by the
     * last call, whose per-token inputs (conv inputs, decays, betas) are in
     * the log.
     *
     * A call starting where the last one ended (plain decoding, prefill, a
     * fully accepted draft) starts from the tip, which becomes the base. A
     * call starting earlier (rejected draft tokens) replays the accepted
     * logged tokens on the base in place. Either way the call's tokens leave
     * their state in the other buffer: the state is read and written once
     * per call, as with an in-place update.
     */
    /* two buffers each, back to back. The DeltaNet state, read and written
       in full for every call, in half precision: half the traffic; its
       rounding stays within the model's own numerical noise (an int16
       format with row scales, 7x more precise, changed nothing measurable) */
    uint16_t *ssm_buf;
    float *conv_buf;
    float *log_mixed, *log_beta, *log_decay;
    int base; /* buffer holding the base */
    int src;  /* this call: buffer the tokens start from */
    uint32_t base_pos, log_n, replay;
    /* the call goes on from the one before it in the same pass (the NPU's
       second half of a block): the state and the convolution's inputs read
       where that one left them, in the other buffer */
    int rec_cont;
    uint32_t n_ctx_train; /* what the model was trained for, 0 if unsaid */

    /*
     * scratch for a pass of up to blk tokens (janas_llm_model_max_block):
     * a prefill block, larger than JANAS_LLM_MAX_BLOCK so that an expert is
     * read once for many tokens; attention takes it 64 at a time
     */
    uint32_t blk;
    uint32_t *tok_first, *tok_fill, *tok_list; /* moe_block's per token rows */
    float *x, *xn, *q, *kk, *vv, *att, *router, *rope_cos, *rope_sin;
    struct janas_attn *attn;
    /* Gemma 4: the sliding-window layers' RoPE table and attention, and the
       residual kept aside while the FFN's output is normalized */
    float *rope_cos_swa, *rope_sin_swa, *xres, *xbr; /* xbr: a branch */
    struct janas_attn *attn_swa;

    /*
     * Multi-token prediction block (optional, a second JNS file): the
     * embedding of token t_p and the final hidden state h_{p-1} of the main
     * model, each RMS-normalized, concatenated and projected, then one
     * decoder layer (attention with its own KV slot, routed and shared
     * experts through their own cache), a norm and the main LM head: a
     * guess at t_{p+1}. hid holds the main model's hidden states of the last
     * call, mtp_hid the MTP's own (to chain guesses).
     */
    struct janas_jns mtp_j;
    /* Gemma 4's assistant, in place of an MTP block (model_assist.c) */
    struct janas_assist *asst;
    uint8_t *mtp_res;
    struct layer mtp_layer;
    struct janas_expert_cache *mtp_cache;
    int mtp;           /* an MTP block: its own file, or in the main one */
    int mtp_own_cache; /* mtp_cache belongs to the MTP file */
    const struct janas_jns_layer *mtp_jl; /* its experts: layer table entry */
    uint32_t mtp_cl;                      /* and layer index in mtp_cache */
    const struct janas_jns_tensor *eh_proj, *enorm, *hnorm, *mtp_norm;
    float *hid, *mtp_hid, *cat;
    struct janas_block_q8k *catq;
    int mtp_open;
    /*
     * Draft head: the rows of the LM head for a subset of the vocabulary,
     * copied into one matrix. A draft only has to be likely, not exact (the
     * main model checks it), so it is chosen among the dh_base lowest token
     * ids (byte-pair merges in order: the most frequent tokens) plus every
     * token seen in the context, at a fraction of the full head's reads.
     */
    uint8_t *dh_w, *dh_in; /* rows; per token id: 1 if in the head */
    const uint8_t *dh_src; /* the full head the rows come from */
    int32_t *dh_ids;
    uint32_t dh_n, dh_cap;
    size_t dh_row;
    float *dh_logits;
    float *big, *qgate, *ba, *mixed, *conv_out, *beta, *decay;
    struct janas_block_q8k *xq, *attq;
    /* routed experts: one pair per (token, chosen expert), sorted by expert,
       then one pair per token for the shared expert */
    uint32_t *pair_tok, *pair_exp, *order, *count, *glist;
    float *pair_w, *gate, *up, *h, *dout;
    uint32_t *hlen; /* per pair row: the rows of its expert's gate */
    /* the Ana-Net probe (model_ana.c): its mode (0 off), the fraction of a
       pair's neurons switched off, each pair row's agreements (ff_buf floats
       a row), the layer being computed, a token's picture and where it goes */
    int ana_mode, ana_tok; /* ana_tok: the fraction is of a token's neurons */
    float ana_frac;
    float *ana_agree;
    /* each pair row's expert (n_expert: the shared one), and the norms of
       the columns of each layer's experts' down matrices, with whether
       computed yet */
    uint32_t *ana_exp;
    float *ana_coln;
    uint8_t *ana_coln_ok;
    /* JANAS_ANA_STATS: every pair's record, and the tokens done so far */
    FILE *ana_stats, *ana_xstats; /* ana_xstats: JANAS_ANA_XSTATS */
    uint64_t ana_base;
    /* JANAS_ANA_COARSE: each pair row's h from gate and up at two bits, the
       choice made on it; the neurons both choices keep, and the true one's */
    float *ana_coarse;
    uint64_t ana_both_kept, ana_true_kept;
    /* JANAS_ANA_GRADE: the fractions of a token's neurons kept at 6, 4 and 2
       bits of down, and each pair row's levels (3, 2, 1; 0 off) */
    float ana_grade[3];
    uint8_t *ana_level;
    /* JANAS_ANA_GULOW: the routed experts of least weight in a token whose
       gate and up are read at two bits, and each pair row's mark */
    int ana_gulow;
    uint8_t *ana_low;
    uint32_t ana_layer, ana_width;
    /* the Janet probe (model_janet.c): a layer's feed-forward recorded, or
       run by Janet's detectors in its place; NULL when off */
    struct janas_janet *janet;
    uint8_t *ana_frame;
    char ana_dir[512];
    int ana_frames, ana_frames_max;
    /* the experts' down matrix by neuron (model_neurons.c): whether a layer
       has it, whether a token's neurons are ranked into levels and the
       fractions read at 6, 4 and 2 bits, each pair row's levels, a pass's
       scores and its pair rows by token */
    int neuron_down, neuron_rank;
    float neuron_frac[3];
    uint8_t *nlevel;
    float *nscore;
    /* a token's chunks of neurons, each its sum, and its length; the pair
       rows whose h needs no eight-bit copy */
    float *npart;
    size_t npart_len;
    uint8_t *nskipq;
    uint32_t *ntok_first, *ntok_fill, *ntok_list;
    struct janas_block_q8k *xg, *hq;
    uint32_t k_use; /* experts used per token, at most k (see set_experts) */
    int exp_level;  /* planes of the experts' down matrix read: 1, 2 or 3 */
    int exp_planes; /* some layer keeps its down matrix in planes */
    int exp_auto;   /* the level was chosen from the memory available */
    int k_auto;     /* and so was the number of experts per token */
    /* an expert not in RAM that every token of the block weighs below this
       is left out, not read (see skip_misses); 0: never. And how many were */
    float miss_skip;
    uint64_t skip_asked, skip_left;
    /* the products' output buffers and the recurrent state and log, whole
       pages, shared with the GPU */
    struct janas_out_buf {
        float *p;
        size_t bytes;
    } out[24];
    int n_out;
    int warm_use;        /* preload the most used experts, and count them */
    uint32_t *use_count; /* experts used this session, n_layer * n_expert */
    uint64_t head_before, head_after; /* output head requantized (head_bits) */
    /* the head's type in the file (BF16 30, F16 1) when rewritten as Q8_0 at
       load (head16_to_q8_0), else 0; and the bytes before and after */
    int head16;
    uint64_t head16_before, head16_after;
    /* the caller's CPUs before the load pinned it (a cpu_set_t), given
       back when the model is freed */
    unsigned char caller_cpus[128];
    int caller_pinned;
    janas_llm_route_fn route_fn; /* janas_llm_model_route_observer */
    void *route_ctx;
    janas_llm_route_fn pred_fn; /* janas_llm_model_route_predictor */
    void *pred_ctx;
    uint32_t pred_k;
    int pred_early; /* ahead of layer l's experts, not after layer l */
    /* reads ahead (janas_m_prefetch): the next layer's router applied
       before this layer's experts, of its first r experts at most b read
       ahead. pf_mode 0 never, 1 where it pays (pf_on, decided every
       PF_WINDOW tokens from the experts the cache would have missed), 2
       always (JANAS_PREFETCH=0, unset, =r,b) */
    uint32_t pf_rank, pf_max;
    int pf_mode, pf_on;
    uint32_t pf_tokens;
    uint64_t pf_would;
    /* reads ahead in a prefill block (janas_m_prefetch_block): pfb_mode 0
       never, 1 after a block that missed many experts (pfb_on), 2 always
       (JANAS_PREFETCH_PREFILL=0, unset, =2) */
    int pfb_mode, pfb_on;
    uint64_t warm_experts; /* experts preloaded at open from the profile */
    double warm_seconds;

    double phase[JANAS_PH_COUNT];
};

/* Shared by the files of the forward pass (model*.c). */

static inline double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* The bytes of a resident tensor, from the main file or the MTP file. */
static inline const void *data(const struct janas_llm_model *m,
                               const struct janas_jns_tensor *t)
{
    if (m->mtp_res && t >= m->mtp_j.tensors &&
        t < m->mtp_j.tensors + m->mtp_j.h.n_tensor)
        return m->mtp_res + (t->offset - m->mtp_j.h.resident_offset);
    return m->resident + (t->offset - m->j.h.resident_offset);
}

static inline const float *f32(const struct janas_llm_model *m,
                               const struct janas_jns_tensor *t)
{
    return data(m, t);
}

/* The tensors of layer l of file j (attention or DeltaNet, then the FFN). */
/* JANAS_TRACE=2: a checkpoint of one row, to compare with another
   implementation, layer by layer. */
static inline void trace_row(const struct janas_llm_model *m, const char *label,
                             const float *v, uint32_t dim)
{
    if (m->trace < 2)
        return;
    double nn = 0;
    for (uint32_t i = 0; i < dim; i++)
        nn += (double)v[i] * v[i];
    fprintf(stderr, "  %-14s dim %5u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
            label, dim, sqrt(nn), v[0], v[1], v[2], v[3]);
}

static inline void rms_norm(float *out, const float *x, const float *w,
                            uint32_t n, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (uint32_t i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + janas_expf(-x));
}

/*
 * The per-token work of a block, spread over the pool: every token (or every
 * head of every token) is done by one thread with the same operations in the
 * same order as on its own, so the results are the same bits. A single
 * token stays on the caller: there is nothing to spread.
 */
struct tok_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    const float *w; /* the norm's weights */
    uint32_t first, n, pos0;
    float *kc_scale, *vc_scale;
    int8_t *kc, *vc;
    uint32_t cap, ring; /* the slot's positions per head, and whether a ring */
    float *out;         /* rows the job rewrites (the soft-capped logits) */
};

void janas_m_init_tuner(struct janas_llm_model *m);
/* Memory the GPU will import collapsed into 2 MB pages (Linux; see
   model_load.c), before the import pins it. */
/* The Ana-Net probe (model_ana.c, JANAS_ANA_*): set up at load (0, or -1
   out of memory), each run's agreements after its gate and up products, a
   pair row's neurons switched off before its down product, a token's
   picture written once its layers are done, freed. */
int janas_m_ana_init(struct janas_llm_model *m, size_t pairs);
void janas_m_ana_agree(struct janas_llm_model *m, const uint8_t *wg,
                       const uint8_t *wu, const struct janas_jns_matrix *mx,
                       uint32_t a, uint32_t b);
void janas_m_ana_mask(struct janas_llm_model *m, size_t g, float *h,
                      uint32_t ff);
void janas_m_ana_coln(struct janas_llm_model *m, uint32_t e,
                      const uint8_t *down, const uint8_t *p1, const uint8_t *p0,
                      const struct janas_jns_matrix *mx);
void janas_m_ana_tok(struct janas_llm_model *m, uint32_t n);
void janas_m_ana_lowmark(struct janas_llm_model *m, uint32_t n);
void janas_m_ana_grade(struct janas_llm_model *m, const uint8_t *down,
                       const uint8_t *p1, const uint8_t *p0,
                       const struct janas_jns_matrix *mx, uint32_t a,
                       uint32_t b);
void janas_m_ana_frame(struct janas_llm_model *m, uint32_t n);
void janas_m_ana_free(struct janas_llm_model *m);
int janas_m_janet_init(struct janas_llm_model *m);
void janas_m_janet_rec(struct janas_llm_model *m, uint32_t l, uint32_t n,
                       int after);
int janas_m_janet_ffn(struct janas_llm_model *m, uint32_t l, uint32_t n);
void janas_m_janet_free(struct janas_llm_model *m);
/* The experts' down matrix by neuron (model_neurons.c): set up at load (0,
   or -1 out of memory), the levels of a pass's neurons, its experts' down
   products, freed. An expert whose down matrix is by neuron, applied to the
   pair rows [a, b): */
struct janas_neuron_run {
    const float *scale; /* each output channel's, after the sum */
    const struct janas_block_q6kp *base;
    const uint8_t *p1, *p0; /* the planes the cache holds, or NULL */
    uint32_t rows, cols;    /* neurons, outputs */
    uint32_t a, b;
};
int janas_m_neurons_init(struct janas_llm_model *m, size_t pairs);
void janas_m_neuron_levels(struct janas_llm_model *m, uint32_t ng);
void janas_m_neuron_down(struct janas_llm_model *m,
                         const struct janas_neuron_run *runs, uint32_t n);
void janas_m_neurons_free(struct janas_llm_model *m);
void janas_m_huge_pages(void *p, uint64_t bytes);
/* The longest region of memory the GPU imports (gpu.c janas_gpu_import): a
   page short of 4 GiB, which Mesa's Intel driver gave the kernel as 0 bytes
   (21 Oct 2026). */
#define JANAS_M_REGION_MAX (((uint64_t)4 << 30) - 4096)
/* The next region the GPU imports of bytes, from at: its first byte into
   *from (at, or the 2 MB boundary before it where huge: memory allocated on
   one), and its end, returned: bytes, or past at a multiple of unit (unit
   0: the farthest it may reach, for the caller to cut), never more than
   JANAS_M_REGION_MAX from *from. */
uint64_t janas_m_import_next(uint64_t at, uint64_t bytes, uint64_t unit,
                             int huge, uint64_t *from);
int janas_m_gpu_import(struct janas_llm_model *m, const uint8_t *p,
                       uint64_t bytes, uint64_t unit);
void janas_m_arena_to_gpu(struct janas_llm_model *m);
/* A block on the GPU whole (model_gpu.c): 0; -1, the CPU to run it; -2,
   the GPU failed after replaying on the base in place, the state lost. */
int janas_m_gpu_token(struct janas_llm_model *m, uint32_t pos, uint32_t n,
                      int all, float *logits);
void janas_m_apply_candidate(struct janas_llm_model *m, int cand, uint32_t nv);
/* Whether the whole token is asked of the GPU (JANAS_GPU_TOKEN=1). */
int janas_m_gpu_token_asked(void);
/* model_npu.c: the NPU for the products of blocks (JANAS_NPU=1), opened
   once the resident region is read; janas_m_products runs a group of
   products as janas_gpu_matvec_group does, those the NPU takes there. */
void janas_m_npu_open(struct janas_llm_model *m);
void janas_m_npu_close(struct janas_llm_model *m);
/* Whether the NPU takes products of a pass of n tokens (its candidate has
   it, and n reaches JANAS_NPU_MIN_N). */
int janas_m_npu_takes(const struct janas_llm_model *m, uint32_t n);
/* Whether the tuner may give the NPU the blocks by itself: where its
   answers measured at par with the CPU's (janas_llm_model_npu_auto). */
int janas_m_npu_auto(const struct janas_llm_model *m);
/* The NPU kept awake through a pass of n tokens it takes products of
   (between them the CPU works long enough for it to be suspended), let
   sleep otherwise. */
void janas_m_npu_awake(struct janas_llm_model *m, uint32_t n);
/* A pass is over: a preparation of the NPU's graphs it began, too
   (janas_llm_model_npu_preparing). */
void janas_m_npu_pass_end(struct janas_llm_model *m);
/*
 * A feed-forward run (an expert, a shared expert, a dense model's) on the
 * NPU in one trip: gate, up, the activation and down, its n_vec rows of x
 * (Q8_K) in, n_vec rows of d_model floats out at dout once finished. w:
 * the slots of gate, up and down (each matrix at its mx offset in it),
 * plane: down's planes (Q6_K_P), mx their matrices. 0 with the run in flight
 * (the caller computes the others, then janas_m_npu_ffn_finish), -1 if it stays
 * on the CPU. One at a time.
 */
struct janas_m_npu_ffn {
    struct npu_entry *e;
    float *dout;
    size_t n_vec, d;
};
int janas_m_npu_ffn_begin(struct janas_llm_model *m, const uint8_t *const w[3],
                          const uint8_t *const plane[2],
                          const struct janas_jns_matrix *mx,
                          const struct janas_block_q8k *x, uint32_t n_vec,
                          float *dout, struct janas_m_npu_ffn *job);
/* Waits for it and stores its rows; -1 if the NPU failed (the run is then
   the caller's to compute). */
int janas_m_npu_ffn_finish(struct janas_llm_model *m,
                           struct janas_m_npu_ffn *job);
/*
 * The block's first layers on the NPU, a graph each (norms, projections,
 * the FFN, the residual), the attention or the recurrence between them on
 * the CPU: for dense models laid out as Qwen3, Qwen3.5 or Gemma 4, when at
 * least nine layers in ten fit the NPU (JANAS_NPU_LAYERS=2: the first ones
 * alone, however few; 0: never). The number of layers done
 * (m->n_layer: all), m->x their output, the caller's loop going on from
 * there; -1 if not done (nothing touched), -2 if the NPU failed halfway
 * (the block cannot be finished).
 */
int janas_m_npu_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0);
/* The calibration of the NPU's INT8 activations (JANAS_NPU_A8=1): rows of
   width values (stride apart) of an input of layer l's products, at site
   0 the attention's output (wo), 1 the FFN's norm (gate, up), 2 the FFN's
   intermediate (down), 3 the norm before the attention (q, k, v); their
   channel maxima kept. Nothing outside a calibration's pass. */
void janas_m_npu_calib(struct janas_llm_model *m, int site, uint32_t l,
                       const float *x, uint32_t rows, uint32_t width,
                       size_t stride);
void janas_m_products(struct janas_llm_model *m, struct janas_gpu *g,
                      const struct janas_matvec_task *t, size_t n);
uint32_t *janas_m_profile_read(const struct janas_llm_model *m, size_t *used);
void janas_m_profile_write(const struct janas_llm_model *m);
void janas_m_norm_quant(struct janas_llm_model *m, const float *w,
                        uint32_t first, uint32_t n);
void janas_m_residual(struct janas_llm_model *m, uint32_t n);
void janas_m_post_norm(struct janas_llm_model *m, const float *w, uint32_t n);
void janas_m_attn_core(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n, uint32_t pos0, double ta);
void janas_m_attn_layer(struct janas_llm_model *m, const struct layer *ly,
                        uint32_t n, uint32_t pos0);
void janas_m_rec_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n);
/* Its recurrence on a block whose input products are in m->big and betas
   and alphas in m->ba: the convolution, the state, the gated norm, into
   m->att. ta: when the layer began (its products counted with it). */
void janas_m_rec_core(struct janas_llm_model *m, const struct layer *ly,
                      uint32_t n, double ta);
/* Qwen3.5's betas then alphas into m->ba, from n normalized rows of xn. */
void janas_m_rec_ba(struct janas_llm_model *m, const struct layer *ly,
                    const float *xn, uint32_t n);
void janas_m_embed(struct janas_llm_model *m, int32_t token, float *x);
/* sum of exp(l[i] - mx), i < n, vectorized where the CPU can */
double janas_m_exp_sum(const float *l, uint32_t n, float mx);
/* the draft head: the first base rows of the head src (rows of row_bytes) */
int janas_m_init_draft_head(struct janas_llm_model *m, const uint8_t *src,
                            size_t row_bytes, uint32_t base);
int janas_m_assist_is(const char *path);
int janas_m_assist_load(struct janas_llm_model *m, const char *path,
                        int attn_scores, char *err, size_t err_len);
void janas_m_assist_free(struct janas_llm_model *m);
int janas_m_assist_head(struct janas_llm_model *m, uint32_t base);
int janas_m_assist_draft(struct janas_llm_model *m, int32_t token,
                         const float *h, uint32_t p, float *hnext, int32_t *out,
                         float *conf);
int janas_m_ple_inputs(struct janas_llm_model *m, const int32_t *tokens,
                       uint32_t n);
void janas_m_ple_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t l, uint32_t n);
int janas_m_exp_tracing(void);
void janas_m_exp_report(void);
/* which experts a moe_block call runs */
enum { MOE_ALL, MOE_SHARED, MOE_ROUTED };
/* The router of layer l applied ahead to the state x (for the predictor
   probe, janas_llm_model_route_predictor): 0 or -1. */
int janas_m_route_predict(struct janas_llm_model *m, uint32_t l, uint32_t n);
/* The router of layer l applied to the state x of n tokens: its first k
   experts for each token into ids (n x k), best first. 0, or -1 when the
   layer has no router to apply ahead. Uses xn, xq and router. */
int janas_m_route_rank(struct janas_llm_model *m, uint32_t l, uint32_t n,
                       uint32_t k, uint32_t *ids);
/* Reads ahead for layer l + 1 of a single token (JANAS_PREFETCH). */
void janas_m_prefetch(struct janas_llm_model *m, uint32_t l);
/* The same for a prefill block of n tokens (JANAS_PREFETCH_PREFILL). */
void janas_m_prefetch_block(struct janas_llm_model *m, uint32_t l, uint32_t n);
int janas_m_moe_block(struct janas_llm_model *m, const struct layer *ly,
                      struct janas_expert_cache *cache,
                      const struct janas_jns_layer *jl, uint32_t l, uint32_t n,
                      int part);

/*
 * The GPU a product may use, or NULL: the tuner's choice for this pass
 * (gpu_use); with the fast prompt the GPU too where the tuner chose the
 * CPU alone, for the fast prompt's products only (janas_gpu_set_only_fast),
 * which must always run there to give the same results. arena: a product
 * of the experts' arena, kept off the GPU while its copy there is stale.
 */
static inline struct janas_gpu *janas_m_gpu(const struct janas_llm_model *m,
                                            int arena)
{
    if (!m->gpu || m->gpu_off || (arena && m->arena_stale))
        return NULL;
    if (m->gpu_use ||
        (m->tuner.allow_gpu && janas_gpu_fast_prompt() && !m->prompts_off_gpu))
        return m->gpu;
    return NULL;
}

#endif
