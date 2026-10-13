/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.h - a whole token on the GPU, with one submission: every layer's
 * norms, products, RoPE, KV cache and attention, the feed-forward and the
 * output head recorded in one command buffer, the logits back at the end
 * (private to the library).
 *
 * For now the dense attention models (qwen3's shape: q and k norms, NeoX
 * RoPE, an eight-bit KV cache, SiLU feed-forward), and qwen3.5's (Gated
 * DeltaNet layers between them, the query with an output gate) for blocks,
 * both with a mixture of experts too (qwen3moe, qwen35moe: a submission a
 * layer, the CPU sorting the router's choice in between),
 * on a GPU that shares the host's memory: the weights, the experts' arena and
 * the KV cache are the CPU's own memory, imported, so the CPU's passes (a
 * prompt in blocks) and the GPU's tokens read and write the same cache. The
 * products are the CPU's bit for bit (their shaders); the norms and the
 * attention sum in the GPU's order.
 */
#ifndef JANAS_LLM_GPU_TOKEN_H
#define JANAS_LLM_GPU_TOKEN_H

#include <stdint.h>

#include "gpu.h"
#include "matvec.h"

/* One layer: host pointers, all in regions the GPU holds. The products'
   tasks give type, weights (and planes), rows and cols; their x, y and
   n_vec are the GPU's own. */
struct janas_gpu_tok_layer {
    const float *attn_norm, *q_norm, *k_norm, *ffn_norm;
    struct janas_matvec_task wq, wk, wv, wo, gate, up, down;
    int8_t *kc, *vc; /* the layer's KV cache: n_kv x cap x hd bytes */
    float *ks, *vs;  /* its scales: n_kv x cap */
    /* a Gated DeltaNet layer (qwen3.5) in place of the attention: q|k|v and
       z products, beta then alpha (2 nv x dm, f32), the convolution
       (d_conv x conv_ch, transposed), dt and A (nv), the heads' norm (ds),
       the output product; its state slot */
    int rec;
    struct janas_matvec_task ssm_qkv, ssm_z, ssm_out;
    const float *ssm_beta, *ssm_alpha, *conv_w, *dt, *ssm_a, *ssm_norm;
    uint32_t slot;
    /* a mixture's layer (the model's n_expert > 1): the router (n_expert x
       dm, f16) and the shared expert's gate (dm floats, or NULL); gate, up
       and down the experts' types and shapes, with no weights - those come
       from the model's slots (janas_gpu_tok_model fetch): off, each
       matrix's bytes into a slot, poff the down matrix's two bare planes'
       (Q6_K_P: 0, a plane the cache does not hold); sh_gate_w, sh_up, sh_down
       the shared expert (type 0: none) */
    const uint16_t *router16;
    const float *sh_gate;
    size_t off[3], poff[2];
    struct janas_matvec_task sh_gate_w, sh_up, sh_down;
};

struct janas_gpu_tok_model {
    uint32_t n_layer, dm, n_head, n_kv, hd, n_rot, ff, n_vocab, cap;
    float eps, attn_scale;
    const struct janas_gpu_tok_layer *layers;
    const float *out_norm;
    struct janas_matvec_task output;
    /* for blocks of more than one token, where layers and output hold the
       GPU's single-vector copies (JANAS_GPU_REPACK): the weights as they
       are; NULL: layers and output */
    const struct janas_gpu_tok_layer *blayers;
    struct janas_matvec_task boutput;
    /* qwen3.5: wq's rows head by head, each head's q then its output gate;
       the recurrent layers' shapes and their states, two halves each (the
       model's double buffer): n_rec x nv x ds x ds sixteen-bit numbers and
       n_rec x (d_conv - 1) x conv_ch floats */
    int gated_q;
    uint32_t n_rec, ds, nv, nkh, d_conv, conv_ch, d_inner;
    int vmap_mod;
    uint16_t *ssm_state;
    float *conv_state;
    /* their log (the model's, model_impl.h): per recurrent layer log_rows
       rows of conv_ch floats (the convolution's inputs) and of nv floats
       (betas, decays), what a call replays of the last one when that one's
       last tokens were taken back (rejected drafts); NULL: blocks short
       enough to be logged stay on the CPU */
    float *log_mixed, *log_beta, *log_decay;
    uint32_t log_rows;
    /* a mixture (n_expert > 1, 0 for a dense model): k experts a token, of
       eff rows each; ff above is the shared expert's (eff without one).
       Its layers run one at a time, the CPU between two of them: the
       router's choice read, its experts' slots asked of fetch (0, or -1:
       the block goes to the CPU), the pairs given to routed (or NULL) */
    uint32_t n_expert, k, eff;
    int (*fetch)(void *ctx, uint32_t layer, const uint32_t *ids, size_t n,
                 const uint8_t **slots);
    void (*routed)(void *ctx, uint32_t layer, uint32_t n, uint32_t k,
                   const uint32_t *ids);
    /* reads every expert of the layer that is not in the cache, while the
       caller goes on (or NULL): a block of many tokens chooses nearly all
       of them, so none is read for nothing */
    void (*prefetch)(void *ctx, uint32_t layer);
    void *ctx;
};

struct janas_gpu_tok;

/* The most tokens of the last call a block replays before its own (see
   janas_gpu_tok_run): a chain of drafts is shorter. */
#define JANAS_GPU_TOK_MAX_REPLAY 16

/* Buffers and shaders for a model of that shape; NULL (the GPU cannot, or
   out of memory) with the reason in err. */
struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len);
void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);

/* The most tokens janas_gpu_tok_run takes at once (0 without t). */
uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t);
/* ... starting at position pos: fewer as the context grows (the
   attention's runs take room). */
uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos);

/*
 * A block of n tokens (1 .. janas_gpu_tok_max_block) whose embeddings are x
 * (n x dm floats), token j at position pos + j (its keys and values go to
 * row pos + j of the cache; positions 0 .. pos + j attended), cs and sn
 * their RoPE tables (n x n_rot / 2 each): the logits (n_vocab each) of
 * the last token (all 0), of every token (1) or of none (2) into logits;
 * with hidden, the final hidden states (n x dm: the last norm's output, an
 * MTP block's input) there. The recurrent layers read their state from
 * half src and write it to half dst (the same for a block's later pieces);
 * first they replay the first replay tokens (at most
 * JANAS_GPU_TOK_MAX_REPLAY) of the log on half src in place, as the CPU
 * does when a call starts before the end of the last one (src is then the
 * base); with log_row other than UINT32_MAX they log the block's tokens
 * from that row of the log on.
 * 0; -1 with nothing changed but the cache rows pos .. pos + n - 1 (the
 * caller runs the block on the CPU then); -2 the GPU failed after it had
 * started, with the state no longer what it was.
 */
int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row);

#endif
