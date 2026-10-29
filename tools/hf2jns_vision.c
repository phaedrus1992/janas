/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf2jns_vision.c - the vision encoder of a multimodal model, from its
 * Hugging Face checkpoint (safetensors) into a JNS file of resident
 * tensors (no layers, no experts) beside the model's: <model>.vision.jns.
 *
 * Qwen3.5 and Qwen3-VL (model.visual.*): a ViT - a 3D convolution over
 * patches of 16x16 pixels and two frames (a still picture is given twice),
 * learned positions on a 48x48 grid interpolated to the picture's, blocks
 * of LayerNorm, attention with a 2D rotary embedding, and a GELU MLP - and
 * a merger that joins 2x2 patches into one token of the language model's
 * width. The matrices are kept in bfloat16 as the checkpoint has them, the
 * norms and biases in float; the metadata holds the sizes, the processor's
 * mean, deviation and pixel bounds, and the ids of the image tokens.
 *
 * Gemma 4 (model.vision_tower.*): see gemma() below.
 *
 * Usage: hf2jns_vision <checkpoint directory> <out.vision.jns>
 *   the directory holding config.json, preprocessor_config.json (or
 *   processor_config.json) and the
 *   safetensors (one file, or the shards of model.safetensors.index.json).
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/hf_safetensors.h"
#include "llm/jns_write.h"
#include "llm/json.h"

#define GGML_F32 0
#define GGML_BF16 30

static _Noreturn void die(const char *msg, const char *arg)
{
    fprintf(stderr, "hf2jns_vision: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(1);
}

/* ---- the checkpoint ---- */

static struct janas_st shards[64];
static int n_shards;

static void open_shards(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d)
        die("cannot open", dir);
    struct dirent *e;
    char p[4096], err[300];
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors"))
            continue;
        if (n_shards == 64)
            die("more than 64 shards in", dir);
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (janas_st_open(&shards[n_shards], p, err, sizeof err) != 0)
            die(err, NULL);
        n_shards++;
    }
    closedir(d);
    if (!n_shards)
        die("no safetensors in", dir);
}

static const struct janas_st *find(const char *name, struct janas_st_tensor *t)
{
    for (int i = 0; i < n_shards; i++)
        if (janas_st_find(&shards[i], name, t) == 0)
            return &shards[i];
    return NULL;
}

static struct janas_json_doc *read_json_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot read", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1);
    if (!t || fread(t, 1, (size_t)n, f) != (size_t)n)
        die("cannot read", path);
    fclose(f);
    char err[200];
    struct janas_json_doc *d = janas_json_parse(t, (size_t)n, err, sizeof err);
    free(t);
    if (!d)
        die(err, path);
    return d;
}

/* ---- the output ---- */

static struct janas_jw W;

static void put(const char *dst, uint32_t type, uint32_t n_dims,
                const uint64_t *dims, void *data, uint64_t bytes)
{
    if (!data)
        die("cannot read or out of memory", dst);
    if (janas_jw_tensor(&W, dst, type, n_dims, dims, data, bytes) != 0)
        die("out of memory, or a name too long", dst);
}

/* a tensor of the checkpoint as one of the file: a matrix (rows x cols,
   the rest of its dimensions flattened into the row; matrix 2: all but the
   last flattened into the rows) in bfloat16, anything else in float */
static void add(const char *src, const char *dst, int matrix)
{
    struct janas_st_tensor t;
    const struct janas_st *s = find(src, &t);
    if (!s)
        die("tensor missing or not F32/F16/BF16", src);
    if (matrix && t.dims < 2)
        die("not a matrix", src);
    if (matrix) {
        uint64_t cols =
            matrix == 2 ? t.shape[t.dims - 1] : t.count / t.shape[0];
        /* ggml order: the row length first */
        uint64_t dims[2] = {cols, t.count / cols};
        put(dst, GGML_BF16, 2, dims, janas_st_bf16(s, &t), 2 * t.count);
    } else {
        uint64_t dims[1] = {t.count};
        put(dst, GGML_F32, 1, dims, janas_st_floats(s, &t), 4 * t.count);
    }
}

/* the scalars of the checkpoint in names[] (each %u the layer) packed into
   one float tensor */
static void add_scalars(const char *const *names, int n, uint32_t l,
                        const char *dst)
{
    float *x = malloc(sizeof(float) * (size_t)n);
    if (!x)
        die("out of memory", NULL);
    for (int i = 0; i < n; i++) {
        char s[160];
        snprintf(s, sizeof s, names[i], l);
        struct janas_st_tensor t;
        const struct janas_st *st = find(s, &t);
        float *v = st && t.count == 1 ? janas_st_floats(st, &t) : NULL;
        if (!v)
            die("scalar missing", s);
        x[i] = v[0];
        free(v);
    }
    uint64_t dims[1] = {(uint64_t)n};
    put(dst, GGML_F32, 1, dims, x, 4 * (uint64_t)n);
}

/* bounds that clamp nothing, for linears without them (Gemma 4 26B) */
static void add_bounds(const char *dst)
{
    float *x = malloc(sizeof(float) * 7 * 4);
    if (!x)
        die("out of memory", NULL);
    for (int i = 0; i < 7 * 4; i++)
        x[i] = i % 2 ? FLT_MAX : -FLT_MAX;
    uint64_t dims[1] = {7 * 4};
    put(dst, GGML_F32, 1, dims, x, 4 * 7 * 4);
}

static void m_u32(const char *k, uint32_t v)
{
    janas_jw_u32(&W, k, v);
}

static void m_f32(const char *k, float v)
{
    janas_jw_f32(&W, k, v);
}

static void m_str(const char *k, const char *v)
{
    janas_jw_str(&W, k, v);
}

static uint32_t cfg_u32(const struct janas_json *o, const char *k, int need,
                        uint32_t def)
{
    const struct janas_json *v = janas_json_get(o, k);
    if (!v || v->type != JANAS_JSON_NUMBER) {
        if (need)
            die("config.json lacks", k);
        return def;
    }
    return (uint32_t)janas_json_num(v, 0);
}

/* Qwen3.5 and Qwen3-VL: the tensors and the metadata; the architecture,
   the width, a line of what was converted */
static const char *qwen(const struct janas_json *cfg,
                        const struct janas_json *pre,
                        const struct janas_json *vc, const char *mt,
                        uint32_t *width, char *what, size_t what_len)
{
    const struct janas_json *ds =
        janas_json_get(vc, "deepstack_visual_indexes");
    if (ds && ds->n)
        die("deepstack layers are not converted yet", mt);
    uint32_t depth = cfg_u32(vc, "depth", 1, 0);
    uint32_t hidden = cfg_u32(vc, "hidden_size", 1, 0);
    uint32_t ff = cfg_u32(vc, "intermediate_size", 1, 0);
    uint32_t heads = cfg_u32(vc, "num_heads", 1, 0);
    uint32_t patch = cfg_u32(vc, "patch_size", 1, 0);
    uint32_t tpatch = cfg_u32(vc, "temporal_patch_size", 1, 0);
    uint32_t merge = cfg_u32(vc, "spatial_merge_size", 1, 0);
    uint32_t npos = cfg_u32(vc, "num_position_embeddings", 1, 0);
    uint32_t out_dim = cfg_u32(vc, "out_hidden_size", 1, 0);
    const char *act = janas_json_str(janas_json_get(vc, "hidden_act"));
    if (!act || strcmp(act, "gelu_pytorch_tanh"))
        die("an activation this does not run", act);
    if (!depth || !hidden || !heads || hidden % heads || !patch || !tpatch ||
        !merge || !npos || !out_dim)
        die("vision sizes out of range", NULL);

    /* tensors */
    add("model.visual.patch_embed.proj.weight", "v.patch_embd", 1);
    add("model.visual.patch_embed.proj.bias", "v.patch_embd.bias", 0);
    add("model.visual.pos_embed.weight", "v.pos_embd", 1);
    char s[160], d[64];
    static const char *const PARTS[][3] = {
        {"norm1.weight", "ln1.weight", "0"},
        {"norm1.bias", "ln1.bias", "0"},
        {"norm2.weight", "ln2.weight", "0"},
        {"norm2.bias", "ln2.bias", "0"},
        {"attn.qkv.weight", "attn_qkv", "1"},
        {"attn.qkv.bias", "attn_qkv.bias", "0"},
        {"attn.proj.weight", "attn_out", "1"},
        {"attn.proj.bias", "attn_out.bias", "0"},
        {"mlp.linear_fc1.weight", "ffn_up", "1"},
        {"mlp.linear_fc1.bias", "ffn_up.bias", "0"},
        {"mlp.linear_fc2.weight", "ffn_down", "1"},
        {"mlp.linear_fc2.bias", "ffn_down.bias", "0"}};
    for (uint32_t l = 0; l < depth; l++)
        for (size_t i = 0; i < sizeof PARTS / sizeof PARTS[0]; i++) {
            snprintf(s, sizeof s, "model.visual.blocks.%u.%s", l, PARTS[i][0]);
            snprintf(d, sizeof d, "v.blk.%u.%s", l, PARTS[i][1]);
            add(s, d, PARTS[i][2][0] == '1');
        }
    add("model.visual.merger.norm.weight", "v.merger.ln.weight", 0);
    add("model.visual.merger.norm.bias", "v.merger.ln.bias", 0);
    add("model.visual.merger.linear_fc1.weight", "v.merger.fc1", 1);
    add("model.visual.merger.linear_fc1.bias", "v.merger.fc1.bias", 0);
    add("model.visual.merger.linear_fc2.weight", "v.merger.fc2", 1);
    add("model.visual.merger.linear_fc2.bias", "v.merger.fc2.bias", 0);

    /* the shapes the encoder needs, checked here once */
    if (W.t[0].dims[1] != hidden ||
        W.t[0].dims[0] != 3ull * tpatch * patch * patch ||
        W.t[2].dims[1] != npos || W.t[2].dims[0] != hidden)
        die("the patch embedding or the positions are not of the sizes the "
            "config says",
            NULL);

    m_str("general.architecture", "qwen3vl_vision");
    m_str("vision.model_type", mt);
    m_u32("vision.block_count", depth);
    m_u32("vision.embedding_length", hidden);
    m_u32("vision.feed_forward_length", ff);
    m_u32("vision.attention.head_count", heads);
    m_f32("vision.attention.layer_norm_epsilon", 1e-6f);
    m_f32("vision.rope.freq_base", 10000.0f);
    m_u32("vision.patch_size", patch);
    m_u32("vision.temporal_patch_size", tpatch);
    m_u32("vision.spatial_merge_size", merge);
    m_u32("vision.position_count", npos);
    m_u32("vision.projection_dim", out_dim);
    m_str("vision.activation", "gelu_tanh");
    float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    const struct janas_json *a = janas_json_get(pre, "image_mean");
    for (int i = 0; a && a->type == JANAS_JSON_ARRAY && i < 3; i++) {
        const struct janas_json *v = a->child;
        for (int k = 0; v && k < i; k++)
            v = v->next;
        mean[i] = (float)janas_json_num(v, mean[i]);
    }
    a = janas_json_get(pre, "image_std");
    for (int i = 0; a && a->type == JANAS_JSON_ARRAY && i < 3; i++) {
        const struct janas_json *v = a->child;
        for (int k = 0; v && k < i; k++)
            v = v->next;
        stdv[i] = (float)janas_json_num(v, stdv[i]);
    }
    for (int i = 0; i < 3; i++) { /* per channel: r, g, b */
        char k[40];
        snprintf(k, sizeof k, "vision.image_mean.%d", i);
        m_f32(k, mean[i]);
        snprintf(k, sizeof k, "vision.image_std.%d", i);
        m_f32(k, stdv[i]);
    }
    const struct janas_json *size = janas_json_get(pre, "size");
    m_u32("vision.min_pixels", cfg_u32(size, "shortest_edge", 0, 56 * 56));
    m_u32("vision.max_pixels",
          cfg_u32(size, "longest_edge", 0, 28 * 28 * 1280));
    m_u32("vision.token.image", cfg_u32(cfg, "image_token_id", 1, 0));
    m_u32("vision.token.start", cfg_u32(cfg, "vision_start_token_id", 1, 0));
    m_u32("vision.token.end", cfg_u32(cfg, "vision_end_token_id", 1, 0));
    *width = hidden;
    snprintf(what, what_len,
             "%u blocks of %u, %u heads, patch %u, merge %u "
             "-> %u",
             depth, hidden, heads, patch, merge, out_dim);
    return "qwen3vl_vision";
}

/* Gemma 4 (model.vision_tower.*, model.embed_vision.*): a ViT of RMS
   norms, clamped linears and a 2D rotary embedding over learned positions
   of the patches' columns and rows, pooled 3 x 3 and projected; the
   clamps of a block's seven linears packed into one tensor of 28 floats (bounds
   that clamp nothing for a model without them) (each linear: input min, max,
   output min, max) */
static const char *gemma(const struct janas_json *cfg,
                         const struct janas_json *pre,
                         const struct janas_json *vc, uint32_t *width,
                         char *what, size_t what_len)
{
    const struct janas_json *tc = janas_json_get(cfg, "text_config");
    uint32_t depth = cfg_u32(vc, "num_hidden_layers", 1, 0);
    uint32_t hidden = cfg_u32(vc, "hidden_size", 1, 0);
    uint32_t ff = cfg_u32(vc, "intermediate_size", 1, 0);
    uint32_t heads = cfg_u32(vc, "num_attention_heads", 1, 0);
    uint32_t kv = cfg_u32(vc, "num_key_value_heads", 0, heads);
    uint32_t hd = cfg_u32(vc, "head_dim", 0, heads ? hidden / heads : 0);
    uint32_t patch = cfg_u32(vc, "patch_size", 1, 0);
    uint32_t pool = cfg_u32(vc, "pooling_kernel_size", 1, 0);
    uint32_t npos = cfg_u32(vc, "position_embedding_size", 1, 0);
    uint32_t out_dim = cfg_u32(tc, "hidden_size", 1, 0);
    uint32_t soft = cfg_u32(pre, "max_soft_tokens", 0,
                            cfg_u32(vc, "default_output_length", 1, 0));
    const char *act = janas_json_str(janas_json_get(vc, "hidden_activation"));
    const struct janas_json *rp = janas_json_get(vc, "rope_parameters");
    const struct janas_json *cl = janas_json_get(vc, "use_clipped_linears");
    const struct janas_json *sz = janas_json_get(vc, "standardize");
    int stdz = sz && sz->type == JANAS_JSON_TRUE;
    if (!act || strcmp(act, "gelu_pytorch_tanh"))
        die("an activation this does not run", act);
    int clipped = cl && cl->type == JANAS_JSON_TRUE;
    if (!depth || !hidden || !heads || hidden % heads || kv != heads ||
        hd * heads != hidden || !patch || !pool || !npos || !out_dim || !soft)
        die("vision sizes out of range", NULL);
    double eps = janas_json_num(janas_json_get(vc, "rms_norm_eps"), 1e-6);
    double theta = janas_json_num(janas_json_get(rp, "rope_theta"), 100.0);

    add("model.vision_tower.patch_embedder.input_proj.weight", "v.patch_embd",
        1);
    add("model.vision_tower.patch_embedder.position_embedding_table",
        "v.pos_embd", 2);
    add("model.embed_vision.embedding_projection.weight", "v.proj", 1);
    if (W.t[0].dims[1] != hidden || W.t[0].dims[0] != 3ull * patch * patch ||
        W.t[1].dims[1] != 2ull * npos || W.t[1].dims[0] != hidden ||
        W.t[2].dims[1] != out_dim || W.t[2].dims[0] != hidden)
        die("the patch embedding, the positions or the projection are not "
            "of the sizes the config says",
            NULL);
    if (stdz) {
        add("model.vision_tower.std_bias", "v.std_bias", 0);
        add("model.vision_tower.std_scale", "v.std_scale", 0);
    }
    static const char *const NORMS[][2] = {
        {"input_layernorm", "in_norm"},
        {"post_attention_layernorm", "post_att_norm"},
        {"pre_feedforward_layernorm", "pre_ff_norm"},
        {"post_feedforward_layernorm", "post_ff_norm"},
        {"self_attn.q_norm", "q_norm"},
        {"self_attn.k_norm", "k_norm"}};
    /* in the order of vision_impl.h's GL_* */
    static const char *const LIN[][2] = {
        {"self_attn.q_proj", "q"}, {"self_attn.k_proj", "k"},
        {"self_attn.v_proj", "v"}, {"self_attn.o_proj", "o"},
        {"mlp.gate_proj", "gate"}, {"mlp.up_proj", "up"},
        {"mlp.down_proj", "down"}};
    static const char *const BOUND[] = {"input_min", "input_max", "output_min",
                                        "output_max"};
    char s[160], d[64];
    static char clip_names[7 * 4][128];
    const char *clips[7 * 4];
    for (uint32_t l = 0; l < depth; l++) {
        const char *pf = "model.vision_tower.encoder.layers";
        for (size_t i = 0; i < sizeof NORMS / sizeof NORMS[0]; i++) {
            snprintf(s, sizeof s, "%s.%u.%s.weight", pf, l, NORMS[i][0]);
            snprintf(d, sizeof d, "v.blk.%u.%s", l, NORMS[i][1]);
            add(s, d, 0);
        }
        for (size_t i = 0; i < 7; i++) {
            snprintf(s, sizeof s, "%s.%u.%s.linear.weight", pf, l, LIN[i][0]);
            snprintf(d, sizeof d, "v.blk.%u.%s", l, LIN[i][1]);
            add(s, d, 1);
            for (int b = 0; b < 4; b++) {
                snprintf(clip_names[i * 4 + b], sizeof clip_names[0],
                         "%s.%%u.%s.%s", pf, LIN[i][0], BOUND[b]);
                clips[i * 4 + b] = clip_names[i * 4 + b];
            }
        }
        snprintf(d, sizeof d, "v.blk.%u.clip", l);
        if (clipped)
            add_scalars(clips, 7 * 4, l, d);
        else
            add_bounds(d);
    }

    m_str("general.architecture", "gemma4_vision");
    m_str("vision.model_type", "gemma4");
    m_u32("vision.block_count", depth);
    m_u32("vision.embedding_length", hidden);
    m_u32("vision.feed_forward_length", ff);
    m_u32("vision.attention.head_count", heads);
    m_f32("vision.attention.layer_norm_epsilon", (float)eps);
    m_f32("vision.rope.freq_base", (float)theta);
    m_u32("vision.patch_size", patch);
    m_u32("vision.pooling_kernel_size", pool);
    m_u32("vision.position_size", npos);
    m_u32("vision.max_soft_tokens", soft);
    m_u32("vision.projection_dim", out_dim);
    m_u32("vision.standardize", (uint32_t)stdz);
    /* the language model's attention over a picture's tokens: both ways
       (12B, 26B) or causal (E2B, E4B) */
    const char *bi =
        janas_json_str(janas_json_get(tc, "use_bidirectional_attention"));
    m_u32("vision.bidirectional", bi && !strcmp(bi, "vision"));
    m_str("vision.activation", "gelu_tanh");
    m_u32("vision.token.image", cfg_u32(cfg, "image_token_id", 1, 0));
    m_u32("vision.token.start", cfg_u32(cfg, "boi_token_id", 1, 0));
    m_u32("vision.token.end", cfg_u32(cfg, "eoi_token_id", 1, 0));
    *width = hidden;
    snprintf(what, what_len,
             "%u blocks of %u, %u heads, patch %u, pooling "
             "%u -> %u",
             depth, hidden, heads, patch, pool, out_dim);
    return "gemma4_vision";
}

/* Gemma 4 12B (gemma4_unified_vision, model.vision_embedder.*): no
   blocks, a 48x48 patch layer-normed, a dense layer, a norm, the positions
   and a norm; then the same projection */
static const char *gemma_unified(const struct janas_json *cfg,
                                 const struct janas_json *pre,
                                 const struct janas_json *vc, uint32_t *width,
                                 char *what, size_t what_len)
{
    const struct janas_json *tc = janas_json_get(cfg, "text_config");
    uint32_t patch = cfg_u32(vc, "patch_size", 1, 0);
    uint32_t pool = cfg_u32(vc, "pooling_kernel_size", 1, 0);
    uint32_t side = cfg_u32(vc, "model_patch_size", 0, patch * pool);
    uint32_t dim = cfg_u32(vc, "mm_embed_dim", 1, 0);
    uint32_t npos = cfg_u32(vc, "mm_posemb_size", 1, 0);
    uint32_t out_dim = cfg_u32(tc, "hidden_size", 1, 0);
    uint32_t soft = cfg_u32(pre, "max_soft_tokens", 0,
                            cfg_u32(vc, "num_soft_tokens", 1, 0));
    if (!patch || !pool || side != patch * pool || !dim || !npos || !out_dim ||
        !soft)
        die("vision sizes out of range", NULL);
    double eps = janas_json_num(janas_json_get(vc, "rms_norm_eps"), 1e-6);
    const char *pf = "model.vision_embedder";
    char s[160], d[64];
    static const char *const LN[][2] = {{"patch_ln1", "v.u.ln1"},
                                        {"patch_ln2", "v.u.ln2"},
                                        {"pos_norm", "v.u.ln3"}};
    for (int i = 0; i < 3; i++)
        for (int b = 0; b < 2; b++) {
            snprintf(s, sizeof s, "%s.%s.%s", pf, LN[i][0],
                     b ? "bias" : "weight");
            snprintf(d, sizeof d, "%s.%s", LN[i][1], b ? "bias" : "weight");
            add(s, d, 0);
        }
    add("model.vision_embedder.patch_dense.weight", "v.u.dense", 1);
    add("model.vision_embedder.patch_dense.bias", "v.u.dense.bias", 0);
    add("model.vision_embedder.pos_embedding", "v.pos_embd", 2);
    add("model.embed_vision.embedding_projection.weight", "v.proj", 1);
    const struct janas_jw_tensor *dn = &W.t[6], *pe = &W.t[8], *pj = &W.t[9];
    if (dn->dims[1] != dim || dn->dims[0] != 3ull * side * side ||
        pe->dims[1] != 2ull * npos || pe->dims[0] != dim ||
        pj->dims[1] != out_dim || pj->dims[0] != dim)
        die("the dense layer, the positions or the projection are not of "
            "the sizes the config says",
            NULL);
    m_str("general.architecture", "gemma4_vision");
    m_str("vision.model_type", "gemma4_unified");
    m_u32("vision.unified", 1);
    m_u32("vision.embedding_length", dim);
    m_u32("vision.patch_size", patch);
    m_u32("vision.pooling_kernel_size", pool);
    m_u32("vision.position_size", npos);
    m_u32("vision.max_soft_tokens", soft);
    m_u32("vision.projection_dim", out_dim);
    m_f32("vision.attention.layer_norm_epsilon", (float)eps);
    /* the embedder's norms are torch's LayerNorm, of its default epsilon */
    m_f32("vision.patch_norm_epsilon", 1e-5f);
    const char *bi =
        janas_json_str(janas_json_get(tc, "use_bidirectional_attention"));
    m_u32("vision.bidirectional", bi && !strcmp(bi, "vision"));
    m_u32("vision.token.image", cfg_u32(cfg, "image_token_id", 1, 0));
    m_u32("vision.token.start", cfg_u32(cfg, "boi_token_id", 1, 0));
    m_u32("vision.token.end", cfg_u32(cfg, "eoi_token_id", 1, 0));
    *width = dim;
    snprintf(what, what_len,
             "no blocks, patches of %u, %u wide, positions %u -> %u", side, dim,
             npos, out_dim);
    return "gemma4_vision";
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: hf2jns_vision <checkpoint directory> "
                        "<out.vision.jns>\n");
        return 2;
    }
    const char *dir = argv[1], *out_path = argv[2];
    char p[4096];
    snprintf(p, sizeof p, "%s/config.json", dir);
    struct janas_json_doc *cd = read_json_file(p);
    const struct janas_json *cfg = janas_json_root(cd);
    const struct janas_json *vc = janas_json_get(cfg, "vision_config");
    const char *mt = janas_json_str(janas_json_get(cfg, "model_type"));
    int is_gemma =
        mt && (!strcmp(mt, "gemma4") || !strcmp(mt, "gemma4_unified"));
    if (!vc || !mt ||
        (!is_gemma && strcmp(mt, "qwen3_5") && strcmp(mt, "qwen3_vl") &&
         strcmp(mt, "qwen3_5_moe") && strcmp(mt, "qwen3_vl_moe")))
        die("not a model whose vision this converts (Qwen3.5, Qwen3-VL, "
            "Gemma 4)",
            mt);
    /* the image processor: preprocessor_config.json, or the part of
       processor_config.json */
    snprintf(p, sizeof p, "%s/preprocessor_config.json", dir);
    if (access(p, R_OK) != 0)
        snprintf(p, sizeof p, "%s/processor_config.json", dir);
    struct janas_json_doc *pd = read_json_file(p);
    const struct janas_json *pre = janas_json_root(pd);
    if (janas_json_get(pre, "image_processor"))
        pre = janas_json_get(pre, "image_processor");
    open_shards(dir);
    uint32_t hidden = 0;
    char what[160];
    const char *vt = janas_json_str(janas_json_get(vc, "model_type"));
    int unified = is_gemma && vt && !strcmp(vt, "gemma4_unified_vision");
    const char *arch =
        unified    ? gemma_unified(cfg, pre, vc, &hidden, what, sizeof what)
        : is_gemma ? gemma(cfg, pre, vc, &hidden, what, sizeof what)
                   : qwen(cfg, pre, vc, mt, &hidden, what, sizeof what);
    uint64_t file_bytes = 0;
    char err[300];
    if (janas_jw_write(&W, out_path, arch, hidden, &file_bytes, err,
                       sizeof err) != 0)
        die(err, NULL);
    printf("written %s: %d tensors, %.1f MB (%s)\n", out_path, W.n,
           file_bytes / 1e6, what);
    janas_jw_free(&W);
    for (int i = 0; i < n_shards; i++)
        janas_st_close(&shards[i]);
    janas_json_free(cd);
    janas_json_free(pd);
    return 0;
}
