/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_hc.c - Qwen3.8-Flash-Next (qwen4exp): the residual in several
 * streams (hyper-connections) and the n-gram embeddings.
 *
 * The residual is hc_n vectors of d_model (four), not one. Before each
 * sublayer - the token mixer, DeltaNet or attention, and the feed-forward -
 * a mix makes its input in place of a norm: every stream normalized with
 * weights of its own, then a gate of low rank over all of them, gate =
 * up(silu(down(streams) / hc_n)), and the input is the mean over the
 * streams of each one times sigmoid(its gate); beside it inject = W
 * streams, hc_n numbers. After the sublayer every stream gets its output
 * times 2 sigmoid(inject / hc_n), a weight centred on 1: a zero inject is
 * the plain residual. The last mix, without inject, is the output norm.
 *
 * The n-gram embeddings (the file's "ple", at layer 1, before its mix):
 * each token's context - itself and the ng_len - 1 tokens before it, an EOS
 * or the start of the sequence cutting it, what lies before reading as EOS
 * - hashed for each length from 2 up into ng_per rows of one table of
 * ng_dim numbers each (16 rows of 160, d_model, a token: 51 billion numbers
 * in the table, read a row at a time from the file). Their concatenation e
 * gives a key per stream and one value: stream c's gate is
 * sigmoid(signed sqrt(norm(key_c) . norm(stream_c) / sqrt(d_model))), and
 * the stream gets value times its gate, plus the silu of a causal conv over
 * that normalized - kernel ng_kernel, dilated by ng_len: positions p, p - 3,
 * p - 6, p - 9, the conv's inputs of earlier positions kept in a ring.
 *
 * What is computed is llama.cpp's src/models/qwen4exp.cpp (build_hc_mix,
 * build_hc_combine, build_ple, llm_graph_input_ple), the reference the
 * logits are compared against; its bf16 matrices meet their inputs rounded
 * to bf16, as there.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/aligned.h"
#include "matvec.h"
#include "vmath.h"

#include "bf16.h"
#include "model_impl.h"

/* ------------------------------------------------------------------ load */

static int meta_u32(struct janas_llm_model *m, const char *suffix, uint32_t *v)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_u32(&m->j, key, v);
}

/* Element i of an integer array of the architecture's metadata. */
static int meta_at(struct janas_llm_model *m, const char *suffix, uint64_t i,
                   uint64_t *v, uint64_t *count)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_uint_at(&m->j, key, i, v, count);
}

/* A tensor of file j of type type (0 f32, 1 f16, 30 bf16, -1 any the
   products read, -2 any float: f32, f16, bf16) and shape d0 x d1 (d1 0:
   one dimension). */
static const struct janas_jns_tensor *tensor_of(const struct janas_jns *j,
                                                const char *name, int type,
                                                uint64_t d0, uint64_t d1,
                                                char *err, size_t err_len)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(j, name);
    int ok = t && t->dims[0] == d0 && (d1 ? t->dims[1] == d1 : t->n_dims == 1);
    if (ok && type >= 0)
        ok = t->type == (uint32_t)type;
    else if (ok && type == -2)
        ok = t->type == 0 || t->type == 1 || t->type == 30;
    else if (ok)
        ok = t->type == 30 || janas_qtype_block_size((int)t->type) != 0;
    if (!ok)
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
    return ok ? t : NULL;
}

/* ... of the main model's file */
static const struct janas_jns_tensor *tensor(struct janas_llm_model *m,
                                             const char *name, int type,
                                             uint64_t d0, uint64_t d1,
                                             char *err, size_t err_len)
{
    return tensor_of(&m->j, name, type, d0, d1, err, err_len);
}

int janas_m_hc_load(struct janas_llm_model *m, char *err, size_t err_len)
{
    uint32_t dm = m->d_model;
    m->ng_layer = m->n_layer;
    if (meta_u32(m, "hyper_connection.count", &m->hc_n) ||
        meta_u32(m, "hyper_connection.low_rank", &m->hc_rank) || m->hc_n < 2 ||
        m->hc_n > 16 || m->hc_rank == 0) {
        snprintf(err, err_len, "hyper-connections: inconsistent counts");
        return -1;
    }
    uint32_t hd = m->hc_n * dm;
    if (!(m->hco_norm =
              tensor(m, "output_hc_norm.weight", 0, hd, 0, err, err_len)) ||
        !(m->hco_down = tensor(m, "output_hc_down.weight", -1, hd, m->hc_rank,
                               err, err_len)) ||
        !(m->hco_up = tensor(m, "output_hc_up.weight", -1, m->hc_rank, hd, err,
                             err_len)))
        return -1;
    /* the n-gram embeddings: one layer, as llama.cpp reads them */
    uint64_t v, count = 0;
    if (meta_at(m, "ple.layers", 0, &v, &count) != 0)
        return 0; /* none */
    if (count != 1 || v >= m->n_layer ||
        meta_u32(m, "ple.ngram_size", &m->ng_len) ||
        meta_u32(m, "ple.heads_per_ngram", &m->ng_per) ||
        meta_u32(m, "ple.conv_kernel", &m->ng_kernel) ||
        meta_u32(m, "embedding_length_per_layer_input", &m->ng_dim) ||
        m->ng_len < 2 || m->ng_len > JANAS_NG_MAX_LEN || m->ng_per == 0 ||
        (m->ng_len - 1) * m->ng_per > JANAS_NG_MAX_HEADS || m->ng_kernel == 0 ||
        m->ng_kernel > 8 || (m->ng_len - 1) * m->ng_per * m->ng_dim != dm) {
        snprintf(err, err_len, "n-gram embeddings: inconsistent sizes");
        return -1;
    }
    m->ng_layer = (uint32_t)v;
    uint32_t eos;
    if (meta_u32(m, "ple.eos_token_id", &eos)) {
        snprintf(err, err_len, "n-gram embeddings: no EOS");
        return -1;
    }
    m->ng_eos = (int32_t)eos;
    uint32_t heads = (m->ng_len - 1) * m->ng_per;
    uint64_t rows = 0;
    for (uint32_t i = 0; i < m->ng_len; i++)
        if (meta_at(m, "ple.layer_multipliers", i, &m->ng_mult[i], NULL)) {
            snprintf(err, err_len, "n-gram embeddings: multipliers");
            return -1;
        }
    for (uint32_t h = 0; h < heads; h++) {
        uint64_t off, size;
        if (meta_at(m, "ple.head_offsets", h, &off, NULL) ||
            meta_at(m, "ple.head_vocab_sizes", h, &size, NULL) || size == 0 ||
            off + size > INT32_MAX) {
            snprintf(err, err_len, "n-gram embeddings: head ranges");
            return -1;
        }
        m->ng_off[h] = (uint32_t)off;
        m->ng_vocab[h] = (uint32_t)size;
        if (off + size > rows)
            rows = off + size;
    }
    const struct janas_jns_tensor *t =
        janas_jns_tensor(&m->j, "per_layer_token_embd.weight");
    if (!t || t->dims[0] != m->ng_dim || t->dims[1] < rows ||
        janas_qtype_row_bytes((int)t->type, m->ng_dim) == 0 ||
        janas_qtype_row_bytes((int)t->type, m->ng_dim) * t->dims[1] !=
            t->bytes) {
        snprintf(err, err_len,
                 "tensor per_layer_token_embd.weight missing or "
                 "unexpected");
        return -1;
    }
    m->ng_tab = t;
    /* the table is read a row at a time where it is (model_load.c) */
    m->ple_tok = t;
    return 0;
}

int janas_m_hc_layer_load(struct janas_llm_model *m, const struct janas_jns *j,
                          uint32_t l, struct layer *ly, char *err,
                          size_t err_len)
{
    uint32_t dm = m->d_model, hd = m->hc_n * dm;
    static const char *const part[2] = {"attn", "ffn"};
    char name[64];
    /* the mixes' matrices bf16 (the trunk's) or quantized (unsloth's MTP
       block: Q4_K, Q5_0, Q6_K) */
    for (int p = 0; p < 2; p++) {
        snprintf(name, sizeof(name), "blk.%u.hc_%s_norm.weight", l, part[p]);
        if (!(ly->hc_norm[p] = tensor_of(j, name, 0, hd, 0, err, err_len)))
            return -1;
        snprintf(name, sizeof(name), "blk.%u.hc_%s_down.weight", l, part[p]);
        if (!(ly->hc_down[p] =
                  tensor_of(j, name, -1, hd, m->hc_rank, err, err_len)))
            return -1;
        snprintf(name, sizeof(name), "blk.%u.hc_%s_up.weight", l, part[p]);
        if (!(ly->hc_up[p] =
                  tensor_of(j, name, -1, m->hc_rank, hd, err, err_len)))
            return -1;
        snprintf(name, sizeof(name), "blk.%u.hc_%s_inject.weight", l, part[p]);
        if (!(ly->hc_inject[p] =
                  tensor_of(j, name, -1, hd, m->hc_n, err, err_len)))
            return -1;
    }
    /* the n-gram layer is the trunk's */
    if (j != &m->j || l != m->ng_layer)
        return 0;
    if (!ly->rec) {
        /* llama.cpp keeps the conv's history in the recurrent state */
        snprintf(err, err_len, "n-gram embeddings on an attention layer");
        return -1;
    }
    const struct janas_jns_tensor *conv;
    uint32_t ed = (m->ng_len - 1) * m->ng_per * m->ng_dim; /* = dm */
#define NG(field, suffix, type, d0, d1)                                        \
    snprintf(name, sizeof(name), "blk.%u." suffix, l);                         \
    if (!(field = tensor(m, name, type, d0, d1, err, err_len)))                \
        return -1;
    NG(ly->ng_key, "ple_key.weight", -1, ed, hd)
    NG(ly->ng_value, "ple_value.weight", -1, ed, dm)
    NG(ly->ng_nkey, "ple_norm_key.weight", 0, hd, 0)
    NG(ly->ng_nquery, "ple_norm_query.weight", 0, hd, 0)
    NG(ly->ng_nconv, "ple_norm_conv.weight", 0, hd, 0)
    NG(conv, "ple_conv1d.weight", -2, m->ng_kernel, hd)
#undef NG
    size_t nw = (size_t)m->ng_kernel * hd;
    if (!(ly->ng_conv = malloc(nw * sizeof(float)))) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    /* channel c's taps k at c * kernel + k in the file, as f32 */
    const uint16_t *h16 = data(m, conv);
    for (size_t i = 0; i < nw; i++)
        ly->ng_conv[i] = conv->type == 0   ? f32(m, conv)[i]
                         : conv->type == 1 ? janas_fp16_to_fp32(h16[i])
                                           : janas_u2f((uint32_t)h16[i] << 16);
    return 0;
}

/* The buffers of a block of blk tokens; the rings hold the conv's history
   and the tokens' contexts for a block and what comes before it. */
int janas_m_hc_alloc(struct janas_llm_model *m, char *err, size_t err_len)
{
    size_t B = m->blk, hd = (size_t)m->hc_n * m->d_model;
    /* the bf16 products' inputs and outputs where the GPU can be given
       them (janas_gpu_dense_mm: a prompt's blocks) */
    m->hc_res = malloc(B * hd * sizeof(float));
    /* the norms' streams, and at bf16 after them */
    m->hc_xn = janas_m_out_alloc(m, 2 * B * hd * sizeof(float));
    m->hc_lo = janas_m_out_alloc(m, 2 * B * m->hc_rank * sizeof(float));
    m->hc_gate = janas_m_out_alloc(m, B * hd * sizeof(float));
    m->hc_inj = janas_m_out_alloc(m, B * m->hc_n * sizeof(float));
    m->hc_q = malloc(B * hd / JANAS_QK * sizeof(*m->hc_q));
    if (!m->hc_res || !m->hc_xn || !m->hc_lo || !m->hc_gate || !m->hc_inj ||
        !m->hc_q) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    if (m->ng_layer < m->n_layer) {
        size_t dm = m->d_model, hist = (size_t)(m->ng_kernel - 1) * m->ng_len;
        m->ng_cap = (uint32_t)(B + hist + m->ng_len);
        size_t rb = janas_qtype_row_bytes((int)m->ng_tab->type, m->ng_dim);
        m->ng_tok = malloc((size_t)m->ng_cap * sizeof(int32_t));
        /* whole pages: the GPU may be given it (model_gpu.c) */
        m->ng_hist = janas_aligned_alloc(
            4096, ((size_t)m->ng_cap * hd * sizeof(float) + 4095) & ~4095ul);
        /* the embeddings, and at bf16 after them */
        m->ng_emb = janas_m_out_alloc(m, 2 * B * dm * sizeof(float));
        m->ng_kq = janas_m_out_alloc(m, B * hd * sizeof(float));
        m->ng_val = janas_m_out_alloc(m, B * dm * sizeof(float));
        m->ng_gv = malloc(B * hd * sizeof(float));
        m->ng_raw = malloc(rb + 64);
        m->ng_q = malloc(B * dm / JANAS_QK * sizeof(*m->ng_q));
        if (!m->ng_tok || !m->ng_hist || !m->ng_emb || !m->ng_kq ||
            !m->ng_val || !m->ng_gv || !m->ng_raw || !m->ng_q) {
            snprintf(err, err_len, "out of memory");
            return -1;
        }
    }
    return 0;
}

void janas_m_hc_free(struct janas_llm_model *m)
{
    free(m->hc_res);
    free(m->hc_q);
    janas_aligned_free(m->hc_xn);
    janas_aligned_free(m->hc_lo);
    janas_aligned_free(m->hc_gate);
    janas_aligned_free(m->hc_inj);
    free(m->ng_tok);
    janas_aligned_free(m->ng_hist);
    janas_aligned_free(m->ng_emb);
    janas_aligned_free(m->ng_kq);
    janas_aligned_free(m->ng_val);
    free(m->ng_gv);
    free(m->ng_raw);
    free(m->ng_q);
    for (uint32_t l = 0; m->layers && l < m->n_layer; l++)
        free(m->layers[l].ng_conv);
}

/* --------------------------------------------------------- the products */

/* y (n rows of t's rows) = scale t x (n rows of t's columns): bf16 on x at
   bf16 (xb), or a quantized matrix on x in Q8_K (q) */
static void product(struct janas_llm_model *m, const struct janas_jns_tensor *t,
                    const float *x, const float *xb, struct janas_block_q8k *q,
                    float *y, uint32_t n, float scale)
{
    uint32_t cols = (uint32_t)t->dims[0], rows = (uint32_t)t->dims[1];
    if (t->type == 30) {
        janas_m_bf16_product(m, data(m, t), xb, y, rows, cols, n, scale);
        return;
    }
    /* a bf16 mix rewritten as Q8_0 at load (model_hc_quant.c): its inputs
       at bf16 still, not quantized */
    if (t->reserved == JANAS_TENSOR_FLOAT_IN && t->type == JANAS_Q8_0 &&
        m->hc_quant[0]) {
        janas_m_q8w_product(m, data(m, t), xb, y, rows, cols, n, scale);
        return;
    }
    /* a width no multiple of 256 (the gates' rank, 320, in an MTP block's
       Q5_0): the last group padded with zeros, as model_ffn.c does */
    size_t nb = (cols + JANAS_QK - 1) / JANAS_QK;
    for (uint32_t j = 0; j < n; j++) {
        if (cols % JANAS_QK == 0) {
            janas_q8k_quantize(x + (size_t)j * cols, q + j * nb, cols);
            continue;
        }
        float pad[JANAS_QK];
        size_t whole = cols / JANAS_QK * JANAS_QK;
        if (whole)
            janas_q8k_quantize(x + (size_t)j * cols, q + j * nb,
                               (uint32_t)whole);
        memset(pad, 0, sizeof(pad));
        memcpy(pad, x + (size_t)j * cols + whole,
               (cols - whole) * sizeof(float));
        janas_q8k_quantize(pad, q + j * nb + nb - 1, JANAS_QK);
    }
    struct janas_matvec_task tk = {.type = (int)t->type,
                                   .w = data(m, t),
                                   .x = q,
                                   .y = y,
                                   .rows = rows,
                                   .cols = cols,
                                   .n_vec = n,
                                   .x_stride = nb};
    janas_m_products(m, janas_m_gpu(m, 0), &tk, 1);
    if (scale != 1.0f)
        for (size_t i = 0; i < (size_t)n * rows; i++)
            y[i] *= scale;
}

static void to_bf16(const float *x, float *xb, size_t n)
{
    janas_round_bf16_n(x, xb, n);
}

/* ---------------------------------------------------- the streams' mixes */

struct hc_job {
    struct janas_llm_model *m;
    const float *w; /* a norm's weights, hc_n x d_model */
    uint32_t first, n;
};

/* each stream normalized with its own weights, then the copy at bf16; a
   thread a stream of a token, so that a single token's four go side by
   side (their sums, in double, one after the other) */
static void norm_worker(void *arg, int tid, int n_threads)
{
    struct hc_job *t = arg;
    struct janas_llm_model *m = t->m;
    size_t dm = m->d_model, hd = m->hc_n * dm;
    float *xb = m->hc_xn + (size_t)m->blk * hd;
    uint32_t hc = m->hc_n, end = t->n * hc;
    for (uint32_t i = t->first * hc + (uint32_t)tid; i < end;
         i += (uint32_t)n_threads) {
        size_t o = (size_t)i * dm; /* token i / hc, stream i % hc */
        rms_norm(m->hc_xn + o, m->hc_res + o, t->w + (i % hc) * dm,
                 (uint32_t)dm, m->eps);
        to_bf16(m->hc_xn + o, xb + o, dm);
    }
}

#if defined(__x86_64__)
/* the same eight at a time: products and sums in the same order */
__attribute__((target("avx2,fma"))) static size_t
mean_avx2(const float *xn, const float *g, float *out, size_t dm, size_t len,
          uint32_t hc, float inv)
{
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 s = zero;
        for (uint32_t c = 0; c < hc; c++) {
            __m256 e = janas_exp8(
                _mm256_sub_ps(zero, _mm256_loadu_ps(g + (size_t)c * dm + i)));
            __m256 p = _mm256_mul_ps(_mm256_loadu_ps(xn + (size_t)c * dm + i),
                                     _mm256_div_ps(one, _mm256_add_ps(one, e)));
            s = c ? _mm256_add_ps(s, p) : p;
        }
        _mm256_storeu_ps(out + i, _mm256_mul_ps(s, _mm256_set1_ps(inv)));
    }
    return i;
}
#endif

/* elements a .. a + len of token j's mean */
static void mean_part(struct janas_llm_model *m, uint32_t j, size_t a,
                      size_t len)
{
    size_t dm = m->d_model, hd = m->hc_n * dm;
    float inv = 1.0f / (float)m->hc_n;
    const float *xn = m->hc_xn + j * hd + a, *g = m->hc_gate + j * hd + a;
    float *out = m->xn + j * dm + a;
    size_t i = 0;
#if defined(__x86_64__)
    if (m->avx2)
        i = mean_avx2(xn, g, out, dm, len, m->hc_n, inv);
#endif
    for (; i < len; i++) {
        float s = xn[i] * sigmoid(g[i]);
        for (uint32_t c = 1; c < m->hc_n; c++)
            s += xn[c * dm + i] * sigmoid(g[c * dm + i]);
        out[i] = s * inv;
    }
}

/* the input: the mean of the streams, each times sigmoid(its gate), a
   thread a token */
static void mean_worker(void *arg, int tid, int n_threads)
{
    struct hc_job *t = arg;
    struct janas_llm_model *m = t->m;
    size_t dm = m->d_model;
    for (uint32_t j = t->first + (uint32_t)tid; j < t->n;
         j += (uint32_t)n_threads) {
        mean_part(m, j, 0, dm);
        janas_q8k_quantize(m->xn + j * dm, m->xq + j * (dm / JANAS_QK),
                           (uint32_t)dm);
    }
}

/* a single token's mean, a thread a piece of MEAN_PIECE elements (each
   element the same operations as on its own); quantized by the caller */
#define MEAN_PIECE 256
static void mean_one_worker(void *arg, int tid, int n_threads)
{
    struct hc_job *t = arg;
    struct janas_llm_model *m = t->m;
    size_t dm = m->d_model;
    for (size_t a = (size_t)tid * MEAN_PIECE; a < dm;
         a += (size_t)n_threads * MEAN_PIECE)
        mean_part(m, t->first, a, dm - a < MEAN_PIECE ? dm - a : MEAN_PIECE);
}

static void run(struct janas_llm_model *m, void (*fn)(void *, int, int),
                struct hc_job *t)
{
    if (t->n - t->first < 2)
        fn(t, 0, 1);
    else
        janas_pool_run(m->compute, fn, t);
}

/*
 * A mix of the block's streams, tokens first .. n - 1: the sublayer's input
 * in xn (and xq), inject (NULL: none, the last mix) in hc_inj.
 */
static void
hc_mix(struct janas_llm_model *m, const struct janas_jns_tensor *norm,
       const struct janas_jns_tensor *down, const struct janas_jns_tensor *up,
       const struct janas_jns_tensor *inject, uint32_t first, uint32_t n)
{
    double t0 = now();
    size_t hd = (size_t)m->hc_n * m->d_model, r = m->hc_rank, k = n - first;
    struct hc_job t = {.m = m, .w = f32(m, norm), .first = first, .n = n};
    if ((size_t)k * m->hc_n < 2)
        norm_worker(&t, 0, 1);
    else
        janas_pool_run(m->compute, norm_worker, &t);
    /* each product on the inputs at bf16 for a bf16 matrix (the trunk's),
       in Q8_K for a quantized one (an MTP block's), as llama.cpp gives
       them */
    const float *x = m->hc_xn + first * hd;
    const float *xb = m->hc_xn + (size_t)m->blk * hd + first * hd;
    float *lo = m->hc_lo + first * r, *lob = m->hc_lo + (size_t)m->blk * r;
    product(m, down, x, xb, m->hc_q, lo, (uint32_t)k, 1.0f / (float)m->hc_n);
    for (size_t i = 0; i < k * r; i++) {
        lo[i] = lo[i] / (1.0f + janas_expf(-lo[i]));
        lob[i] = janas_m_round_bf16(lo[i]);
    }
    product(m, up, lo, lob, m->hc_q, m->hc_gate + first * hd, (uint32_t)k,
            1.0f);
    if (k == 1) {
        size_t dm = m->d_model;
        janas_pool_run(m->compute, mean_one_worker, &t);
        janas_q8k_quantize(m->xn + (size_t)first * dm,
                           m->xq + first * (dm / JANAS_QK), (uint32_t)dm);
    } else
        run(m, mean_worker, &t);
    if (inject)
        product(m, inject, x, xb, m->hc_q, m->hc_inj + first * m->hc_n,
                (uint32_t)k, 1.0f);
    m->phase[JANAS_PH_MIX] += now() - t0;
}

/* every stream += the sublayer's output (in x) times 2 sigmoid(inject /
   hc_n) */
static void combine_worker(void *arg, int tid, int n_threads)
{
    struct hc_job *t = arg;
    struct janas_llm_model *m = t->m;
    size_t dm = m->d_model, hd = m->hc_n * dm;
    float inv = 1.0f / (float)m->hc_n;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        const float *b = m->x + j * dm;
        for (uint32_t c = 0; c < m->hc_n; c++) {
            float w = 2.0f * sigmoid(m->hc_inj[j * m->hc_n + c] * inv);
            float *s = m->hc_res + j * hd + c * dm;
            for (size_t i = 0; i < dm; i++)
                s[i] += b[i] * w;
        }
    }
}

static void hc_combine(struct janas_llm_model *m, uint32_t n)
{
    struct hc_job t = {.m = m, .n = n};
    run(m, combine_worker, &t);
}

void janas_m_hc_head(struct janas_llm_model *m, uint32_t first, uint32_t n)
{
    hc_mix(m, m->hco_norm, m->hco_down, m->hco_up, NULL, first, n);
}

/* ------------------------------------------------------- the MTP block */

/*
 * As llama.cpp's graph_mtp (PR #28243): each stream of the trunk's residual
 * normalized with weights of its own, beside the token's normalized
 * embedding - "per stream, not pooled: pooling before the projection
 * discards the hyper-connection residual" - projected back to d_model: the
 * block's own streams; then its layer as the trunk's, its attention dense
 * (QSA left for later there too, "a numerical superset below the
 * 2048-token budget"); its streams kept for the next draft, then its final
 * mix.
 */
int janas_m_hc_mtp(struct janas_llm_model *m, const float *hidden, uint32_t n,
                   uint32_t pos0, uint32_t first)
{
    size_t dm = m->d_model, hd = (size_t)m->hc_n * dm, nbc = 2 * dm / JANAS_QK;
    const struct layer *ly = &m->mtp_layer;
    const float *en = f32(m, m->enorm), *hn = f32(m, m->hnorm);
    float *c = m->cat; /* a row of [embedding; stream] */
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(c, m->x + (size_t)j * dm, en, (uint32_t)dm, m->eps);
        for (uint32_t s = 0; s < m->hc_n; s++) {
            rms_norm(c + dm, hidden + (size_t)j * hd + s * dm, hn + s * dm,
                     (uint32_t)dm, m->eps);
            janas_q8k_quantize(c, m->catq + ((size_t)j * m->hc_n + s) * nbc,
                               (uint32_t)(2 * dm));
        }
    }
    /* stream s of token j at row j hc_n + s: the residual's layout */
    struct janas_matvec_task tp = {.type = (int)m->eh_proj->type,
                                   .w = data(m, m->eh_proj),
                                   .x = m->catq,
                                   .y = m->hc_res,
                                   .rows = dm,
                                   .cols = 2 * dm,
                                   .n_vec = (size_t)n * m->hc_n};
    janas_m_products(m, janas_m_gpu(m, 0), &tp, 1);
    hc_mix(m, ly->hc_norm[0], ly->hc_down[0], ly->hc_up[0], ly->hc_inject[0], 0,
           n);
    memset(m->x, 0, (size_t)n * dm * sizeof(float));
    janas_m_attn_layer(m, ly, n, pos0);
    hc_combine(m, n);
    hc_mix(m, ly->hc_norm[1], ly->hc_down[1], ly->hc_up[1], ly->hc_inject[1], 0,
           n);
    memset(m->x, 0, (size_t)n * dm * sizeof(float));
    if (janas_m_moe_block(m, ly, m->mtp_cache, m->mtp_jl, m->mtp_cl, n,
                          MOE_ALL) != 0)
        return -1;
    hc_combine(m, n);
    /* the wide streams, which the next draft step reads (h_nextn) */
    memcpy(m->mtp_hid, m->hc_res, (size_t)n * hd * sizeof(float));
    hc_mix(m, m->mtp_hco_norm, m->mtp_hco_down, m->mtp_hco_up, NULL, first, n);
    return 0;
}

/* ---------------------------------------------------- the n-gram layer */

/* token at position p (ng_hi and what came before it, in the ring) */
static int32_t ring_tok(const struct janas_llm_model *m, uint32_t p)
{
    return m->ng_tok[p % m->ng_cap];
}

/* the rows of the table the token at position p reads: heads in the order
   of the lengths, 2 first */
static void ng_rows(const struct janas_llm_model *m, uint32_t p, uint32_t *row)
{
    int64_t ctx[JANAS_NG_MAX_LEN];
    ctx[0] = ring_tok(m, p);
    int cut = 0;
    for (uint32_t s = 1; s < m->ng_len; s++) {
        /* the token s back; none before the start: EOS from there on, as
           an EOS among them - not the token's own */
        int32_t t = !cut && p >= s ? ring_tok(m, p - s) : -1;
        cut = cut || t < 0 || t == m->ng_eos;
        ctx[s] = cut ? m->ng_eos : t;
    }
    for (uint32_t len = 2; len <= m->ng_len; len++) {
        uint64_t mixed = (uint64_t)ctx[0] * m->ng_mult[0];
        for (uint32_t j = 1; j < len; j++)
            mixed ^= (uint64_t)ctx[j] * m->ng_mult[j];
        for (uint32_t g = 0; g < m->ng_per; g++) {
            uint32_t h = (len - 2) * m->ng_per + g;
            row[h] = (uint32_t)(mixed % m->ng_vocab[h]) + m->ng_off[h];
        }
    }
}

/* a token's embeddings: its rows of the table, read from the file */
static int ng_embed(struct janas_llm_model *m, uint32_t p, float *e)
{
    uint32_t row[JANAS_NG_MAX_HEADS], heads = (m->ng_len - 1) * m->ng_per;
    size_t rb = janas_qtype_row_bytes((int)m->ng_tab->type, m->ng_dim);
    ng_rows(m, p, row);
    for (uint32_t h = 0; h < heads; h++) {
        if (m->ple_ram) { /* the table in memory (janas_llm_options) */
            janas_dequantize((int)m->ng_tab->type,
                             (const uint8_t *)data(m, m->ng_tab) +
                                 (uint64_t)row[h] * rb,
                             e + (size_t)h * m->ng_dim, m->ng_dim);
            continue;
        }
        off_t at = (off_t)(m->ng_tab->offset + (uint64_t)row[h] * rb);
        for (size_t done = 0; done < rb;) {
            ssize_t r =
                pread(m->j.fd, m->ng_raw + done, rb - done, at + (off_t)done);
            if (r <= 0)
                return -1;
            done += (size_t)r;
        }
        janas_dequantize((int)m->ng_tab->type, m->ng_raw,
                         e + (size_t)h * m->ng_dim, m->ng_dim);
    }
    return 0;
}

struct ng_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    uint32_t n, pos0;
};

/* the streams' gates, the gated values and the conv's inputs */
static void gate_worker(void *arg, int tid, int n_threads)
{
    struct ng_job *a = arg;
    struct janas_llm_model *m = a->m;
    const struct layer *ly = a->ly;
    size_t dm = m->d_model, hd = m->hc_n * dm;
    const float *nk = f32(m, ly->ng_nkey), *nq = f32(m, ly->ng_nquery),
                *nc = f32(m, ly->ng_nconv);
    float isd = 1.0f / sqrtf((float)dm);
    for (uint32_t j = (uint32_t)tid; j < a->n; j += (uint32_t)n_threads) {
        float *key = m->ng_kq + j * hd, *gv = m->ng_gv + j * hd;
        float *in = m->ng_hist + (size_t)((a->pos0 + j) % m->ng_cap) * hd;
        const float *val = m->ng_val + j * dm;
        for (uint32_t c = 0; c < m->hc_n; c++) {
            float *kc = key + c * dm, *g = gv + c * dm;
            rms_norm(kc, kc, nk + c * dm, (uint32_t)dm, m->eps);
            /* the stream as a query, normalized, into gv for a moment */
            rms_norm(g, m->hc_res + j * hd + c * dm, nq + c * dm, (uint32_t)dm,
                     m->eps);
            double dot = 0.0;
            for (size_t i = 0; i < dm; i++)
                dot += (double)kc[i] * g[i];
            float s = (float)dot * isd;
            float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            float gt = sigmoid(s > 0 ? mag : s < 0 ? -mag : 0.0f);
            for (size_t i = 0; i < dm; i++)
                g[i] = val[i] * gt;
            rms_norm(in + c * dm, g, nc + c * dm, (uint32_t)dm, m->eps);
        }
    }
}

/* the conv over positions, dilated by ng_len, its silu, and the streams'
   new values: stream += gated value + silu(conv) */
static void conv_worker(void *arg, int tid, int n_threads)
{
    struct ng_job *a = arg;
    struct janas_llm_model *m = a->m;
    const float *w = a->ly->ng_conv;
    size_t hd = (size_t)m->hc_n * m->d_model, K = m->ng_kernel;
    for (uint32_t j = (uint32_t)tid; j < a->n; j += (uint32_t)n_threads) {
        uint32_t p = a->pos0 + j;
        const float *tap[8];
        for (size_t k = 0; k < K; k++) {
            /* tap k reads (K - 1 - k) * ng_len positions back */
            uint32_t back = (uint32_t)(K - 1 - k) * m->ng_len;
            tap[k] = p >= back
                         ? m->ng_hist + (size_t)((p - back) % m->ng_cap) * hd
                         : NULL; /* before the start: zeros */
        }
        float *s = m->hc_res + j * hd;
        const float *gv = m->ng_gv + j * hd;
        for (size_t c = 0; c < hd; c++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float term = tap[k] ? w[c * K + k] * tap[k][c] : 0.0f;
                acc = k ? acc + term : term;
            }
            float silu = acc / (1.0f + janas_expf(-acc));
            s[c] += gv[c] + silu;
        }
    }
}

int janas_m_hc_ng_begin(struct janas_llm_model *m, const int32_t *tokens,
                        uint32_t n, uint32_t pos0)
{
    if (m->ng_layer >= m->n_layer)
        return 0;
    uint32_t hist = (m->ng_kernel - 1) * m->ng_len;
    uint32_t need = pos0 < hist ? 0 : pos0 - hist; /* oldest position read */
    if (pos0 == 0)
        m->ng_hi = 0;
    /* the history must be there: not past what the ring has seen, nor
       older than it keeps */
    if (pos0 > m->ng_hi ||
        (pos0 && m->ng_hi > m->ng_cap && need < m->ng_hi - m->ng_cap))
        return -1;
    for (uint32_t j = 0; j < n; j++)
        m->ng_tok[(pos0 + j) % m->ng_cap] = tokens[j];
    for (uint32_t j = 0; j < n; j++)
        if (ng_embed(m, pos0 + j, m->ng_emb + (size_t)j * m->d_model) != 0)
            return -1;
    return 0;
}

static int ng_layer(struct janas_llm_model *m, const struct layer *ly,
                    const int32_t *tokens, uint32_t n, uint32_t pos0)
{
    if (janas_m_hc_ng_begin(m, tokens, n, pos0) != 0)
        return -1;
    size_t dm = m->d_model, hd = (size_t)m->hc_n * dm;
    float *eb = m->ng_emb + (size_t)m->blk * dm;
    to_bf16(m->ng_emb, eb, (size_t)n * dm);
    product(m, ly->ng_key, m->ng_emb, eb, m->ng_q, m->ng_kq, n, 1.0f);
    product(m, ly->ng_value, m->ng_emb, eb, m->ng_q, m->ng_val, n, 1.0f);
    struct ng_job a = {.m = m, .ly = ly, .n = n, .pos0 = pos0};
    if (n < 2)
        gate_worker(&a, 0, 1);
    else
        janas_pool_run(m->compute, gate_worker, &a);
    if (n < 2)
        conv_worker(&a, 0, 1);
    else
        janas_pool_run(m->compute, conv_worker, &a);
    m->ng_hi = pos0 + n;
    (void)hd;
    return 0;
}

/*
 * The context moved along (janas_llm_model_shift): positions keep .. keep +
 * drop - 1 forgotten, those after them drop places down. The conv's
 * history is a summary of the last positions, as llama.cpp keeps it in its
 * recurrent state: the last ones, whatever their places, are moved to their
 * new places - from a copy, since in the ring a place may still hold an
 * entry to be moved - and the tokens with them.
 */
int janas_m_hc_shift(struct janas_llm_model *m, uint32_t keep, uint32_t drop,
                     uint32_t n_kv)
{
    if (m->ng_layer >= m->n_layer || n_kv == 0)
        return 0;
    uint32_t hist = (m->ng_kernel - 1) * m->ng_len, n = n_kv - drop;
    uint32_t from = n > hist ? n - hist : 0;
    size_t hd = (size_t)m->hc_n * m->d_model;
    /* the ring must reach n_kv (places past it, drafts taken back, are
       computed again) and still hold every place read */
    if (m->ng_hi < n_kv)
        return -1;
    float *tmp = malloc((size_t)(n - from) * hd * sizeof(float));
    int32_t *tok = malloc((size_t)(n - from) * sizeof(int32_t) + 1);
    if (!tmp || !tok) {
        free(tmp);
        free(tok);
        return -1;
    }
    for (uint32_t q = from; q < n; q++) {
        uint32_t old = q < keep ? q : q + drop; /* where it was */
        if (m->ng_hi > m->ng_cap && old < m->ng_hi - m->ng_cap) {
            free(tmp);
            free(tok);
            return -1;
        }
        memcpy(tmp + (size_t)(q - from) * hd,
               m->ng_hist + (size_t)(old % m->ng_cap) * hd, hd * sizeof(float));
        tok[q - from] = m->ng_tok[old % m->ng_cap];
    }
    for (uint32_t q = from; q < n; q++) {
        memcpy(m->ng_hist + (size_t)(q % m->ng_cap) * hd,
               tmp + (size_t)(q - from) * hd, hd * sizeof(float));
        m->ng_tok[q % m->ng_cap] = tok[q - from];
    }
    free(tmp);
    free(tok);
    m->ng_hi = n;
    return 0;
}

/* ------------------------------------------------------------ the layers */

int janas_m_hc_layers(struct janas_llm_model *m, const int32_t *tokens,
                      uint32_t n, uint32_t pos0)
{
    size_t dm = m->d_model, hd = (size_t)m->hc_n * dm;
    /* every stream starts as the embedding */
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t c = 0; c < m->hc_n; c++)
            memcpy(m->hc_res + j * hd + c * dm, m->x + j * dm,
                   dm * sizeof(float));
    for (uint32_t l = 0; l < m->n_layer; l++) {
        m->ana_layer = l;
        const struct layer *ly = &m->layers[l];
        if (l == m->ng_layer && ng_layer(m, ly, tokens, n, pos0) != 0)
            return -1;
        hc_mix(m, ly->hc_norm[0], ly->hc_down[0], ly->hc_up[0],
               ly->hc_inject[0], 0, n);
        trace_row(m, "hc_attn_mixed", m->xn + (size_t)(n - 1) * dm,
                  (uint32_t)dm);
        /* the sublayers add their outputs to x: zero, the output alone */
        memset(m->x, 0, (size_t)n * dm * sizeof(float));
        if (ly->rec)
            janas_m_rec_layer(m, ly, n);
        else
            janas_m_attn_layer(m, ly, n, pos0);
        hc_combine(m, n);
        hc_mix(m, ly->hc_norm[1], ly->hc_down[1], ly->hc_up[1],
               ly->hc_inject[1], 0, n);
        trace_row(m, "hc_ffn_mixed", m->xn + (size_t)(n - 1) * dm,
                  (uint32_t)dm);
        memset(m->x, 0, (size_t)n * dm * sizeof(float));
        if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                              MOE_ALL) != 0)
            return -1;
        trace_row(m, "ffn_out", m->x + (size_t)(n - 1) * dm, (uint32_t)dm);
        hc_combine(m, n);
        if (m->trace) {
            /* JANAS_TRACE=1: the last stream after each layer, last token,
               as llama.cpp's l_last row (job/tests/c/layer_trace.c) */
            const float *v = m->hc_res + (size_t)(n - 1) * hd + hd - dm;
            double nn = 0;
            for (size_t i = 0; i < dm; i++)
                nn += (double)v[i] * v[i];
            fprintf(stderr, "l_out-%-3u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
                    l, sqrt(nn), v[0], v[1], v[2], v[3]);
        }
    }
    return 0;
}
