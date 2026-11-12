/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_neurons.c - the experts' down matrix by neuron (JANAS_Q6_K_PN,
 * tools/jns_neurons.c): each neuron of a token read at the depth its
 * contribution earns - Ana-Net's Alpha (job/docs/progetto_ana.md; the name
 * is provisional).
 *
 * A token's neurons, all the pairs of the pass together (its routed experts
 * and the shared one), are ranked by w |h|: the expert's weight (the
 * router's, or the shared expert's gate) times the neuron's activation,
 * what the neuron gives the output. The first neuron_frac[0] of them read
 * the three planes of their row of down, the next neuron_frac[1] two, the
 * next neuron_frac[2] the base alone, the rest nothing. The shared expert,
 * resident and by rows, is ranked with the others but computed whole.
 * JANAS_NEURON_LEVELS=f6,f4,f2 sets the fractions; without it every neuron
 * reads all three planes. The planes the cache holds (JANAS_EXPERT_BITS)
 * bound every level.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"
#include "model_impl.h"
#include "quant.h"

/* Neurons are summed in chunks of JANAS_PN_CHUNK, each in integers from
   zero, the chunks then added in order: the same sums, bit for bit, whether
   a token comes alone (a chunk a thread) or with others (a block of outputs
   a thread). */
#define NEURON_CHUNK JANAS_PN_CHUNK

int janas_m_neurons_init(struct janas_llm_model *m, size_t pairs)
{
    m->neuron_down = 0;
    for (uint32_t l = 0; l < m->j.h.n_layer; l++)
        if (m->j.layers[l].m[JANAS_JNS_DOWN].type == JANAS_Q6_K_PN)
            m->neuron_down = 1;
    if (!m->neuron_down)
        return 0;
    /* a token alone: k experts' chunks of neurons, each its sum of the
       outputs; and which pair rows need no h in eight bits */
    for (uint32_t l = 0; l < m->j.h.n_layer; l++) {
        const struct janas_jns_matrix *d = &m->j.layers[l].m[JANAS_JNS_DOWN];
        size_t len = (size_t)m->k *
                     ((d->rows + NEURON_CHUNK - 1) / NEURON_CHUNK) * d->cols;
        if (d->type == JANAS_Q6_K_PN && len > m->npart_len)
            m->npart_len = len;
    }
    m->npart = malloc(m->npart_len * sizeof(float));
    m->nskipq = calloc(pairs, 1);
    if (!m->npart || !m->nskipq)
        return -1;
    const char *e = getenv("JANAS_NEURON_LEVELS");
    float *f = m->neuron_frac;
    m->neuron_rank = e && sscanf(e, "%f,%f,%f", &f[0], &f[1], &f[2]) == 3 &&
                     f[0] >= 0 && f[1] >= 0 && f[2] >= 0 &&
                     f[0] + f[1] + f[2] <= 1.0001f && f[0] + f[1] + f[2] > 0.0f;
    if (e && !m->neuron_rank)
        fprintf(stderr,
                "JANAS_NEURON_LEVELS=%s: three fractions, f6,f4,f2, "
                "summing to at most 1; every neuron whole\n",
                e);
    if (!m->neuron_rank)
        return 0;
    m->nlevel = malloc(pairs * m->ff_buf);
    m->nscore = malloc(pairs * m->ff_buf * sizeof(float));
    m->ntok_first = malloc((m->blk + 1) * sizeof(uint32_t));
    m->ntok_fill = malloc(m->blk * sizeof(uint32_t));
    m->ntok_list = malloc(pairs * sizeof(uint32_t));
    if (!m->nlevel || !m->nscore || !m->ntok_first || !m->ntok_fill ||
        !m->ntok_list)
        return -1;
    fprintf(stderr,
            "neurons: down at 6, 4, 2 bits for %.0f%%, %.0f%%, %.0f%% of a "
            "token's neurons, the rest off\n",
            100.0 * f[0], 100.0 * f[1], 100.0 * f[2]);
    return 0;
}

void janas_m_neurons_free(struct janas_llm_model *m)
{
    free(m->npart);
    free(m->nskipq);
    m->npart = NULL;
    m->nskipq = NULL;
    m->npart_len = 0;
    free(m->nlevel);
    free(m->nscore);
    free(m->ntok_first);
    free(m->ntok_fill);
    free(m->ntok_list);
    m->nlevel = NULL;
    m->nscore = NULL;
    m->ntok_first = m->ntok_fill = m->ntok_list = NULL;
}

/* Rearranges a[0 .. n) so that its k largest come first (1 <= k <= n) and
   returns the least of them: Hoare's selection. */
static float top_k(float *a, long n, long k)
{
    long lo = 0, hi = n - 1, want = k - 1;
    while (lo < hi) {
        float p = a[lo + (hi - lo) / 2];
        long i = lo, j = hi;
        do {
            while (a[i] > p)
                i++;
            while (a[j] < p)
                j--;
            if (i <= j) {
                float t = a[i];
                a[i++] = a[j];
                a[j--] = t;
            }
        } while (i <= j);
        if (want <= j)
            hi = j;
        else if (want >= i)
            lo = i;
        else
            return a[want];
    }
    return a[want];
}

struct level_job {
    struct janas_llm_model *m;
};

/* Token t's pair rows, ntok_list[ntok_first[t] .. [t + 1]): their neurons'
   scores gathered in the token's share of nscore, the thresholds of the
   three levels selected among them, the levels set. */
static void level_token(struct janas_llm_model *m, uint32_t t)
{
    uint32_t stride = m->ff_buf, a = m->ntok_first[t], b = m->ntok_first[t + 1];
    float *s = m->nscore + (size_t)a * stride;
    long n = 0;
    for (uint32_t i = a; i < b; i++) {
        size_t g = m->ntok_list[i];
        float w = fabsf(m->pair_w[m->order[g]]);
        const float *h = m->h + g * stride;
        for (uint32_t j = 0; j < m->hlen[g]; j++)
            s[n++] = w * fabsf(h[j]);
    }
    if (!n)
        return;
    /* the thresholds from one score in step, some 512 of them, where a
       token has more than 1024 neurons: the selection runs on a tenth of
       them or less, and the quotas hold to a couple of hundredths */
    long step = n > 1024 ? n / 512 : 1;
    if (step > 1) {
        long ns = 0;
        for (long i = 0; i < n; i += step)
            s[ns++] = s[i];
        n = ns;
    }
    /* from the top: the k-th largest score of each level's end, each search
       among the ones the last left first */
    long k[3];
    float th[3];
    float c = 0.0f;
    for (int q = 0; q < 3; q++) {
        c += m->neuron_frac[q];
        k[q] = (long)(c * (float)n + 0.5f);
        if (k[q] > n)
            k[q] = n;
    }
    for (int q = 2; q >= 0; q--) {
        long kq = k[q], within = q == 2 ? n : k[q + 1];
        th[q] = kq > 0 ? top_k(s, within, kq) : INFINITY;
    }
    int top = m->exp_level;
    for (uint32_t i = a; i < b; i++) {
        size_t g = m->ntok_list[i];
        float w = fabsf(m->pair_w[m->order[g]]);
        const float *h = m->h + g * stride;
        uint8_t *lv = m->nlevel + g * stride;
        for (uint32_t j = 0; j < m->hlen[g]; j++) {
            float v = w * fabsf(h[j]);
            int l = v <= 0.0f    ? 0
                    : v >= th[0] ? 3
                    : v >= th[1] ? 2
                    : v >= th[2] ? 1
                                 : 0;
            lv[j] = (uint8_t)(l < top ? l : top);
        }
    }
}

static void level_worker(void *arg, int tid, int n_threads)
{
    struct level_job *L = arg;
    for (uint32_t t = (uint32_t)tid; t < L->m->blk; t += (uint32_t)n_threads)
        if (L->m->ntok_first[t + 1] > L->m->ntok_first[t])
            level_token(L->m, t);
}

void janas_m_neuron_levels(struct janas_llm_model *m, uint32_t ng)
{
    uint32_t B = m->blk, *first = m->ntok_first, *fill = m->ntok_fill;
    memset(first, 0, (B + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < ng; i++)
        first[m->pair_tok[m->order[m->glist[i]]] + 1]++;
    for (uint32_t t = 0; t < B; t++)
        first[t + 1] += first[t];
    memcpy(fill, first, B * sizeof(uint32_t));
    for (uint32_t i = 0; i < ng; i++) {
        uint32_t g = m->glist[i];
        m->ntok_list[fill[m->pair_tok[m->order[g]]]++] = g;
    }
    struct level_job L = {m};
    if (first[1] == ng) /* a single token: no threads to wake */
        level_token(m, 0);
    else
        janas_pool_run(m->compute, level_worker, &L);
}

/* Neurons [j0, j1) of pair row g, those it reads (a level above 0, h not
   zero) gathered by level, highest first - so that the kernel's branches
   follow one another - and added to y in integers (janas_q6kpn_chunk),
   nbk blocks of 256 outputs from block ob on. */
static void chunk_rows(const struct janas_llm_model *m,
                       const struct janas_neuron_run *r, uint32_t g,
                       uint32_t j0, uint32_t j1, uint32_t ob, uint32_t nbk,
                       float *y)
{
    uint32_t stride = m->ff_buf, n = 0;
    const float *h = m->h + (size_t)g * stride;
    const uint8_t *lv = m->neuron_rank ? m->nlevel + (size_t)g * stride : NULL;
    uint32_t js[JANAS_PN_CHUNK];
    uint8_t ls[JANAS_PN_CHUNK];
    float hs[JANAS_PN_CHUNK];
    for (int l = 3; l >= 1; l--)
        for (uint32_t j = j0; j < j1; j++)
            if ((lv ? lv[j] : m->exp_level) == l && h[j] != 0.0f) {
                js[n] = j;
                ls[n] = (uint8_t)l;
                hs[n++] = h[j];
            }
    for (uint32_t k = 0; n && k < nbk; k++)
        janas_q6kpn_chunk(r->base, r->p1, r->p0, r->rows, js, ls, hs, n, ob + k,
                          y + (size_t)k * JANAS_QK);
}

/* A token alone: an item a run and a chunk of its neurons, each neuron's
   row read whole - its planes contiguous - into the chunk's own sum; then
   an item a run and a block of outputs, the chunks' sums added in order,
   times the channels' scales. */
struct alone_job {
    struct janas_llm_model *m;
    const struct janas_neuron_run *runs;
    uint32_t n_runs, chunks;
};

static void tok_worker(void *arg, int tid, int n_threads)
{
    struct alone_job *J = arg;
    for (uint32_t i = (uint32_t)tid; i < J->n_runs * J->chunks;
         i += (uint32_t)n_threads) {
        const struct janas_neuron_run *r = &J->runs[i / J->chunks];
        uint32_t j0 = i % J->chunks * NEURON_CHUNK;
        uint32_t j1 = j0 + NEURON_CHUNK < r->rows ? j0 + NEURON_CHUNK : r->rows;
        float *y = J->m->npart + (size_t)i * r->cols;
        memset(y, 0, r->cols * sizeof(float));
        chunk_rows(J->m, r, r->a, j0, j1, 0, r->cols / JANAS_QK, y);
    }
}

static void sum_worker(void *arg, int tid, int n_threads)
{
    struct alone_job *J = arg;
    struct janas_llm_model *m = J->m;
    uint32_t nb = J->runs[0].cols / JANAS_QK;
    for (uint32_t i = (uint32_t)tid; i < J->n_runs * nb;
         i += (uint32_t)n_threads) {
        uint32_t ri = i / nb, ob = i % nb;
        const struct janas_neuron_run *r = &J->runs[ri];
        float *y = m->dout + (size_t)r->a * m->d_model + (size_t)ob * JANAS_QK;
        memset(y, 0, JANAS_QK * sizeof(float));
        for (uint32_t c = 0; c < J->chunks; c++) {
            const float *p = m->npart + ((size_t)ri * J->chunks + c) * r->cols +
                             (size_t)ob * JANAS_QK;
            for (int v = 0; v < JANAS_QK; v++)
                y[v] += p[v];
        }
        const float *sc = r->scale + (size_t)ob * JANAS_QK;
        for (int v = 0; v < JANAS_QK; v++)
            y[v] *= sc[v];
    }
}

/* Tokens together: an item a run and a block of outputs, each pair's
   chunks summed apart and added in order, as for a token alone. */
struct down_job {
    struct janas_llm_model *m;
    const struct janas_neuron_run *runs;
    uint32_t n_runs, nb;
};

static void down_worker(void *arg, int tid, int n_threads)
{
    struct down_job *J = arg;
    struct janas_llm_model *m = J->m;
    for (uint32_t i = (uint32_t)tid; i < J->n_runs * J->nb;
         i += (uint32_t)n_threads) {
        const struct janas_neuron_run *r = &J->runs[i / J->nb];
        uint32_t ob = i % J->nb;
        const float *sc = r->scale + (size_t)ob * JANAS_QK;
        for (uint32_t g = r->a; g < r->b; g++) {
            float *y = m->dout + (size_t)g * m->d_model + (size_t)ob * JANAS_QK;
            float part[JANAS_QK];
            memset(y, 0, JANAS_QK * sizeof(float));
            for (uint32_t j0 = 0; j0 < r->rows; j0 += NEURON_CHUNK) {
                uint32_t j1 =
                    j0 + NEURON_CHUNK < r->rows ? j0 + NEURON_CHUNK : r->rows;
                memset(part, 0, sizeof(part));
                chunk_rows(m, r, g, j0, j1, ob, 1, part);
                for (int v = 0; v < JANAS_QK; v++)
                    y[v] += part[v];
            }
            for (int v = 0; v < JANAS_QK; v++)
                y[v] *= sc[v];
        }
    }
}

void janas_m_neuron_down(struct janas_llm_model *m,
                         const struct janas_neuron_run *runs, uint32_t n)
{
    if (!n)
        return;
    uint32_t chunks = (runs[0].rows + NEURON_CHUNK - 1) / NEURON_CHUNK;
    int alone = (size_t)n * chunks * runs[0].cols <= m->npart_len;
    for (uint32_t r = 0; alone && r < n; r++)
        alone = runs[r].b - runs[r].a == 1;
    if (alone) {
        struct alone_job J = {m, runs, n, chunks};
        janas_pool_run(m->compute, tok_worker, &J);
        janas_pool_run(m->compute, sum_worker, &J);
        return;
    }
    struct down_job J = {m, runs, n, runs[0].cols / JANAS_QK};
    janas_pool_run(m->compute, down_worker, &J);
}
