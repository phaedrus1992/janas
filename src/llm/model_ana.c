/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_ana.c - the Ana-Net probe (job/docs/progetto_ana.md; the name is
 * provisional). A fraction of the feed-forward neurons is switched off
 * before the down product, chosen by a score, to measure what each choice
 * costs the answers against a reference. A probe: nothing is saved (the
 * down product still reads every column), and only the CPU's path has it
 * (JANAS_GPU=0).
 *
 * JANAS_ANA_MODE, a pair's neurons (a token through one expert, or through
 * the dense feed-forward) ranked by themselves:
 *   mag   |h|, the known way;
 *   agree how much the neuron's inputs agree: for gate and for up,
 *         |sum w x| / sum |w x| (1 when every contribution pulls the same
 *         way, near 0 when they cancel), the two multiplied;
 *   both  |h| times the agreement;
 *   magw  |h| times the norm of the neuron's column of down: what switching
 *         it off takes from the output;
 *   bothw magw times the square root of the agreement;
 *   random.
 * Or a token's neurons, all its experts' together, ranked by what each
 * gives the output, its expert's weight w (the router's, or the shared
 * expert's gate) in it, so that a light expert loses more neurons than a
 * heavy one:
 *   tokmag  w |h|;  tokw  w |h| times the column's norm;
 *   tokboth tokw times the square root of the agreement.
 * JANAS_ANA_FRAC: the fraction off (0.4).
 * JANAS_ANA_COARSE=1 (the per-token modes): the choice made on h from gate
 * and up read at two bits a weight (Q4_K's nibbles, Q8_0's bytes, cut to
 * their two high bits), the neurons chosen then computed whole - a
 * predictor that reads a third of gate and up; at the end, how many of the
 * neurons the true choice keeps the two-bit one kept too.
 * JANAS_ANA_GULOW=n (the per-token modes): the n routed experts of least
 * weight in each token have their gate and up read at two bits a weight,
 * as JANAS_ANA_COARSE reads them; the shared expert whole.
 * JANAS_ANA_GRADE=f6,f4,f2 (the per-token modes): of a token's neurons
 * ranked, the first f6 keep all of down, the next f4 down at four bits, the
 * next f2 at two (its bit planes, or Q8_0's bytes cut), the rest off.
 * JANAS_ANA_STATS=file (the per-token modes): for every pair of every token
 * a record - its token, layer, expert (n_expert: the shared one), neurons
 * ff and weight w as five 32-bit words (w a float), then ff floats of h
 * before anything is off, then ff bytes, 1 for a neuron kept - for
 * job/tests/py/ana_stats.py.
 * JANAS_ANA_XSTATS=file (the per-token modes): each token's input to the
 * feed-forward of each layer - its token, layer and width as three 32-bit
 * words, then that many floats - once for every pass that ranks it (a
 * layer's experts in two passes write it twice).
 * JANAS_ANA_FRAMES=dir: for each token computed alone, a picture of the
 * network (dir/frame_NNNNN.ppm, a row a layer, a column a neuron of a pair
 * of the token: red its agreement against the pair's highest, green its
 * magnitude against the pair's largest, blue where it was switched off),
 * the first JANAS_ANA_FRAMES_MAX (32) of them.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"
#include "quant.h"

enum {
    ANA_MAG = 1,
    ANA_AGREE,
    ANA_BOTH,
    ANA_MAGW,
    ANA_BOTHW,
    ANA_RANDOM,
    ANA_TOKMAG,
    ANA_TOKW,
    ANA_TOKBOTH
};

static const char *const NAMES[] = {"",     "mag",    "agree",  "both",
                                    "magw", "bothw",  "random", "tokmag",
                                    "tokw", "tokboth"};

static int needs_agree(int mode)
{
    return mode == ANA_AGREE || mode == ANA_BOTH || mode == ANA_BOTHW ||
           mode == ANA_TOKBOTH;
}

static int needs_coln(int mode)
{
    return mode == ANA_MAGW || mode == ANA_BOTHW || mode == ANA_TOKW ||
           mode == ANA_TOKBOTH;
}

int janas_m_ana_init(struct janas_llm_model *m, size_t pairs)
{
    const char *e = getenv("JANAS_ANA_MODE");
    m->ana_mode = 0;
    for (int i = 1; e && i < (int)(sizeof(NAMES) / sizeof(NAMES[0])); i++)
        if (!strcmp(e, NAMES[i]))
            m->ana_mode = i;
    m->ana_tok = m->ana_mode >= ANA_TOKMAG;
    e = getenv("JANAS_ANA_FRAC");
    m->ana_frac = e ? (float)atof(e) : 0.4f;
    if (m->ana_frac < 0.0f)
        m->ana_frac = 0.0f;
    if (m->ana_frac > 0.95f)
        m->ana_frac = 0.95f;
    e = getenv("JANAS_ANA_FRAMES");
    if (e && *e)
        snprintf(m->ana_dir, sizeof(m->ana_dir), "%s", e);
    e = getenv("JANAS_ANA_FRAMES_MAX");
    m->ana_frames_max = e ? atoi(e) : 32;
    e = getenv("JANAS_ANA_GRADE");
    if (m->ana_tok && e &&
        sscanf(e, "%f,%f,%f", &m->ana_grade[0], &m->ana_grade[1],
               &m->ana_grade[2]) == 3) {
        m->ana_frac =
            1.0f - (m->ana_grade[0] + m->ana_grade[1] + m->ana_grade[2]);
        if (m->ana_frac < 0.0f)
            m->ana_frac = 0.0f;
        if (!(m->ana_level = calloc(pairs * m->ff_buf, 1)))
            return -1;
    }
    e = getenv("JANAS_ANA_GULOW");
    m->ana_gulow = m->ana_tok && e ? atoi(e) : 0;
    if (m->ana_gulow > 0 && !(m->ana_low = calloc(pairs, 1)))
        return -1;
    if (m->ana_gulow > 0)
        fprintf(stderr,
                "ana: gate and up at two bits for the %d experts of "
                "least weight in a token\n",
                m->ana_gulow);
    e = getenv("JANAS_ANA_COARSE");
    if (m->ana_tok && e && atoi(e) &&
        !(m->ana_coarse = calloc(pairs * m->ff_buf, sizeof(float))))
        return -1;
    e = getenv("JANAS_ANA_STATS");
    if (e && *e && m->ana_tok && !(m->ana_stats = fopen(e, "wb")))
        return -1;
    e = getenv("JANAS_ANA_XSTATS");
    if (e && *e && m->ana_tok && !(m->ana_xstats = fopen(e, "wb")))
        return -1;
    if (!m->ana_mode && !m->ana_dir[0])
        return 0;
    m->ana_exp = calloc(pairs, sizeof(uint32_t));
    if (!m->ana_exp)
        return -1;
    if (needs_agree(m->ana_mode) || m->ana_dir[0]) {
        m->ana_agree = calloc(pairs * m->ff_buf, sizeof(float));
        if (!m->ana_agree)
            return -1;
    }
    if (needs_coln(m->ana_mode)) {
        size_t slots = (size_t)m->n_layer * (m->n_expert + 1);
        m->ana_coln = calloc(slots * m->ff_buf, sizeof(float));
        m->ana_coln_ok = calloc(slots, 1);
        if (!m->ana_coln || !m->ana_coln_ok)
            return -1;
    }
    if (m->ana_dir[0]) {
        /* the routed experts of a token and a shared one, or a dense
           feed-forward */
        m->ana_width = m->ff_buf * (m->n_expert > 1 ? m->k + 1 : 1);
        m->ana_frame = calloc((size_t)m->n_layer * m->ana_width, 3);
        if (!m->ana_frame)
            return -1;
    }
    if (m->ana_level)
        fprintf(stderr,
                "ana: down at 6, 4, 2 bits for %.0f%%, %.0f%%, %.0f%% "
                "of a token's neurons\n",
                100.0 * m->ana_grade[0], 100.0 * m->ana_grade[1],
                100.0 * m->ana_grade[2]);
    if (m->ana_coarse)
        fprintf(stderr, "ana: the choice made on gate and up at two bits\n");
    fprintf(stderr,
            "ana: %s, %.0f%% of each %s's neurons off%s%s (a probe: the CPU's "
            "path only)\n",
            m->ana_mode ? NAMES[m->ana_mode] : "nothing switched off",
            m->ana_mode ? 100.0 * m->ana_frac : 0.0,
            m->ana_tok ? "token" : "pair",
            m->ana_dir[0] ? ", pictures in " : "", m->ana_dir);
    return 0;
}

void janas_m_ana_free(struct janas_llm_model *m)
{
    if (m->ana_true_kept)
        fprintf(stderr,
                "ana: the two-bit choice kept %.1f%% of the neurons the true "
                "one kept\n",
                100.0 * (double)m->ana_both_kept / (double)m->ana_true_kept);
    free(m->ana_coarse);
    free(m->ana_level);
    free(m->ana_low);
    m->ana_coarse = NULL;
    m->ana_level = NULL;
    m->ana_low = NULL;
    free(m->ana_agree);
    free(m->ana_frame);
    free(m->ana_exp);
    free(m->ana_coln);
    free(m->ana_coln_ok);
    if (m->ana_stats)
        fclose(m->ana_stats);
    if (m->ana_xstats)
        fclose(m->ana_xstats);
    m->ana_stats = m->ana_xstats = NULL;
    m->ana_agree = m->ana_coln = NULL;
    m->ana_frame = m->ana_coln_ok = NULL;
    m->ana_exp = NULL;
}

/* A run's agreements: for each neuron (a row of gate and of up), for each of
   its pairs, how much the products w x of the token's inputs agree. The
   rows unpacked once for all the run's pairs; neurons split among the
   threads. */
struct agree_job {
    struct janas_llm_model *m;
    const uint8_t *w[2];
    const struct janas_jns_matrix *mx;
    uint32_t a, b;
};

/* A row of gate or up at two bits a weight: Q4_K's nibbles and Q8_0's bytes
   cut to their two high bits, the middle of what those leave; other types
   as they are. tmp: room for the row's blocks. */
static void coarse_row(int type, const uint8_t *w, float *y, uint32_t n,
                       uint8_t *tmp)
{
    size_t nb = n / JANAS_QK, bs = janas_qtype_block_size(type);
    memcpy(tmp, w, nb * bs);
    if (type == JANAS_Q4_K) {
        for (size_t b = 0; b < nb; b++) {
            struct janas_block_q4k *k =
                (struct janas_block_q4k *)(tmp + b * bs);
            for (int i = 0; i < JANAS_QK / 2; i++)
                k->qs[i] = (uint8_t)(((k->qs[i] & 0x0C) | 0x01) |
                                     ((k->qs[i] & 0xC0) | 0x20));
        }
    } else if (type == JANAS_Q8_0) {
        struct janas_block_q8_0 *k = (struct janas_block_q8_0 *)tmp;
        for (size_t b = 0; b < nb * 8; b++)
            for (int i = 0; i < 32; i++)
                k[b].qs[i] = (int8_t)(((uint8_t)k[b].qs[i] & 0xC0) | 0x20);
    }
    janas_dequantize(type, tmp, y, n);
}

static void agree_worker(void *arg, int tid, int n_threads)
{
    struct agree_job *J = arg;
    struct janas_llm_model *m = J->m;
    uint32_t rows = J->mx[0].rows, cols = J->mx[0].cols, dm = m->d_model;
    size_t rb[2];
    for (int p = 0; p < 2; p++)
        rb[p] = (size_t)J->mx[p].cols / JANAS_QK *
                janas_qtype_block_size((int)J->mx[p].type);
    float *row = malloc(4 * (size_t)cols * sizeof(float));
    uint8_t *tmp = malloc(rb[0] > rb[1] ? rb[0] : rb[1]);
    if (!row || !tmp) {
        free(row);
        free(tmp);
        return;
    }
    for (uint32_t j = (uint32_t)tid; j < rows; j += (uint32_t)n_threads) {
        for (int p = 0; p < 2; p++)
            janas_dequantize((int)J->mx[p].type, J->w[p] + j * rb[p],
                             row + (size_t)p * cols, cols);
        int two = m->ana_coarse != NULL;
        for (uint32_t g = J->a; m->ana_low && !two && g < J->b; g++)
            two = m->ana_low[g];
        for (int p = 0; two && p < 2; p++)
            coarse_row((int)J->mx[p].type, J->w[p] + j * rb[p],
                       row + (size_t)(2 + p) * cols, cols, tmp);
        for (uint32_t g = J->a; two && g < J->b; g++) {
            if (!m->ana_coarse && !m->ana_low[g])
                continue;
            const float *x = m->xn + (size_t)m->pair_tok[m->order[g]] * dm;
            double v[2] = {0, 0};
            for (int p = 0; p < 2; p++)
                for (uint32_t i = 0; i < cols; i++)
                    v[p] += (double)row[(size_t)(2 + p) * cols + i] * x[i];
            if (m->ana_coarse) {
                double gs = v[0] / (1.0 + exp(-v[0]));
                m->ana_coarse[(size_t)g * m->ff_buf + j] = (float)(gs * v[1]);
            }
            if (m->ana_low && m->ana_low[g]) {
                /* this pair's gate and up are the two-bit ones */
                m->gate[(size_t)g * m->ff_buf + j] = (float)v[0];
                m->up[(size_t)g * m->ff_buf + j] = (float)v[1];
            }
        }
        if (!m->ana_agree)
            continue;
        for (uint32_t g = J->a; g < J->b; g++) {
            const float *x = m->xn + (size_t)m->pair_tok[m->order[g]] * dm;
            float c[2];
            for (int p = 0; p < 2; p++) {
                const float *w = row + (size_t)p * cols;
                double s = 0, sa = 0;
                for (uint32_t i = 0; i < cols; i++) {
                    double v = (double)w[i] * x[i];
                    s += v;
                    sa += fabs(v);
                }
                c[p] = sa > 0 ? (float)(fabs(s) / sa) : 0.0f;
            }
            m->ana_agree[(size_t)g * m->ff_buf + j] = c[0] * c[1];
        }
    }
    free(row);
    free(tmp);
}

void janas_m_ana_agree(struct janas_llm_model *m, const uint8_t *wg,
                       const uint8_t *wu, const struct janas_jns_matrix *mx,
                       uint32_t a, uint32_t b)
{
    if ((!m->ana_agree && !m->ana_coarse && !m->ana_low) ||
        mx[0].cols != m->d_model || mx[1].cols != m->d_model ||
        mx[0].rows != mx[1].rows || mx[0].rows > m->ff_buf ||
        !janas_qtype_block_size((int)mx[0].type) ||
        !janas_qtype_block_size((int)mx[1].type))
        return;
    struct agree_job J = {.m = m, .w = {wg, wu}, .mx = mx, .a = a, .b = b};
    janas_pool_run(m->compute, agree_worker, &J);
}

/* The norms of the columns of expert e's down matrix in the current layer
   (rows the model's width, a column a neuron), once: what switching a
   neuron off takes from the output, per unit of |h|. Bit planes unpacked
   with those the cache holds; a type that cannot be unpacked, norms of 1. */
void janas_m_ana_coln(struct janas_llm_model *m, uint32_t e,
                      const uint8_t *down, const uint8_t *p1, const uint8_t *p0,
                      const struct janas_jns_matrix *mx)
{
    size_t slot = (size_t)m->ana_layer * (m->n_expert + 1) + e;
    if (!m->ana_coln || m->ana_coln_ok[slot] || mx->cols > m->ff_buf)
        return;
    float *out = m->ana_coln + slot * m->ff_buf;
    uint32_t rows = mx->rows, cols = mx->cols, nb = cols / JANAS_QK;
    int planes = mx->type == JANAS_Q6_K_P;
    size_t bs = planes ? sizeof(struct janas_block_q6kp)
                       : janas_qtype_block_size((int)mx->type);
    float *y = malloc(cols * sizeof(float));
    double *acc = calloc(cols, sizeof(double));
    if (!y || !acc || !bs) {
        for (uint32_t j = 0; j < cols; j++)
            out[j] = 1.0f;
    } else {
        for (uint32_t r = 0; r < rows; r++) {
            if (planes)
                janas_q6kp_dequantize(
                    (const struct janas_block_q6kp *)(down +
                                                      (size_t)r * nb * bs),
                    p1 ? p1 + (size_t)r * nb * JANAS_Q6KP_PLANE : NULL,
                    p1 && p0 ? p0 + (size_t)r * nb * JANAS_Q6KP_PLANE : NULL, y,
                    cols);
            else
                janas_dequantize((int)mx->type, down + (size_t)r * nb * bs, y,
                                 cols);
            for (uint32_t j = 0; j < cols; j++)
                acc[j] += (double)y[j] * y[j];
        }
        for (uint32_t j = 0; j < cols; j++)
            out[j] = (float)sqrt(acc[j]);
    }
    free(y);
    free(acc);
    m->ana_coln_ok[slot] = 1;
}

/* A fixed pseudo-random number of a neuron of a pair in a layer. */
static float noise(uint32_t layer, size_t g, uint32_t j)
{
    uint64_t z = ((uint64_t)layer << 40) ^ ((uint64_t)g << 20) ^ j;
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float)(z >> 40) / (float)(1u << 24);
}

/* Neuron j of pair row g by the mode's score (higher: kept). */
static float score(const struct janas_llm_model *m, size_t g, uint32_t j,
                   float h)
{
    const float *agree = m->ana_agree ? m->ana_agree + g * m->ff_buf : NULL;
    float a = agree ? agree[j] : 1.0f, mag = fabsf(h);
    float col = 1.0f;
    if (m->ana_coln && m->ana_exp) {
        size_t slot = (size_t)m->ana_layer * (m->n_expert + 1) + m->ana_exp[g];
        col = m->ana_coln[slot * m->ff_buf + j];
    }
    float w = m->ana_tok ? m->pair_w[m->order[g]] : 1.0f;
    switch (m->ana_mode) {
    case ANA_MAG:
        return mag;
    case ANA_AGREE:
        return a;
    case ANA_BOTH:
        return mag * a;
    case ANA_MAGW:
        return mag * col;
    case ANA_BOTHW:
        return mag * col * sqrtf(a);
    case ANA_TOKMAG:
        return w * mag;
    case ANA_TOKW:
        return w * mag * col;
    case ANA_TOKBOTH:
        return w * mag * col * sqrtf(a);
    default:
        return noise(m->ana_layer, g, j);
    }
}

struct ranked {
    float score;
    uint32_t g, j;
};

static int by_score(const void *a, const void *b)
{
    const struct ranked *x = a, *y = b;
    if (x->score != y->score)
        return x->score < y->score ? -1 : 1;
    if (x->g != y->g)
        return x->g < y->g ? -1 : 1;
    return x->j < y->j ? -1 : x->j > y->j ? 1 : 0;
}

/* Pair row g in the picture: red its agreements against the pair's highest,
   green its magnitudes (before any is off) against the largest; blue, its
   neurons switched off, set after. */
static void frame_pair(struct janas_llm_model *m, size_t g, const float *h,
                       uint32_t ff)
{
    if (!m->ana_frame || (g + 1) * m->ff_buf > m->ana_width)
        return;
    const float *agree = m->ana_agree ? m->ana_agree + g * m->ff_buf : NULL;
    float big = 0.0f, top = 0.0f;
    for (uint32_t j = 0; j < ff; j++) {
        if (fabsf(h[j]) > big)
            big = fabsf(h[j]);
        if (agree && agree[j] > top)
            top = agree[j];
    }
    uint8_t *px = m->ana_frame +
                  ((size_t)m->ana_layer * m->ana_width + g * m->ff_buf) * 3;
    for (uint32_t j = 0; j < ff; j++) {
        px[3 * j] = top > 0 ? (uint8_t)(255.0f * agree[j] / top + 0.5f) : 0;
        px[3 * j + 1] =
            big > 0 ? (uint8_t)(255.0f * sqrtf(fabsf(h[j]) / big) + 0.5f) : 0;
        px[3 * j + 2] = 0;
    }
}

static void frame_off(struct janas_llm_model *m, size_t g, uint32_t j)
{
    if (m->ana_frame && (g + 1) * m->ff_buf <= m->ana_width)
        m->ana_frame[((size_t)m->ana_layer * m->ana_width + g * m->ff_buf + j) *
                         3 +
                     2] = 160;
}

void janas_m_ana_mask(struct janas_llm_model *m, size_t g, float *h,
                      uint32_t ff)
{
    uint32_t k = (uint32_t)(m->ana_frac * (float)ff + 0.5f);
    struct ranked *r = NULL;
    if (m->ana_mode && k > 0 && (r = malloc(ff * sizeof(*r)))) {
        for (uint32_t j = 0; j < ff; j++)
            r[j] = (struct ranked){score(m, g, j, h[j]), (uint32_t)g, j};
        qsort(r, ff, sizeof(*r), by_score);
    }
    frame_pair(m, g, h, ff);
    for (uint32_t i = 0; r && i < k; i++) {
        frame_off(m, g, r[i].j);
        h[r[i].j] = 0.0f;
    }
    free(r);
}

static int by_token(const void *a, const void *b)
{
    const uint64_t *x = a, *y = b;
    return *x < *y ? -1 : *x > *y ? 1 : 0;
}

/* The ana_tok modes: the pair rows glist[0 .. n) of this pass grouped by
   token, each token's neurons ranked together - on h, or on the two-bit h
   with JANAS_ANA_COARSE - the fraction of them with the least score
   switched off (and, with JANAS_ANA_GRADE, the others given their levels),
   then every pair's h quantized. Pairs of a token computed in another pass
   (an expert read from the disk while the others ran) are ranked apart. */
static size_t rank_token(struct janas_llm_model *m, const uint64_t *key,
                         uint32_t i, uint32_t e, size_t total, int coarse,
                         struct ranked *r)
{
    uint32_t stride = m->ff_buf;
    size_t at = 0;
    for (uint32_t q = i; q < e; q++) {
        size_t g = (uint32_t)key[q];
        const float *h =
            coarse ? m->ana_coarse + g * stride : m->h + g * stride;
        for (uint32_t j = 0; j < m->hlen[g]; j++)
            r[at++] = (struct ranked){score(m, g, j, h[j]), (uint32_t)g, j};
    }
    qsort(r, total, sizeof(*r), by_score);
    return at;
}

void janas_m_ana_tok(struct janas_llm_model *m, uint32_t n)
{
    uint32_t stride = m->ff_buf;
    uint64_t *key = malloc(n * sizeof(uint64_t));
    if (key) {
        for (uint32_t i = 0; i < n; i++) {
            size_t g = m->glist[i];
            key[i] = ((uint64_t)m->pair_tok[m->order[g]] << 32) | (uint32_t)g;
        }
        qsort(key, n, sizeof(*key), by_token);
        for (uint32_t i = 0; i < n;) {
            uint32_t tok = (uint32_t)(key[i] >> 32), e = i;
            size_t total = 0;
            while (e < n && (uint32_t)(key[e] >> 32) == tok)
                total += m->hlen[(uint32_t)key[e++]];
            if (m->ana_xstats) {
                uint32_t head[3] = {(uint32_t)(m->ana_base + tok), m->ana_layer,
                                    m->d_model};
                fwrite(head, sizeof(head), 1, m->ana_xstats);
                fwrite(m->xn + (size_t)tok * m->d_model, sizeof(float),
                       m->d_model, m->ana_xstats);
            }
            struct ranked *r = malloc(total * sizeof(*r));
            if (!r) {
                i = e;
                continue;
            }
            for (uint32_t q = i; q < e; q++) {
                size_t g = (uint32_t)key[q];
                frame_pair(m, g, m->h + g * stride, m->hlen[g]);
            }
            size_t k = (size_t)(m->ana_frac * (float)total + 0.5f);
            rank_token(m, key, i, e, total, m->ana_coarse != NULL, r);
            /* with the two-bit choice, how much of the true one it keeps:
               levels mark the true choice's kept neurons for a moment */
            if (m->ana_coarse && m->ana_level == NULL) {
                struct ranked *t = malloc(total * sizeof(*t));
                uint8_t *mark = calloc((size_t)(e - i) * stride, 1);
                if (t && mark) {
                    rank_token(m, key, i, e, total, 0, t);
                    for (size_t q = k; q < total; q++)
                        for (uint32_t z = i; z < e; z++)
                            if ((uint32_t)key[z] == t[q].g) {
                                mark[(size_t)(z - i) * stride + t[q].j] = 1;
                                break;
                            }
                    for (size_t q = k; q < total; q++)
                        for (uint32_t z = i; z < e; z++)
                            if ((uint32_t)key[z] == r[q].g) {
                                m->ana_both_kept +=
                                    mark[(size_t)(z - i) * stride + r[q].j];
                                break;
                            }
                    m->ana_true_kept += total - k;
                }
                free(t);
                free(mark);
            }
            /* the levels: from the top, f6, f4 and f2 of the token's
               neurons; the rest, level 0, is what goes off below */
            if (m->ana_level) {
                size_t n6 = (size_t)(m->ana_grade[0] * (float)total + 0.5f),
                       n4 = (size_t)(m->ana_grade[1] * (float)total + 0.5f),
                       n2 = (size_t)(m->ana_grade[2] * (float)total + 0.5f);
                for (size_t q = 0; q < total; q++) {
                    size_t top = total - 1 - q;
                    m->ana_level[(size_t)r[q].g * stride + r[q].j] =
                        top < n6             ? 3
                        : top < n6 + n4      ? 2
                        : top < n6 + n4 + n2 ? 1
                                             : 0;
                }
            }
            /* the record first, with h as it was */
            uint8_t *keep = m->ana_stats ? malloc(stride) : NULL;
            for (uint32_t q = i; keep && q < e; q++) {
                size_t g = (uint32_t)key[q];
                uint32_t ff = m->hlen[g];
                memset(keep, 1, ff);
                for (size_t z = 0; z < k; z++)
                    if (r[z].g == g)
                        keep[r[z].j] = 0;
                float w = m->pair_w[m->order[g]];
                uint32_t head[5] = {(uint32_t)(m->ana_base + tok), m->ana_layer,
                                    m->ana_exp[g], ff, 0};
                memcpy(&head[4], &w, sizeof(w));
                fwrite(head, sizeof(head), 1, m->ana_stats);
                fwrite(m->h + g * stride, sizeof(float), ff, m->ana_stats);
                fwrite(keep, 1, ff, m->ana_stats);
            }
            free(keep);
            for (size_t q = 0; q < k; q++) {
                frame_off(m, r[q].g, r[q].j);
                m->h[(size_t)r[q].g * stride + r[q].j] = 0.0f;
            }
            free(r);
            i = e;
        }
        free(key);
    }
    for (uint32_t i = 0; i < n; i++) {
        size_t g = m->glist[i];
        janas_q8k_quantize_det(m->h + g * stride, m->hq + g * stride / JANAS_QK,
                               m->hlen[g]);
    }
}

/* The pair rows glist[0 .. n) of this pass whose gate and up go to two
   bits: in each token, its ana_gulow routed experts of least weight. */
void janas_m_ana_lowmark(struct janas_llm_model *m, uint32_t n)
{
    uint64_t *key = malloc(n * sizeof(uint64_t));
    if (!key)
        return;
    uint32_t nk = 0;
    for (uint32_t i = 0; i < n; i++) {
        size_t g = m->glist[i];
        m->ana_low[g] = 0;
        if (m->ana_exp[g] == m->n_expert) /* the shared expert: whole */
            continue;
        /* by token, then by weight (a positive float's bits sort as it) */
        float w = m->pair_w[m->order[g]];
        uint32_t wb;
        memcpy(&wb, &w, sizeof(wb));
        key[nk++] = ((uint64_t)m->pair_tok[m->order[g]] << 40) |
                    ((uint64_t)(wb >> 8) << 16) | (g & 0xFFFF);
    }
    qsort(key, nk, sizeof(*key), by_token);
    for (uint32_t i = 0; i < nk;) {
        uint32_t e = i;
        while (e < nk && (key[e] >> 40) == (key[i] >> 40))
            e++;
        for (uint32_t q = i; q < e && q < i + (uint32_t)m->ana_gulow; q++)
            m->ana_low[key[q] & 0xFFFF] = 1;
        i = e;
    }
    free(key);
}

/* Down at fewer bits for the neurons of the run's pairs whose level is 1 or
   2: their contribution corrected, after the product at full precision,
   by h times the difference of the row read at two or four bits. Q6_K_P:
   its planes; Q8_0: its bytes cut to two or four high bits; other types
   left whole. Rows split among the threads. */
struct grade_job {
    struct janas_llm_model *m;
    const uint8_t *down, *p1, *p0;
    const struct janas_jns_matrix *mx;
    uint32_t a, b;
};

static void cut_q8_0(uint8_t *blocks, size_t nb32, int bits)
{
    struct janas_block_q8_0 *k = (struct janas_block_q8_0 *)blocks;
    uint8_t keep = bits == 4 ? 0xF0 : 0xC0, mid = bits == 4 ? 0x08 : 0x20;
    for (size_t b = 0; b < nb32; b++)
        for (int i = 0; i < 32; i++)
            k[b].qs[i] = (int8_t)(((uint8_t)k[b].qs[i] & keep) | mid);
}

static void grade_worker(void *arg, int tid, int n_threads)
{
    struct grade_job *J = arg;
    struct janas_llm_model *m = J->m;
    uint32_t rows = J->mx->rows, cols = J->mx->cols, nb = cols / JANAS_QK;
    int planes = J->mx->type == JANAS_Q6_K_P, q8 = J->mx->type == JANAS_Q8_0;
    size_t bs = planes ? sizeof(struct janas_block_q6kp)
                       : janas_qtype_block_size((int)J->mx->type);
    float *w = malloc(3 * (size_t)cols * sizeof(float));
    uint8_t *tmp = malloc(nb * bs);
    if (!w || !tmp || !(planes || q8)) {
        free(w);
        free(tmp);
        return;
    }
    float *w6 = w, *w4 = w + cols, *w2 = w + 2 * (size_t)cols;
    for (uint32_t r = (uint32_t)tid; r < rows; r += (uint32_t)n_threads) {
        const uint8_t *row = J->down + (size_t)r * nb * bs;
        if (planes) {
            const struct janas_block_q6kp *base =
                (const struct janas_block_q6kp *)row;
            const uint8_t *p1 =
                J->p1 ? J->p1 + (size_t)r * nb * JANAS_Q6KP_PLANE : NULL;
            const uint8_t *p0 = J->p1 && J->p0
                                    ? J->p0 + (size_t)r * nb * JANAS_Q6KP_PLANE
                                    : NULL;
            janas_q6kp_dequantize(base, p1, p0, w6, cols);
            janas_q6kp_dequantize(base, p1, NULL, w4, cols);
            janas_q6kp_dequantize(base, NULL, NULL, w2, cols);
        } else {
            janas_dequantize(JANAS_Q8_0, row, w6, cols);
            memcpy(tmp, row, nb * bs);
            cut_q8_0(tmp, (size_t)nb * 8, 4);
            janas_dequantize(JANAS_Q8_0, tmp, w4, cols);
            memcpy(tmp, row, nb * bs);
            cut_q8_0(tmp, (size_t)nb * 8, 2);
            janas_dequantize(JANAS_Q8_0, tmp, w2, cols);
        }
        for (uint32_t g = J->a; g < J->b; g++) {
            const uint8_t *lv = m->ana_level + (size_t)g * m->ff_buf;
            const float *h = m->h + (size_t)g * m->ff_buf;
            double d = 0;
            for (uint32_t j = 0; j < cols; j++)
                if (lv[j] == 2)
                    d += (double)h[j] * (w4[j] - w6[j]);
                else if (lv[j] == 1)
                    d += (double)h[j] * (w2[j] - w6[j]);
            m->dout[(size_t)g * m->d_model + r] += (float)d;
        }
    }
    free(w);
    free(tmp);
}

void janas_m_ana_grade(struct janas_llm_model *m, const uint8_t *down,
                       const uint8_t *p1, const uint8_t *p0,
                       const struct janas_jns_matrix *mx, uint32_t a,
                       uint32_t b)
{
    if (!m->ana_level || mx->rows != m->d_model || mx->cols > m->ff_buf)
        return;
    struct grade_job J = {
        .m = m, .down = down, .p1 = p1, .p0 = p0, .mx = mx, .a = a, .b = b};
    janas_pool_run(m->compute, grade_worker, &J);
}

void janas_m_ana_frame(struct janas_llm_model *m, uint32_t n)
{
    m->ana_base += n;
    if (!m->ana_frame)
        return;
    if (n == 1 && m->ana_frames < m->ana_frames_max) {
        char path[600];
        snprintf(path, sizeof(path), "%s/frame_%05d.ppm", m->ana_dir,
                 m->ana_frames++);
        FILE *f = fopen(path, "wb");
        if (f) {
            fprintf(f, "P6\n%u %u\n255\n", m->ana_width, m->n_layer);
            fwrite(m->ana_frame, 3, (size_t)m->n_layer * m->ana_width, f);
            fclose(f);
        }
    }
    memset(m->ana_frame, 0, (size_t)m->n_layer * m->ana_width * 3);
}
