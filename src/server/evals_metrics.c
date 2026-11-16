/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * evals_metrics.c - the text similarity metrics of the evals' graders (see
 * evals_metrics.h): words lower-cased and split at what is not a letter or
 * a digit (bytes of UTF-8 characters count as letters), then the usual
 * formulas - fuzzy match as Levenshtein's ratio over characters, BLEU with
 * add-one smoothing past the unigrams, GLEU as Google's sentence one,
 * METEOR on exact unigram matches with its fragmentation penalty, ROUGE-n
 * and ROUGE-L as F1, cosine over word counts.
 */
#include "evals_metrics.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS 4096

struct words {
    char *buf;
    char *w[MAX_WORDS];
    int n;
};

static void split(const char *s, struct words *o)
{
    size_t n = strlen(s);
    o->buf = malloc(n + 1);
    o->n = 0;
    if (!o->buf)
        return;
    size_t k = 0;
    for (size_t i = 0; i <= n && o->n < MAX_WORDS;) {
        while (i < n &&
               !(isalnum((unsigned char)s[i]) || (unsigned char)s[i] >= 0x80))
            i++;
        if (i >= n)
            break;
        o->w[o->n++] = o->buf + k;
        while (i < n &&
               (isalnum((unsigned char)s[i]) || (unsigned char)s[i] >= 0x80))
            o->buf[k++] = (char)tolower((unsigned char)s[i++]);
        o->buf[k++] = 0;
    }
}

static int same_gram(char *const *a, char *const *b, int n)
{
    for (int i = 0; i < n; i++)
        if (strcmp(a[i], b[i]) != 0)
            return 0;
    return 1;
}

/* n-grams of h also in r, clipped by their count in r */
static int clipped(const struct words *h, const struct words *r, int n)
{
    if (h->n < n || r->n < n)
        return 0;
    char *used = calloc((size_t)(r->n - n + 1), 1);
    int m = 0;
    for (int i = 0; used && i + n <= h->n; i++)
        for (int j = 0; j + n <= r->n; j++)
            if (!used[j] && same_gram(h->w + i, r->w + j, n)) {
                used[j] = 1;
                m++;
                break;
            }
    free(used);
    return m;
}

static double f1(double match, double a, double b)
{
    if (match <= 0 || a <= 0 || b <= 0)
        return 0;
    double p = match / a, r = match / b;
    return 2 * p * r / (p + r);
}

static double lcs(const struct words *a, const struct words *b)
{
    int *prev = calloc((size_t)b->n + 1, sizeof(int)),
        *cur = calloc((size_t)b->n + 1, sizeof(int));
    if (!prev || !cur) {
        free(prev);
        free(cur);
        return 0;
    }
    for (int i = 1; i <= a->n; i++) {
        for (int j = 1; j <= b->n; j++)
            cur[j] = !strcmp(a->w[i - 1], b->w[j - 1])
                         ? prev[j - 1] + 1
                         : (prev[j] > cur[j - 1] ? prev[j] : cur[j - 1]);
        int *t = prev;
        prev = cur;
        cur = t;
    }
    double v = prev[b->n];
    free(prev);
    free(cur);
    return v;
}

static double levenshtein_ratio(const char *a, const char *b)
{
    size_t n = strlen(a), m = strlen(b);
    if (!n && !m)
        return 1;
    if (n > 20000 || m > 20000)
        return a[0] == b[0] && !strcmp(a, b);
    size_t *prev = malloc((m + 1) * sizeof(size_t)),
           *cur = malloc((m + 1) * sizeof(size_t));
    if (!prev || !cur) {
        free(prev);
        free(cur);
        return 0;
    }
    for (size_t j = 0; j <= m; j++)
        prev[j] = j;
    for (size_t i = 1; i <= n; i++) {
        cur[0] = i;
        for (size_t j = 1; j <= m; j++) {
            size_t sub = prev[j - 1] + (a[i - 1] != b[j - 1]);
            size_t del = prev[j] + 1, ins = cur[j - 1] + 1;
            cur[j] =
                sub < del ? (sub < ins ? sub : ins) : (del < ins ? del : ins);
        }
        size_t *t = prev;
        prev = cur;
        cur = t;
    }
    double d = (double)prev[m];
    free(prev);
    free(cur);
    return 1.0 - d / (double)(n > m ? n : m);
}

static double bleu(const struct words *h, const struct words *r)
{
    if (!h->n || !r->n)
        return 0;
    double logp = 0;
    for (int n = 1; n <= 4; n++) {
        double grams = h->n - n + 1;
        double m = clipped(h, r, n);
        if (grams <= 0)
            grams = 0;
        /* add-one past the unigrams, so a short sentence is not 0 */
        double p = n == 1 ? (grams ? m / grams : 0) : (m + 1) / (grams + 1);
        if (p <= 0)
            return 0;
        logp += log(p) / 4;
    }
    double bp = h->n >= r->n ? 1 : exp(1 - (double)r->n / h->n);
    return bp * exp(logp);
}

static double gleu(const struct words *h, const struct words *r)
{
    double match = 0, hg = 0, rg = 0;
    for (int n = 1; n <= 4; n++) {
        match += clipped(h, r, n);
        hg += h->n >= n ? h->n - n + 1 : 0;
        rg += r->n >= n ? r->n - n + 1 : 0;
    }
    if (!hg || !rg)
        return 0;
    double p = match / hg, rc = match / rg;
    return p < rc ? p : rc;
}

static double meteor(const struct words *h, const struct words *r)
{
    double m = clipped(h, r, 1);
    if (m <= 0)
        return 0;
    double p = m / h->n, rc = m / r->n;
    double fmean = 10 * p * rc / (rc + 9 * p);
    /* chunks: runs of matched words in the same order in both */
    int chunks = 0, in = 0;
    for (int i = 0; i < h->n; i++) {
        int hit = 0;
        for (int j = 0; j < r->n && !hit; j++)
            hit = !strcmp(h->w[i], r->w[j]);
        if (hit && !in)
            chunks++;
        in = hit;
    }
    double pen = 0.5 * pow(chunks / m, 3);
    return fmean * (1 - pen);
}

static double cosine_words(const struct words *a, const struct words *b)
{
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < a->n; i++) {
        int seen = 0;
        for (int k = 0; k < i && !seen; k++)
            seen = !strcmp(a->w[k], a->w[i]);
        if (seen)
            continue;
        double ca = 0, cb = 0;
        for (int k = 0; k < a->n; k++)
            ca += !strcmp(a->w[k], a->w[i]);
        for (int k = 0; k < b->n; k++)
            cb += !strcmp(b->w[k], a->w[i]);
        dot += ca * cb;
        na += ca * ca;
    }
    for (int i = 0; i < b->n; i++) {
        int seen = 0;
        for (int k = 0; k < i && !seen; k++)
            seen = !strcmp(b->w[k], b->w[i]);
        if (seen)
            continue;
        double cb = 0;
        for (int k = 0; k < b->n; k++)
            cb += !strcmp(b->w[k], b->w[i]);
        nb += cb * cb;
    }
    return na > 0 && nb > 0 ? dot / sqrt(na * nb) : 0;
}

double srv_eval_similarity(const char *metric, const char *input,
                           const char *reference)
{
    if (!strcmp(metric, "fuzzy_match"))
        return levenshtein_ratio(input, reference);
    struct words h, r;
    split(input, &h);
    split(reference, &r);
    double v = 0;
    if (!h.buf || !r.buf)
        v = 0;
    else if (!strcmp(metric, "bleu"))
        v = bleu(&h, &r);
    else if (!strcmp(metric, "gleu"))
        v = gleu(&h, &r);
    else if (!strcmp(metric, "meteor"))
        v = meteor(&h, &r);
    else if (!strcmp(metric, "rouge_l"))
        v = f1(lcs(&h, &r), h.n, r.n);
    else if (!strncmp(metric, "rouge_", 6) && metric[6] >= '1' &&
             metric[6] <= '5' && !metric[7]) {
        int n = metric[6] - '0';
        v = f1(clipped(&h, &r, n), h.n >= n ? h.n - n + 1 : 0,
               r.n >= n ? r.n - n + 1 : 0);
    } else if (!strcmp(metric, "cosine"))
        v = cosine_words(&h, &r);
    free(h.buf);
    free(r.buf);
    return v < 0 ? 0 : v > 1 ? 1 : v;
}

int srv_eval_metric_ok(const char *metric)
{
    static const char *const M[] = {
        "cosine",  "fuzzy_match", "bleu",    "gleu",    "meteor", "rouge_1",
        "rouge_2", "rouge_3",     "rouge_4", "rouge_5", "rouge_l"};
    for (size_t i = 0; metric && i < sizeof(M) / sizeof(M[0]); i++)
        if (!strcmp(metric, M[i]))
            return 1;
    return 0;
}
