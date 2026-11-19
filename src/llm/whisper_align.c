/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper_align.c - the times of the words, as OpenAI's timing.py finds
 * them: the window's text decoded again after the start of transcript,
 * its language, its task and "no timestamps", keeping the scores of the
 * alignment heads' attention over the sound; their softmax over the
 * sound's frames, standardized across the tokens, a median filter of 7
 * frames, their mean; the path of least cost through it (dynamic time
 * warping) gives each token the frame where it starts; tokens joined into
 * words as OpenAI's tokenizer joins them, punctuation into its word, and
 * OpenAI's corrections of words too long at pauses and at the segments'
 * ends.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "whisper.h"

#define TOKENS_PER_SECOND 50.0
#define MEDFILT 7

static const char PUNCTUATION[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
static const char PREPEND[] = "\"'\xe2\x80\x9c\xc2\xbf([{-";
static const char APPEND[] = "\"'.\xe3\x80\x82,\xef\xbc\x8c!\xef\xbc\x81?"
                             "\xef\xbc\x9f:\xef\xbc\x9a\xe2\x80\x9d)]}"
                             "\xe3\x80\x81";
static const char SENTENCE_END[] = ".\xe3\x80\x82!\xef\xbc\x81?\xef\xbc\x9f";

/* Python's "a in b" for strings: a substring (the empty one too) */
static int in(const char *a, size_t n, const char *b)
{
    if (!n)
        return 1;
    size_t bn = strlen(b);
    for (size_t i = 0; i + n <= bn; i++)
        if (!memcmp(b + i, a, n))
            return 1;
    return 0;
}

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
}

/* str.strip(): the part between the spaces */
static const char *strip(const char *s, size_t *n)
{
    size_t a = 0, b = strlen(s);
    while (a < b && is_space(s[a]))
        a++;
    while (b > a && is_space(s[b - 1]))
        b--;
    *n = b - a;
    return s + a;
}

/* bytes that end a whole UTF-8 sequence, or that no more bytes could make
   one of (what Python decodes with a replacement character for good) */
static int utf8_done(const unsigned char *s, size_t n)
{
    size_t i = 0;
    while (i < n) {
        unsigned c = s[i];
        size_t need = c < 0x80   ? 0
                      : c < 0xC2 ? (size_t)-1
                      : c < 0xE0 ? 1
                      : c < 0xF0 ? 2
                      : c < 0xF5 ? 3
                                 : (size_t)-1;
        if (need == (size_t)-1) {
            i++;
            continue;
        }
        size_t k = 1;
        while (k <= need && i + k < n && (s[i + k] & 0xC0) == 0x80)
            k++;
        if (k <= need && i + k == n)
            return 0; /* a sequence the next token may end */
        i += k;
    }
    return 1;
}

struct word {
    char *text;
    int32_t *tok;
    uint32_t n_tok;
    double t0, t1;
    float prob;
};

static void words_free(struct word *w, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        free(w[i].text);
        free(w[i].tok);
    }
    free(w);
}

static int append(struct word *w, const char *s, size_t sn, const int32_t *t,
                  uint32_t tn)
{
    size_t old = w->text ? strlen(w->text) : 0;
    char *x = realloc(w->text, old + sn + 1);
    if (!x)
        return -1;
    memcpy(x + old, s, sn);
    x[old + sn] = 0;
    w->text = x;
    int32_t *y = realloc(w->tok, sizeof(int32_t) * (w->n_tok + tn + 1));
    if (!y)
        return -1;
    memcpy(y + w->n_tok, t, sizeof(int32_t) * tn);
    w->tok = y;
    w->n_tok += tn;
    return 0;
}

/* tokenizer.split_to_word_tokens: the words of tokens (the end of text
   last), into *out; their count, or -1 */
static long split_words(struct janas_whisper *w, const int32_t *tok, uint32_t n,
                        int no_spaces, struct word **out)
{
    struct word *ws = calloc(n + 1, sizeof *ws);
    long nw = 0;
    char *buf = NULL;
    size_t cap = 0, len = 0;
    uint32_t start = 0;
    if (!ws)
        return -1;
    for (uint32_t i = 0; i < n; i++) {
        char none[1];
        size_t k = janas_tokenizer_decode(w->tk, tok[i], none, 0);
        if (len + k + 1 > cap) {
            size_t nc = (len + k + 1) * 2;
            char *b = realloc(buf, nc);
            if (!b)
                goto fail;
            buf = b;
            cap = nc;
        }
        janas_tokenizer_decode(w->tk, tok[i], buf + len, k);
        len += k;
        buf[len] = 0;
        int special = tok[i] >= w->eot;
        if (!special && !utf8_done((const unsigned char *)buf, len) &&
            i + 1 < n)
            continue;
        /* a subword: tokens start .. i */
        size_t sn;
        const char *st = strip(buf, &sn);
        int new_word = no_spaces || special || buf[0] == ' ' ||
                       in(st, sn, PUNCTUATION) || nw == 0;
        if (new_word)
            nw++;
        if (append(&ws[nw - 1], buf, len, tok + start, i + 1 - start) != 0)
            goto fail;
        len = 0;
        start = i + 1;
    }
    free(buf);
    *out = ws;
    return nw;
fail:
    free(buf);
    words_free(ws, n + 1);
    return -1;
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* dtw(-matrix) of n x m: the path's (row, column) pairs, from the start;
   their count, or -1 */
static long dtw(const float *mat, uint32_t n, uint32_t m, uint32_t **ti,
                uint32_t **fi)
{
    size_t W = (size_t)m + 1;
    float *cost = malloc(sizeof(float) * (n + 1) * W);
    signed char *trace = malloc((size_t)(n + 1) * W);
    *ti = malloc(sizeof(uint32_t) * (n + m + 2));
    *fi = malloc(sizeof(uint32_t) * (n + m + 2));
    long cnt = -1;
    if (!cost || !trace || !*ti || !*fi)
        goto end;
    for (size_t i = 0; i < (n + 1) * W; i++) {
        cost[i] = INFINITY;
        trace[i] = -1;
    }
    cost[0] = 0;
    for (uint32_t j = 1; j <= m; j++)
        for (uint32_t i = 1; i <= n; i++) {
            float c0 = cost[(i - 1) * W + j - 1], c1 = cost[(i - 1) * W + j];
            float c2 = cost[i * W + j - 1], c;
            int t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }
            cost[i * W + j] =
                (float)(-(double)mat[(size_t)(i - 1) * m + j - 1] + c);
            trace[i * W + j] = (signed char)t;
        }
    for (uint32_t j = 0; j <= m; j++)
        trace[j] = 2;
    for (uint32_t i = 0; i <= n; i++)
        trace[i * W] = 1;
    uint32_t i = n, j = m;
    long k = 0;
    while (i > 0 || j > 0) {
        (*ti)[k] = i - 1;
        (*fi)[k] = j - 1;
        k++;
        int t = trace[i * W + j];
        if (t == 0) {
            i--;
            j--;
        } else if (t == 1) {
            i--;
        } else {
            j--;
        }
    }
    for (long a = 0, b = k - 1; a < b; a++, b--) {
        uint32_t x = (*ti)[a];
        (*ti)[a] = (*ti)[b];
        (*ti)[b] = x;
        x = (*fi)[a];
        (*fi)[a] = (*fi)[b];
        (*fi)[b] = x;
    }
    cnt = k;
end:
    free(cost);
    free(trace);
    if (cnt < 0) {
        free(*ti);
        free(*fi);
        *ti = *fi = NULL;
    }
    return cnt;
}

static void merge_punctuation(struct word *al, uint32_t n)
{
    if (n < 2)
        return;
    /* prepended: into the word after */
    long i = (long)n - 2, j = (long)n - 1;
    while (i >= 0) {
        struct word *p = &al[i], *f = &al[j];
        size_t sn;
        const char *st = strip(p->text, &sn);
        if (p->text[0] == ' ' && in(st, sn, PREPEND)) {
            size_t a = strlen(p->text), b = strlen(f->text);
            char *x = malloc(a + b + 1);
            int32_t *t = malloc(sizeof(int32_t) * (p->n_tok + f->n_tok + 1));
            if (x && t) {
                memcpy(x, p->text, a);
                memcpy(x + a, f->text, b + 1);
                memcpy(t, p->tok, sizeof(int32_t) * p->n_tok);
                memcpy(t + p->n_tok, f->tok, sizeof(int32_t) * f->n_tok);
                free(f->text);
                free(f->tok);
                f->text = x;
                f->tok = t;
                f->n_tok += p->n_tok;
                p->text[0] = 0;
                p->n_tok = 0;
            } else {
                free(x);
                free(t);
            }
        } else {
            j = i;
        }
        i--;
    }
    /* appended: into the word before */
    i = 0;
    j = 1;
    while ((uint32_t)j < n) {
        struct word *p = &al[i], *f = &al[j];
        size_t pl = strlen(p->text);
        if (!(pl && p->text[pl - 1] == ' ') &&
            in(f->text, strlen(f->text), APPEND)) {
            if (append(p, f->text, strlen(f->text), f->tok, f->n_tok) == 0) {
                f->text[0] = 0;
                f->n_tok = 0;
            }
        } else {
            i = j;
        }
        j++;
    }
}

static double round2(double x)
{
    return round(x * 100.0) / 100.0;
}

int janas_whisper_align(struct janas_whisper *w, int lang, int translate,
                        struct janas_whisper_seg *segs, uint32_t first,
                        uint32_t n_segs, double offset, uint32_t frames,
                        double *last_speech)
{
    if (first >= n_segs)
        return 0;
    /* the text tokens of the window's segments */
    uint32_t n_text = 0;
    for (uint32_t s = first; s < n_segs; s++)
        for (uint32_t i = 0; i < segs[s].n_tokens; i++)
            n_text += segs[s].tokens[i] < w->eot;
    int32_t sot_seq[3];
    uint32_t n_sot = 0;
    sot_seq[n_sot++] = w->sot;
    if (w->multilingual) {
        sot_seq[n_sot++] = w->sot + 1 + (lang >= 0 ? lang : 0);
        sot_seq[n_sot++] = translate ? w->translate : w->transcribe;
    }
    uint32_t n = n_sot + 1 + n_text + 1;
    if (!n_text || n > w->t_ctx)
        return 0;
    uint32_t nf = frames / 2;
    if (nf > w->a_ctx)
        nf = w->a_ctx;
    if (!nf)
        return 0;
    int32_t *tok = malloc(sizeof(int32_t) * n);
    float *logits = malloc(sizeof(float) * (size_t)n * w->vocab);
    float *cap = malloc(sizeof(float) * (size_t)w->n_aheads * n * w->a_ctx);
    float *mat = malloc(sizeof(float) * (size_t)n * nf);
    float *probs = malloc(sizeof(float) * (n_text + 1));
    struct word *ws = NULL;
    long nw = 0;
    uint32_t *ti = NULL, *fi = NULL;
    int rc = -1;
    if (!tok || !logits || !cap || !mat || !probs)
        goto end;
    uint32_t k = 0;
    for (uint32_t i = 0; i < n_sot; i++)
        tok[k++] = sot_seq[i];
    tok[k++] = w->no_timestamps;
    for (uint32_t s = first; s < n_segs; s++)
        for (uint32_t i = 0; i < segs[s].n_tokens; i++)
            if (segs[s].tokens[i] < w->eot)
                tok[k++] = segs[s].tokens[i];
    tok[k++] = w->eot;
    w->cap = cap;
    int dec = janas_whisper_decode(w, tok, n, 0, logits, 1);
    w->cap = NULL;
    if (dec != 0)
        goto end;
    /* each text token's probability among the text tokens */
    for (uint32_t i = 0; i < n_text; i++) {
        const float *l = logits + (size_t)(n_sot + i) * w->vocab;
        float mx = -INFINITY;
        for (int32_t v = 0; v < w->eot; v++)
            if (l[v] > mx)
                mx = l[v];
        double sum = 0;
        for (int32_t v = 0; v < w->eot; v++)
            sum += exp((double)l[v] - mx);
        probs[i] = (float)(exp((double)l[tok[n_sot + 1 + i]] - mx) / sum);
    }
    /* the heads' softmax over the frames, standardized over the tokens,
       median-filtered, averaged */
    memset(mat, 0, sizeof(float) * (size_t)n * nf);
    float *row = malloc(sizeof(float) * (nf + MEDFILT));
    float *col = malloc(sizeof(float) * n);
    if (!row || !col) {
        free(row);
        free(col);
        goto end;
    }
    for (uint32_t h = 0; h < w->n_aheads; h++) {
        float *c = cap + (size_t)h * n * w->a_ctx;
        for (uint32_t i = 0; i < n; i++) {
            float *r = c + (size_t)i * w->a_ctx, mx = -INFINITY;
            for (uint32_t f = 0; f < nf; f++)
                if (r[f] > mx)
                    mx = r[f];
            float sum = 0;
            for (uint32_t f = 0; f < nf; f++) {
                r[f] = expf(r[f] - mx);
                sum += r[f];
            }
            for (uint32_t f = 0; f < nf; f++)
                r[f] /= sum;
        }
        for (uint32_t f = 0; f < nf; f++) {
            double mean = 0, var = 0;
            for (uint32_t i = 0; i < n; i++)
                mean += c[(size_t)i * w->a_ctx + f];
            mean /= n;
            for (uint32_t i = 0; i < n; i++) {
                double dv = c[(size_t)i * w->a_ctx + f] - mean;
                var += dv * dv;
            }
            float sd = (float)sqrt(var / n);
            for (uint32_t i = 0; i < n; i++) {
                float *v = &c[(size_t)i * w->a_ctx + f];
                *v = (float)((*v - (float)mean) / sd);
            }
        }
        int pad = MEDFILT / 2;
        for (uint32_t i = 0; i < n; i++) {
            const float *r = c + (size_t)i * w->a_ctx;
            float *out = mat + (size_t)i * nf;
            if (nf <= (uint32_t)pad) {
                for (uint32_t f = 0; f < nf; f++)
                    out[f] += r[f];
                continue;
            }
            /* reflected at both ends */
            for (int f = -pad; f < (int)nf + pad; f++) {
                int g = f < 0 ? -f : f >= (int)nf ? 2 * ((int)nf - 1) - f : f;
                row[f + pad] = r[g];
            }
            for (uint32_t f = 0; f < nf; f++) {
                float win[MEDFILT];
                memcpy(win, row + f, sizeof win);
                qsort(win, MEDFILT, sizeof(float), cmp_float);
                out[f] += win[MEDFILT / 2];
            }
        }
    }
    free(row);
    free(col);
    for (size_t i = 0; i < (size_t)n * nf; i++)
        mat[i] /= (float)w->n_aheads;
    /* the rows of "no timestamps" and the text: the time path */
    long np = dtw(mat + (size_t)n_sot * nf, n_text + 1, nf, &ti, &fi);
    if (np < 0)
        goto end;
    double *jump = malloc(sizeof(double) * (n_text + 2));
    if (!jump)
        goto end;
    uint32_t nj = 0;
    for (long p = 0; p < np; p++)
        if (p == 0 || ti[p] != ti[p - 1])
            jump[nj++] = fi[p] / TOKENS_PER_SECOND;
    /* the words, the end of text last */
    const char *code = NULL;
    char lc[16] = "en";
    if (w->multilingual && lang >= 0)
        janas_whisper_lang(w, (uint32_t)lang, lc, sizeof lc);
    code = lc;
    int no_spaces = !strcmp(code, "zh") || !strcmp(code, "ja") ||
                    !strcmp(code, "th") || !strcmp(code, "lo") ||
                    !strcmp(code, "my") || !strcmp(code, "yue");
    nw = split_words(w, tok + n_sot + 1, n_text + 1, no_spaces, &ws);
    if (nw < 0) {
        free(jump);
        goto end;
    }
    if (nw <= 1) {
        free(jump);
        rc = 0;
        goto end;
    }
    uint32_t at = 0, nal = (uint32_t)nw - 1;
    for (uint32_t i = 0; i < nal; i++) {
        uint32_t a = at, b = at + ws[i].n_tok;
        ws[i].t0 = a < nj ? jump[a] : jump[nj - 1];
        ws[i].t1 = b < nj ? jump[b] : jump[nj - 1];
        double sp = 0;
        for (uint32_t t = a; t < b && t < n_text; t++)
            sp += probs[t];
        ws[i].prob = b > a ? (float)(sp / (b - a)) : 0;
        at = b;
    }
    free(jump);
    /* OpenAI's corrections */
    double *dur = malloc(sizeof(double) * nal);
    if (!dur)
        goto end;
    uint32_t nd = 0;
    for (uint32_t i = 0; i < nal; i++)
        if (ws[i].t1 - ws[i].t0 != 0)
            dur[nd++] = ws[i].t1 - ws[i].t0;
    double median = 0;
    if (nd) {
        qsort(dur, nd, sizeof(double), cmp_double);
        median = nd & 1 ? dur[nd / 2] : (dur[nd / 2 - 1] + dur[nd / 2]) / 2;
    }
    free(dur);
    if (median > 0.7)
        median = 0.7;
    double max_dur = median * 2;
    if (nd)
        for (uint32_t i = 1; i < nal; i++)
            if (ws[i].t1 - ws[i].t0 > max_dur) {
                if (in(ws[i].text, strlen(ws[i].text), SENTENCE_END))
                    ws[i].t1 = ws[i].t0 + max_dur;
                else if (in(ws[i - 1].text, strlen(ws[i - 1].text),
                            SENTENCE_END))
                    ws[i].t0 = ws[i].t1 - max_dur;
            }
    merge_punctuation(ws, nal);
    /* into the segments */
    uint32_t wi = 0;
    for (uint32_t s = first; s < n_segs; s++) {
        struct janas_whisper_seg *sg = &segs[s];
        uint32_t seg_text = 0, saved = 0, cnt = 0;
        for (uint32_t i = 0; i < sg->n_tokens; i++)
            seg_text += sg->tokens[i] < w->eot;
        sg->words = calloc(nal ? nal : 1, sizeof *sg->words);
        if (!sg->words)
            goto end;
        while (wi < nal && saved < seg_text) {
            struct word *t = &ws[wi];
            if (t->text[0]) {
                struct janas_whisper_word *o = &sg->words[cnt];
                o->text = strdup(t->text);
                if (!o->text)
                    goto end;
                o->t0 = round2(offset + t->t0);
                o->t1 = round2(offset + t->t1);
                o->probability = t->prob;
                cnt++;
                sg->n_words = cnt;
            }
            saved += t->n_tok;
            wi++;
        }
        if (!cnt)
            continue;
        struct janas_whisper_word *wd = sg->words;
        if (wd[0].t1 - *last_speech > median * 4 &&
            (wd[0].t1 - wd[0].t0 > max_dur ||
             (cnt > 1 && wd[1].t1 - wd[0].t0 > max_dur * 2))) {
            if (cnt > 1 && wd[1].t1 - wd[1].t0 > max_dur) {
                double boundary = wd[1].t1 / 2 > wd[1].t1 - max_dur
                                      ? wd[1].t1 / 2
                                      : wd[1].t1 - max_dur;
                wd[0].t1 = wd[1].t0 = boundary;
            }
            wd[0].t0 = wd[0].t1 - max_dur > 0 ? wd[0].t1 - max_dur : 0;
        }
        if (sg->t0 < wd[0].t1 && sg->t0 - 0.5 > wd[0].t0) {
            double v = wd[0].t1 - median < sg->t0 ? wd[0].t1 - median : sg->t0;
            wd[0].t0 = v > 0 ? v : 0;
        } else {
            sg->t0 = wd[0].t0;
        }
        if (sg->t1 > wd[cnt - 1].t0 && sg->t1 + 0.5 < wd[cnt - 1].t1) {
            double v = wd[cnt - 1].t0 + median;
            wd[cnt - 1].t1 = v > sg->t1 ? v : sg->t1;
        } else {
            sg->t1 = wd[cnt - 1].t1;
        }
        *last_speech = sg->t1;
    }
    rc = 0;
end:
    if (ws)
        words_free(ws, nw > 0 ? (uint32_t)nw : 0);
    free(ti);
    free(fi);
    free(tok);
    free(logits);
    free(cap);
    free(mat);
    free(probs);
    return rc;
}
