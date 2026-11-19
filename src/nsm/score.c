/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * score.c - an explication's words counted (score.h).
 */
#include "score.h"

#include <stdlib.h>
#include <string.h>

/* Python's \w for the characters an explication has: ASCII letters,
   digits and '_', and any character past ASCII except the punctuation
   models write (curly quotes, dashes, the ellipsis). */
static size_t word_char(const unsigned char *s, size_t n)
{
    if (s[0] < 0x80)
        return (s[0] >= 'a' && s[0] <= 'z') || (s[0] >= 'A' && s[0] <= 'Z') ||
                       (s[0] >= '0' && s[0] <= '9') || s[0] == '_'
                   ? 1
                   : 0;
    size_t len = s[0] >= 0xf0 ? 4 : s[0] >= 0xe0 ? 3 : 2;
    if (len > n)
        return 0;
    if (len == 3 && s[0] == 0xe2 && s[1] == 0x80 &&
        ((s[2] >= 0x90 && s[2] <= 0xa7) || (s[2] >= 0xb0 && s[2] <= 0xbf)))
        return 0; /* U+2010..2027, U+2030..203F: dashes, quotes, ... */
    if (len == 2 && s[0] == 0xc2 &&
        (s[1] == 0xab || s[1] == 0xbb || s[1] == 0xa0 || s[1] == 0xb7))
        return 0; /* guillemets, no-break space, middle dot */
    return len;
}

static int is_space(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

static void copy_word(const char *w, char *out, size_t cap)
{
    size_t l = strlen(w);
    if (l >= cap)
        l = cap - 1;
    memcpy(out, w, l);
    out[l] = 0;
}

static const char *const irregular[][2] = {
    {"was", "be"},        {"were", "be"},        {"is", "be"},
    {"are", "be"},        {"been", "be"},        {"am", "be"},
    {"had", "have"},      {"has", "have"},       {"did", "do"},
    {"does", "do"},       {"done", "do"},        {"said", "say"},
    {"says", "say"},      {"saw", "see"},        {"seen", "see"},
    {"felt", "feel"},     {"knew", "know"},      {"known", "know"},
    {"thought", "think"}, {"made", "make"},      {"went", "go"},
    {"gone", "go"},       {"came", "come"},      {"took", "take"},
    {"taken", "take"},    {"children", "child"}, {"men", "man"},
    {"women", "woman"},   {"feet", "foot"},      {"teeth", "tooth"},
    {"mice", "mouse"},    {"geese", "goose"},    {"left", "leave"},
    {"got", "get"},       {"gave", "give"},      {"given", "give"},
    {"ran", "run"},       {"ate", "eat"},        {"eaten", "eat"},
    {"fell", "fall"},     {"fallen", "fall"},    {"dug", "dig"}};

void nsm_lemma(const char *w, char *out, size_t cap)
{
    for (size_t i = 0; i < sizeof(irregular) / sizeof(irregular[0]); i++)
        if (!strcmp(w, irregular[i][0])) {
            copy_word(irregular[i][1], out, cap);
            return;
        }
    static const char *const suf[][2] = {
        {"ies", "y"},   {"ied", "y"}, {"sses", "ss"}, {"ches", "ch"},
        {"shes", "sh"}, {"xes", "x"}, {"ing", ""},    {"ed", ""},
        {"es", "e"},    {"s", ""}};
    size_t n = strlen(w);
    size_t ends_ss = n >= 2 && w[n - 1] == 's' && w[n - 2] == 's';
    for (size_t i = 0; i < sizeof(suf) / sizeof(suf[0]); i++) {
        size_t ls = strlen(suf[i][0]), lr = strlen(suf[i][1]);
        /* a stem of two letters before the endings that say what they took
           off (flies, boxes), of three before the bare ones (thing, eyes) */
        size_t min = i < 6 ? 2 : 3;
        if (n < ls + min || strcmp(w + n - ls, suf[i][0]) || ends_ss)
            continue;
        size_t b = n - ls;
        if (b + lr >= cap)
            break;
        memcpy(out, w, b);
        memcpy(out + b, suf[i][1], lr);
        b += lr;
        /* a doubled consonant before -ing, -ed: stopped, running */
        if ((i == 6 || i == 7) && b > 3 && out[b - 1] == out[b - 2])
            b--;
        out[b] = 0;
        return;
    }
    copy_word(w, out, cap);
}

static int ends(const char *w, const char *suffix)
{
    size_t n = strlen(w), ls = strlen(suffix);
    return n > ls && !strcmp(w + n - ls, suffix);
}

/* The rules take a silent e off with -ing and -ed (making, promised: "mak",
   "promis"): one lemma with an e more than the other's matches when the
   other's word had one of those endings. */
static int same_lemma(const char *w, const char *wl, const char *t,
                      const char *tl)
{
    if (!strcmp(wl, tl))
        return 1;
    size_t a = strlen(wl), b = strlen(tl);
    if (a + 1 == b && tl[a] == 'e' && !strncmp(wl, tl, a))
        return ends(w, "ing") || ends(w, "ed");
    if (b + 1 == a && wl[b] == 'e' && !strncmp(wl, tl, b))
        return ends(t, "ing") || ends(t, "ed");
    return 0;
}

void nsm_score_text(enum nsm_lex lex, const char *text, size_t n,
                    const char *target, struct nsm_score *s)
{
    memset(s, 0, sizeof(*s));
    /* lower-cased, the punctuation dropped, then split */
    char *buf = malloc(n + 1);
    char **word = malloc(sizeof(char *) * (n / 2 + 2));
    if (!buf || !word) {
        free(buf);
        free(word);
        return;
    }
    size_t w = 0, nw = 0;
    int in = 0;
    for (size_t i = 0; i < n;) {
        const unsigned char *c = (const unsigned char *)text + i;
        size_t k = word_char(c, n - i);
        int keep_apos = lex == NSM_LEX_FULL &&
                        (c[0] == '\'' || (n - i >= 3 && c[0] == 0xe2 &&
                                          c[1] == 0x80 && c[2] == 0x99)) &&
                        in;
        if (k || keep_apos) {
            if (!in) {
                word[nw++] = buf + w;
                in = 1;
            }
            if (keep_apos) {
                buf[w++] = '\'';
                i += c[0] == '\'' ? 1 : 3;
                continue;
            }
            for (size_t j = 0; j < k; j++)
                buf[w++] =
                    (char)(k == 1 && c[j] >= 'A' && c[j] <= 'Z' ? c[j] + 32
                                                                : c[j]);
            i += k;
        } else if (is_space(c[0])) {
            if (in)
                buf[w++] = 0;
            in = 0;
            i++;
        } else { /* punctuation: removed, the word goes on */
            i += c[0] >= 0xf0 ? 4 : c[0] >= 0xe0 ? 3 : c[0] >= 0xc0 ? 2 : 1;
        }
    }
    if (in)
        buf[w++] = 0;
    enum nsm_kind *kind = malloc(sizeof(*kind) * (nw + 1));
    size_t *span = malloc(sizeof(*span) * (nw + 1)), nu = 0;
    if (kind && span) {
        nsm_classify(lex, word, nw, kind, span, nw + 1, &nu);
        for (size_t u = 0; u < nu; u++) {
            if (kind[u] == NSM_PRIME)
                s->primes += span[u];
            else if (kind[u] == NSM_STOP)
                s->stops += span[u];
            else
                s->molecules += span[u];
        }
        s->words = nw;
        if (nw) {
            s->primes_ratio = (double)s->primes / (double)nw;
            s->molecules_ratio = (double)s->molecules / (double)nw;
        }
        s->legality = 10 * (s->primes_ratio - s->molecules_ratio);
        char tl[128], wl[128], low[128];
        size_t lt = strlen(target);
        if (lt >= sizeof(low))
            lt = sizeof(low) - 1;
        for (size_t i = 0; i < lt; i++)
            low[i] =
                (char)(target[i] >= 'A' && target[i] <= 'Z' ? target[i] + 32
                                                            : target[i]);
        low[lt] = 0;
        nsm_lemma(low, tl, sizeof(tl));
        for (size_t i = 0; i < nw && !s->circular; i++) {
            nsm_lemma(word[i], wl, sizeof(wl));
            s->circular = same_lemma(word[i], wl, low, tl);
        }
    }
    free(kind);
    free(span);
    free(word);
    free(buf);
}
