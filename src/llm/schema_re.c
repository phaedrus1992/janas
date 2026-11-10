/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schema_re.c - a regular expression as grammar rules (see schema_impl.h).
 *
 * The expression is read into a tree - characters and classes, sequences,
 * alternatives, repetitions - and each part is made a rule that matches it
 * and then goes on with the rule after it, so every rule recurses on its
 * right only, as the grammars want. A bounded repetition is written out
 * ({2,4}: two, then up to two more), an open one loops on itself.
 *
 * The text is the inside of a JSON string, so the bytes that would end it or
 * start an escape (the quote, the backslash, the control characters) are
 * matched as their escapes (\" \\ \n \u0001); a class that holds no byte
 * at all is not read. Bytes are matched as bytes: a class of non-ASCII
 * characters is not read.
 */
#include "schema_impl.h"

#include <stdlib.h>
#include <string.h>

#define NONE UINT32_MAX
#define REP_MAX 256 /* a bounded repetition written out up to this */

enum { RE_SET, RE_SEQ, RE_ALT, RE_REP };

struct re {
    int kind;
    uint8_t set[32];
    struct re **kids;
    int n;
    int lo, hi; /* RE_REP: hi < 0 open */
};

struct parser {
    const char *p, *end;
    int bad; /* not an expression this reads */
};

static struct re *node(int kind)
{
    struct re *r = calloc(1, sizeof(*r));
    if (r)
        r->kind = kind;
    return r;
}

static void re_free(struct re *r)
{
    if (!r)
        return;
    for (int i = 0; i < r->n; i++)
        re_free(r->kids[i]);
    free(r->kids);
    free(r);
}

static int add_kid(struct re *r, struct re *k)
{
    if (!k)
        return -1;
    struct re **t = realloc(r->kids, (size_t)(r->n + 1) * sizeof(*t));
    if (!t) {
        re_free(k);
        return -1;
    }
    r->kids = t;
    r->kids[r->n++] = k;
    return 0;
}

/* What a JSON string may hold as it is. */
static void json_plain(uint8_t set[32])
{
    uint8_t keep[32];
    janas_set_clear(keep);
    janas_set_add(keep, 0x20, 0xff);
    keep['"' >> 3] &= (uint8_t)~(1u << ('"' & 7));
    keep['\\' >> 3] &= (uint8_t)~(1u << ('\\' & 7));
    for (int i = 0; i < 32; i++)
        set[i] &= keep[i];
}

static int set_empty(const uint8_t set[32])
{
    for (int i = 0; i < 32; i++)
        if (set[i])
            return 0;
    return 1;
}

/* The escape a JSON string writes byte c with (c not plain). */
static size_t escape_of(uint8_t c, char out[8])
{
    static const char hex[] = "0123456789abcdef";
    const char *e = c == '"'    ? "\\\""
                    : c == '\\' ? "\\\\"
                    : c == '\n' ? "\\n"
                    : c == '\t' ? "\\t"
                    : c == '\r' ? "\\r"
                    : c == '\b' ? "\\b"
                    : c == '\f' ? "\\f"
                                : NULL;
    if (e) {
        memcpy(out, e, 2);
        return 2;
    }
    memcpy(out, "\\u00", 4);
    out[4] = hex[c >> 4];
    out[5] = hex[c & 15];
    return 6;
}

static void class_escape(char c, uint8_t set[32], int *ok)
{
    switch (c) {
    case 'd':
        janas_set_add(set, '0', '9');
        break;
    case 'w':
        janas_set_add(set, '0', '9');
        janas_set_add(set, 'a', 'z');
        janas_set_add(set, 'A', 'Z');
        janas_set_add(set, '_', '_');
        break;
    case 's':
        janas_set_add(set, ' ', ' ');
        janas_set_add(set, '\t', '\r');
        break;
    case 'D':
    case 'W':
    case 'S': {
        uint8_t s[32];
        janas_set_clear(s);
        class_escape((char)(c + 32), s, ok);
        janas_set_invert(s);
        for (int i = 0; i < 32; i++)
            set[i] |= s[i];
        break;
    }
    case 'n':
        janas_set_add(set, '\n', '\n');
        break;
    case 't':
        janas_set_add(set, '\t', '\t');
        break;
    case 'r':
        janas_set_add(set, '\r', '\r');
        break;
    default:
        /* an escaped punctuation character: itself; letters and digits
           with another meaning (\b, \p{...}, back references) are not read */
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9'))
            *ok = 0;
        else
            janas_set_add(set, (uint8_t)c, (uint8_t)c);
    }
}

/* [...] after its '[' */
static struct re *parse_class(struct parser *ps)
{
    struct re *r = node(RE_SET);
    if (!r)
        return NULL;
    int neg = ps->p < ps->end && *ps->p == '^';
    if (neg)
        ps->p++;
    int ok = 1, first = 1;
    while (ps->p < ps->end && (*ps->p != ']' || first)) {
        first = 0;
        uint8_t c = (uint8_t)*ps->p++;
        if (c == '\\' && ps->p < ps->end) {
            char e = *ps->p++;
            if (strchr("dwsDWS", e)) { /* a class of its own: no range */
                class_escape(e, r->set, &ok);
                continue;
            }
            c = e == 'n'   ? '\n'
                : e == 't' ? '\t'
                : e == 'r' ? '\r'
                           : (uint8_t)e;
            if (!strchr("ntr", e) &&
                ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') ||
                 (e >= '0' && e <= '9')))
                ok = 0;
        }
        uint8_t hi = c;
        if (ps->p + 1 < ps->end && *ps->p == '-' && ps->p[1] != ']') {
            ps->p++;
            hi = (uint8_t)*ps->p++;
            if (hi == '\\' && ps->p < ps->end)
                hi = (uint8_t)*ps->p++;
            if (hi < c)
                ok = 0;
        }
        if (ok)
            janas_set_add(r->set, c, hi);
    }
    if (ps->p >= ps->end)
        ok = 0;
    else
        ps->p++; /* ']' */
    if (!ok) {
        ps->bad = 1;
        re_free(r);
        return NULL;
    }
    if (neg)
        janas_set_invert(r->set);
    if (set_empty(r->set)) {
        ps->bad = 1;
        re_free(r);
        return NULL;
    }
    return r;
}

static struct re *parse_alt(struct parser *ps);

static struct re *parse_atom(struct parser *ps)
{
    char c = *ps->p++;
    if (c == '(') {
        if (ps->p + 1 < ps->end && ps->p[0] == '?' && ps->p[1] == ':')
            ps->p += 2;
        struct re *r = parse_alt(ps);
        if (!r || ps->p >= ps->end || *ps->p != ')') {
            ps->bad = 1;
            re_free(r);
            return NULL;
        }
        ps->p++;
        return r;
    }
    if (c == '[')
        return parse_class(ps);
    struct re *r = node(RE_SET);
    if (!r)
        return NULL;
    int ok = 1;
    if (c == '.') {
        janas_set_add(r->set, 0x20, 0xff);
    } else if (c == '\\' && ps->p < ps->end) {
        class_escape(*ps->p++, r->set, &ok);
    } else if (strchr(")|*+?{}^$", c)) {
        ok = 0; /* an anchor or a misplaced operator */
    } else {
        janas_set_add(r->set, (uint8_t)c, (uint8_t)c);
    }
    if (!ok) {
        ps->bad = 1;
        re_free(r);
        return NULL;
    }
    return r;
}

static int number(struct parser *ps, int *v)
{
    if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9')
        return -1;
    long x = 0;
    while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9' && x < 100000)
        x = x * 10 + (*ps->p++ - '0');
    *v = (int)x;
    return 0;
}

static struct re *parse_rep(struct parser *ps)
{
    struct re *a = parse_atom(ps);
    while (a && ps->p < ps->end && strchr("*+?{", *ps->p)) {
        int lo, hi;
        char q = *ps->p++;
        if (q == '*')
            lo = 0, hi = -1;
        else if (q == '+')
            lo = 1, hi = -1;
        else if (q == '?')
            lo = 0, hi = 1;
        else {
            if (number(ps, &lo) != 0) {
                ps->bad = 1;
                re_free(a);
                return NULL;
            }
            hi = lo;
            if (ps->p < ps->end && *ps->p == ',') {
                ps->p++;
                hi = -1;
                if (ps->p < ps->end && *ps->p != '}' && number(ps, &hi) != 0)
                    hi = -2;
            }
            if (ps->p >= ps->end || *ps->p != '}' || hi == -2 ||
                (hi >= 0 && hi < lo) || lo > REP_MAX || hi > REP_MAX) {
                ps->bad = 1;
                re_free(a);
                return NULL;
            }
            ps->p++;
        }
        if (ps->p < ps->end && *ps->p == '?')
            ps->p++; /* lazy: the same language */
        struct re *r = node(RE_REP);
        if (!r) {
            re_free(a);
            return NULL;
        }
        r->lo = lo;
        r->hi = hi;
        if (add_kid(r, a) != 0) {
            re_free(r);
            return NULL;
        }
        a = r;
    }
    return a;
}

static struct re *parse_seq(struct parser *ps)
{
    struct re *s = node(RE_SEQ);
    if (!s)
        return NULL;
    while (ps->p < ps->end && *ps->p != '|' && *ps->p != ')') {
        if (add_kid(s, parse_rep(ps)) != 0) {
            ps->bad = 1;
            re_free(s);
            return NULL;
        }
    }
    return s;
}

static struct re *parse_alt(struct parser *ps)
{
    struct re *a = node(RE_ALT);
    if (!a)
        return NULL;
    for (;;) {
        if (add_kid(a, parse_seq(ps)) != 0) {
            ps->bad = 1;
            re_free(a);
            return NULL;
        }
        if (ps->p < ps->end && *ps->p == '|') {
            ps->p++;
            continue;
        }
        return a;
    }
}

/* Whether r can match nothing. */
static int nullable(const struct re *r)
{
    switch (r->kind) {
    case RE_SET:
        return 0;
    case RE_SEQ:
        for (int i = 0; i < r->n; i++)
            if (!nullable(r->kids[i]))
                return 0;
        return 1;
    case RE_ALT:
        for (int i = 0; i < r->n; i++)
            if (nullable(r->kids[i]))
                return 1;
        return 0;
    default:
        return r->lo == 0 || nullable(r->kids[0]);
    }
}

/* An open repetition of what can be empty would be a rule that reaches
   itself without reading (left recursion, which the grammars refuse). */
static int loops_empty(const struct re *r)
{
    if (r->kind == RE_REP && r->hi < 0 && nullable(r->kids[0]))
        return 1;
    for (int i = 0; i < r->n; i++)
        if (loops_empty(r->kids[i]))
            return 1;
    return 0;
}

/* A rule matching r and then going on with k. */
static uint32_t make(struct janas_gbuild *b, const struct re *r, uint32_t k)
{
    switch (r->kind) {
    case RE_SET: {
        uint32_t x = janas_gb_rule(b);
        uint8_t plain[32];
        memcpy(plain, r->set, sizeof plain);
        json_plain(plain);
        if (!set_empty(plain)) {
            uint32_t a = janas_gb_alt(b, x);
            janas_gb_set(b, a, plain);
            janas_gb_ref(b, a, k);
        }
        for (int c = 0; c < 256; c++)
            if ((r->set[c >> 3] >> (c & 7) & 1) &&
                !(plain[c >> 3] >> (c & 7) & 1)) {
                char e[8];
                size_t n = escape_of((uint8_t)c, e);
                uint32_t a = janas_gb_alt(b, x);
                janas_gb_lit(b, a, e, n);
                janas_gb_ref(b, a, k);
            }
        return x;
    }
    case RE_SEQ:
        for (int i = r->n - 1; i >= 0; i--)
            k = make(b, r->kids[i], k);
        return k;
    case RE_ALT: {
        if (r->n == 1)
            return make(b, r->kids[0], k);
        uint32_t x = janas_gb_rule(b);
        for (int i = 0; i < r->n; i++)
            janas_gb_ref(b, janas_gb_alt(b, x), make(b, r->kids[i], k));
        return x;
    }
    default: { /* RE_REP */
        const struct re *in = r->kids[0];
        uint32_t c = k;
        if (r->hi < 0) { /* L = in L | k */
            c = janas_gb_rule(b);
            janas_gb_ref(b, janas_gb_alt(b, c), make(b, in, c));
            janas_gb_ref(b, janas_gb_alt(b, c), k);
        } else {
            for (int i = r->lo; i < r->hi; i++) { /* O = in O' | k */
                uint32_t o = janas_gb_rule(b);
                janas_gb_ref(b, janas_gb_alt(b, o), make(b, in, c));
                janas_gb_ref(b, janas_gb_alt(b, o), k);
                c = o;
            }
        }
        for (int i = 0; i < r->lo; i++)
            c = make(b, in, c);
        return c;
    }
    }
}

uint32_t janas_jsg_regex(struct janas_gbuild *b, const char *expr, size_t n,
                         uint32_t then)
{
    struct parser ps = {.p = expr, .end = expr + n};
    struct re *r = parse_alt(&ps);
    if (!r || ps.bad || ps.p != ps.end || loops_empty(r)) {
        re_free(r);
        return NONE;
    }
    uint32_t x = make(b, r, then);
    re_free(r);
    return x;
}

/* Any bytes a JSON string holds as they are, as many as wanted, then then. */
static uint32_t any_then(struct janas_gbuild *b, uint32_t then)
{
    uint8_t any[32];
    janas_set_clear(any);
    janas_set_add(any, 0x20, 0xff);
    json_plain(any);
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_set(b, a, any);
    janas_gb_ref(b, a, r);
    janas_gb_ref(b, janas_gb_alt(b, r), then);
    return r;
}

uint32_t janas_jsg_pattern(struct janas_gbuild *b, const char *expr, size_t n,
                           uint32_t then)
{
    /* Written from the string's start even when it may be found later: a
       model let to write before the match may never come to it, and could
       not close the string meanwhile. A string that begins with the match
       contains it. */
    if (n && expr[0] == '^')
        expr++, n--;
    /* a '$' at the end, not escaped */
    size_t bs = 0;
    while (n >= 2 + bs && expr[n - 2 - bs] == '\\')
        bs++;
    int end = n && expr[n - 1] == '$' && bs % 2 == 0;
    if (end)
        n--;
    uint32_t tail = end ? then : any_then(b, then);
    return janas_jsg_regex(b, expr, n, tail);
}
