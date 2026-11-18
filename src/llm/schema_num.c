/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schema_num.c - numbers with bounds as grammar rules (see schema_impl.h).
 *
 * An integer between two bounds, and a multiple of m, is a regular language
 * of its digits: read left to right, what matters is whether the digits so
 * far are still those of the lower bound, still those of the upper, and the
 * remainder by m. Each such state is a rule whose alternatives are the
 * digits allowed there (those leading to the same state in one set), so the
 * grammar is as exact as a check after the fact would be, and it never
 * lets a number start that cannot end inside the bounds.
 *
 * A real number with bounds is its integer part with such a rule and a
 * fraction where any fraction keeps it inside: below an integer upper
 * bound any fraction, at it none (5 for <= 5, never 5.2); above an
 * exclusive integer lower bound a fraction that is not all zeros. Where a
 * bound is not an integer the part between it and the next integer is left
 * out (>= 1.5 writes 2 and more): what is written is always valid, though
 * not every valid number can be written. No exponent.
 */
#include "schema_impl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NONE UINT32_MAX
#define MAG_DIGITS 18 /* magnitudes below 10^18: an int64 holds them */
#define MAG_MAX 999999999999999999ull
#define MULT_MAX 50 /* multipleOf enforced up to this (states: m each) */
#define FRAC_MAX 19 /* a fraction's digits at most */

/* The digit-by-digit rules of one length, made as they are asked. */
struct dp {
    struct janas_gbuild *b;
    int len;
    char lo[MAG_DIGITS + 1], hi[MAG_DIGITS + 1];
    uint32_t m;
    uint32_t *edge;  /* [pos][tl][th][r], tl or th set */
    uint32_t *free_; /* shared: [remaining][r], neither bound tight */
};

static uint32_t state(struct dp *d, int pos, int tl, int th, uint32_t r);

/* Digits left, neither bound in the way: any digits, the remainder r so
   far. Shared by every length (the caller keeps the table). */
static uint32_t free_state(struct dp *d, int left, uint32_t r)
{
    uint32_t *slot = &d->free_[(size_t)left * d->m + r];
    if (*slot != NONE)
        return *slot;
    uint32_t rule = *slot = janas_gb_rule(d->b);
    if (left == 0) {
        if (r == 0)
            janas_gb_alt(d->b, rule); /* the end */
        return rule;
    }
    /* the digits by the remainder they lead to, one set each */
    int done[10] = {0};
    for (int dg = 0; dg <= 9; dg++) {
        if (done[dg])
            continue;
        uint32_t nr = (r * 10 + (uint32_t)dg) % d->m;
        uint8_t set[32];
        janas_set_clear(set);
        for (int e = dg; e <= 9; e++)
            if (!done[e] && (r * 10 + (uint32_t)e) % d->m == nr) {
                janas_set_add(set, (uint8_t)('0' + e), (uint8_t)('0' + e));
                done[e] = 1;
            }
        uint32_t a = janas_gb_alt(d->b, rule);
        janas_gb_set(d->b, a, set);
        janas_gb_ref(d->b, a, free_state(d, left - 1, nr));
    }
    return rule;
}

static uint32_t state(struct dp *d, int pos, int tl, int th, uint32_t r)
{
    if (!tl && !th)
        return free_state(d, d->len - pos, r);
    size_t at = (((size_t)pos * 2 + (size_t)tl) * 2 + (size_t)th) * d->m + r;
    if (d->edge[at] != NONE)
        return d->edge[at];
    uint32_t rule = d->edge[at] = janas_gb_rule(d->b);
    if (pos == d->len) {
        if (r == 0)
            janas_gb_alt(d->b, rule);
        return rule;
    }
    int from = tl ? d->lo[pos] - '0' : 0, to = th ? d->hi[pos] - '0' : 9;
    /* runs of digits with the same next state, one alternative each */
    for (int dg = from; dg <= to;) {
        int ntl = tl && dg == d->lo[pos] - '0';
        int nth = th && dg == d->hi[pos] - '0';
        uint32_t nr = (r * 10 + (uint32_t)dg) % d->m;
        int e = dg;
        if (d->m == 1)
            while (e + 1 <= to && (tl && e + 1 == d->lo[pos] - '0') == ntl &&
                   (th && e + 1 == d->hi[pos] - '0') == nth)
                e++;
        uint8_t set[32];
        janas_set_clear(set);
        janas_set_add(set, (uint8_t)('0' + dg), (uint8_t)('0' + e));
        uint32_t a = janas_gb_alt(d->b, rule);
        janas_gb_set(d->b, a, set);
        janas_gb_ref(d->b, a, state(d, pos + 1, ntl, nth, nr));
        dg = e + 1;
    }
    return rule;
}

static int ndigits(uint64_t x)
{
    int n = 1;
    while (x >= 10) {
        x /= 10;
        n++;
    }
    return n;
}

/* Alternatives of rule `into`: the magnitudes in [a, b] (b <= MAG_MAX)
   that are multiples of m, as their digits without leading zeros. */
static int magnitudes(struct janas_gbuild *b, uint32_t into, uint64_t a,
                      uint64_t hi, uint32_t m)
{
    if (a > hi)
        return 0;
    struct dp d = {.b = b, .m = m};
    size_t nfree = (size_t)(MAG_DIGITS + 1) * m;
    size_t nedge = (size_t)(MAG_DIGITS + 1) * 4 * m;
    d.free_ = malloc(nfree * sizeof(uint32_t));
    d.edge = malloc(nedge * sizeof(uint32_t));
    if (!d.free_ || !d.edge) {
        free(d.free_);
        free(d.edge);
        return -1;
    }
    memset(d.free_, 0xff, nfree * sizeof(uint32_t));
    uint64_t p10 = 1;
    for (int len = 1; len <= MAG_DIGITS; len++, p10 *= 10) {
        uint64_t top = len == MAG_DIGITS ? MAG_MAX : p10 * 10 - 1;
        uint64_t lo = len == 1 ? 0 : p10, up = top;
        if (lo < a)
            lo = a;
        if (up > hi)
            up = hi;
        if (lo > up)
            continue;
        d.len = len;
        int nl = ndigits(lo), nh = ndigits(up);
        if (nl != len || nh != len)
            continue; /* cannot happen: lo, up of this length */
        uint64_t x = lo, y = up;
        for (int k = len - 1; k >= 0; k--, x /= 10, y /= 10) {
            d.lo[k] = (char)('0' + x % 10);
            d.hi[k] = (char)('0' + y % 10);
        }
        memset(d.edge, 0xff, nedge * sizeof(uint32_t));
        janas_gb_ref(b, janas_gb_alt(b, into), state(&d, 0, 1, 1, 0));
    }
    free(d.free_);
    free(d.edge);
    return 0;
}

/* A bound as a magnitude: the integer, clamped to what the rules take. */
static uint64_t clamp_mag(double x)
{
    return x <= 0 ? 0 : x >= (double)MAG_MAX ? MAG_MAX : (uint64_t)x;
}

uint32_t janas_jsg_int_bounded(struct janas_gbuild *b,
                               const struct janas_jsg_bounds *bd, int *relaxed)
{
    uint32_t m = 1;
    if (bd->has_mult) {
        double mv = bd->mult;
        if (mv >= 1 && mv <= MULT_MAX && mv == floor(mv))
            m = (uint32_t)mv;
        else if (!(mv > 0 && mv < 1 && floor(1 / mv) == 1 / mv))
            (*relaxed)++; /* 0.5 lets every integer through */
    }
    /* the integers in [L, H] */
    double L = bd->has_lo ? (bd->lo_x ? floor(bd->lo) + 1 : ceil(bd->lo))
                          : -(double)MAG_MAX;
    double H = bd->has_hi ? (bd->hi_x ? ceil(bd->hi) - 1 : floor(bd->hi))
                          : (double)MAG_MAX;
    if (L < -(double)MAG_MAX)
        L = -(double)MAG_MAX;
    if (H > (double)MAG_MAX)
        H = (double)MAG_MAX;
    uint32_t r = janas_gb_rule(b);
    if (L > H)
        return r; /* nothing: the schema allows no integer */
    if (H >= 0 && magnitudes(b, r, clamp_mag(L), clamp_mag(H), m) != 0)
        return NONE;
    if (L < 0) { /* -x: x in [max(1, -H), -L] */
        uint32_t neg = janas_gb_rule(b);
        uint64_t a = H < 0 ? clamp_mag(-H) : 1;
        if (magnitudes(b, neg, a ? a : 1, clamp_mag(-L), m) != 0)
            return NONE;
        uint32_t alt = janas_gb_alt(b, r);
        janas_gb_byte(b, alt, '-');
        janas_gb_ref(b, alt, neg);
    }
    return r;
}

/* "." and up to FRAC_MAX digits, nz: not all zeros. */
static uint32_t fraction(struct janas_gbuild *b, int nz)
{
    uint8_t dg[32], d19[32];
    janas_set_clear(dg);
    janas_set_add(dg, '0', '9');
    janas_set_clear(d19);
    janas_set_add(d19, '1', '9');
    /* D_k: up to k more digits; Z_k: up to k digits, one of them not 0 */
    uint32_t d = janas_gb_empty(b), z = NONE;
    uint32_t dk[FRAC_MAX + 1];
    dk[0] = d;
    for (int k = 1; k <= FRAC_MAX; k++) {
        uint32_t r = janas_gb_rule(b);
        janas_gb_alt(b, r);
        uint32_t a = janas_gb_alt(b, r);
        janas_gb_set(b, a, dg);
        janas_gb_ref(b, a, dk[k - 1]);
        dk[k] = r;
    }
    if (nz) {
        for (int k = 1; k <= FRAC_MAX; k++) {
            uint32_t r = janas_gb_rule(b);
            uint32_t a = janas_gb_alt(b, r);
            janas_gb_set(b, a, d19);
            janas_gb_ref(b, a, dk[k - 1]);
            if (z != NONE) {
                a = janas_gb_alt(b, r);
                janas_gb_byte(b, a, '0');
                janas_gb_ref(b, a, z);
            }
            z = r;
        }
    }
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, '.');
    if (nz) {
        janas_gb_ref(b, a, z);
    } else {
        janas_gb_set(b, a, dg);
        janas_gb_ref(b, a, dk[FRAC_MAX - 1]);
    }
    return r;
}

/* An integer part from the magnitudes [a, b] followed by `tail` (NONE:
   nothing), as one alternative of rule into. */
static int part(struct janas_gbuild *b, uint32_t into, uint64_t a, uint64_t hi,
                uint32_t tail)
{
    if (a > hi)
        return 0;
    uint32_t ip = janas_gb_rule(b);
    if (magnitudes(b, ip, a, hi, 1) != 0)
        return -1;
    uint32_t alt = janas_gb_alt(b, into);
    janas_gb_ref(b, alt, ip);
    if (tail != NONE)
        janas_gb_ref(b, alt, tail);
    return 0;
}

/*
 * Magnitudes v in a range: A (excl: A < v), B (excl: v < B; inf: none), the
 * alternatives of `into`. positive: v > 0 (the negative side: never -0).
 */
static int real_side(struct janas_gbuild *b, uint32_t into, double A, int ax,
                     double B, int bx, int inf, int positive, uint32_t any_f,
                     uint32_t nz_f)
{
    int64_t i1, q = -1, p = -1, i2;
    if (A == floor(A)) {
        i1 = (int64_t)A + (ax ? 1 : 0);
        if (ax)
            q = (int64_t)A; /* A itself only with a fraction not zero */
    } else {
        i1 = (int64_t)ceil(A);
    }
    if (positive && i1 == 0 && q < 0) {
        q = 0; /* 0.x, never 0 */
        i1 = 1;
    }
    if (inf) {
        i2 = (int64_t)MAG_MAX;
    } else if (B == floor(B)) {
        i2 = (int64_t)B - 1;
        if (!bx)
            p = (int64_t)B;
    } else {
        i2 = (int64_t)floor(B) - 1;
        p = (int64_t)floor(B);
    }
    if (p >= 0 && (p < i1 || (positive && p == 0)))
        p = -1; /* not inside */
    if (q >= 0 && !inf && (double)q + 1 > B)
        q = -1; /* q.9 could pass the upper bound */
    /* any fraction or none, then the exact point, then the point above the
       exclusive bound */
    if (i1 <= i2) {
        uint32_t opt = janas_gb_rule(b);
        janas_gb_alt(b, opt);
        janas_gb_ref(b, janas_gb_alt(b, opt), any_f);
        if (part(b, into, (uint64_t)i1, clamp_mag((double)i2), opt) != 0)
            return -1;
    }
    if (p >= 0 && part(b, into, (uint64_t)p, (uint64_t)p, NONE) != 0)
        return -1;
    if (q >= 0 && part(b, into, (uint64_t)q, (uint64_t)q, nz_f) != 0)
        return -1;
    return 0;
}

uint32_t janas_jsg_real_bounded(struct janas_gbuild *b,
                                const struct janas_jsg_bounds *bd, int *relaxed)
{
    if (bd->has_mult) {
        double mv = bd->mult;
        if (mv >= 1 && mv == floor(mv)) /* integers only: a valid subset */
            return janas_jsg_int_bounded(b, bd, relaxed);
        (*relaxed)++; /* a fractional step: not enforced */
    }
    uint32_t any_f = fraction(b, 0), nz_f = fraction(b, 1);
    uint32_t r = janas_gb_rule(b);
    double lo = bd->lo, hi = bd->hi;
    /* v >= 0 */
    if (!bd->has_hi || hi > 0 || (hi == 0 && !bd->hi_x)) {
        double A = bd->has_lo && lo > 0 ? lo : 0;
        int ax = bd->has_lo && lo >= 0 && bd->lo_x;
        if (real_side(b, r, A, ax, bd->has_hi ? hi : 0, bd->hi_x, !bd->has_hi,
                      0, any_f, nz_f) != 0)
            return NONE;
    }
    /* v < 0: -m, m in (max(0, -hi), -lo) */
    if (!bd->has_lo || lo < 0) {
        double A = bd->has_hi && hi < 0 ? -hi : 0;
        int ax = bd->has_hi && hi < 0 ? bd->hi_x : 1;
        uint32_t neg = janas_gb_rule(b);
        if (real_side(b, neg, A, ax, bd->has_lo ? -lo : 0, bd->lo_x,
                      !bd->has_lo, 1, any_f, nz_f) != 0)
            return NONE;
        uint32_t alt = janas_gb_alt(b, r);
        janas_gb_byte(b, alt, '-');
        janas_gb_ref(b, alt, neg);
    }
    return r;
}
