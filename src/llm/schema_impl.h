/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schema_impl.h - the parts of schema.c in files of their own: numbers with
 * bounds (schema_num.c) and regular expressions (schema_re.c), as grammar
 * rules.
 */
#ifndef JANAS_LLM_SCHEMA_IMPL_H
#define JANAS_LLM_SCHEMA_IMPL_H

#include <stddef.h>
#include <stdint.h>

#include "grammar.h"

/* minimum / maximum (exclusive: exclusiveMinimum, exclusiveMaximum as
   numbers) and multipleOf, as a schema gives them */
struct janas_jsg_bounds {
    int has_lo, has_hi, lo_x, hi_x, has_mult;
    double lo, hi, mult;
};

/* An integer inside the bounds and a multiple (up to 50; larger, or a
   fraction other than 1/n, counted in *relaxed and not enforced). A rule
   with no alternatives when no integer fits; NONE (UINT32_MAX) on memory. */
uint32_t janas_jsg_int_bounded(struct janas_gbuild *b,
                               const struct janas_jsg_bounds *bd, int *relaxed);
/* A number inside the bounds, without exponent; a fraction left out where
   it could pass a bound that is not an integer (>= 1.5 writes 2 and more).
   An integer multipleOf makes it an integer; a fractional one is counted in
   *relaxed. */
uint32_t janas_jsg_real_bounded(struct janas_gbuild *b,
                                const struct janas_jsg_bounds *bd,
                                int *relaxed);

/*
 * The n bytes of expression expr (character classes, \d \w \s, groups,
 * alternatives, * + ? {m,n}; no anchors, back references or look-arounds)
 * as a rule matching the inside of a JSON string, then going on with rule
 * then. NONE when the expression is not one this reads.
 */
uint32_t janas_jsg_regex(struct janas_gbuild *b, const char *expr, size_t n,
                         uint32_t then);
/* The same for a schema's pattern: written from the string's start
   whether anchored with ^ or not (the string then contains it), followed
   by any text unless anchored with $ at the end (anchors elsewhere are not
   read). */
uint32_t janas_jsg_pattern(struct janas_gbuild *b, const char *expr, size_t n,
                           uint32_t then);

#endif
