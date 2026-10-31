/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * score.h - what an NSM explication is made of, counted.
 *
 * With NSM_LEX_PAPER the counts are DeepNSM's (prompts.py, legality_score):
 * the text lower-cased, every character that is neither a word character
 * nor a space removed (apostrophes too: "don't" becomes "dont"), split at
 * spaces; each word a prime, a function word or else a "molecule". Their
 * legality is 10 (primes - molecules) / words, -10 to 10; an explication
 * is circular when a word of it has the target word's lemma (theirs with
 * spaCy, here with English suffix rules: the same verdict on all 1341 of
 * their published records). With NSM_LEX_FULL the apostrophes stay and
 * the primes' phrases count as primes.
 */
#ifndef JANAS_NSM_SCORE_H
#define JANAS_NSM_SCORE_H

#include <stddef.h>

#include "lexicon.h"

struct nsm_score {
    size_t words, primes, stops, molecules; /* primes: words in primes */
    double primes_ratio, molecules_ratio, legality;
    int circular;
};

void nsm_score_text(enum nsm_lex lex, const char *text, size_t n,
                    const char *target, struct nsm_score *s);

/* An English lemma by suffix rules, into out (cap bytes). */
void nsm_lemma(const char *w, char *out, size_t cap);

#endif
