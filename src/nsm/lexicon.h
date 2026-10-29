/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * lexicon.h - the words of the Natural Semantic Metalanguage (NSM) in
 * English, as janas-nsm counts them.
 *
 * Two lists of primes:
 *  - NSM_LEX_PAPER: the one DeepNSM's published scores were computed with
 *    (arXiv 2505.11764; the words of utils.py's NSM_PRIMES without "i" and
 *    "said": with them its records do not come back, without them they do,
 *    in 1337 of 1341 explications), single words only, as its counting
 *    never matches a phrase;
 *  - NSM_LEX_FULL: the 65 primes of Goddard and Wierzbicka (2014) with
 *    their allolexes and inflected forms, phrases included ("a long time",
 *    "don't want", "there is"), matched longest first.
 * And the function words: NLTK's English stop words (179), less the primes.
 */
#ifndef JANAS_NSM_LEXICON_H
#define JANAS_NSM_LEXICON_H

#include <stddef.h>

enum nsm_lex { NSM_LEX_PAPER, NSM_LEX_FULL };

enum nsm_kind { NSM_PRIME, NSM_STOP, NSM_MOLECULE };

/* The words of a lower-cased, punctuation-free text, and how each counts:
   a prime phrase is one unit of as many words. *n_units gets the units;
   kind[i] and words[i] (the words the unit spans) per unit, room for
   max. Returns the number of words. */
size_t nsm_classify(enum nsm_lex lex, char *const *word, size_t n_words,
                    enum nsm_kind *kind, size_t *span, size_t max,
                    size_t *n_units);

/* 1 when w is a prime (single word) of the list. */
int nsm_is_prime(enum nsm_lex lex, const char *w);
/* 1 when w is a function word that is not a prime of the list. */
int nsm_is_stop(enum nsm_lex lex, const char *w);

/* The English exponents of the 65 primes and their forms, one table: the
   prime's name and a NULL-terminated list of forms (lower case, phrases
   with single spaces). For the restricted vocabulary. */
struct nsm_prime {
    const char *name;
    const char *const *forms;
};
extern const struct nsm_prime nsm_primes[];
extern const size_t nsm_n_primes;

/* NLTK's English stop words, NULL-terminated. */
extern const char *const nsm_stop_words[];

#endif
