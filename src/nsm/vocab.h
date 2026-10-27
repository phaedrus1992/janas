/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vocab.h - the tokens an NSM explication may be written with (janas-nsm
 * --strict): the primes' forms, the function words and a list of semantic
 * molecules, each as the model's tokens of "w", " w", "W" and " W", and
 * every token made of white space and punctuation alone.
 */
#ifndef JANAS_NSM_VOCAB_H
#define JANAS_NSM_VOCAB_H

#include <stddef.h>
#include <stdint.h>

#include "janas/llm.h"

/* The default molecules, NULL-terminated: body parts, physical qualities,
   nature, people, everyday actions and things - the words NSM's
   explications of concrete meanings lean on (Goddard and Wierzbicka's
   molecules are of these kinds), written for Janas. */
extern const char *const nsm_default_molecules[];

/* The token ids, malloc'd, into *ids; their count, or -1. molecules NULL:
   the default ones; else a NULL-terminated list. */
long nsm_vocab_tokens(const janas_llm *m, const char *const *molecules,
                      int32_t **ids);

#endif
