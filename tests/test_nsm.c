/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_nsm.c - janas-nsm's counts (src/nsm/score.c): explications counted
 * by hand, with DeepNSM's list (punctuation and apostrophes dropped, single
 * words) and with the full one (phrases as primes, apostrophes kept), UTF-8
 * quotes taken off whole, the circularity of a word's inflected forms, and
 * the lemma's rules.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "nsm/lexicon.c"
#include "nsm/score.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void counts(enum nsm_lex lex, const char *text, const char *target,
                   size_t words, size_t primes, size_t stops, size_t mols,
                   double legality, int circular)
{
    struct nsm_score s;
    nsm_score_text(lex, text, strlen(text), target, &s);
    CHECK(s.words == words && s.primes == primes && s.stops == stops &&
              s.molecules == mols,
          "%s \"%s\": %zu words %zu primes %zu stops %zu molecules, "
          "want %zu %zu %zu %zu",
          lex == NSM_LEX_PAPER ? "paper" : "full", text, s.words, s.primes,
          s.stops, s.molecules, words, primes, stops, mols);
    CHECK(fabs(s.legality - legality) < 1e-9,
          "\"%s\": legality %.6f, want %.6f", text, s.legality, legality);
    CHECK(s.circular == circular, "\"%s\" for %s: circular %d, want %d", text,
          target, s.circular, circular);
}

static void lemma(const char *w, const char *want)
{
    char b[64];
    nsm_lemma(w, b, sizeof(b));
    CHECK(!strcmp(b, want), "lemma of %s: %s, want %s", w, b, want);
}

int main(void)
{
    alarm(60);
    /* someone can see this kind = primes, is too in their list; of it it
       on the: function words; animal long moves ground: molecules ("move"
       is a prime, "moves" not in their list) */
    counts(NSM_LEX_PAPER,
           "Someone can see this kind of animal. It is long, it moves on "
           "the ground.",
           "snake", 15, 6, 5, 4, 10.0 * (6 - 4) / 15, 0);
    /* the same in the full list: "moves" a prime's form, "is" alone a
       function word */
    counts(NSM_LEX_FULL,
           "Someone can see this kind of animal. It is long, it moves on "
           "the ground.",
           "snake", 15, 6, 6, 3, 10.0 * (6 - 3) / 15, 0);
    /* their counting: "doesn't" becomes "doesnt", a molecule; "time" a
       prime, "long" a molecule, phrases never matched */
    counts(NSM_LEX_PAPER, "Someone doesn't want this for a long time.", "wait",
           8, 4, 2, 2, 10.0 * (4 - 2) / 8, 0);
    /* the full list: "doesn't want" and "a long time" primes, whole */
    counts(NSM_LEX_FULL, "Someone doesn't want this for a long time.", "wait",
           8, 7, 1, 0, 10.0 * 7 / 8, 0);
    /* curly quotes and a dash off whole, not as stray bytes */
    counts(NSM_LEX_PAPER,
           "\xe2\x80\x9cSomeone\xe2\x80\x9d sees \xe2\x80\x94 "
           "this.",
           "x", 3, 2, 0, 1, 10.0 * (2 - 1) / 3, 0);
    /* circular: the target or an inflected form of it */
    counts(NSM_LEX_PAPER, "Snakes move.", "snake", 2, 1, 0, 1,
           10.0 * (1 - 1) / 2, 1);
    counts(NSM_LEX_PAPER, "It is running.", "run", 3, 1, 1, 1,
           10.0 * (1 - 1) / 3, 1);
    counts(NSM_LEX_PAPER, "Children play.", "child", 2, 0, 0, 2,
           10.0 * (0 - 2) / 2, 1);
    counts(NSM_LEX_PAPER, "Something sneaky.", "snake", 2, 1, 0, 1, 0.0, 0);
    /* a silent e the rules took off the target, or the word */
    counts(NSM_LEX_PAPER, "They make it.", "making", 3, 0, 2, 1, -10.0 / 3, 1);
    counts(NSM_LEX_PAPER, "Someone was making it.", "make", 4, 1, 2, 1, 0.0, 1);
    counts(NSM_LEX_PAPER, "They use it.", "us", 3, 0, 2, 1, -10.0 / 3, 0);
    /* nothing to count */
    counts(NSM_LEX_PAPER, " ... ", "x", 0, 0, 0, 0, 0.0, 0);

    lemma("snakes", "snake");
    lemma("running", "run");
    lemma("stopped", "stop");
    lemma("flies", "fly");
    lemma("tried", "try");
    lemma("boxes", "box");
    lemma("watches", "watch");
    lemma("classes", "class");
    lemma("glass", "glass");
    lemma("went", "go");
    lemma("feet", "foot");
    lemma("dog", "dog");

    if (failures) {
        printf("test_nsm: %d failures\n", failures);
        return 1;
    }
    printf("test_nsm: ok\n");
    return 0;
}
