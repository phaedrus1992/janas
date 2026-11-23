/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * lexicon.c - the English words of NSM (lexicon.h).
 */
#include "lexicon.h"

#include <string.h>

#define FORMS(...) ((const char *const[]){__VA_ARGS__, NULL})

/* Goddard and Wierzbicka's chart of 2014, its allolexes (~) and the forms
   the explications inflect them into */
const struct nsm_prime nsm_primes[] = {
    {"I", FORMS("i", "me")},
    {"YOU", FORMS("you")},
    {"SOMEONE", FORMS("someone", "somebody", "person", "persons")},
    {"PEOPLE", FORMS("people")},
    {"SOMETHING", FORMS("something", "thing", "things")},
    {"BODY", FORMS("body", "bodies")},
    {"KIND", FORMS("kind", "kinds")},
    {"PART", FORMS("part", "parts")},
    {"THIS", FORMS("this", "these")},
    {"THE SAME", FORMS("the same", "same")},
    {"OTHER", FORMS("other", "others", "else", "another")},
    {"ONE", FORMS("one")},
    {"TWO", FORMS("two")},
    {"SOME", FORMS("some")},
    {"ALL", FORMS("all")},
    {"MUCH", FORMS("much", "many")},
    {"LITTLE", FORMS("little", "few")},
    {"GOOD", FORMS("good", "well")},
    {"BAD", FORMS("bad", "badly")},
    {"BIG", FORMS("big")},
    {"SMALL", FORMS("small")},
    {"THINK", FORMS("think", "thinks", "thinking", "thought")},
    {"KNOW", FORMS("know", "knows", "knowing", "knew", "known")},
    {"WANT", FORMS("want", "wants", "wanting", "wanted")},
    {"DON'T WANT", FORMS("don't want", "doesn't want", "didn't want",
                         "do not want", "does not want", "did not want")},
    {"FEEL", FORMS("feel", "feels", "feeling", "felt")},
    {"SEE", FORMS("see", "sees", "seeing", "saw", "seen")},
    {"HEAR", FORMS("hear", "hears", "hearing", "heard")},
    {"SAY", FORMS("say", "says", "saying", "said")},
    {"WORDS", FORMS("words", "word")},
    {"TRUE", FORMS("true")},
    {"DO", FORMS("do", "does", "doing", "did", "done")},
    {"HAPPEN", FORMS("happen", "happens", "happening", "happened")},
    {"MOVE", FORMS("move", "moves", "moving", "moved")},
    {"BE SOMEWHERE", FORMS("be somewhere", "is somewhere", "are somewhere",
                           "was somewhere", "were somewhere", "somewhere")},
    {"THERE IS", FORMS("there is", "there are", "there was", "there were",
                       "there can be", "there will be")},
    {"BE SOMEONE/SOMETHING",
     FORMS("is someone", "be someone", "is something", "be something")},
    {"MINE", FORMS("mine")},
    {"LIVE", FORMS("live", "lives", "living", "lived")},
    {"DIE", FORMS("die", "dies", "dying", "died")},
    {"WHEN", FORMS("when", "time", "times")},
    {"NOW", FORMS("now")},
    {"BEFORE", FORMS("before")},
    {"AFTER", FORMS("after")},
    {"A LONG TIME", FORMS("a long time")},
    {"A SHORT TIME", FORMS("a short time")},
    {"FOR SOME TIME", FORMS("for some time")},
    {"MOMENT", FORMS("moment", "moments")},
    {"WHERE", FORMS("where", "place", "places")},
    {"HERE", FORMS("here")},
    {"ABOVE", FORMS("above")},
    {"BELOW", FORMS("below")},
    {"FAR", FORMS("far")},
    {"NEAR", FORMS("near")},
    {"SIDE", FORMS("side", "sides")},
    {"INSIDE", FORMS("inside")},
    {"TOUCH", FORMS("touch", "touches", "touching", "touched")},
    {"NOT", FORMS("not", "no", "don't", "doesn't", "didn't", "isn't", "aren't",
                  "wasn't", "weren't", "can't", "cannot")},
    {"MAYBE", FORMS("maybe")},
    {"CAN", FORMS("can", "could")},
    {"BECAUSE", FORMS("because")},
    {"IF", FORMS("if")},
    {"VERY", FORMS("very")},
    {"MORE", FORMS("more")},
    {"LIKE", FORMS("like", "as", "way", "ways")},
};
const size_t nsm_n_primes = sizeof(nsm_primes) / sizeof(nsm_primes[0]);

_Static_assert(sizeof(nsm_primes) / sizeof(nsm_primes[0]) == 65,
               "the 65 primes");

/* DeepNSM's list as its scores were computed (lexicon.h) */
static const char *const paper_primes[] = {
    "you",    "someone", "people", "something", "thing",   "body",    "kind",
    "part",   "this",    "other",  "else",      "another", "one",     "two",
    "some",   "all",     "much",   "many",      "little",  "few",     "good",
    "bad",    "big",     "small",  "think",     "know",    "want",    "feel",
    "see",    "hear",    "say",    "words",     "true",    "do",      "happen",
    "move",   "there",   "is",     "be",        "mine",    "live",    "die",
    "when",   "time",    "now",    "before",    "after",   "moment",  "where",
    "place",  "here",    "above",  "below",     "far",     "near",    "side",
    "inside", "touch",   "not",    "maybe",     "can",     "because", "if",
    "very",   "more",    "like",   "as",        "way",     NULL};

const char *const nsm_stop_words[] = {
    "i",        "me",        "my",        "myself",   "we",         "our",
    "ours",     "ourselves", "you",       "you're",   "you've",     "you'll",
    "you'd",    "your",      "yours",     "yourself", "yourselves", "he",
    "him",      "his",       "himself",   "she",      "she's",      "her",
    "hers",     "herself",   "it",        "it's",     "its",        "itself",
    "they",     "them",      "their",     "theirs",   "themselves", "what",
    "which",    "who",       "whom",      "this",     "that",       "that'll",
    "these",    "those",     "am",        "is",       "are",        "was",
    "were",     "be",        "been",      "being",    "have",       "has",
    "had",      "having",    "do",        "does",     "did",        "doing",
    "a",        "an",        "the",       "and",      "but",        "if",
    "or",       "because",   "as",        "until",    "while",      "of",
    "at",       "by",        "for",       "with",     "about",      "against",
    "between",  "into",      "through",   "during",   "before",     "after",
    "above",    "below",     "to",        "from",     "up",         "down",
    "in",       "out",       "on",        "off",      "over",       "under",
    "again",    "further",   "then",      "once",     "here",       "there",
    "when",     "where",     "why",       "how",      "all",        "any",
    "both",     "each",      "few",       "more",     "most",       "other",
    "some",     "such",      "no",        "nor",      "not",        "only",
    "own",      "same",      "so",        "than",     "too",        "very",
    "s",        "t",         "can",       "will",     "just",       "don",
    "don't",    "should",    "should've", "now",      "d",          "ll",
    "m",        "o",         "re",        "ve",       "y",          "ain",
    "aren",     "aren't",    "couldn",    "couldn't", "didn",       "didn't",
    "doesn",    "doesn't",   "hadn",      "hadn't",   "hasn",       "hasn't",
    "haven",    "haven't",   "isn",       "isn't",    "ma",         "mightn",
    "mightn't", "mustn",     "mustn't",   "needn",    "needn't",    "shan",
    "shan't",   "shouldn",   "shouldn't", "wasn",     "wasn't",     "weren",
    "weren't",  "won",       "won't",     "wouldn",   "wouldn't",   NULL};

static int in_list(const char *const *l, const char *w)
{
    for (; *l; l++)
        if (!strcmp(*l, w))
            return 1;
    return 0;
}

static int full_single(const char *w)
{
    for (size_t p = 0; p < nsm_n_primes; p++)
        for (const char *const *f = nsm_primes[p].forms; *f; f++)
            if (!strchr(*f, ' ') && !strcmp(*f, w))
                return 1;
    return 0;
}

int nsm_is_prime(enum nsm_lex lex, const char *w)
{
    return lex == NSM_LEX_PAPER ? in_list(paper_primes, w) : full_single(w);
}

int nsm_is_stop(enum nsm_lex lex, const char *w)
{
    return in_list(nsm_stop_words, w) && !nsm_is_prime(lex, w);
}

/* The number of words of the longest prime phrase starting at word[0], 0
   when none (single words are not phrases). */
static size_t phrase_at(char *const *word, size_t n)
{
    size_t best = 0;
    for (size_t p = 0; p < nsm_n_primes; p++)
        for (const char *const *f = nsm_primes[p].forms; *f; f++) {
            if (!strchr(*f, ' '))
                continue;
            const char *s = *f;
            size_t k = 0;
            while (k < n) {
                size_t l = strlen(word[k]);
                if (strncmp(s, word[k], l) || (s[l] != ' ' && s[l] != 0))
                    break;
                k++;
                s += l;
                if (!*s)
                    break;
                s++;
            }
            if (!*s && k > best)
                best = k;
        }
    return best;
}

size_t nsm_classify(enum nsm_lex lex, char *const *word, size_t n_words,
                    enum nsm_kind *kind, size_t *span, size_t max,
                    size_t *n_units)
{
    size_t u = 0;
    for (size_t i = 0; i < n_words && u < max;) {
        size_t k = lex == NSM_LEX_FULL ? phrase_at(word + i, n_words - i) : 0;
        if (k) {
            kind[u] = NSM_PRIME;
        } else {
            k = 1;
            kind[u] = nsm_is_prime(lex, word[i])  ? NSM_PRIME
                      : nsm_is_stop(lex, word[i]) ? NSM_STOP
                                                  : NSM_MOLECULE;
        }
        span[u++] = k;
        i += k;
    }
    *n_units = u;
    return n_words;
}
