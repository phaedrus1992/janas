/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vocab.c - the tokens of an NSM explication (vocab.h).
 */
#include "vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lexicon.h"

const char *const nsm_default_molecules[] = {
    /* the body */
    "head", "face", "eyes", "eye", "ears", "ear", "nose", "mouth", "teeth",
    "tongue", "hair", "skin", "hands", "hand", "fingers", "arms", "arm", "legs",
    "leg", "feet", "foot", "back", "belly", "heart", "blood", "bones", "wings",
    "tail",
    /* physical qualities */
    "long", "short", "round", "flat", "thin", "thick", "hard", "soft", "sharp",
    "heavy", "light", "hot", "cold", "warm", "wet", "dry", "smooth", "straight",
    "high", "low", "fast", "slow", "loud", "quiet", "bright", "dark", "colour",
    "color", "red", "white", "black", "green", "blue", "yellow", "sweet",
    "sour",
    /* nature */
    "water", "ground", "earth", "sky", "sun", "moon", "stars", "fire", "rain",
    "wind", "air", "stone", "stones", "wood", "tree", "trees", "plant",
    "plants", "grass", "leaves", "flowers", "fruit", "seeds", "animal",
    "animals", "creature", "creatures", "bird", "birds", "fish", "egg", "eggs",
    "sea", "river", "mountain", "day", "days", "night", "nights", "year",
    "years",
    /* people */
    "man", "men", "woman", "women", "child", "children", "baby", "babies",
    "boy", "girl", "mother", "father", "family", "husband", "wife", "friend",
    "friends", "born",
    /* doing */
    "make", "makes", "made", "eat", "eats", "eating", "drink", "sleep", "sit",
    "stand", "lie", "walk", "run", "go", "goes", "going", "went", "come",
    "comes", "came", "give", "gives", "take", "takes", "hold", "put", "puts",
    "kill", "cut", "break", "hit", "carry", "wear", "cook", "play", "work",
    "help", "need", "try", "use", "uses", "call", "called", "laugh", "cry",
    "sing", "read", "write", "show", "look", "looks", "find", "keep", "open",
    "close", "start", "stop", "get", "gets", "let", "become", "becomes",
    "change", "changes", "grow", "grows", "living",
    /* things and places */
    "house", "home", "food", "money", "clothes", "tool", "tools", "machine",
    "machines", "town", "country", "school", "road", "car", "door", "room",
    "bed", "table",
    /* time, space, number */
    "together", "around", "again", "always", "often", "sometimes", "never",
    "also", "first", "second", "last", "end", "outside", "top", "bottom",
    "front", "middle", "number", "numbers", "three", "four", "five", "ten",
    "hundred", "enough", "almost", "every", "everything", "everyone", "anyone",
    "anything", "nothing", "nobody",
    /* the mind, and how things are */
    "mind", "idea", "ideas", "reason", "question", "answer", "problem", "name",
    "story", "language", "sound", "sounds", "voice", "picture", "sign", "rule",
    "rules", "real", "different", "special", "important", "easy", "difficult",
    "possible", "happy", "sad", "afraid", "angry", "sorry", "pain", "sick",
    "dead", "alive", "young", "old", "new", "able", "free", "safe", "dangerous",
    "beautiful", "clean", "dirty", "full", "empty", "ready", "strong", "weak",
    "rich", "poor", "wrong", "right", NULL};

struct set {
    uint8_t *in;
    int32_t *ids;
    long n;
    int32_t nv;
};

static void add_text(const janas_llm *m, struct set *s, const char *t)
{
    int32_t ids[64], n = 0;
    if (janas_llm_tokenize(m, t, (int32_t)strlen(t), 0, ids, 64, &n) !=
        JANAS_LLM_OK)
        return;
    for (int32_t i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < s->nv && !s->in[ids[i]]) {
            s->in[ids[i]] = 1;
            s->ids[s->n++] = ids[i];
        }
}

/* A word's four spellings: at a line's start, after a space, capitalized. */
static void add_word(const janas_llm *m, struct set *s, const char *w)
{
    char b[128];
    size_t n = strlen(w);
    if (!n || n + 2 > sizeof(b))
        return;
    for (int cap = 0; cap < 2; cap++)
        for (int sp = 0; sp < 2; sp++) {
            snprintf(b, sizeof(b), "%s%s", sp ? " " : "", w);
            if (cap && b[sp] >= 'a' && b[sp] <= 'z')
                b[sp] = (char)(b[sp] - 32);
            add_text(m, s, b);
        }
}

/* A token of white space and punctuation alone (curly quotes included). */
static int only_punct(const char *t, int32_t n)
{
    if (n <= 0)
        return 0;
    for (int32_t i = 0; i < n;) {
        unsigned char c = (unsigned char)t[i];
        if (c < 0x80) {
            if (!strchr(" \n\t.,:;!?\"'()-", c) || !c)
                return 0;
            i++;
        } else if (n - i >= 3 && c == 0xe2 && (unsigned char)t[i + 1] == 0x80 &&
                   ((unsigned char)t[i + 2] == 0x9c ||
                    (unsigned char)t[i + 2] == 0x9d ||
                    (unsigned char)t[i + 2] == 0x98 ||
                    (unsigned char)t[i + 2] == 0x99)) {
            i += 3;
        } else
            return 0;
    }
    return 1;
}

long nsm_vocab_tokens(const janas_llm *m, const char *const *molecules,
                      int32_t **ids)
{
    /* the vocabulary's size: the first id token_text refuses */
    char t[1024];
    int32_t nv = 0, len;
    while (nv < (1 << 24) &&
           janas_llm_token_text(m, nv, t, sizeof(t), &len) == JANAS_LLM_OK)
        nv++;
    if (nv <= 0)
        return -1;
    struct set s = {.in = calloc((size_t)nv, 1),
                    .ids = malloc((size_t)nv * sizeof(int32_t)),
                    .nv = nv};
    if (!s.in || !s.ids) {
        free(s.in);
        free(s.ids);
        return -1;
    }
    for (size_t p = 0; p < nsm_n_primes; p++)
        for (const char *const *f = nsm_primes[p].forms; *f; f++) {
            /* a phrase's words one by one */
            char b[64];
            snprintf(b, sizeof(b), "%s", *f);
            for (char *w = strtok(b, " "); w; w = strtok(NULL, " "))
                add_word(m, &s, w);
        }
    for (const char *const *w = nsm_stop_words; *w; w++)
        add_word(m, &s, *w);
    for (const char *const *w = molecules ? molecules : nsm_default_molecules;
         *w; w++)
        add_word(m, &s, *w);
    for (int32_t i = 0; i < nv; i++) {
        len = 0;
        if (s.in[i] ||
            janas_llm_token_text(m, i, t, sizeof(t), &len) != JANAS_LLM_OK ||
            len > (int32_t)sizeof(t))
            continue;
        if (only_punct(t, len)) {
            s.in[i] = 1;
            s.ids[s.n++] = i;
        }
    }
    free(s.in);
    *ids = s.ids;
    return s.n;
}
