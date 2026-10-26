/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vocab_check.c - the output head restricted to a vocabulary
 * (janas_llm_model_set_vocab, janas-nsm --strict) against the whole head:
 * the same token ids through both, one at a time and in blocks with every
 * token's logits; the vocabulary's logits must be the same to the bit, the
 * others -inf, and the whole head must come back unchanged after.
 *
 * Usage: vocab_check <model.jns> [compute_cpus] [every=K] [block=B]
 * The vocabulary is every K-th token id (default 37) and the stretch 0 .. 299
 * (the bytes and the commonest tokens); the text is 64 token ids spread over
 * the vocabulary, the same every run.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/model.h"

#define N_TOK 64

static int parse_cpus(const char *s, int *cpus)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", s);
    int n = 0;
    for (char *t = strtok(buf, ","); t && n < 256; t = strtok(NULL, ","))
        cpus[n++] = atoi(t);
    return n;
}

/* the logits of all N_TOK tokens, one at a time (block 1) or in blocks */
static int run(struct janas_llm_model *m, const int32_t *tok, uint32_t block,
               float *out, uint32_t nv)
{
    for (uint32_t p = 0; p < N_TOK; p += block) {
        uint32_t b = N_TOK - p < block ? N_TOK - p : block;
        if (janas_llm_model_forward(m, tok + p, b, p, out + (size_t)p * nv,
                                    1) != 0)
            return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: vocab_check <model.jns> [compute_cpus] "
                        "[every=K] [block=B]\n");
        return 2;
    }
    uint32_t every = 37, block = 16;
    for (int i = 2; i < argc; i++) {
        if (!strncmp(argv[i], "every=", 6))
            every = (uint32_t)atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "block=", 6))
            block = (uint32_t)atoi(argv[i] + 6);
    }
    int cpus[256], io[2] = {12, 13};
    int nc = parse_cpus(argc > 2 && strchr("0123456789", argv[2][0])
                            ? argv[2]
                            : "0,1,2,3,4,5,6,7,8,9,10,11",
                        cpus);
    struct janas_llm_options o = {.cache_bytes = (uint64_t)4096 << 20,
                                  .n_ctx = 256,
                                  .compute_cpus = cpus,
                                  .n_compute = nc,
                                  .io_cpus = io,
                                  .n_io = 2};
    char err[256] = "";
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    uint32_t nv = janas_llm_model_n_vocab(m);
    if (block > janas_llm_model_max_block(m))
        block = janas_llm_model_max_block(m);
    int32_t tok[N_TOK];
    for (uint32_t i = 0; i < N_TOK; i++)
        tok[i] = (int32_t)((i * 2654435761u + 977u) % nv);
    int32_t *ids = malloc(nv * sizeof(int32_t));
    uint8_t *in = calloc(nv, 1);
    float *full = malloc((size_t)N_TOK * nv * sizeof(float));
    float *got = malloc((size_t)N_TOK * nv * sizeof(float));
    if (!ids || !in || !full || !got) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    uint32_t n = 0;
    for (uint32_t i = 0; i < nv; i++)
        if (i < 300 || i % every == 0) {
            ids[n++] = (int32_t)i;
            in[i] = 1;
        }
    printf("%s: %u tokens of %u in the vocabulary, %d tokens, blocks of %u\n",
           argv[1], n, nv, N_TOK, block);
    /* a pass first, not compared: with the GPU the first passes run another
       way until the weights reach it (janas_m_gpu_token waits for the
       arena), the later ones all on it */
    if (run(m, tok, 1, full, nv) != 0 || run(m, tok, 1, full, nv) != 0) {
        fprintf(stderr, "the whole head's pass failed\n");
        return 1;
    }
    if (janas_llm_model_set_vocab(m, ids, n) != 0) {
        fprintf(stderr, "set_vocab refused this model's head\n");
        return 1;
    }
    int bad = 0;
    for (int pass = 0; pass < 3; pass++) {
        /* 0: one at a time, 1: blocks, 2: the whole head again */
        if (pass == 2)
            janas_llm_model_set_vocab(m, NULL, 0);
        if (run(m, tok, pass == 1 ? block : 1, got, nv) != 0) {
            fprintf(stderr, "pass %d failed\n", pass);
            return 1;
        }
        size_t same = 0, differ = 0, not_inf = 0;
        for (size_t t = 0; t < N_TOK; t++)
            for (uint32_t i = 0; i < nv; i++) {
                float a = full[t * nv + i], b = got[t * nv + i];
                if (pass == 2 || in[i]) {
                    if (memcmp(&a, &b, sizeof(float)))
                        differ++;
                    else
                        same++;
                } else if (!(isinf(b) && b < 0)) {
                    not_inf++;
                }
            }
        static const char *names[] = {"token by token", "in blocks",
                                      "whole head again"};
        printf("%-17s %zu logits the same, %zu differ, %zu outside the "
               "vocabulary not -inf\n",
               names[pass], same, differ, not_inf);
        bad |= differ || not_inf;
    }
    janas_llm_model_free(m);
    free(ids);
    free(in);
    free(full);
    free(got);
    printf("%s\n", bad ? "FAIL" : "ok");
    return bad;
}
