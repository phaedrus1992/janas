/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_hc_quant.c - Qwen3.8-Flash-Next's mixes of the streams rewritten
 * as Q8_0 once, at load (JANAS_HC_MIX=q8).
 *
 * Every layer mixes its four streams twice, before the token mixer and
 * before the feed-forward, with matrices kept in BF16 in the file - down
 * (10240 x 320), up (320 x 10240) - and the output with two more: 1.26 GB
 * read at every token, a third of what a token reads besides the experts.
 * As Q8_0 they are read at half the bytes. Their products keep the inputs
 * as the BF16 products have them, rounded to bf16, not quantized
 * (janas_m_q8w_product): against llama.cpp on 400 tokens of C and 400 of
 * Italian prose, the most likely token agreed at 393 and 370 positions
 * (mean logit difference 0.137 and 0.164) where in BF16 at 390 and 367
 * (0.135, 0.166); with the inputs quantized to eight bits as the quantized
 * kernels take them, 388 and 359 (0.185, 0.204), and with the weights in
 * Q4_K and Q5_0, as unsloth gave them in its MTP block, 382 and 349 (0.310,
 * 0.294) (23 Oct 2026). The inject matrices (10240 x 4, a fraction of a
 * percent of the bytes) stay in BF16.
 *
 * Each matrix is rewritten in place, Q8_0 being the smaller, before the GPU
 * and the NPU look at the types; those products are the CPU's, and in the
 * whole token on the GPU dense_mm.comp's (gpu_token_hc.c).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vmath.h"

#include "model_impl.h"

struct hcq_job {
    struct janas_llm_model *m;
    struct janas_jns_tensor **t;
    int n, next, failed;
    pthread_mutex_t lock;
};

/* t, BF16, rewritten as Q8_0 in place: 0, or -1 (out of memory) */
static int requant(struct janas_llm_model *m, struct janas_jns_tensor *t)
{
    size_t n = t->dims[0] * t->dims[1], bytes = n / 32 * 34;
    float *f = malloc(n * sizeof(float));
    struct janas_block_q8_0 *q = malloc(n / 32 * sizeof(*q));
    if (!f || !q) {
        free(f);
        free(q);
        return -1;
    }
    uint16_t *src = (uint16_t *)data(m, t);
    for (size_t i = 0; i < n; i++)
        f[i] = janas_u2f((uint32_t)src[i] << 16);
    /* rows of a multiple of 32: the matrix quantizes as one run of blocks */
    janas_q8_0_quantize(f, q, n);
    memcpy(src, q, bytes);
    t->type = JANAS_Q8_0;
    t->bytes = bytes;
    t->reserved = JANAS_TENSOR_FLOAT_IN;
    free(f);
    free(q);
    return 0;
}

static void *hcq_worker(void *arg)
{
    struct hcq_job *jb = arg;
    for (;;) {
        pthread_mutex_lock(&jb->lock);
        int i = jb->next++;
        pthread_mutex_unlock(&jb->lock);
        if (i >= jb->n)
            return NULL;
        if (requant(jb->m, jb->t[i]) != 0) {
            pthread_mutex_lock(&jb->lock);
            jb->failed = 1;
            pthread_mutex_unlock(&jb->lock);
        }
    }
}

/* the tensor if it is BF16, resident, its rows a multiple of 32, else
   NULL */
static struct janas_jns_tensor *mix(struct janas_llm_model *m, const char *name)
{
    struct janas_jns_tensor *t =
        (struct janas_jns_tensor *)janas_jns_tensor(&m->j, name);
    if (!t || t->type != 30 || t->n_dims != 2 || t->dims[0] % 32 ||
        t->offset < m->j.h.resident_offset ||
        t->offset + t->bytes > m->j.h.resident_offset + m->j.h.resident_bytes)
        return NULL;
    return t;
}

int janas_m_hc_requant(struct janas_llm_model *m, char *err, size_t err_len)
{
    const char *e = getenv("JANAS_HC_MIX");
    if (!e || !strcmp(e, "bf16"))
        return 0;
    if (strcmp(e, "q8") != 0) {
        snprintf(err, err_len, "JANAS_HC_MIX=%s: bf16 or q8", e);
        return -1;
    }
    size_t cap = (size_t)m->n_layer * 4 + 2;
    struct janas_jns_tensor **t = malloc(cap * sizeof(*t));
    if (!t) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    static const char *const part[2] = {"attn", "ffn"}, *const what[2] = {
                                                            "down", "up"};
    char name[64];
    int n = 0;
    uint64_t before = 0, after = 0;
    for (uint32_t l = 0; l <= m->n_layer; l++)
        for (int p = 0; p < (l < m->n_layer ? 2 : 1); p++)
            for (int k = 0; k < 2; k++) {
                if (l < m->n_layer)
                    snprintf(name, sizeof(name), "blk.%u.hc_%s_%s.weight", l,
                             part[p], what[k]);
                else /* the output's mix */
                    snprintf(name, sizeof(name), "output_hc_%s.weight",
                             what[k]);
                struct janas_jns_tensor *x = mix(m, name);
                if (x) {
                    before += x->bytes;
                    t[n++] = x;
                }
            }
    enum { NT = 16 };
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int nt = cpus < 1 ? 1 : cpus > NT ? NT : (int)cpus;
    struct hcq_job jb = {.m = m, .t = t, .n = n};
    pthread_mutex_init(&jb.lock, NULL);
    pthread_t th[NT];
    int started = 0;
    for (int k = 0; k < nt; k++)
        if (pthread_create(&th[k], NULL, hcq_worker, &jb) == 0)
            started |= 1 << k;
    if (!started)
        hcq_worker(&jb);
    for (int k = 0; k < nt; k++)
        if (started & 1 << k)
            pthread_join(th[k], NULL);
    pthread_mutex_destroy(&jb.lock);
    for (int i = 0; i < n; i++)
        after += t[i]->bytes;
    free(t);
    if (jb.failed) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    snprintf(m->hc_quant, sizeof(m->hc_quant), "%s", e);
    m->hc_quant_before = before;
    m->hc_quant_after = after;
    return 0;
}

const char *janas_llm_model_hc_quant(const struct janas_llm_model *m,
                                     uint64_t *before, uint64_t *after)
{
    *before = m->hc_quant_before;
    *after = m->hc_quant_after;
    return m->hc_quant[0] ? m->hc_quant : NULL;
}
