/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_gpu_output_copy.c - on a GPU with memory of its own the engine's
 * buffers, imported, are copies on the card: a product whose output is one
 * of them must still reach the host (issue #17, reported with this test's
 * first form by xpil: a Quadro RTX 5000's outputs stayed on the card and
 * the CPU read what was there before). JANAS_GPU_COPY=1 gives any GPU that
 * path. A Q4_K row of 256 ones times 256 ones is 256; a dense product of
 * ones over 64 columns is 64; the host's buffers start at -12345.
 * Every buffer lives until the GPU is gone: a region is never dropped, and
 * a buffer freed and allocated again at its address would be read from
 * the old one's copy (the staged product read the first one's output as
 * its weights, 13 Oct 2026). Skipped without a GPU.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"
#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static void *kept[8];
static int n_kept;

/* aligned, and freed only after the GPU (see above) */
static void *buffer(size_t bytes)
{
    void *p = aligned_alloc(4096, bytes);
    if (p && n_kept < 8)
        kept[n_kept++] = p;
    return p;
}

static int product(struct janas_gpu *g, struct janas_pool *pool, int imported)
{
    const size_t rows = 1, cols = 256, bytes = 4096;
    struct janas_block_q4k *w = buffer(bytes);
    float *y = buffer(4096);
    memset(w, 0, bytes);
    w[0].d = janas_fp32_to_fp16(1.0f);
    for (int i = 0; i < 4; i++)
        w[0].scales[i] = 1;
    for (int i = 8; i < 12; i++)
        w[0].scales[i] = 1;
    memset(w[0].qs, 0x11, sizeof(w[0].qs));
    float xf[256], ref = 0;
    for (int i = 0; i < 256; i++)
        xf[i] = 1.0f;
    struct janas_block_q8k x;
    janas_q8k_quantize(xf, &x, cols);
    for (int i = 0; i < 1024; i++)
        y[i] = -12345.0f;
    struct janas_matvec_task t = {.type = JANAS_Q4_K,
                                  .w = w,
                                  .x = &x,
                                  .y = &ref,
                                  .rows = rows,
                                  .cols = cols,
                                  .n_vec = 1};
    janas_matvec_q4k_group(pool, &t, 1);
    t.y = y;
    int rc = 0, waited = 0;
    size_t nr = rows;
    if (janas_gpu_import(g, w, bytes) != 0 ||
        (imported && janas_gpu_import(g, y, 4096) != 0) ||
        janas_gpu_begin(g, &t, &nr, 1) < 0 ||
        janas_gpu_finish(g, &waited) != 0) {
        printf("FAIL: the %s product did not run\n",
               imported ? "imported" : "staged");
        rc = 1;
    } else if (y[0] != ref) {
        printf("FAIL: %s output %g, the CPU's %g%s\n",
               imported ? "imported" : "staged", y[0], ref,
               y[0] == -12345.0f ? " (never written back)" : "");
        rc = 1;
    }
    return rc;
}

static int dense(struct janas_gpu *g)
{
    const size_t rows = 64, cols = 64, n = 64;
    float *w = buffer(16384), *x = buffer(16384), *y = buffer(16384);
    for (size_t i = 0; i < rows * cols; i++)
        w[i] = x[i] = 1.0f;
    for (size_t i = 0; i < 4096; i++)
        y[i] = -12345.0f;
    int rc = 0;
    if (janas_gpu_import(g, w, 16384) != 0 ||
        janas_gpu_import(g, x, 16384) != 0 ||
        janas_gpu_import(g, y, 16384) != 0) {
        printf("FAIL: dense buffers not imported\n");
        rc = 1;
    } else if (janas_gpu_dense_mm(g, 0, w, x, y, rows, cols, n, 1.0f) == 0) {
        /* run, then right: refusing (the CPU's then) is right too */
        for (size_t i = 0; i < rows * n && !rc; i++)
            if (y[i] != 64.0f) {
                printf("FAIL: dense output %zu is %g, not 64\n", i, y[i]);
                rc = 1;
            }
    }
    return rc;
}

int main(void)
{
    const char *e = getenv("JANAS_GPU");
    if (e && atoi(e) == 0) {
        printf("test_gpu_output_copy: GPU disabled, skipped\n");
        return 0;
    }
    setenv("JANAS_GPU_COPY", "1", 1); /* a card's path, on any GPU */
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("test_gpu_output_copy: no GPU (%s), skipped\n", err);
        return 0;
    }
    struct janas_pool *pool = janas_pool_create(4, NULL);
    int rc = product(g, pool, 1) | product(g, pool, 0) | dense(g);
    janas_pool_destroy(pool);
    janas_gpu_destroy(g);
    for (int i = 0; i < n_kept; i++)
        free(kept[i]);
    if (!rc)
        printf("test_gpu_output_copy: ok\n");
    return rc;
}
