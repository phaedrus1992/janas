/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_gemm.c - throughput of the quantized products over several vectors
 * (the tokens of a block), on random weights.
 *
 * Decode is bound by memory; a block of n tokens reads every weight once for
 * all of them, so beyond a few tokens the product is bound by arithmetic and
 * the figure of merit is multiply-accumulates per second.
 *
 * Usage: bench_gemm [rows cols
 * q4k|q5k|q6k|q3k|q80|iq4xs|iq3s|iq4nl|q6kp1|q6kp2|q6kp3 threads [gpu|gpufew]]
 * q6kpL is Q6_K in bit planes with L of the three planes in memory.
 * Defaults: 12288 x 2048 Q4_K (Qwen3-Next ssm_in), 12 threads on CPUs 0-11.
 * gpu: the products of a prompt's blocks on the GPU (the fast prompt's
 * tiled shaders, *_mmq.comp), 64 to 512 vectors, the weights imported.
 * BENCH_MEM=huge2m puts the weights on 2 MB pages before the GPU imports
 * them, as the engine does; BENCH_MEM=driver in memory allocated by the
 * GPU's driver (default: pages of 4 KB).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <time.h>

#include "common/pool.h"
#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

/* where the weights go (BENCH_MEM): 0 pages of 4 KB, 1 the GPU driver's
   memory, 2 pages of 2 MB */
static int mem_kind;
static struct janas_gpu *gpu;

/* bytes of weights rounded up to whole pages of their kind */
static size_t whole(size_t bytes)
{
    size_t al = mem_kind == 2 ? (size_t)2 << 20 : 4096;
    return (bytes + al - 1) / al * al;
}

static uint8_t *weights(size_t bytes)
{
    if (mem_kind == 1)
        return janas_gpu_alloc(gpu, bytes);
    return aligned_alloc(mem_kind == 2 ? (size_t)2 << 20 : 4096, bytes);
}

/* Written weights collapsed into 2 MB pages, then imported (pinned): the
   engine's way (janas_m_huge_pages); the driver's memory needs no import. */
static int to_gpu(uint8_t *p, size_t bytes)
{
#if defined(__linux__)
    const size_t piece = (size_t)32 << 20;
    if (mem_kind == 2)
        for (size_t at = 0; at < bytes; at += piece)
            madvise(p + at, bytes - at < piece ? bytes - at : piece,
                    MADV_COLLAPSE);
#endif
    if (mem_kind == 1) {
        janas_gpu_flush(gpu, p, bytes);
        return 0;
    }
    return janas_gpu_import(gpu, p, bytes);
}

int main(int argc, char **argv)
{
    size_t rows = argc > 1 ? (size_t)atol(argv[1]) : 12288;
    size_t cols = argc > 2 ? (size_t)atol(argv[2]) : 2048;
    const char *kind = argc > 3 ? argv[3] : "q4k";
    int level = 0;
    int type = JANAS_Q4_K;
    if (!strcmp(kind, "q6k"))
        type = JANAS_Q6_K;
    else if (!strcmp(kind, "q3k"))
        type = JANAS_Q3_K;
    else if (!strcmp(kind, "q5k"))
        type = JANAS_Q5_K;
    else if (!strcmp(kind, "q80"))
        type = JANAS_Q8_0;
    else if (!strcmp(kind, "iq4xs"))
        type = JANAS_IQ4_XS;
    else if (!strcmp(kind, "iq3s"))
        type = JANAS_IQ3_S;
    else if (!strcmp(kind, "iq4nl"))
        type = JANAS_IQ4_NL;
    else if (!strncmp(kind, "q6kp", 4)) {
        type = JANAS_Q6_K_P;
        level = kind[4] ? kind[4] - '0' : 3;
        if (level < 1 || level > 3)
            return 2;
    }
    int threads = argc > 4 ? atoi(argv[4]) : 12;
    /* gpufew: the products of a few vectors on the GPU (a chain of drafts) */
    int few = argc > 5 && !strcmp(argv[5], "gpufew");
    int on_gpu = argc > 5 && (!strcmp(argv[5], "gpu") || few);
    int cpus[256];
    for (int i = 0; i < threads && i < 256; i++)
        cpus[i] = i;
    const char *bm = getenv("BENCH_MEM");
    mem_kind = !bm                     ? 0
               : !strcmp(bm, "driver") ? 1
               : !strcmp(bm, "huge2m") ? 2
                                       : 0;
    if (on_gpu || mem_kind == 1) {
        char err[200];
        gpu = janas_gpu_create(err, sizeof(err));
        if (!gpu) {
            fprintf(stderr, "bench_gemm: no GPU: %s\n", err);
            return 1;
        }
    }
    struct janas_pool *pool = janas_pool_create(threads, cpus);
    janas_pin_current_thread(0);
    size_t nb = cols / JANAS_QK, bsz = janas_qtype_block_size(type);
    size_t nvmax = on_gpu ? 512 : 64;
    /* aligned and whole pages: what the GPU imports */
    size_t wbytes = whole(rows * nb * bsz + 4);
    uint8_t *w = weights(wbytes);
    size_t plane_bytes = whole(rows * nb * JANAS_Q6KP_PLANE);
    uint8_t *pl[2] = {NULL, NULL};
    for (int i = 0; i < level - 1; i++) {
        pl[i] = weights(plane_bytes);
        if (!pl[i])
            return 1;
        for (size_t j = 0; j < plane_bytes; j++)
            pl[i][j] = (uint8_t)rand();
    }
    struct janas_block_q8k *x = malloc(nvmax * nb * sizeof(*x));
    float *xf = malloc(cols * sizeof(float));
    float *y = malloc(nvmax * rows * sizeof(float));
    float *yc = malloc(nvmax * rows * sizeof(float));
    if (!pool || !w || !x || !xf || !y || !yc)
        return 1;
    srand(1);
    for (size_t i = 0; i < wbytes; i++)
        w[i] = (uint8_t)rand();
    /* blocks of 32 weights: eight to a block of 256 */
    for (size_t b = 0;
         b < rows * nb * (type == JANAS_IQ4_NL || type == JANAS_Q8_0 ? 8 : 1);
         b++) {
        if (type == JANAS_Q8_0) {
            ((struct janas_block_q8_0 *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_Q5_K) {
            ((struct janas_block_q5k *)w)[b].d = janas_fp32_to_fp16(0.001f);
            ((struct janas_block_q5k *)w)[b].dmin = janas_fp32_to_fp16(0.0005f);
        } else if (type == JANAS_IQ4_NL) {
            ((struct janas_block_iq4nl *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_IQ3_S) {
            ((struct janas_block_iq3s *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_Q6_K_P) {
            ((struct janas_block_q6kp *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_Q6_K) {
            ((struct janas_block_q6k *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_Q3_K) {
            ((struct janas_block_q3k *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else if (type == JANAS_IQ4_XS) {
            ((struct janas_block_iq4xs *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else {
            ((struct janas_block_q4k *)w)[b].d = janas_fp32_to_fp16(0.001f);
            ((struct janas_block_q4k *)w)[b].dmin = janas_fp32_to_fp16(0.0005f);
        }
    }
    for (size_t v = 0; v < nvmax; v++) {
        for (size_t i = 0; i < cols; i++)
            xf[i] = (float)((double)rand() / RAND_MAX) - 0.5f;
        janas_q8k_quantize(xf, x + v * nb, cols);
    }
    double mb = (rows * nb * bsz + (size_t)(level ? level - 1 : 0) * rows * nb *
                                       JANAS_Q6KP_PLANE) /
                1e6;
    if (on_gpu &&
        (to_gpu(w, wbytes) != 0 || (pl[0] && to_gpu(pl[0], plane_bytes) != 0) ||
         (pl[1] && to_gpu(pl[1], plane_bytes) != 0))) {
        fprintf(stderr, "bench_gemm: the GPU cannot read the weights\n");
        return 1;
    }
    printf("%s %zu x %zu (%.1f MB), %s, weights in %s\n", kind, rows, cols, mb,
           on_gpu ? janas_gpu_name(gpu) : "CPU",
           mem_kind == 1   ? "the GPU driver's memory"
           : mem_kind == 2 ? "2 MB pages"
                           : "4 KB pages");
    if (!on_gpu)
        printf("%d threads\n", threads);
    size_t nvs_cpu[] = {1, 2, 4, 8, 16, 32, 64},
           nvs_gpu[] = {64, 128, 256, 512}, nvs_few[] = {1, 2, 4, 8, 9, 16};
    size_t *nvs = few      ? nvs_few
                  : on_gpu ? nvs_gpu
                           : nvs_cpu,
           n_nvs = few      ? sizeof(nvs_few) / sizeof(*nvs_few)
                   : on_gpu ? sizeof(nvs_gpu) / sizeof(*nvs_gpu)
                            : sizeof(nvs_cpu) / sizeof(*nvs_cpu);
    for (size_t k = 0; k < n_nvs; k++) {
        size_t nv = nvs[k];
        struct janas_matvec_task t = {.type = type,
                                      .w = w,
                                      .plane = {pl[0], pl[1]},
                                      .x = x,
                                      .y = y,
                                      .rows = rows,
                                      .cols = cols,
                                      .n_vec = nv};
        int reps = (int)(200 / nv) + 5;
        if (on_gpu) {
            janas_gpu_matvec_group(gpu, pool, &t, 1); /* warm-up */
            reps = (int)(2048 / nv) + 5;
        } else
            janas_matvec_q4k_group(pool, &t, 1);
        double t0 = now();
        for (int r = 0; r < reps; r++) {
            if (on_gpu)
                janas_gpu_matvec_group(gpu, pool, &t, 1);
            else
                janas_matvec_q4k_group(pool, &t, 1);
        }
        double dt = (now() - t0) / reps;
        /* the outputs' sum, to tell two versions of a shader apart */
        double sum = 0;
        for (size_t i = 0; i < nv * rows; i++)
            sum += y[i];
        printf("n %2zu: %8.3f ms, %7.3f ms per vector, %6.1f GMAC/s, "
               "%5.1f GB/s, sum %.17g\n",
               nv, dt * 1e3, dt * 1e3 / nv, rows * cols * nv / dt / 1e9,
               mb / 1e3 / dt, sum);
        if (on_gpu) {
            /* the GPU's outputs against the CPU's, one by one: the largest
               difference over the largest output */
            struct janas_matvec_task tc = t;
            tc.y = yc;
            janas_matvec_q4k_group(pool, &tc, 1);
            double dmax = 0, ymax = 0;
            for (size_t i = 0; i < nv * rows; i++) {
                double d = fabs((double)y[i] - yc[i]);
                dmax = d > dmax ? d : dmax;
                ymax = fabs(yc[i]) > ymax ? fabs(yc[i]) : ymax;
            }
            printf("      against the CPU: largest difference %.3g, %.3g of "
                   "the largest output\n",
                   dmax, ymax > 0 ? dmax / ymax : 0);
        }
    }
    janas_gpu_destroy(gpu); /* the driver's memory with it */
    janas_pool_destroy(pool);
    if (mem_kind != 1) {
        free(w);
        free(pl[0]);
        free(pl[1]);
    }
    free(x);
    free(xf);
    free(y);
    free(yc);
    return 0;
}
