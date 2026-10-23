/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns_neurons.c - rewrite a model file with the experts' down matrix by
 * neuron (JANAS_Q6_K_PN, quant.h): a row a neuron, its outputs across the
 * model's width, in Q6_K's three bit planes, so that the engine reads each
 * neuron of a token at the depth its contribution earns - six, four or two
 * bits, or none (model_neurons.c) - where by rows it reads every column.
 *
 * The rows are stored by block of 256 outputs - every neuron's block 0,
 * then every neuron's block 1 - since the engine reads a block of outputs
 * of a chunk of neurons at a time (quant.h janas_q6kpn_chunk): read by
 * neuron, the blocks of a chunk lay 820 bytes apart (Qwen3-4B) and the
 * hardware prefetcher did not follow them.
 *
 * Each output channel is divided by its root mean square over the neurons,
 * kept as a float before the rows (quant.h JANAS_Q6_K_PN) and multiplied
 * back after the sum: the few channels of the residual stream far larger
 * than the others would otherwise set the scales of every group of 16 they
 * fall in, and the others there would lose their bits.
 *
 * Layers whose down matrix is Q6_K, in planes or not, are converted; with
 * --q4k those in Q4_K too, whose six bits then read more bytes than the
 * Q4_K they come from. Each expert's down is unpacked, turned over and
 * quantized again to Q6_K by Janas's own quantizer (janas_q6k_quantize):
 * a second rounding at six bits, so the weights are close to the original
 * ones, not bit for bit. Gate, up, the other layers and the resident
 * weights are copied as they are.
 *
 * Usage: jns_neurons [--q4k] [--threads N] <in.jns> <out.jns>
 */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/pool.h"
#include "llm/jns.h"
#include "llm/quant.h"

#define ALIGN JANAS_JNS_ALIGN

static uint64_t align_up(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}

/* Expert slots are sorted by the offset they have in the source file. */
static const uint64_t *sort_key;

static int by_offset(const void *a, const void *b)
{
    uint64_t x = sort_key[*(const uint32_t *)a];
    uint64_t y = sort_key[*(const uint32_t *)b];
    return x < y ? -1 : x > y ? 1 : 0;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int write_at(int fd, const void *buf, size_t n, uint64_t off)
{
    const char *p = buf;
    while (n) {
        ssize_t w = pwrite(fd, p, n, (off_t)off);
        if (w <= 0)
            return -1;
        p += w;
        n -= (size_t)w;
        off += (uint64_t)w;
    }
    return 0;
}

static int read_at(int fd, void *buf, size_t n, uint64_t off)
{
    char *p = buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0)
            return -1;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return 0;
}

/* Copies n bytes between two files, through a buffer. */
static int copy_region(int in, uint64_t src, int out, uint64_t dst, uint64_t n)
{
    enum { CHUNK = 8u << 20 };
    char *buf = malloc(CHUNK);
    if (!buf)
        return -1;
    while (n) {
        size_t c = n < CHUNK ? (size_t)n : CHUNK;
        if (read_at(in, buf, c, src) || write_at(out, buf, c, dst)) {
            free(buf);
            return -1;
        }
        src += c;
        dst += c;
        n -= c;
    }
    free(buf);
    return 0;
}

/* Whether layer l's down matrix is turned into neurons. */
static int converted(const struct janas_jns_layer *ly, int q4k)
{
    uint32_t t = ly->m[JANAS_JNS_DOWN].type;
    const struct janas_jns_matrix *d = &ly->m[JANAS_JNS_DOWN];
    return (t == JANAS_Q6_K || t == JANAS_Q6_K_P || (q4k && t == JANAS_Q4_K)) &&
           d->rows % JANAS_QK == 0 && d->cols % JANAS_QK == 0;
}

/* The source's down matrix of a slot, rows x cols floats (converted() has
   checked its type). */
static void unpack_down(const struct janas_jns_layer *so, const uint8_t *src,
                        float *w)
{
    const struct janas_jns_matrix *d = &so->m[JANAS_JNS_DOWN];
    size_t nb = d->cols / JANAS_QK;
    for (uint32_t r = 0; r < d->rows; r++) {
        float *y = w + (size_t)r * d->cols;
        if (d->type == JANAS_Q6_K_P) {
            const struct janas_block_q6kp *base =
                (const struct janas_block_q6kp *)(src + d->offset) + r * nb;
            janas_q6kp_dequantize(
                base, src + so->plane[0].offset + r * nb * JANAS_Q6KP_PLANE,
                src + so->plane[1].offset + r * nb * JANAS_Q6KP_PLANE, y,
                d->cols);
        } else {
            size_t rb = nb * janas_qtype_block_size((int)d->type);
            janas_dequantize((int)d->type, src + d->offset + r * rb, y,
                             d->cols);
        }
    }
}

struct job {
    const struct janas_jns *j;
    const struct janas_jns_layer *ly; /* the new layer table */
    const uint32_t *order;
    const uint64_t *new_off;
    uint64_t *new_sum;
    uint64_t n_slots, max_src, max_slot, total;
    int out;
    double t0;
    uint64_t next, done; /* atomic */
    int failed;          /* atomic */
};

/* A slot rewritten: gate and up as they are, down by neuron. */
static int convert_slot(struct job *J, uint32_t s, uint8_t *src, uint8_t *dst,
                        float *w, float *t, struct janas_block_q6k *q)
{
    const struct janas_jns *j = J->j;
    uint32_t l = s / j->h.n_expert;
    const struct janas_jns_layer *so = &j->layers[l], *ly = &J->ly[l];
    if (read_at(j->fd, src, so->slot_bytes, j->expert_offset[s]) != 0)
        return -1;
    memset(dst, 0, ly->slot_bytes);
    if (ly->m[JANAS_JNS_DOWN].type != JANAS_Q6_K_PN) {
        memcpy(dst, src, so->slot_bytes);
    } else {
        for (int m = 0; m < 2; m++)
            memcpy(dst + ly->m[m].offset, src + so->m[m].offset,
                   so->m[m].bytes);
        const struct janas_jns_matrix *d = &so->m[JANAS_JNS_DOWN];
        uint32_t dm = d->rows, ff = d->cols;
        unpack_down(so, src, w);
        /* each output channel over its root mean square, which goes first:
           the channels of the residual stream far larger than the others
           would set the scales of every group they fall in */
        float *scale = (float *)(dst + ly->m[JANAS_JNS_DOWN].offset);
        for (uint32_t r = 0; r < dm; r++) {
            double ss = 0;
            for (uint32_t c = 0; c < ff; c++)
                ss += (double)w[(size_t)r * ff + c] * w[(size_t)r * ff + c];
            float rms = (float)sqrt(ss / ff);
            scale[r] = rms > 0.0f ? rms : 1.0f;
        }
        for (uint32_t r = 0; r < dm; r++) /* a row a neuron */
            for (uint32_t c = 0; c < ff; c++)
                t[(size_t)c * dm + r] = w[(size_t)r * ff + c] / scale[r];
        size_t nb = (size_t)ff * (dm / JANAS_QK), bpr = dm / JANAS_QK;
        janas_q6k_quantize(t, q, (size_t)ff * dm);
        /* by block of outputs: every neuron's block 0, then every neuron's
           block 1 - what the engine reads together lies together (w, done
           with, holds them) */
        struct janas_block_q6k *qb = (struct janas_block_q6k *)(void *)w;
        for (size_t jn = 0; jn < ff; jn++)
            for (size_t b = 0; b < bpr; b++)
                qb[b * ff + jn] = q[jn * bpr + b];
        janas_q6k_to_planes(
            qb, nb,
            (struct janas_block_q6kp *)(dst + ly->m[JANAS_JNS_DOWN].offset +
                                        dm * sizeof(float)),
            dst + ly->plane[0].offset, dst + ly->plane[1].offset);
    }
    J->new_sum[s] = janas_crc32(0, dst, (size_t)ly->slot_bytes);
    return write_at(J->out, dst, (size_t)ly->slot_bytes, J->new_off[s]);
}

static void worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct job *J = arg;
    const struct janas_jns *j = J->j;
    size_t floats = 0;
    for (uint32_t l = 0; l < j->h.n_layer; l++) {
        const struct janas_jns_matrix *d = &j->layers[l].m[JANAS_JNS_DOWN];
        if (J->ly[l].m[JANAS_JNS_DOWN].type == JANAS_Q6_K_PN &&
            (size_t)d->rows * d->cols > floats)
            floats = (size_t)d->rows * d->cols;
    }
    uint8_t *src = malloc(J->max_src), *dst = malloc(J->max_slot);
    float *w = malloc((floats ? floats : 1) * sizeof(float));
    float *t = malloc((floats ? floats : 1) * sizeof(float));
    struct janas_block_q6k *q =
        malloc((floats / JANAS_QK + 1) * sizeof(struct janas_block_q6k));
    if (!src || !dst || !w || !t || !q)
        __atomic_store_n(&J->failed, 1, __ATOMIC_RELAXED);
    while (!__atomic_load_n(&J->failed, __ATOMIC_RELAXED)) {
        uint64_t i = __atomic_fetch_add(&J->next, 1, __ATOMIC_RELAXED);
        if (i >= J->n_slots)
            break;
        uint32_t s = J->order[i];
        if (convert_slot(J, s, src, dst, w, t, q) != 0) {
            fprintf(stderr, "\nexpert %u of layer %u: cannot convert\n",
                    s % j->h.n_expert, s / j->h.n_expert);
            __atomic_store_n(&J->failed, 1, __ATOMIC_RELAXED);
            break;
        }
        uint64_t done = __atomic_add_fetch(&J->done, 1, __ATOMIC_RELAXED);
        if ((done & 255) == 0 || done == J->n_slots) {
            double dt = now() - J->t0;
            fprintf(stderr, "\r%5.1f%%  %.1f GB in %.0f s   ",
                    100.0 * (double)done / (double)J->n_slots,
                    (double)J->total * (double)done / (double)J->n_slots / 1e9,
                    dt);
        }
    }
    free(src);
    free(dst);
    free(w);
    free(t);
    free(q);
}

int main(int argc, char **argv)
{
    int q4k = 0, threads = (int)sysconf(_SC_NPROCESSORS_ONLN), a = 1;
    for (; a < argc && argv[a][0] == '-'; a++) {
        if (!strcmp(argv[a], "--q4k"))
            q4k = 1;
        else if (!strcmp(argv[a], "--threads") && a + 1 < argc)
            threads = atoi(argv[++a]);
        else
            break;
    }
    if (argc - a != 2 || threads < 1) {
        fprintf(
            stderr,
            "usage: jns_neurons [--q4k] [--threads N] <in.jns> <out.jns>\n");
        return 2;
    }
    const char *in = argv[a], *out_path = argv[a + 1];
    struct janas_jns j;
    char err[256];
    if (janas_jns_open(in, &j, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s: %s\n", in, err);
        return 1;
    }
    uint32_t n_layer = j.h.n_layer, n_expert = j.h.n_expert;
    uint64_t n_slots = (uint64_t)n_layer * n_expert;

    /* the new layer table: gate, up and down's base plane first, then its
       other two planes, each level padded to a whole number of pages */
    struct janas_jns_layer *ly = calloc(n_layer, sizeof(*ly));
    uint64_t *new_off = malloc(n_slots * sizeof(uint64_t));
    uint64_t *new_sum = malloc(n_slots * sizeof(uint64_t));
    uint32_t *order = malloc(n_slots * sizeof(uint32_t));
    if (!ly || !new_off || !new_sum || !order)
        return 1;
    uint32_t turned = 0;
    for (uint32_t l = 0; l < n_layer; l++) {
        const struct janas_jns_layer *src = &j.layers[l];
        ly[l] = *src;
        if (!converted(src, q4k))
            continue;
        turned++;
        memset(ly[l].plane, 0, sizeof(ly[l].plane));
        const struct janas_jns_matrix *dn = &src->m[JANAS_JNS_DOWN];
        uint64_t nb = (uint64_t)dn->cols * (dn->rows / JANAS_QK);
        uint64_t off = 0;
        for (int m = 0; m < 3; m++) {
            ly[l].m[m].offset = off;
            if (m == JANAS_JNS_DOWN) {
                ly[l].m[m] = (struct janas_jns_matrix){
                    .type = JANAS_Q6_K_PN,
                    .rows = dn->cols, /* the neurons */
                    .cols = dn->rows, /* the outputs */
                    .offset = off,
                    .bytes = (uint64_t)dn->rows * sizeof(float) +
                             nb * sizeof(struct janas_block_q6kp)};
            }
            off = align_up(off + ly[l].m[m].bytes, 256);
        }
        ly[l].level_bytes[0] = align_up(off, ALIGN);
        for (int p = 0; p < 2; p++) {
            ly[l].plane[p] = ly[l].m[JANAS_JNS_DOWN];
            ly[l].plane[p].offset = ly[l].level_bytes[p];
            ly[l].plane[p].bytes = nb * JANAS_Q6KP_PLANE;
            ly[l].level_bytes[p + 1] =
                align_up(ly[l].plane[p].offset + ly[l].plane[p].bytes, ALIGN);
        }
        ly[l].slot_bytes = ly[l].level_bytes[2];
    }
    if (!turned) {
        fprintf(stderr, "%s: no down matrix to turn into neurons%s\n", in,
                q4k ? "" : " (Q4_K ones only with --q4k)");
        return 1;
    }

    /* header and tables: as the source's, the offsets after the resident
       weights moved by what the slots grew */
    struct janas_jns_header h = j.h;
    h.version = JANAS_JNS_VERSION;
    uint64_t layer_len = (uint64_t)n_layer * sizeof(struct janas_jns_layer);
    uint64_t slot_len = n_slots * sizeof(uint64_t);
    uint64_t dir_len = j.h.n_tensor * sizeof(struct janas_jns_tensor);
    h.layer_table_offset = ALIGN;
    h.expert_table_offset = h.layer_table_offset + layer_len;
    h.expert_sums_offset = h.expert_table_offset + slot_len;
    h.tensor_dir_offset = h.expert_sums_offset + slot_len;
    h.metadata_offset = h.tensor_dir_offset + dir_len;
    h.resident_offset = align_up(h.metadata_offset + h.metadata_bytes, ALIGN);
    h.experts_offset = align_up(h.resident_offset + h.resident_bytes, ALIGN);

    struct janas_jns_tensor *tensors =
        malloc(dir_len ? (size_t)dir_len : (size_t)1);
    if (!tensors)
        return 1;
    memcpy(tensors, j.tensors, (size_t)dir_len);
    int64_t shift = (int64_t)h.resident_offset - (int64_t)j.h.resident_offset;
    for (uint64_t t = 0; t < h.n_tensor; t++)
        tensors[t].offset = (uint64_t)((int64_t)tensors[t].offset + shift);

    /* the experts keep the order they have on disk (co-activation) */
    for (uint64_t i = 0; i < n_slots; i++)
        order[i] = (uint32_t)i;
    sort_key = j.expert_offset;
    qsort(order, (size_t)n_slots, sizeof(uint32_t), by_offset);

    uint64_t at = h.experts_offset;
    for (uint64_t i = 0; i < n_slots; i++) {
        uint32_t s = order[i];
        new_off[s] = at;
        at += ly[s / n_expert].slot_bytes;
    }
    h.experts_bytes = at - h.experts_offset;
    h.file_bytes = at;

    int out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        fprintf(stderr, "cannot create %s\n", out_path);
        return 1;
    }
    if (copy_region(j.fd, j.h.resident_offset, out, h.resident_offset,
                    h.resident_bytes) != 0) {
        fprintf(stderr, "cannot copy the resident weights\n");
        return 1;
    }

    struct job J = {.j = &j,
                    .ly = ly,
                    .order = order,
                    .new_off = new_off,
                    .new_sum = new_sum,
                    .n_slots = n_slots,
                    .total = h.experts_bytes,
                    .out = out,
                    .t0 = now()};
    for (uint32_t l = 0; l < n_layer; l++) {
        if (ly[l].slot_bytes > J.max_slot)
            J.max_slot = ly[l].slot_bytes;
        if (j.layers[l].slot_bytes > J.max_src)
            J.max_src = j.layers[l].slot_bytes;
    }
    struct janas_pool *pool = janas_pool_create(threads, NULL);
    if (!pool)
        return 1;
    janas_pool_run(pool, worker, &J);
    janas_pool_destroy(pool);
    fprintf(stderr, "\n");
    if (J.failed || J.done != n_slots)
        return 1;

    /* tables last, with the checksums over what was written */
    uint64_t sum = JANAS_FNV_INIT;
    sum = janas_fnv1a(ly, (size_t)layer_len, sum);
    sum = janas_fnv1a(new_off, (size_t)slot_len, sum);
    sum = janas_fnv1a(new_sum, (size_t)slot_len, sum);
    sum = janas_fnv1a(tensors, (size_t)dir_len, sum);
    h.tables_checksum = sum;
    h.metadata_checksum = janas_crc32(0, j.metadata, (size_t)h.metadata_bytes);
    h.header_checksum = 0;
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);

    uint8_t head[ALIGN];
    memset(head, 0, sizeof(head));
    memcpy(head, &h, sizeof(h));
    if (write_at(out, head, sizeof(head), 0) ||
        write_at(out, ly, (size_t)layer_len, h.layer_table_offset) ||
        write_at(out, new_off, (size_t)slot_len, h.expert_table_offset) ||
        write_at(out, new_sum, (size_t)slot_len, h.expert_sums_offset) ||
        write_at(out, tensors, (size_t)dir_len, h.tensor_dir_offset) ||
        write_at(out, j.metadata, (size_t)h.metadata_bytes,
                 h.metadata_offset)) {
        fprintf(stderr, "cannot write the tables\n");
        return 1;
    }
    if (ftruncate(out, (off_t)h.file_bytes) != 0 || close(out) != 0) {
        fprintf(stderr, "cannot close %s\n", out_path);
        return 1;
    }
    printf("%s: %u layers, %u of them with the down matrix by neuron, "
           "%.1f GB\n",
           out_path, n_layer, turned, (double)h.file_bytes / 1e9);
    free(ly);
    free(new_off);
    free(new_sum);
    free(order);
    free(tensors);
    janas_jns_close(&j);
    return 0;
}
