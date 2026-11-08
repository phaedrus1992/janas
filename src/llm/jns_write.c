/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns_write.c - a JNS file of resident tensors only (jns_write.h).
 */
#define _GNU_SOURCE
#include "jns_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jns.h"

#define GG_I32 5
#define GG_STRING 8
#define GG_ARRAY 9

static uint64_t align_up(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

int janas_jw_tensor(struct janas_jw *w, const char *name, uint32_t type,
                    uint32_t n_dims, const uint64_t *dims, void *data,
                    uint64_t bytes)
{
    if (strlen(name) >= 64 || n_dims < 1 || n_dims > 4 || !data) {
        free(data);
        return -1;
    }
    if (w->n == w->cap) {
        int cap = w->cap ? 2 * w->cap : 64;
        struct janas_jw_tensor *t = realloc(w->t, sizeof(*t) * (size_t)cap);
        if (!t) {
            free(data);
            return -1;
        }
        w->t = t;
        w->cap = cap;
    }
    struct janas_jw_tensor *t = &w->t[w->n++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", name);
    t->type = type;
    t->n_dims = n_dims;
    memcpy(t->dims, dims, sizeof(uint64_t) * n_dims);
    t->data = data;
    t->bytes = bytes;
    return 0;
}

static void key(struct janas_jw *w, const char *k, uint32_t type)
{
    uint64_t n = strlen(k);
    janas_buf_put(&w->meta, (const char *)&n, 8);
    janas_buf_put(&w->meta, k, n);
    janas_buf_put(&w->meta, (const char *)&type, 4);
    w->n_meta++;
}

void janas_jw_u32(struct janas_jw *w, const char *k, uint32_t v)
{
    key(w, k, 4);
    janas_buf_put(&w->meta, (const char *)&v, 4);
}

void janas_jw_f32(struct janas_jw *w, const char *k, float v)
{
    key(w, k, 6);
    janas_buf_put(&w->meta, (const char *)&v, 4);
}

void janas_jw_str(struct janas_jw *w, const char *k, const char *v)
{
    key(w, k, GG_STRING);
    uint64_t n = strlen(v);
    janas_buf_put(&w->meta, (const char *)&n, 8);
    janas_buf_put(&w->meta, v, n);
}

void janas_jw_strings(struct janas_jw *w, const char *k, uint64_t n,
                      const char *const *v, const uint64_t *lens)
{
    uint32_t et = GG_STRING;
    key(w, k, GG_ARRAY);
    janas_buf_put(&w->meta, (const char *)&et, 4);
    janas_buf_put(&w->meta, (const char *)&n, 8);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t len = lens ? lens[i] : strlen(v[i]);
        janas_buf_put(&w->meta, (const char *)&len, 8);
        janas_buf_put(&w->meta, v[i], len);
    }
}

void janas_jw_ints(struct janas_jw *w, const char *k, uint64_t n,
                   const int32_t *v)
{
    uint32_t et = GG_I32;
    key(w, k, GG_ARRAY);
    janas_buf_put(&w->meta, (const char *)&et, 4);
    janas_buf_put(&w->meta, (const char *)&n, 8);
    janas_buf_put(&w->meta, (const char *)v, 4 * n);
}

int janas_jw_write(struct janas_jw *w, const char *path, const char *arch,
                   uint32_t d_model, uint64_t *file_bytes, char *err,
                   size_t err_len)
{
    struct janas_buf mreg = {0};
    janas_buf_put(&mreg, (const char *)&w->n_meta, 8);
    janas_buf_put(&mreg, w->meta.p, w->meta.n);
    uint64_t A = JANAS_JNS_ALIGN, dir_off = A;
    uint64_t dir_len = sizeof(struct janas_jns_tensor) * (uint64_t)w->n;
    uint8_t *dirb = calloc(1, dir_len ? dir_len : 1);
    FILE *f = NULL;
    int rc = -1;
    if (w->meta.oom || mreg.oom || !dirb) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    /* layout: header, tensor directory, metadata, resident data */
    uint64_t meta_off = align_up(dir_off + dir_len, A);
    uint64_t res_off = align_up(meta_off + mreg.n, A), pos = res_off;
    for (int i = 0; i < w->n; i++) {
        w->t[i].offset = pos;
        pos = align_up(pos + w->t[i].bytes, A);
    }
    f = fopen(path, "w+b");
    if (!f || ftruncate(fileno(f), (off_t)pos)) {
        snprintf(err, err_len, "cannot create %s", path);
        goto done;
    }
    for (int i = 0; i < w->n; i++) {
        struct janas_jns_tensor t = {0};
        memcpy(t.name, w->t[i].name, sizeof t.name);
        t.type = w->t[i].type;
        t.n_dims = w->t[i].n_dims;
        memcpy(t.dims, w->t[i].dims, sizeof t.dims);
        t.offset = w->t[i].offset;
        t.bytes = w->t[i].bytes;
        memcpy(dirb + (size_t)i * sizeof t, &t, sizeof t);
        if (fseeko(f, (off_t)t.offset, SEEK_SET) ||
            fwrite(w->t[i].data, 1, t.bytes, f) != t.bytes)
            goto failed;
    }
    if (fseeko(f, (off_t)dir_off, SEEK_SET) ||
        fwrite(dirb, 1, dir_len, f) != dir_len ||
        fseeko(f, (off_t)meta_off, SEEK_SET) ||
        fwrite(mreg.p, 1, mreg.n, f) != mreg.n)
        goto failed;
    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = JANAS_JNS_VERSION;
    h.header_bytes = JANAS_JNS_ALIGN;
    h.file_bytes = pos;
    snprintf(h.arch, sizeof h.arch, "%s", arch);
    h.d_model = d_model;
    h.layer_table_offset = dir_off;
    h.expert_table_offset = dir_off;
    h.expert_sums_offset = dir_off;
    h.tensor_dir_offset = dir_off;
    h.n_tensor = (uint64_t)w->n;
    h.metadata_offset = meta_off;
    h.metadata_bytes = mreg.n;
    h.resident_offset = res_off;
    h.resident_bytes = pos - res_off;
    h.experts_offset = pos;
    h.tables_checksum = janas_fnv1a(dirb, dir_len, JANAS_FNV_INIT);
    h.metadata_checksum = janas_crc32(0, mreg.p, mreg.n);
    h.header_checksum = janas_fnv1a(&h, sizeof h, JANAS_FNV_INIT);
    if (fseeko(f, 0, SEEK_SET) || fwrite(&h, 1, sizeof h, f) != sizeof h)
        goto failed;
    int closed = fclose(f);
    f = NULL;
    if (closed != 0)
        goto failed;
    /* read back as the engine reads it */
    struct janas_jns j;
    char e[256];
    if (janas_jns_open(path, &j, e, sizeof e) != 0) {
        snprintf(err, err_len, "the file written does not read back: %s", e);
        goto done;
    }
    janas_jns_close(&j);
    if (file_bytes)
        *file_bytes = pos;
    rc = 0;
    goto done;
failed:
    snprintf(err, err_len, "write failed: %s", path);
done:
    if (f)
        fclose(f);
    free(dirb);
    free(mreg.p);
    return rc;
}

void janas_jw_free(struct janas_jw *w)
{
    for (int i = 0; i < w->n; i++)
        free(w->t[i].data);
    free(w->t);
    free(w->meta.p);
    memset(w, 0, sizeof *w);
}
