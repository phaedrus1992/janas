/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns_write.h - a JNS file of resident tensors only (no layers, no
 * experts), as the converters of encoders write it (jns_write.c): the
 * tensors and the metadata gathered, then written at once and read back as
 * the engine reads it.
 */
#ifndef JANAS_LLM_JNS_WRITE_H
#define JANAS_LLM_JNS_WRITE_H

#include <stdint.h>

#include "json.h"

struct janas_jw_tensor {
    char name[64];
    uint32_t type, n_dims;
    uint64_t dims[4]; /* ggml order: the row length first */
    void *data;       /* the writer's: freed with it */
    uint64_t bytes, offset;
};

struct janas_jw {
    struct janas_jw_tensor *t;
    int n, cap;
    struct janas_buf meta; /* GGUF's key/value encoding */
    uint64_t n_meta;
};

/* A tensor; data (malloc'd) becomes the writer's, freed on failure too.
   0, or -1 (out of memory, a name of 64 bytes or more). */
int janas_jw_tensor(struct janas_jw *w, const char *name, uint32_t type,
                    uint32_t n_dims, const uint64_t *dims, void *data,
                    uint64_t bytes);
/* Metadata; a failure is remembered (oom) and reported by the write. */
void janas_jw_u32(struct janas_jw *w, const char *k, uint32_t v);
void janas_jw_f32(struct janas_jw *w, const char *k, float v);
void janas_jw_str(struct janas_jw *w, const char *k, const char *v);
/* an array of n strings, each of lens[i] bytes (lens NULL: NUL-ended) */
void janas_jw_strings(struct janas_jw *w, const char *k, uint64_t n,
                      const char *const *v, const uint64_t *lens);
void janas_jw_ints(struct janas_jw *w, const char *k, uint64_t n,
                   const int32_t *v);
/* The file: header (arch, d_model), tensor directory, metadata, data.
   0, or -1 with the reason in err. */
int janas_jw_write(struct janas_jw *w, const char *path, const char *arch,
                   uint32_t d_model, uint64_t *file_bytes, char *err,
                   size_t err_len);
void janas_jw_free(struct janas_jw *w);

#endif
