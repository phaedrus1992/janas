/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf_safetensors.h - a safetensors file (Hugging Face's checkpoints) read
 * for the tools (hf_safetensors.c): an 8-byte header length, a JSON header
 * naming each tensor's type, shape and byte range, the data. Every field
 * is checked against the file before it is used.
 */
#ifndef JANAS_COMMON_HF_SAFETENSORS_H
#define JANAS_COMMON_HF_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct janas_json_doc;

struct janas_st {
    FILE *f;
    struct janas_json_doc *doc; /* the header */
    uint64_t data_start, size;
};

enum { JANAS_ST_F32, JANAS_ST_F16, JANAS_ST_BF16 };

struct janas_st_tensor {
    int dtype; /* JANAS_ST_... */
    int dims;
    uint64_t shape[8]; /* as the checkpoint gives it, outermost first;
                          none for a scalar (dims 0, count 1) */
    uint64_t count;    /* elements */
    uint64_t begin, end;
};

/* 0, or -1 with the reason in err. */
int janas_st_open(struct janas_st *s, const char *path, char *err,
                  size_t err_len);
void janas_st_close(struct janas_st *s);
/* The tensor of that name: 0, or -1 when it is not there or its entry is
   not one this reads (F32, F16, BF16; one to eight dimensions; its bytes
   inside the file and as many as its shape says). */
int janas_st_find(const struct janas_st *s, const char *name,
                  struct janas_st_tensor *t);
/* Its elements as float (malloc'd), or NULL. */
float *janas_st_floats(const struct janas_st *s,
                       const struct janas_st_tensor *t);
/* Its elements as bfloat16 (malloc'd; F32 and F16 rounded to nearest even),
   or NULL. */
uint16_t *janas_st_bf16(const struct janas_st *s,
                        const struct janas_st_tensor *t);

#endif
