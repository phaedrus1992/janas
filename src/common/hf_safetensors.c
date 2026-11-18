/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf_safetensors.c - safetensors read for the tools (hf_safetensors.h).
 */
#define _FILE_OFFSET_BITS 64
#include "hf_safetensors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "llm/json.h"

int janas_st_open(struct janas_st *s, const char *path, char *err,
                  size_t err_len)
{
    memset(s, 0, sizeof(*s));
    s->f = fopen(path, "rb");
    uint64_t n = 0;
    if (!s->f || fread(&n, 8, 1, s->f) != 1 || n > ((uint64_t)1 << 30)) {
        snprintf(err, err_len, "%s: not a safetensors file", path);
        janas_st_close(s);
        return -1;
    }
    char *json = malloc(n + 1);
    if (!json || fread(json, 1, n, s->f) != n) {
        free(json);
        snprintf(err, err_len, "%s: cannot read its header", path);
        janas_st_close(s);
        return -1;
    }
    char why[160];
    s->doc = janas_json_parse(json, n, why, sizeof why);
    free(json);
    if (!s->doc || janas_json_root(s->doc)->type != JANAS_JSON_OBJECT) {
        snprintf(err, err_len, "%s: its header is not JSON (%s)", path,
                 s->doc ? "not an object" : why);
        janas_st_close(s);
        return -1;
    }
    s->data_start = 8 + n;
    if (fseeko(s->f, 0, SEEK_END) != 0 || ftello(s->f) < (off_t)s->data_start) {
        snprintf(err, err_len, "%s: shorter than its header", path);
        janas_st_close(s);
        return -1;
    }
    s->size = (uint64_t)ftello(s->f);
    return 0;
}

void janas_st_close(struct janas_st *s)
{
    if (s->f)
        fclose(s->f);
    janas_json_free(s->doc);
    memset(s, 0, sizeof(*s));
}

static int u64_of(const struct janas_json *v, uint64_t *out)
{
    if (!v || v->type != JANAS_JSON_NUMBER || !v->n || v->n > 20)
        return -1;
    uint64_t x = 0;
    for (size_t i = 0; i < v->n; i++) {
        if (v->s[i] < '0' || v->s[i] > '9' || x > (UINT64_MAX - 9) / 10)
            return -1;
        x = x * 10 + (uint64_t)(v->s[i] - '0');
    }
    *out = x;
    return 0;
}

int janas_st_find(const struct janas_st *s, const char *name,
                  struct janas_st_tensor *t)
{
    const struct janas_json *e = janas_json_get(janas_json_root(s->doc), name);
    const struct janas_json *dt = janas_json_get(e, "dtype");
    const struct janas_json *sh = janas_json_get(e, "shape");
    const struct janas_json *of = janas_json_get(e, "data_offsets");
    memset(t, 0, sizeof(*t));
    size_t esize;
    if (janas_json_is(dt, "F32"))
        t->dtype = JANAS_ST_F32, esize = 4;
    else if (janas_json_is(dt, "F16"))
        t->dtype = JANAS_ST_F16, esize = 2;
    else if (janas_json_is(dt, "BF16"))
        t->dtype = JANAS_ST_BF16, esize = 2;
    else
        return -1;
    if (!sh || sh->type != JANAS_JSON_ARRAY || sh->n > 8 || !of ||
        of->type != JANAS_JSON_ARRAY || of->n != 2)
        return -1;
    t->count = 1;
    for (const struct janas_json *d = sh->child; d; d = d->next) {
        if (u64_of(d, &t->shape[t->dims]) || !t->shape[t->dims] ||
            __builtin_mul_overflow(t->count, t->shape[t->dims], &t->count))
            return -1;
        t->dims++;
    }
    if (u64_of(of->child, &t->begin) || u64_of(of->child->next, &t->end) ||
        t->end < t->begin || t->end > s->size - s->data_start ||
        t->count > UINT64_MAX / esize || t->end - t->begin != t->count * esize)
        return -1;
    return 0;
}

static void *read_raw(const struct janas_st *s, const struct janas_st_tensor *t)
{
    size_t n = (size_t)(t->end - t->begin);
    void *raw = malloc(n ? n : 1);
    if (!raw ||
        fseeko(s->f, (off_t)(s->data_start + t->begin), SEEK_SET) != 0 ||
        fread(raw, 1, n, s->f) != n) {
        free(raw);
        return NULL;
    }
    return raw;
}

static float half_to_float(uint16_t h)
{
    int e = h >> 10 & 31, m = h & 1023;
    float v = e == 0    ? ldexpf((float)m, -24)
              : e == 31 ? (m ? NAN : INFINITY)
                        : ldexpf((float)(m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}

float *janas_st_floats(const struct janas_st *s,
                       const struct janas_st_tensor *t)
{
    void *raw = read_raw(s, t);
    float *x = raw ? malloc(sizeof(float) * (size_t)t->count) : NULL;
    if (!x) {
        free(raw);
        return NULL;
    }
    for (uint64_t i = 0; i < t->count; i++) {
        if (t->dtype == JANAS_ST_F32) {
            memcpy(&x[i], (char *)raw + 4 * i, 4);
        } else {
            uint16_t h;
            memcpy(&h, (char *)raw + 2 * i, 2);
            if (t->dtype == JANAS_ST_BF16) {
                uint32_t u = (uint32_t)h << 16;
                memcpy(&x[i], &u, 4);
            } else {
                x[i] = half_to_float(h);
            }
        }
    }
    free(raw);
    return x;
}

uint16_t *janas_st_bf16(const struct janas_st *s,
                        const struct janas_st_tensor *t)
{
    if (t->dtype == JANAS_ST_BF16)
        return read_raw(s, t);
    float *x = janas_st_floats(s, t);
    uint16_t *b = x ? malloc(2 * (size_t)t->count) : NULL;
    for (uint64_t i = 0; b && i < t->count; i++) {
        uint32_t u;
        memcpy(&u, &x[i], 4);
        if ((u & 0x7fffffff) > 0x7f800000) /* NaN stays NaN */
            b[i] = (uint16_t)(u >> 16 | 0x40);
        else
            b[i] = (uint16_t)((u + 0x7fff + (u >> 16 & 1)) >> 16);
    }
    free(x);
    return b;
}
