/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper.c - a Whisper model opened (whisper.h): its sizes, special tokens
 * and languages from the metadata, its tensors read into memory as the
 * converter laid them out, the tables of the spectrogram, the state.
 */
#define _GNU_SOURCE
#include "whisper.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/pool.h"
#include "common/sysinfo.h"
#include "tokenizer.h"

#define T_F32 0
#define T_F16 1

/* a resident tensor of that name, type and shape (rows x cols, cols the
   length of a stored row), or NULL */
static const void *tensor(struct janas_whisper *w, const char *name,
                          uint32_t type, uint64_t rows, uint64_t cols)
{
    for (uint64_t i = 0; i < w->j.h.n_tensor; i++) {
        const struct janas_jns_tensor *t = &w->j.tensors[i];
        if (strncmp(t->name, name, sizeof t->name))
            continue;
        uint64_t n = t->n_dims == 2 ? t->dims[0] * t->dims[1] : t->dims[0];
        uint64_t esz = type == T_F16 ? 2 : 4;
        if (t->type != type || n != rows * cols || t->bytes != n * esz ||
            (t->n_dims == 2 && t->dims[0] != cols))
            return NULL;
        return w->data + (t->offset - w->j.h.resident_offset);
    }
    return NULL;
}

static const char *read_data(struct janas_whisper *w)
{
    struct janas_jns *j = &w->j;
    w->data = malloc(j->h.resident_bytes ? j->h.resident_bytes : 1);
    if (!w->data)
        return "out of memory";
    size_t got = 0;
    while (got < j->h.resident_bytes) {
        ssize_t k = pread(j->fd, w->data + got, j->h.resident_bytes - got,
                          (off_t)(j->h.resident_offset + got));
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    return got == j->h.resident_bytes ? NULL : "cannot read the weights";
}

static const char *sizes(struct janas_whisper *w)
{
    struct janas_jns *j = &w->j;
    static const struct {
        const char *key;
        size_t at;
    } U[] = {
#define F(k, f) {k, offsetof(struct janas_whisper, f)}
        F("whisper.dim", dim),
        F("whisper.audio.layers", a_layers),
        F("whisper.audio.heads", a_heads),
        F("whisper.audio.ffn", a_ffn),
        F("whisper.audio.ctx", a_ctx),
        F("whisper.text.layers", t_layers),
        F("whisper.text.heads", t_heads),
        F("whisper.text.ffn", t_ffn),
        F("whisper.text.ctx", t_ctx),
        F("whisper.vocab", vocab),
        F("whisper.n_mels", n_mels),
        F("whisper.n_fft", n_fft),
        F("whisper.hop", hop),
        F("whisper.sample_rate", rate),
        F("whisper.chunk_seconds", chunk),
        F("whisper.max_initial_timestamp", max_initial_ts),
#undef F
    };
    for (size_t i = 0; i < sizeof U / sizeof *U; i++)
        if (janas_jns_meta_u32(j, U[i].key, (uint32_t *)((char *)w + U[i].at)))
            return "the model's sizes are missing";
    static const struct {
        const char *key;
        size_t at;
    } S[] = {
#define F(k, f) {"whisper.token." k, offsetof(struct janas_whisper, f)}
        F("eot", eot),
        F("sot", sot),
        F("translate", translate),
        F("transcribe", transcribe),
        F("sot_lm", sot_lm),
        F("sot_prev", sot_prev),
        F("no_speech", no_speech),
        F("no_timestamps", no_timestamps),
        F("timestamp_begin", ts_begin),
#undef F
    };
    for (size_t i = 0; i < sizeof S / sizeof *S; i++) {
        uint32_t v;
        if (janas_jns_meta_u32(j, S[i].key, &v) || v >= w->vocab)
            return "the special tokens are missing";
        *(int32_t *)((char *)w + S[i].at) = (int32_t)v;
    }
    uint32_t multi = 0;
    janas_jns_meta_u32(j, "whisper.multilingual", &multi);
    w->multilingual = multi != 0;
    uint64_t nl;
    if (janas_jns_meta_strings(j, "whisper.languages", &nl, &w->langs) || !nl ||
        nl > 200 || w->sot + nl >= w->vocab)
        return "the languages are missing";
    w->n_lang = (uint32_t)nl;
    if (janas_jns_meta_ints(j, "whisper.suppress", &w->n_suppress,
                            &w->suppress))
        return "the tokens to suppress are missing";
    if (janas_jns_meta_ints(j, "whisper.begin_suppress", &w->n_begin_suppress,
                            &w->begin_suppress))
        w->n_begin_suppress = 0;
    /* the alignment heads; without them OpenAI's default: every head of
       the decoder's second half */
    const int32_t *ah = NULL;
    uint64_t nah = 0;
    if (janas_jns_meta_ints(j, "whisper.alignment_heads", &nah, &ah))
        nah = 0;
    uint64_t pairs = nah / 2;
    if (!pairs)
        pairs = (uint64_t)(w->t_layers - w->t_layers / 2) * w->t_heads;
    w->aheads = malloc(sizeof(int32_t) * 2 * (pairs ? pairs : 1));
    if (!w->aheads)
        return "out of memory";
    w->n_aheads = 0;
    if (nah >= 2) {
        for (uint64_t i = 0; i + 1 < nah; i += 2)
            if (ah[i] >= 0 && (uint32_t)ah[i] < w->t_layers && ah[i + 1] >= 0 &&
                (uint32_t)ah[i + 1] < w->t_heads) {
                w->aheads[2 * w->n_aheads] = ah[i];
                w->aheads[2 * w->n_aheads + 1] = ah[i + 1];
                w->n_aheads++;
            }
    } else {
        for (uint32_t l = w->t_layers / 2; l < w->t_layers; l++)
            for (uint32_t hh = 0; hh < w->t_heads; hh++) {
                w->aheads[2 * w->n_aheads] = (int32_t)l;
                w->aheads[2 * w->n_aheads + 1] = (int32_t)hh;
                w->n_aheads++;
            }
    }
    if (!w->dim || !w->a_heads || !w->t_heads || w->dim % w->a_heads ||
        w->dim % w->t_heads || !w->a_layers || !w->t_layers ||
        w->a_layers > 64 || w->t_layers > 64 || w->t_heads > 64 || !w->a_ctx ||
        !w->t_ctx || !w->n_mels || w->n_fft < 2 || !w->hop || !w->rate ||
        w->a_ctx * 2 * w->hop != w->chunk * w->rate ||
        w->ts_begin >= (int32_t)w->vocab)
        return "the model's sizes are out of range";
    return NULL;
}

static const char *block(struct janas_whisper *w, struct wlayer *L,
                         const char *pre, int cross, uint32_t ffn)
{
    uint32_t d = w->dim;
    char n[64];
    int ok = 1;
#define V(field, part)                                                         \
    snprintf(n, sizeof n, "%s." part, pre);                                    \
    L->field = tensor(w, n, T_F32, 1, d);                                      \
    ok = ok && L->field
#define M(field, part, r, c)                                                   \
    snprintf(n, sizeof n, "%s." part, pre);                                    \
    L->field = tensor(w, n, T_F16, r, c);                                      \
    ok = ok && L->field
    V(ln_w, "ln.w");
    V(ln_b, "ln.b");
    M(q, "q", d, d);
    M(k, "k", d, d);
    M(v, "v", d, d);
    M(o, "o", d, d);
    V(q_b, "q.b");
    V(v_b, "v.b");
    V(o_b, "o.b");
    if (cross) {
        V(xln_w, "xln.w");
        V(xln_b, "xln.b");
        M(xq, "xq", d, d);
        M(xk, "xk", d, d);
        M(xv, "xv", d, d);
        M(xo, "xo", d, d);
        V(xq_b, "xq.b");
        V(xv_b, "xv.b");
        V(xo_b, "xo.b");
    }
    V(mln_w, "mln.w");
    V(mln_b, "mln.b");
    V(fc2_b, "fc2.b");
    M(fc1, "fc1", d, ffn);
    M(fc2, "fc2", ffn, d);
    snprintf(n, sizeof n, "%s.fc1.b", pre);
    L->fc1_b = tensor(w, n, T_F32, 1, ffn);
    ok = ok && L->fc1_b;
#undef V
#undef M
    return ok ? NULL : "a tensor of a block is missing or of another shape";
}

static const char *tensors(struct janas_whisper *w)
{
    uint32_t d = w->dim, bins = w->n_fft / 2 + 1;
    w->mel_filters = tensor(w, "mel_filters", T_F32, w->n_mels, bins);
    w->conv1 = tensor(w, "e.conv1", T_F16, 3ull * w->n_mels, d);
    w->conv1_b = tensor(w, "e.conv1.b", T_F32, 1, d);
    w->conv2 = tensor(w, "e.conv2", T_F16, 3ull * d, d);
    w->conv2_b = tensor(w, "e.conv2.b", T_F32, 1, d);
    w->e_pos = tensor(w, "e.pos", T_F32, 1, (uint64_t)w->a_ctx * d);
    w->e_ln_w = tensor(w, "e.ln.w", T_F32, 1, d);
    w->e_ln_b = tensor(w, "e.ln.b", T_F32, 1, d);
    w->tok = tensor(w, "d.tok", T_F16, d, w->vocab);
    w->d_pos = tensor(w, "d.pos", T_F32, 1, (uint64_t)w->t_ctx * d);
    w->d_ln_w = tensor(w, "d.ln.w", T_F32, 1, d);
    w->d_ln_b = tensor(w, "d.ln.b", T_F32, 1, d);
    if (!w->mel_filters || !w->conv1 || !w->conv1_b || !w->conv2 ||
        !w->conv2_b || !w->e_pos || !w->e_ln_w || !w->e_ln_b || !w->tok ||
        !w->d_pos || !w->d_ln_w || !w->d_ln_b)
        return "a tensor is missing or of another shape";
    w->enc = calloc(w->a_layers, sizeof *w->enc);
    w->dec = calloc(w->t_layers, sizeof *w->dec);
    if (!w->enc || !w->dec)
        return "out of memory";
    const char *why = NULL;
    char pre[16];
    for (uint32_t l = 0; !why && l < w->a_layers; l++) {
        snprintf(pre, sizeof pre, "e.%u", l);
        why = block(w, &w->enc[l], pre, 0, w->a_ffn);
    }
    for (uint32_t l = 0; !why && l < w->t_layers; l++) {
        snprintf(pre, sizeof pre, "d.%u", l);
        why = block(w, &w->dec[l], pre, 1, w->t_ffn);
    }
    return why;
}

/* the DFT's rows with the periodic Hann window folded in */
static const char *tables(struct janas_whisper *w)
{
    uint32_t N = w->n_fft, bins = N / 2 + 1;
    w->dft_cos = malloc(sizeof(float) * bins * N);
    w->dft_sin = malloc(sizeof(float) * bins * N);
    if (!w->dft_cos || !w->dft_sin)
        return "out of memory";
    for (uint32_t k = 0; k < bins; k++)
        for (uint32_t n = 0; n < N; n++) {
            double win = 0.5 - 0.5 * cos(2 * M_PI * n / N);
            /* k n mod N keeps the angle small and exact */
            double a = 2 * M_PI * (double)((uint64_t)k * n % N) / N;
            w->dft_cos[k * N + n] = (float)(win * cos(a));
            w->dft_sin[k * N + n] = (float)(win * sin(a));
        }
    return NULL;
}

static const char *state(struct janas_whisper *w)
{
    size_t d = w->dim, cross = (size_t)w->t_layers * w->a_ctx * d;
    size_t self = (size_t)w->t_layers * w->t_ctx * d;
    w->xk = malloc(sizeof(float) * cross);
    w->xv = malloc(sizeof(float) * cross);
    w->kc = malloc(sizeof(float) * self);
    w->vc = malloc(sizeof(float) * self);
    w->x = malloc(sizeof(float) * w->t_ctx * d);
    w->h = malloc(sizeof(float) * w->t_ctx * d);
    w->q = malloc(sizeof(float) * w->t_ctx * d);
    w->att = malloc(sizeof(float) * w->t_ctx * d);
    w->ff = malloc(sizeof(float) * w->t_ctx * (size_t)w->t_ffn);
    if (!w->xk || !w->xv || !w->kc || !w->vc || !w->x || !w->h || !w->q ||
        !w->att || !w->ff)
        return "out of memory";
    return NULL;
}

struct janas_whisper *janas_whisper_open(const char *path, int n_threads,
                                         char *err, size_t err_len)
{
    struct janas_whisper *w = calloc(1, sizeof(*w));
    if (!w) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    if (janas_jns_open(path, &w->j, err, err_len) != 0) {
        free(w);
        return NULL;
    }
    const char *why = NULL;
    if (strncmp(w->j.h.arch, "whisper", sizeof w->j.h.arch))
        why = "not a Whisper model";
    if (!why)
        why = sizes(w);
    if (!why)
        why = read_data(w);
    if (!why)
        why = tensors(w);
    if (!why)
        why = tables(w);
    if (!why)
        why = state(w);
    if (!why) {
        w->tk = janas_tokenizer_create(&w->j, err, err_len);
        if (!w->tk) {
            janas_whisper_close(w);
            return NULL;
        }
    }
    if (!why) {
        /* one thread a performance core of those the caller may use (its
           threads inherit the caller's CPUs: a caller a chat model pinned
           to one gets one) */
        int n = n_threads;
        struct janas_cpu_layout lay;
        if (n <= 0)
            n = janas_cpu_layout(NULL, 0, &lay) == 0 ? lay.n_p_cores : 1;
        w->pool = janas_pool_create(n > 0 ? n : 1, NULL);
        if (!w->pool)
            why = "cannot start the threads";
    }
    if (why) {
        snprintf(err, err_len, "%s: %s", path, why);
        janas_whisper_close(w);
        return NULL;
    }
    return w;
}

void janas_whisper_close(struct janas_whisper *w)
{
    if (!w)
        return;
    if (w->pool)
        janas_pool_destroy(w->pool);
    if (w->tk)
        janas_tokenizer_destroy(w->tk);
    free(w->enc);
    free(w->dec);
    free(w->dft_cos);
    free(w->dft_sin);
    free(w->xk);
    free(w->xv);
    free(w->kc);
    free(w->vc);
    free(w->x);
    free(w->h);
    free(w->q);
    free(w->att);
    free(w->ff);
    free(w->aheads);
    free(w->data);
    janas_jns_close(&w->j);
    free(w);
}

void janas_whisper_lang(const struct janas_whisper *w, uint32_t i, char *buf,
                        size_t cap)
{
    const uint8_t *p = w->langs;
    if (!cap)
        return;
    buf[0] = 0;
    for (uint32_t k = 0; k < w->n_lang; k++) {
        uint64_t len;
        memcpy(&len, p, 8);
        if (k == i) {
            size_t n = len < cap - 1 ? (size_t)len : cap - 1;
            memcpy(buf, p + 8, n);
            buf[n] = 0;
            return;
        }
        p += 8 + len;
    }
}
