/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper_mel.c - Whisper's log-mel spectrogram (whisper.h), as OpenAI's
 * log_mel_spectrogram computes it: the sound followed by zeros, reflected
 * at both ends by half a window (torch.stft's center), a periodic Hann
 * window of n_fft samples every hop, the power of each frequency, the last
 * frame dropped; the model's mel filters over the powers, log10 (at least
 * 1e-10), nothing below the largest less 8, then (x + 4) / 4.
 */
#include "whisper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"

struct mel_job {
    const struct janas_whisper *w;
    const float *pcm;
    size_t n, total; /* samples, and with the zeros after them */
    float *out;
    uint32_t frames;
    float *maxes; /* each thread's largest */
};

/* sample i of the sound padded with zeros, reflected past either end */
static float at(const struct mel_job *m, long i)
{
    long t = (long)m->total;
    if (i < 0)
        i = -i;
    if (i >= t)
        i = 2 * (t - 1) - i;
    return i >= 0 && (size_t)i < m->n ? m->pcm[i] : 0.0f;
}

static void mel_worker(void *arg, int tid, int nt)
{
    struct mel_job *m = arg;
    const struct janas_whisper *w = m->w;
    uint32_t N = w->n_fft, bins = N / 2 + 1, half = N / 2;
    float *frame = malloc(sizeof(float) * (N + bins));
    float mx = -INFINITY;
    if (!frame) {
        m->maxes[tid] = NAN; /* told to the caller */
        return;
    }
    float *pw = frame + N;
    uint32_t per = (m->frames + (uint32_t)nt - 1) / (uint32_t)nt;
    uint32_t f0 = (uint32_t)tid * per, f1 = f0 + per;
    if (f1 > m->frames)
        f1 = m->frames;
    for (uint32_t f = f0; f < f1; f++) {
        long s0 = (long)f * w->hop - (long)half;
        if (s0 >= 0 && (size_t)s0 + N <= m->n)
            memcpy(frame, m->pcm + s0, sizeof(float) * N);
        else
            for (uint32_t i = 0; i < N; i++)
                frame[i] = at(m, s0 + (long)i);
        for (uint32_t k = 0; k < bins; k++) {
            const float *c = w->dft_cos + (size_t)k * N;
            const float *s = w->dft_sin + (size_t)k * N;
            float re = 0, im = 0;
            for (uint32_t i = 0; i < N; i++) {
                re += frame[i] * c[i];
                im += frame[i] * s[i];
            }
            pw[k] = re * re + im * im;
        }
        for (uint32_t b = 0; b < w->n_mels; b++) {
            const float *fl = w->mel_filters + (size_t)b * bins;
            double e = 0;
            for (uint32_t k = 0; k < bins; k++)
                e += (double)fl[k] * pw[k];
            float v = (float)log10(e > 1e-10 ? e : 1e-10);
            m->out[(size_t)b * m->frames + f] = v;
            if (v > mx)
                mx = v;
        }
    }
    m->maxes[tid] = mx;
    free(frame);
}

float *janas_whisper_mel(struct janas_whisper *w, const float *pcm, size_t n,
                         size_t pad, uint32_t *frames)
{
    size_t total = n + pad, nf = total / w->hop;
    *frames = 0;
    if (nf > UINT32_MAX / 2)
        return NULL;
    int nt = janas_pool_size(w->pool);
    float *out = malloc(sizeof(float) * (nf ? nf : 1) * w->n_mels);
    float *maxes = malloc(sizeof(float) * (size_t)nt);
    if (!out || !maxes) {
        free(out);
        free(maxes);
        return NULL;
    }
    struct mel_job m = {.w = w,
                        .pcm = pcm,
                        .n = n,
                        .total = total,
                        .out = out,
                        .frames = (uint32_t)nf,
                        .maxes = maxes};
    janas_pool_run(w->pool, mel_worker, &m);
    float mx = -INFINITY;
    for (int t = 0; t < nt; t++) {
        if (isnan(maxes[t])) {
            free(out);
            free(maxes);
            return NULL;
        }
        if (maxes[t] > mx)
            mx = maxes[t];
    }
    free(maxes);
    for (size_t i = 0; i < nf * w->n_mels; i++) {
        float v = out[i] < mx - 8.0f ? mx - 8.0f : out[i];
        out[i] = (v + 4.0f) / 4.0f;
    }
    *frames = (uint32_t)nf;
    return out;
}
