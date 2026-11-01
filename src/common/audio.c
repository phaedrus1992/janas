/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio.c - a sound file recognized by its first bytes and decoded
 * (audio.h); resampling.
 */
#include "audio.h"
#include "audio_ogg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int janas_audio_decode(const void *data, size_t n, struct janas_audio *a,
                       char *why, size_t wn)
{
    memset(a, 0, sizeof *a);
    const uint8_t *d = data;
    /* an ID3v2 tag before the sound (MP3's habit, FLAC's sometimes) */
    while (n >= 10 && !memcmp(d, "ID3", 3) && d[3] < 0xFF && d[4] < 0xFF &&
           !((d[6] | d[7] | d[8] | d[9]) & 0x80)) {
        size_t len = 10 +
                     ((size_t)d[6] << 21 | (size_t)d[7] << 14 |
                      (size_t)d[8] << 7 | d[9]) +
                     (d[5] & 0x10 ? 10 : 0); /* a footer */
        if (len > n)
            break;
        d += len;
        n -= len;
    }
    int rc = janas_audio_wav(d, n, a, why, wn);
    if (rc <= 0)
        return rc;
    if ((rc = janas_audio_flac(d, n, a, why, wn)) <= 0)
        return rc;
    if ((rc = janas_audio_ogg(d, n, a, why, wn)) <= 0)
        return rc;
    if ((rc = janas_audio_mp4(d, n, a, why, wn)) <= 0)
        return rc;
    if ((rc = janas_audio_adts(d, n, a, why, wn)) <= 0)
        return rc;
    if ((rc = janas_audio_mkv(d, n, a, why, wn)) <= 0)
        return rc;
    if ((rc = janas_audio_mpeg(d, n, a, why, wn)) <= 0)
        return rc;
    snprintf(why, wn,
             "not a sound file this reads (WAV, FLAC, MP3 and MPEG audio, "
             "Ogg, MP4/M4A, AAC, WebM/Matroska)");
    return -1;
}

void audio_edit_apply(const struct audio_edit *e, uint32_t rate, size_t *skip,
                      size_t *total)
{
    size_t got = *total;
    *skip = 0;
    if (!e || e->media_time < 0 || !e->timescale)
        return;
    uint64_t s = (uint64_t)e->media_time * rate / e->timescale;
    *skip = s < got ? (size_t)s : got;
    *total = got - *skip;
    if (e->length && e->movie_timescale) {
        uint64_t k = e->length * rate / e->movie_timescale;
        if (k < *total)
            *total = (size_t)k;
    }
}

void janas_audio_free(struct janas_audio *a)
{
    free(a->pcm);
    memset(a, 0, sizeof *a);
}

static uint32_t gcd(uint32_t a, uint32_t b)
{
    while (b) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* the zeroth-order modified Bessel function, for the Kaiser window */
static double bessel_i0(double x)
{
    double s = 1, t = 1;
    for (int k = 1; k < 50; k++) {
        t *= (x / (2 * k)) * (x / (2 * k));
        s += t;
        if (t < 1e-12 * s)
            break;
    }
    return s;
}

#define ZEROS 32     /* zero crossings of the sinc on either side */
#define BETA 8.6     /* the Kaiser window's: about 90 dB of rejection */
#define ROLLOFF 0.95 /* the cut-off, of the lower Nyquist frequency */

/* the filter at distance x (in input samples), cut-off fc (cycles a
   sample), half-width hw samples */
static double kernel(double x, double fc, double hw)
{
    if (fabs(x) >= hw)
        return 0;
    double s = x == 0 ? 1 : sin(M_PI * 2 * fc * x) / (M_PI * 2 * fc * x);
    double r = x / hw;
    return 2 * fc * s * bessel_i0(BETA * sqrt(1 - r * r)) / bessel_i0(BETA);
}

int janas_audio_resample(struct janas_audio *a, uint32_t rate)
{
    if (a->rate == rate || !a->n)
        return a->rate = rate, 0;
    double ratio = (double)rate / a->rate;
    double fc = 0.5 * ROLLOFF * (ratio < 1 ? ratio : 1);
    double hw = ZEROS / (2 * fc); /* in input samples */
    size_t out_n = (size_t)floor(a->n * ratio);
    float *out = malloc(sizeof(float) * (out_n ? out_n : 1));
    if (!out)
        return -1;
    /* output i sits at input i * in / out: with in / g and out / g the
       rates' reduced ratio the phases repeat every L = out / g outputs */
    uint32_t g = gcd(a->rate, rate), L = rate / g, M = a->rate / g;
    int taps = (int)ceil(2 * hw) + 1;
    float *table = NULL;
    if ((size_t)L * (size_t)taps <= ((size_t)1 << 22))
        table = malloc(sizeof(float) * L * (size_t)taps);
    if (table)
        for (uint32_t p = 0; p < L; p++) {
            double frac = (double)((uint64_t)p * M % L) / L;
            for (int t = 0; t < taps; t++) {
                long k = t - taps / 2;
                table[(size_t)p * taps + t] =
                    (float)kernel((double)k - frac, fc, hw);
            }
        }
    for (size_t i = 0; i < out_n; i++) {
        uint64_t num = (uint64_t)i * M;
        long base = (long)(num / L);
        uint32_t p = (uint32_t)(i % L);
        double frac = (double)(num % L) / L, s = 0;
        for (int t = 0; t < taps; t++) {
            long k = base + t - taps / 2;
            if (k < 0 || (size_t)k >= a->n)
                continue;
            double h = table ? table[(size_t)p * taps + t]
                             : kernel((double)(t - taps / 2) - frac, fc, hw);
            s += h * a->pcm[k];
        }
        out[i] = (float)s;
    }
    free(table);
    free(a->pcm);
    a->pcm = out;
    a->n = out_n;
    a->rate = rate;
    return 0;
}
