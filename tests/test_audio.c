/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_audio.c - the sound decoder and the resampler (src/common/audio*.c):
 * WAV files made here of every kind it reads (PCM of 8, 16, 24 and 32 bits,
 * float of 32 and 64, stereo, WAVE_FORMAT_EXTENSIBLE, a chunk of odd length
 * before the samples, a data chunk longer than the file) give back the
 * sound written into them; the kinds it does not read are refused with a
 * reason; small clips of FLAC, MP3, MPEG Layers I and II, Ogg Vorbis, Ogg
 * FLAC, Opus (SILK, CELT, hybrid), AAC in M4A and in ADTS frames, ALAC,
 * Opus in WebM and Vorbis in Matroska made by FFmpeg (audio_clips.h) give
 * back FFmpeg's decoding of them, sample for sample; a tone resampled from
 * 44.1 and 8 kHz to 16 kHz is the same tone, one above the new Nyquist
 * frequency is gone; then truncations and thousands of corrupted copies of
 * every file must be read or refused, never crash (the sanitizer builds
 * watch).
 */
#include "audio_clips.h"
#include "common/audio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            fprintf(stderr, "test_audio: " __VA_ARGS__);                       \
            fputc('\n', stderr);                                               \
            failures++;                                                        \
        }                                                                      \
    } while (0)

struct buf {
    unsigned char *p;
    size_t n, cap;
};

static void put(struct buf *b, const void *s, size_t n)
{
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2;
        b->p = realloc(b->p, b->cap);
        if (!b->p)
            exit(1);
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
}

static void u16(struct buf *b, unsigned v)
{
    unsigned char c[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
    put(b, c, 2);
}

static void u32(struct buf *b, unsigned long v)
{
    u16(b, (unsigned)(v & 0xffff));
    u16(b, (unsigned)(v >> 16));
}

static double tone(size_t i, double rate, double hz)
{
    return 0.5 * sin(2 * M_PI * hz * (double)i / rate);
}

/* a WAV of frames samples of the tone in every channel (the second, when
   there is one, its negative plus the tone twice: the mix is the tone);
   fmt 1 PCM, 3 float; ext: WAVE_FORMAT_EXTENSIBLE; odd: a chunk of odd
   length before the data; long_data: a data length past the file */
static struct buf wav(unsigned fmt, unsigned bits, unsigned ch, unsigned rate,
                      size_t frames, int ext, int odd, int long_data)
{
    struct buf b = {0};
    unsigned align = ch * bits / 8;
    put(&b, "RIFF", 4);
    u32(&b, 0); /* a size nobody checks */
    put(&b, "WAVE", 4);
    put(&b, "fmt ", 4);
    u32(&b, ext ? 40 : 16);
    u16(&b, ext ? 0xFFFE : fmt);
    u16(&b, ch);
    u32(&b, rate);
    u32(&b, (unsigned long)rate * align);
    u16(&b, align);
    u16(&b, bits);
    if (ext) {
        u16(&b, 22);
        u16(&b, bits);
        u32(&b, 0);
        u16(&b, fmt); /* the subformat's GUID, its first two bytes */
        put(&b, "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71", 14);
    }
    if (odd) {
        put(&b, "LIST", 4);
        u32(&b, 3);
        put(&b, "abc\0", 4); /* and its pad byte */
    }
    put(&b, "data", 4);
    u32(&b, long_data ? 0xFFFFFFF0ul : (unsigned long)(frames * align));
    for (size_t i = 0; i < frames; i++)
        for (unsigned c = 0; c < ch; c++) {
            double s = tone(i, rate, 440);
            if (c == 1)
                s = -s;
            if (c == 2)
                s = 2 * s;
            if (fmt == 3 && bits == 32) {
                float f = (float)s;
                put(&b, &f, 4);
            } else if (fmt == 3) {
                put(&b, &s, 8);
            } else if (bits == 8) {
                unsigned char v = (unsigned char)lround(s * 127 + 128);
                put(&b, &v, 1);
            } else if (bits == 16) {
                u16(&b, (unsigned)(long)lround(s * 32767) & 0xffff);
            } else if (bits == 24) {
                long v = lround(s * 8388607);
                unsigned char c3[3] = {(unsigned char)v,
                                       (unsigned char)(v >> 8),
                                       (unsigned char)(v >> 16)};
                put(&b, c3, 3);
            } else {
                u32(&b, (unsigned long)lround(s * 2147483647.0) & 0xffffffffu);
            }
        }
    return b;
}

static void kinds(void)
{
    static const struct {
        unsigned fmt, bits, ch;
        int ext, odd, long_data;
        double tol;
    } K[] = {
        {1, 8, 1, 0, 0, 0, 1.0 / 100}, {1, 16, 1, 0, 0, 0, 1e-4},
        {1, 24, 1, 0, 0, 0, 1e-6},     {1, 32, 1, 0, 0, 0, 1e-6},
        {3, 32, 1, 0, 0, 0, 1e-7},     {3, 64, 1, 0, 0, 0, 1e-7},
        {1, 16, 3, 0, 0, 0, 1e-4},     {1, 16, 1, 1, 0, 0, 1e-4},
        {3, 32, 2, 1, 0, 0, 1e-7},     {1, 16, 1, 0, 1, 0, 1e-4},
        {1, 16, 1, 0, 0, 1, 1e-4},
    };
    char why[200];
    for (size_t k = 0; k < sizeof K / sizeof *K; k++) {
        size_t frames = 2205;
        struct buf b = wav(K[k].fmt, K[k].bits, K[k].ch, 22050, frames,
                           K[k].ext, K[k].odd, K[k].long_data);
        struct janas_audio a;
        int rc = janas_audio_decode(b.p, b.n, &a, why, sizeof why);
        CHECK(rc == 0, "kind %zu (%u bits, %u channels) refused: %s", k,
              K[k].bits, K[k].ch, why);
        if (rc == 0) {
            CHECK(a.rate == 22050 && a.n == frames,
                  "kind %zu: %u Hz, %zu samples", k, a.rate, a.n);
            double err = 0;
            /* the channels' mean: s; s and -s; s, -s and 2 s */
            double gain = K[k].ch == 1 ? 1 : K[k].ch == 2 ? 0 : 2.0 / 3;
            for (size_t i = 0; i < a.n && i < frames; i++) {
                double e = fabs(a.pcm[i] - gain * tone(i, 22050, 440));
                if (e > err)
                    err = e;
            }
            CHECK(err <= K[k].tol, "kind %zu: off by %g", k, err);
            janas_audio_free(&a);
        }
        free(b.p);
    }
    /* refused, with a reason */
    struct buf b = wav(2, 4, 1, 8000, 100, 0, 0, 0); /* ADPCM */
    struct janas_audio a;
    CHECK(janas_audio_decode(b.p, b.n, &a, why, sizeof why) != 0 && why[0],
          "ADPCM read");
    free(b.p);
    b = wav(1, 12, 1, 8000, 100, 0, 0, 0); /* 12 bits */
    CHECK(janas_audio_decode(b.p, b.n, &a, why, sizeof why) != 0,
          "12-bit read");
    free(b.p);
    static const char *const NOT[] = {"fLaC....", "ID3.....", "OggS....",
                                      "hello, world"};
    for (size_t i = 0; i < sizeof NOT / sizeof *NOT; i++)
        CHECK(janas_audio_decode(NOT[i], strlen(NOT[i]), &a, why, sizeof why) !=
                      0 &&
                  why[0],
              "'%s' read", NOT[i]);
}

static void formats(void)
{
    char why[200];
    for (size_t k = 0; k < sizeof CLIPS / sizeof *CLIPS; k++) {
        const struct clip *c = &CLIPS[k];
        struct janas_audio a;
        int rc = janas_audio_decode(c->data, c->size, &a, why, sizeof why);
        CHECK(rc == 0, "%s refused: %s", c->what, why);
        if (rc != 0)
            continue;
        CHECK(a.rate == c->rate && a.n == c->n, "%s: %u Hz, %zu samples",
              c->what, a.rate, a.n);
        double err = 0;
        for (size_t i = 0; i * 4 < a.n && i < c->nref; i++) {
            double e = fabs(a.pcm[i * 4] * 32768.0 - c->ref[i]);
            if (e > err)
                err = e;
        }
        CHECK(err <= c->tol, "%s: off by %g / 32768", c->what, err);
        janas_audio_free(&a);
    }
}

/* the largest gap to the tone at the new rate, away from the ends */
static double resampled(unsigned from, unsigned to, double hz, double *rms)
{
    size_t n = (size_t)from; /* a second */
    struct janas_audio a = {
        .pcm = malloc(sizeof(float) * n), .n = n, .rate = from};
    if (!a.pcm)
        exit(1);
    for (size_t i = 0; i < n; i++)
        a.pcm[i] = (float)tone(i, from, hz);
    if (janas_audio_resample(&a, to) != 0)
        exit(1);
    double err = 0, sq = 0;
    size_t edge = to / 50, cnt = 0;
    for (size_t i = edge; i + edge < a.n; i++) {
        double e = fabs(a.pcm[i] - tone(i, to, hz));
        if (e > err)
            err = e;
        sq += (double)a.pcm[i] * a.pcm[i];
        cnt++;
    }
    *rms = sqrt(sq / (cnt ? cnt : 1));
    CHECK(a.n == (size_t)to, "%u -> %u: %zu samples", from, to, a.n);
    janas_audio_free(&a);
    return err;
}

static void resampling(void)
{
    double rms;
    double e = resampled(44100, 16000, 1000, &rms);
    CHECK(e < 2e-3, "44100 -> 16000, 1 kHz: off by %g", e);
    e = resampled(8000, 16000, 1000, &rms);
    CHECK(e < 2e-3, "8000 -> 16000, 1 kHz: off by %g", e);
    e = resampled(48000, 16000, 3000, &rms);
    CHECK(e < 2e-3, "48000 -> 16000, 3 kHz: off by %g", e);
    resampled(44100, 16000, 10000, &rms);
    CHECK(rms < 1e-3, "44100 -> 16000, 10 kHz: left at %g", rms);
}

static unsigned long long rng = 0x9e3779b97f4a7c15ULL;

static unsigned next(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (unsigned)(rng >> 32);
}

int main(void)
{
    kinds();
    formats();
    resampling();
    long read = 0, refused = 0;
    char why[200];
    struct buf w[3] = {wav(1, 16, 2, 8000, 300, 0, 1, 0),
                       wav(3, 32, 1, 8000, 300, 1, 0, 0),
                       wav(1, 24, 1, 8000, 300, 0, 0, 1)};
    for (int k = 0; k < 3; k++) {
        unsigned char *b = malloc(w[k].n);
        if (!b)
            return 1;
        for (size_t cut = 0; cut <= w[k].n; cut++) {
            memcpy(b, w[k].p, cut);
            struct janas_audio a;
            if (janas_audio_decode(b, cut, &a, why, sizeof why) == 0) {
                read++;
                janas_audio_free(&a);
            } else {
                refused++;
            }
        }
        for (int i = 0; i < 3000; i++) {
            memcpy(b, w[k].p, w[k].n);
            int flips = 1 + (int)(next() % 8);
            for (int f = 0; f < flips; f++)
                b[next() % 80 % w[k].n] = (unsigned char)next();
            struct janas_audio a;
            if (janas_audio_decode(b, w[k].n, &a, why, sizeof why) == 0) {
                read++;
                janas_audio_free(&a);
            } else {
                refused++;
            }
        }
        free(b);
        free(w[k].p);
    }
    for (size_t k = 0; k < sizeof CLIPS / sizeof *CLIPS; k++) {
        const struct clip *c = &CLIPS[k];
        unsigned char *b = malloc(c->size);
        if (!b)
            return 1;
        for (size_t cut = 0; cut <= c->size; cut += 1 + cut / 16) {
            memcpy(b, c->data, cut);
            struct janas_audio a;
            if (janas_audio_decode(b, cut, &a, why, sizeof why) == 0) {
                read++;
                janas_audio_free(&a);
            } else {
                refused++;
            }
        }
        for (int i = 0; i < 400; i++) {
            memcpy(b, c->data, c->size);
            int flips = 1 + (int)(next() % 8);
            for (int f = 0; f < flips; f++) {
                /* the headers half of the time, anywhere the rest */
                size_t at = i & 1 ? next() % 200 : next();
                b[at % c->size] = (unsigned char)next();
            }
            struct janas_audio a;
            if (janas_audio_decode(b, c->size, &a, why, sizeof why) == 0) {
                read++;
                janas_audio_free(&a);
            } else {
                refused++;
            }
        }
        free(b);
    }
    if (failures) {
        fprintf(stderr, "test_audio: %d failures\n", failures);
        return 1;
    }
    printf("test_audio: ok (WAV of 11 kinds, %zu clips of other formats, "
           "resampling; broken copies: %ld read, %ld refused)\n",
           sizeof CLIPS / sizeof *CLIPS, read, refused);
    return 0;
}
