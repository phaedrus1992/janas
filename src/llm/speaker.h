/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * speaker.h - who is speaking: WeSpeaker's ResNet34 speaker embeddings
 * (speaker.c) over Kaldi's filter banks, and the diarization that groups
 * the stretches of a recording by their speaker (diarize.c).
 */
#ifndef JANAS_LLM_SPEAKER_H
#define JANAS_LLM_SPEAKER_H

#include <stddef.h>
#include <stdint.h>

struct janas_speaker;

/* a model made by tools/wespeaker2jns, its products on n_threads (0: the
   performance cores), or NULL with the reason in err */
struct janas_speaker *janas_speaker_open(const char *path, int n_threads,
                                         char *err, size_t err_len);
void janas_speaker_close(struct janas_speaker *s);

/* Kaldi's log mel filter banks (80, 25 ms every 10 ms, Hamming, as
   torchaudio's kaldi.fbank) of n samples at 16 kHz in [-1, 1]: *frames x
   80 (malloc'd), or NULL when out of memory */
float *janas_speaker_fbank(const float *pcm, size_t n, uint32_t *frames);

/* the embedding (256) of frames f0 .. f0 + nf of the filter banks, their
   mean taken off first: 0, or -1 when out of memory or nf too short */
int janas_speaker_embed(struct janas_speaker *s, const float *fbank,
                        uint32_t f0, uint32_t nf, float *emb);

/* diarize.c: who speaks when, from stretches of speech (pairs of samples
   at 16 kHz): windows of 1.5 s every 0.75 s embedded, grouped by spectral
   clustering into at most max_speakers (0: 20), each window's group in
   speaker; known speakers (embeddings ref, n_ref, 256 each) take the
   groups nearest them. 0, or -1 when out of memory */
struct janas_diar {
    size_t *win;  /* 2 x n_win: start, end in samples */
    int *speaker; /* n_win: 0, 1... in order of first appearance */
    uint32_t n_win;
    uint32_t n_speakers;
    int *known; /* n_speakers: the known speaker it is, or -1 */
};

int janas_diarize(struct janas_speaker *s, const float *pcm, size_t n,
                  const size_t *regions, size_t n_regions, const float *ref,
                  uint32_t n_ref, uint32_t max_speakers,
                  struct janas_diar *out);
void janas_diar_free(struct janas_diar *d);

#endif
