/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_fft.h - the transforms the sound decoders share (audio_fft.c): a
 * complex FFT of any length made of the factors 2, 3, 4 and 5 (and others,
 * more slowly), and the inverse MDCT on it.
 */
#ifndef JANAS_COMMON_AUDIO_FFT_H
#define JANAS_COMMON_AUDIO_FFT_H

#include <stddef.h>

struct afft;

/* a forward FFT of n points (e^-i), or NULL when out of memory */
struct afft *afft_new(int n);
void afft_free(struct afft *f);
/* in and out: n complex numbers each, (re, im) pairs one after another */
void afft_run(const struct afft *f, const float *in, float *out);

/* The inverse MDCT of n / 2 coefficients into n samples, unscaled:
   y[i] = sum_k x[k] cos(2 pi / n (i + 1/2 + n / 4)(k + 1/2)); n a multiple
   of 4. */
struct aimdct;
struct aimdct *aimdct_new(int n);
void aimdct_free(struct aimdct *m);
void aimdct_run(struct aimdct *m, const float *x, float *y);
/* the DCT-IV the inverse MDCT is made of: u[m] = sum_k x[k]
   cos(pi / M (m + 1/2)(k + 1/2)), M = n / 2, x read every stride */
void aimdct_dct4(struct aimdct *m, const float *x, int stride, float *u);

#endif
