/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * image_resize.c - bicubic resampling of 8-bit RGB (image.h), as Pillow's
 * resize(BICUBIC) computes it, so a vision model gets the pixels its
 * processor gave it in training: separable, rows first then columns, each
 * pass rounded to 8 bits; for each output sample the source span around its
 * centre, the cubic kernel (a = -0.5) widened by the scale when shrinking,
 * the weights normalised to sum 1 and turned into 22-bit fixed point.
 */
#include "image.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PB 22 /* bits of the weights */

static double cubic(double x)
{
    const double a = -0.5;
    x = fabs(x);
    if (x < 1)
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

/* each output sample's first source sample and count, and its weights */
struct taps {
    int ksize;
    int *first, *count;
    int32_t *w; /* out * ksize */
};

static int taps_make(struct taps *t, int in, int out)
{
    double scale = (double)in / out;
    double fs = scale < 1 ? 1 : scale;
    double support = 2.0 * fs;
    t->ksize = (int)ceil(support) * 2 + 1;
    t->first = malloc(sizeof(int) * (size_t)out);
    t->count = malloc(sizeof(int) * (size_t)out);
    t->w = calloc((size_t)out * (size_t)t->ksize, sizeof(int32_t));
    double *k = malloc(sizeof(double) * (size_t)t->ksize);
    if (!t->first || !t->count || !t->w || !k) {
        free(k);
        return -1;
    }
    for (int x = 0; x < out; x++) {
        double centre = (x + 0.5) * scale, sum = 0;
        int lo = (int)(centre - support + 0.5), hi;
        if (lo < 0)
            lo = 0;
        hi = (int)(centre + support + 0.5);
        if (hi > in)
            hi = in;
        int n = hi - lo;
        for (int i = 0; i < n; i++) {
            k[i] = cubic((i + lo - centre + 0.5) / fs);
            sum += k[i];
        }
        for (int i = 0; i < n; i++) {
            double v = sum != 0 ? k[i] / sum : k[i];
            t->w[(size_t)x * (size_t)t->ksize + (size_t)i] =
                (int32_t)(v < 0 ? -0.5 + v * (1 << PB) : 0.5 + v * (1 << PB));
        }
        t->first[x] = lo;
        t->count[x] = n;
    }
    free(k);
    return 0;
}

static void taps_free(struct taps *t)
{
    free(t->first);
    free(t->count);
    free(t->w);
}

static uint8_t clip(int64_t v)
{
    v >>= PB;
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

int janas_image_resize(const struct janas_image *in, int w, int h,
                       struct janas_image *out)
{
    memset(out, 0, sizeof(*out));
    if (w <= 0 || h <= 0 || in->w <= 0 || in->h <= 0)
        return -1;
    struct taps tx = {0}, ty = {0};
    uint8_t *mid = NULL;
    int rc = -1;
    if (taps_make(&tx, in->w, w) || taps_make(&ty, in->h, h))
        goto done;
    /* the rows the columns pass reads: those of any output row's span */
    int y0 = ty.first[0], y1 = ty.first[h - 1] + ty.count[h - 1];
    mid = malloc((size_t)w * (size_t)(y1 - y0) * 3);
    out->rgb = malloc((size_t)w * (size_t)h * 3);
    if (!mid || !out->rgb)
        goto done;
    for (int y = y0; y < y1; y++) { /* rows */
        const uint8_t *src = in->rgb + (size_t)y * (size_t)in->w * 3;
        uint8_t *dst = mid + (size_t)(y - y0) * (size_t)w * 3;
        for (int x = 0; x < w; x++) {
            const int32_t *k = tx.w + (size_t)x * (size_t)tx.ksize;
            for (int c = 0; c < 3; c++) {
                int64_t s = (int64_t)1 << (PB - 1);
                for (int i = 0; i < tx.count[x]; i++)
                    s += (int64_t)src[(size_t)(tx.first[x] + i) * 3 + c] * k[i];
                dst[(size_t)x * 3 + c] = clip(s);
            }
        }
    }
    for (int y = 0; y < h; y++) { /* columns */
        const int32_t *k = ty.w + (size_t)y * (size_t)ty.ksize;
        uint8_t *dst = out->rgb + (size_t)y * (size_t)w * 3;
        for (int x = 0; x < w * 3; x++) {
            int64_t s = (int64_t)1 << (PB - 1);
            for (int i = 0; i < ty.count[y]; i++)
                s += (int64_t)
                         mid[(size_t)(ty.first[y] - y0 + i) * (size_t)w * 3 +
                             (size_t)x] *
                     k[i];
            dst[x] = clip(s);
        }
    }
    out->w = w;
    out->h = h;
    rc = 0;
done:
    if (rc) {
        free(out->rgb);
        memset(out, 0, sizeof(*out));
    }
    free(mid);
    taps_free(&tx);
    taps_free(&ty);
    return rc;
}
