/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_fft.c - a mixed-radix complex FFT, decimation in time, recursive
 * over the factors of its length (4 first, then 2, 3, 5 and whatever is
 * left), butterflies of 2 and 4 written out and any other radix as a small
 * DFT; and the inverse MDCT through a DCT-IV of n / 2 points made of one
 * FFT of n / 4 (audio_fft.h).
 *
 * The inverse MDCT: with M = n / 2, the DCT-IV u[m] = sum_k x[k]
 * cos(pi / M (m + 1/2)(k + 1/2)) splits into even and odd-reversed inputs,
 * z[p] = (x[2p] + i x[M - 1 - 2p]) e^(-i pi p / M), whose FFT Z, turned by
 * e^(-i pi (q + 1/4) / M), gives u[2q] = Re Z[q] and u[M - 1 - 2q] =
 * -Im Z[q]; the MDCT's output is u shifted by n / 4 with its symmetries.
 */
#include "audio_fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct afft {
    int n, nf;
    int fac[64]; /* (radix, the length below it) pairs */
    float *tw;   /* e^(-2 pi i k / n), k < n */
};

struct afft *afft_new(int n)
{
    if (n < 1)
        return NULL;
    struct afft *f = calloc(1, sizeof *f);
    if (!f || !(f->tw = malloc(sizeof(float) * 2 * (size_t)n))) {
        free(f);
        return NULL;
    }
    f->n = n;
    for (int k = 0; k < n; k++) {
        double a = -2 * M_PI * k / n;
        f->tw[2 * k] = (float)cos(a);
        f->tw[2 * k + 1] = (float)sin(a);
    }
    int m = n, p = 4;
    while (m > 1) {
        while (m % p) {
            p = p == 4 ? 2 : p == 2 ? 3 : p + 2;
            if (p * p > m)
                p = m;
        }
        m /= p;
        f->fac[2 * f->nf] = p;
        f->fac[2 * f->nf + 1] = m;
        f->nf++;
    }
    return f;
}

void afft_free(struct afft *f)
{
    if (f) {
        free(f->tw);
        free(f);
    }
}

static inline void cmul(float *r, const float *a, const float *b)
{
    float re = a[0] * b[0] - a[1] * b[1], im = a[0] * b[1] + a[1] * b[0];
    r[0] = re;
    r[1] = im;
}

static void work(const struct afft *f, float *out, const float *in, int fstride,
                 const int *fac)
{
    int p = fac[0], m = fac[1];
    if (m == 1) {
        for (int q = 0; q < p; q++) {
            out[2 * q] = in[2 * q * fstride];
            out[2 * q + 1] = in[2 * q * fstride + 1];
        }
    } else {
        for (int q = 0; q < p; q++)
            work(f, out + 2 * q * m, in + 2 * q * fstride, fstride * p,
                 fac + 2);
    }
    const float *tw = f->tw;
    if (p == 2) {
        for (int k = 0; k < m; k++) {
            float *a = out + 2 * k, *b = out + 2 * (k + m), t[2];
            cmul(t, b, tw + 2 * k * fstride);
            b[0] = a[0] - t[0];
            b[1] = a[1] - t[1];
            a[0] += t[0];
            a[1] += t[1];
        }
    } else if (p == 4) {
        for (int k = 0; k < m; k++) {
            float *a0 = out + 2 * k, *a1 = a0 + 2 * m, *a2 = a1 + 2 * m,
                  *a3 = a2 + 2 * m, b1[2], b2[2], b3[2];
            cmul(b1, a1, tw + 2 * k * fstride);
            cmul(b2, a2, tw + 4 * k * fstride);
            cmul(b3, a3, tw + 6 * k * fstride);
            float s0r = a0[0] + b2[0], s0i = a0[1] + b2[1];
            float s1r = a0[0] - b2[0], s1i = a0[1] - b2[1];
            float s2r = b1[0] + b3[0], s2i = b1[1] + b3[1];
            float s3r = b1[0] - b3[0], s3i = b1[1] - b3[1];
            a0[0] = s0r + s2r;
            a0[1] = s0i + s2i;
            a2[0] = s0r - s2r;
            a2[1] = s0i - s2i;
            /* times -i for the forward transform */
            a1[0] = s1r + s3i;
            a1[1] = s1i - s3r;
            a3[0] = s1r - s3i;
            a3[1] = s1i + s3r;
        }
    } else {
        float sc[2 * 64];
        for (int u = 0; u < m; u++) {
            for (int q = 0; q < p && q < 64; q++) {
                sc[2 * q] = out[2 * (u + q * m)];
                sc[2 * q + 1] = out[2 * (u + q * m) + 1];
            }
            for (int q1 = 0; q1 < p; q1++) {
                int k = u + q1 * m, t = 0;
                float acc[2] = {sc[0], sc[1]}, x[2];
                for (int q = 1; q < p; q++) {
                    t += fstride * k;
                    t %= f->n;
                    cmul(x, sc + 2 * q, tw + 2 * t);
                    acc[0] += x[0];
                    acc[1] += x[1];
                }
                out[2 * k] = acc[0];
                out[2 * k + 1] = acc[1];
            }
        }
    }
}

void afft_run(const struct afft *f, const float *in, float *out)
{
    work(f, out, in, 1, f->fac);
}

struct aimdct {
    int n;
    struct afft *f;
    float *pre, *post, *z, *zz, *u;
};

struct aimdct *aimdct_new(int n)
{
    if (n < 8 || n % 4)
        return NULL;
    struct aimdct *m = calloc(1, sizeof *m);
    int M = n / 2, h = n / 4;
    if (!m)
        return NULL;
    m->n = n;
    m->f = afft_new(h);
    m->pre = malloc(sizeof(float) * 2 * (size_t)h);
    m->post = malloc(sizeof(float) * 2 * (size_t)h);
    m->z = malloc(sizeof(float) * 2 * (size_t)h);
    m->zz = malloc(sizeof(float) * 2 * (size_t)h);
    m->u = malloc(sizeof(float) * (size_t)M);
    if (!m->f || !m->pre || !m->post || !m->z || !m->zz || !m->u) {
        aimdct_free(m);
        return NULL;
    }
    for (int p = 0; p < h; p++) {
        double a = -M_PI * p / M, b = -M_PI * (p + 0.25) / M;
        m->pre[2 * p] = (float)cos(a);
        m->pre[2 * p + 1] = (float)sin(a);
        m->post[2 * p] = (float)cos(b);
        m->post[2 * p + 1] = (float)sin(b);
    }
    return m;
}

void aimdct_free(struct aimdct *m)
{
    if (!m)
        return;
    afft_free(m->f);
    free(m->pre);
    free(m->post);
    free(m->z);
    free(m->zz);
    free(m->u);
    free(m);
}

void aimdct_dct4(struct aimdct *m, const float *x, int stride, float *u)
{
    int n = m->n, M = n / 2, h = n / 4;
    for (int p = 0; p < h; p++) {
        float v[2] = {x[2 * p * stride], x[(M - 1 - 2 * p) * stride]};
        cmul(m->z + 2 * p, v, m->pre + 2 * p);
    }
    afft_run(m->f, m->z, m->zz);
    for (int q = 0; q < h; q++) {
        float t[2];
        cmul(t, m->zz + 2 * q, m->post + 2 * q);
        u[2 * q] = t[0];
        u[M - 1 - 2 * q] = -t[1];
    }
}

void aimdct_run(struct aimdct *m, const float *x, float *y)
{
    int n = m->n, h = n / 4;
    float *u = m->u;
    aimdct_dct4(m, x, 1, u);
    for (int i = 0; i < h; i++)
        y[i] = u[i + h];
    for (int i = h; i < 3 * h; i++)
        y[i] = -u[3 * h - 1 - i];
    for (int i = 3 * h; i < n; i++)
        y[i] = -u[i - 3 * h];
}
