/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_janet.c - the Janet probe (job/docs/progetto_jnet.md): a model's
 * feed-forward replaced, layer by layer, by Janet's detectors trained to
 * imitate it (job/jnet/lm, jlm ffnfit). A probe: only the CPU's path has it
 * (JANAS_GPU=0), and the detectors run in floats, one token at a time.
 *
 * JANAS_JANET_REC=dir with JANAS_JANET_REC_LAYERS=l1,l2,...: for those
 * layers each token's feed-forward input (the normed state, floats) is
 * appended to dir/x<l>.f32 and its output (what the block adds to the
 * residual) to dir/y<l>.f32, d_model floats a token.
 * JANAS_JANET_FFN=file1,file2,...: Janet feed-forwards (jlm ffnfit's files,
 * each naming its layer) run in place of those layers' own; or the control,
 * a SwiGLU trained the same way (its file "SWIGLUF1").
 *
 * A Janet detector u: the input in K zones of contiguous components, each
 * zone's branch sum m_k = its components times the unit's weights plus an
 * offset; the zones' heights h_k = A_k / (A_k + 1), A_k the sum of the
 * zone's components' sizes (each at most 1); the centre the heights'
 * barycentre of the m_k; the support S the sum of h_k (1 - u_k^2) over the
 * hills that cover the centre (u_k = (centre - m_k) / tolerance); Alpha =
 * gain ((1 + kappa) S - kappa T), T the heights' sum; the unit speaks when
 * Alpha reaches its threshold (and, rectified, the centre is positive) and
 * sends min(Alpha, 1) times the centre to its row of down.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"

#define JANET_MAX_LAYERS 256
#define JANET_MAX_K 8

/* One layer's Janet feed-forward, as jlm ffnfit writes it (swiglu: the
   control, wa its gate and wd its down, wu its up). */
struct jffn {
    int d, h, K, rectify, in_full, swiglu;
    float *wu;
    float *wa;  /* the branches, d rows (by input) of h */
    float *wd;  /* the down, h rows (by unit) of d */
    float *inv; /* each unit's inverse tolerance, gain, kappa, threshold */
    float *gain, *kap, *th;
    float *b; /* its K offsets, h rows of K */
};

struct janas_janet {
    char dir[512];
    unsigned char rec[JANET_MAX_LAYERS];
    FILE *fx[JANET_MAX_LAYERS], *fy[JANET_MAX_LAYERS];
    float *xb; /* the residual before a recorded block, blk rows */
    struct jffn *ffn[JANET_MAX_LAYERS];
    float *msum, *out, *yrow; /* a token's branch sums, outputs, its add */
    size_t scratch_h;
    uint64_t rows_rec, rows_run, spoke, units;
};

static void jffn_free(struct jffn *f)
{
    if (!f)
        return;
    free(f->wa), free(f->wd), free(f->inv), free(f->gain), free(f->kap);
    free(f->wu);
    free(f->th), free(f->b), free(f);
}

static int read_floats(FILE *f, float **p, size_t n)
{
    *p = malloc(n * sizeof(float));
    return *p && fread(*p, sizeof(float), n, f) == n ? 0 : -1;
}

/* A file of jlm ffnfit: "JANETFF1", then layer, d, h, K, rectify, in_full
   as 32-bit integers, then the floats. Its layer, or -1. */
static int jffn_load(const char *path, uint32_t dm, struct jffn **out)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "janet: %s: cannot open\n", path);
        return -1;
    }
    char magic[8];
    int32_t hd[6];
    struct jffn *j = calloc(1, sizeof(*j));
    if (j && fread(magic, 1, 8, f) == 8 && !memcmp(magic, "SWIGLUF1", 8) &&
        fread(hd, sizeof(int32_t), 3, f) == 3) { /* the control */
        j->swiglu = 1, j->d = hd[1], j->h = hd[2], j->K = 1;
        size_t d = (size_t)j->d, h = (size_t)j->h;
        if ((uint32_t)j->d != dm || j->h < 1 || hd[0] < 0 ||
            hd[0] >= JANET_MAX_LAYERS || read_floats(f, &j->wa, d * h) ||
            read_floats(f, &j->wu, d * h) || read_floats(f, &j->wd, h * d)) {
            fprintf(stderr, "janet: %s: bad shape or short file\n", path);
            fclose(f), jffn_free(j);
            return -1;
        }
        fclose(f);
        *out = j;
        return hd[0];
    }
    if (!j || memcmp(magic, "JANETFF1", 8) ||
        fread(hd, sizeof(int32_t), 6, f) != 6) {
        fprintf(stderr, "janet: %s: not a Janet feed-forward\n", path);
        fclose(f), free(j);
        return -1;
    }
    j->d = hd[1], j->h = hd[2], j->K = hd[3], j->rectify = hd[4];
    j->in_full = hd[5];
    int layer = hd[0];
    size_t d = (size_t)j->d, h = (size_t)j->h;
    if ((uint32_t)j->d != dm || j->K < 1 || j->K > JANET_MAX_K || j->d % j->K ||
        j->h < 1 || layer < 0 || layer >= JANET_MAX_LAYERS ||
        read_floats(f, &j->wa, d * h) || read_floats(f, &j->wd, h * d) ||
        read_floats(f, &j->inv, h) || read_floats(f, &j->gain, h) ||
        read_floats(f, &j->kap, h) || read_floats(f, &j->th, h) ||
        read_floats(f, &j->b, h * j->K)) {
        fprintf(stderr, "janet: %s: bad shape or short file\n", path);
        fclose(f), jffn_free(j);
        return -1;
    }
    fclose(f);
    *out = j;
    return layer;
}

int janas_m_janet_init(struct janas_llm_model *m)
{
    const char *rec = getenv("JANAS_JANET_REC"),
               *ffn = getenv("JANAS_JANET_FFN");
    if ((!rec || !*rec) && (!ffn || !*ffn))
        return 0;
    struct janas_janet *J = calloc(1, sizeof(*J));
    if (!J)
        return -1;
    m->janet = J;
    uint32_t dm = m->d_model;
    if (rec && *rec) {
        snprintf(J->dir, sizeof(J->dir), "%s", rec);
        const char *ls = getenv("JANAS_JANET_REC_LAYERS");
        for (const char *s = ls ? ls : "0"; *s;) {
            char *e;
            long l = strtol(s, &e, 10);
            if (e == s)
                break;
            if (l >= 0 && l < JANET_MAX_LAYERS && (uint32_t)l < m->n_layer) {
                char p[600];
                J->rec[l] = 1;
                snprintf(p, sizeof(p), "%s/x%ld.f32", J->dir, l);
                J->fx[l] = fopen(p, "wb");
                snprintf(p, sizeof(p), "%s/y%ld.f32", J->dir, l);
                J->fy[l] = fopen(p, "wb");
                if (!J->fx[l] || !J->fy[l]) {
                    fprintf(stderr, "janet: cannot write in %s\n", J->dir);
                    return -1;
                }
            }
            s = *e == ',' ? e + 1 : e;
        }
        J->xb = malloc((size_t)m->blk * dm * sizeof(float));
        if (!J->xb)
            return -1;
    }
    if (ffn && *ffn) {
        char buf[4096];
        snprintf(buf, sizeof(buf), "%s", ffn);
        for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
            struct jffn *f = NULL;
            int l = jffn_load(t, dm, &f);
            if (l < 0)
                return -1;
            if ((uint32_t)l >= m->n_layer) {
                fprintf(stderr, "janet: %s: layer %d of %u\n", t, l,
                        m->n_layer);
                jffn_free(f);
                return -1;
            }
            jffn_free(J->ffn[l]);
            J->ffn[l] = f;
            if ((size_t)f->h > J->scratch_h)
                J->scratch_h = (size_t)f->h;
            if (f->swiglu)
                fprintf(stderr,
                        "janet: layer %d's feed-forward by the control, %d "
                        "SwiGLU units (%s)\n",
                        l, f->h, t);
            else
                fprintf(stderr,
                        "janet: layer %d's feed-forward by %d detectors of %d "
                        "branches (%s)\n",
                        l, f->h, f->K, t);
        }
        J->msum = malloc(J->scratch_h * (JANET_MAX_K + 1) * sizeof(float));
        J->out = malloc(J->scratch_h * sizeof(float));
        J->yrow = malloc((size_t)dm * sizeof(float));
        if (!J->msum || !J->out || !J->yrow)
            return -1;
    }
    return 0;
}

/* The recording around layer l's feed-forward of the n rows: before it
   (after = 0) the inputs written and the residual kept, after it the
   block's output (the residual's change) written. */
void janas_m_janet_rec(struct janas_llm_model *m, uint32_t l, uint32_t n,
                       int after)
{
    struct janas_janet *J = m->janet;
    if (!J || l >= JANET_MAX_LAYERS || !J->rec[l])
        return;
    size_t nd = (size_t)n * m->d_model;
    if (!after) {
        fwrite(m->xn, sizeof(float), nd, J->fx[l]);
        memcpy(J->xb, m->x, nd * sizeof(float));
        return;
    }
    for (size_t i = 0; i < nd; i++)
        J->xb[i] = m->x[i] - J->xb[i];
    fwrite(J->xb, sizeof(float), nd, J->fy[l]);
    J->rows_rec += n;
}

/* The control's: silu(gate) times up, its down rows. */
static void swiglu_row(struct janas_janet *J, const struct jffn *f,
                       const float *x, float *y)
{
    int d = f->d, h = f->h;
    float *g = J->msum, *u = J->msum + h;
    memset(g, 0, sizeof(float) * 2 * (size_t)h);
    for (int i = 0; i < d; i++) {
        float xi = x[i];
        const float *wg = f->wa + (size_t)i * h, *wu = f->wu + (size_t)i * h;
        for (int k = 0; k < h; k++)
            g[k] += xi * wg[k], u[k] += xi * wu[k];
    }
    for (int k = 0; k < h; k++) {
        float o = (float)(g[k] / (1 + exp(-(double)g[k]))) * u[k];
        const float *w = f->wd + (size_t)k * d;
        for (int i = 0; i < d; i++)
            y[i] += o * w[i];
    }
}

/* One token's Janet feed-forward: from x (d) its output added to y. */
static void jffn_row(struct janas_janet *J, const struct jffn *f,
                     const float *x, float *y)
{
    if (f->swiglu) {
        swiglu_row(J, f, x, y);
        return;
    }
    int d = f->d, h = f->h, K = f->K, dz = d / K;
    double hk[JANET_MAX_K], T = 0;
    for (int k = 0; k < K; k++) {
        double A = 0;
        for (int i = k * dz; i < (k + 1) * dz; i++) {
            double ax = fabs((double)x[i]);
            A += f->in_full ? 1 : ax < 1 ? ax : 1;
        }
        hk[k] = A / (A + 1);
        T += hk[k];
    }
    float *ms = J->msum;
    memset(ms, 0, sizeof(float) * (size_t)K * h);
    for (int k = 0; k < K; k++) {
        float *mk = ms + (size_t)k * h;
        for (int i = k * dz; i < (k + 1) * dz; i++) {
            float xi = x[i];
            if (xi == 0)
                continue;
            const float *w = f->wa + (size_t)i * h;
            for (int u = 0; u < h; u++)
                mk[u] += xi * w[u];
        }
    }
    for (int u = 0; u < h; u++) {
        double mk[JANET_MAX_K], xc = 0, S = 0;
        for (int k = 0; k < K; k++) {
            mk[k] = (double)ms[(size_t)k * h + u] + f->b[(size_t)u * K + k];
            xc += hk[k] * mk[k];
        }
        xc = T > 0 ? xc / T : 0;
        for (int k = 0; k < K; k++) {
            double uu = (xc - mk[k]) * f->inv[u];
            S += uu * uu < 1 ? hk[k] * (1 - uu * uu) : 0;
        }
        double A = f->gain[u] * ((1 + f->kap[u]) * S - f->kap[u] * T);
        int speak = A >= f->th[u] && (!f->rectify || xc > 0);
        double cl = A < 0 ? 0 : A > 1 ? 1 : A;
        J->out[u] = speak ? (float)(cl * xc) : 0;
        J->spoke += speak;
    }
    J->units += (uint64_t)h;
    for (int u = 0; u < h; u++) {
        float o = J->out[u];
        if (o == 0)
            continue;
        const float *w = f->wd + (size_t)u * d;
        for (int i = 0; i < d; i++)
            y[i] += o * w[i];
    }
}

/* Layer l's feed-forward by Janet's detectors, when the probe has one for
   it: each of the n rows' output added to the residual; 1 when it ran. */
int janas_m_janet_ffn(struct janas_llm_model *m, uint32_t l, uint32_t n)
{
    struct janas_janet *J = m->janet;
    if (!J || l >= JANET_MAX_LAYERS || !J->ffn[l])
        return 0;
    uint32_t dm = m->d_model;
    for (uint32_t t = 0; t < n; t++) {
        memset(J->yrow, 0, sizeof(float) * dm);
        jffn_row(J, J->ffn[l], m->xn + (size_t)t * dm, J->yrow);
        float *x = m->x + (size_t)t * dm;
        for (uint32_t i = 0; i < dm; i++)
            x[i] += J->yrow[i];
    }
    J->rows_run += n;
    return 1;
}

void janas_m_janet_free(struct janas_llm_model *m)
{
    struct janas_janet *J = m->janet;
    if (!J)
        return;
    if (J->rows_rec)
        fprintf(stderr, "janet: %llu rows recorded in %s (all layers)\n",
                (unsigned long long)J->rows_rec, J->dir);
    if (J->rows_run)
        fprintf(stderr,
                "janet: %llu rows through Janet's feed-forward, %.1f%% of "
                "its detectors speaking\n",
                (unsigned long long)J->rows_run,
                J->units ? 100.0 * (double)J->spoke / (double)J->units : 0);
    for (int l = 0; l < JANET_MAX_LAYERS; l++) {
        if (J->fx[l])
            fclose(J->fx[l]);
        if (J->fy[l])
            fclose(J->fy[l]);
        jffn_free(J->ffn[l]);
    }
    free(J->xb), free(J->msum), free(J->out), free(J->yrow);
    free(J);
    m->janet = NULL;
}
