/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac_sbr.c - spectral band replication (ISO/IEC 14496-3, 4.6.18),
 * the upper half of HE-AAC's band made from the lower: the header and the
 * frequency band tables (master, high and low resolution, noise floor,
 * limiter, the patches), the grid of envelopes and noise floors, their
 * Huffman-coded factors, the 32-band analysis QMF bank of the core's
 * samples, the inverse filtering and the high band generated from the
 * low, its envelope adjusted to the factors with noise and sinusoids
 * added, limited and smoothed, and the 64-band synthesis QMF bank at twice
 * the core's rate. Parametric stereo (audio_aac_ps.c) comes in its
 * extension. In single precision as FFmpeg's decoder, whose order of
 * operations it keeps where it changes the rounding.
 */
#include "audio_aac.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "audio_aac_tab.h"
#include "audio_fft.h"

enum { FIXFIX, FIXVAR, VARFIX, VARVAR };
enum {
    T_ENV_1_5,
    F_ENV_1_5,
    T_ENV_BAL_1_5,
    F_ENV_BAL_1_5,
    T_ENV_3_0,
    F_ENV_3_0,
    T_ENV_BAL_3_0,
    F_ENV_BAL_3_0,
    T_NOISE_3_0,
    T_NOISE_BAL_3_0,
};

#define ENV_OFF 2 /* ENVELOPE_ADJUSTMENT_OFFSET: X_high's slots lead */
#define NOISE_FLOOR_OFFSET 6

/* the state of a channel */
struct sbr_ch {
    float x_ana[320 + 1024]; /* the analysis' input, oldest first */
    float W[2][32][32][2];   /* its output, this frame's and the last */
    int ypos;
    float Y[2][38][64][2]; /* the adjusted high band, two frames */
    float v[1280];         /* the synthesis' state, newest first */
    float bw[5];
    int invf[2][5];
    /* the grid */
    int num_env, num_noise, frame_class, amp_res;
    int freq_res[6]; /* [0]: the last of the frame before */
    int t_env[6], t_q[3], t_env_old;
    int e_a[2];
    int df_env[5], df_noise[2];
    int env_q[6][48], noise_q[3][5];
    float env[6][48], noise[3][5];
    int add_harm_flag, add_harm[48];
    int s_index[7][48];
    float g_temp[42][48], q_temp[42][48];
    int index_noise, index_sine;
};

struct sbr {
    int rate; /* SBR's, twice the core's */
    int ch, down;
    int start, reset, ready;
    /* the header */
    int amp_res_header;
    int params[6]; /* start, stop, xover, freq_scale, alter_scale,
                      noise_bands */
    int limiter_bands, limiter_gains, interpol_freq, smoothing;
    /* the tables */
    int k[3], kx[2], m[2], pushed;
    int n_master, n[2], n_q, n_lim;
    int f_master[49], f_high[49], f_low[25], f_noise[6], f_lim[30];
    int num_patches, patch_num[6], patch_start[6];
    int coupling;
    struct sbr_ch c[2];
    /* the frame's work */
    float X_low[32][40][2];
    float X_high[64][40][2];
    float alpha0[64][2], alpha1[64][2];
    float e_orig[7][48], q_mapped[7][48], e_curr[7][48];
    float q_m[7][48], s_m[7][48], gain[7][48];
    int s_mapped[7][48];
    float X[2][2][38][64]; /* the synthesis' input: channel, re/im */
    /* the QMF banks: DCT-IVs of 32 (analysis), inverse MDCTs of 128
       (synthesis), the rotations of the analysis' bands; the downsampled
       synthesis' kernel */
    struct aimdct *m64, *m128;
    float rot[32][2];
    float ds_cos[64][32], ds_sin[64][32];
    struct vlc vlc[10];
    struct ps *ps;
    int ps_found;
};

static float exp2fi(int x)
{
    return ldexpf(1.0f, x);
}

/* ---- the header and the frequency band tables ---- */

static void turnoff(struct sbr *s)
{
    s->start = 0;
    s->ready = 0;
    s->kx[1] = 32;
    s->m[1] = 0;
    s->c[0].e_a[1] = s->c[1].e_a[1] = -1;
    for (int i = 0; i < 6; i++)
        s->params[i] = -1;
}

static int cmp_int(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

static void make_bands(int *bands, int start, int stop, int num)
{
    float base = powf((float)stop / (float)start, 1.0f / (float)num);
    float prod = (float)start;
    int previous = start;
    for (int k = 0; k < num - 1; k++) {
        prod *= base;
        int present = (int)lrintf(prod);
        bands[k] = present - previous;
        previous = present;
    }
    bands[num - 1] = stop - previous;
}

static int in_table(const int *t, int last, int v)
{
    for (int i = 0; i <= last; i++)
        if (t[i] == v)
            return 1;
    return 0;
}

static void make_f_lim(struct sbr *s)
{
    if (s->limiter_bands > 0) {
        static const float WARP[3] = {1.32715174233856803909f,
                                      1.18509277094158210129f,
                                      1.11987160404675912501f};
        float lim = WARP[s->limiter_bands - 1];
        int patch_borders[7];
        patch_borders[0] = s->kx[1];
        for (int k = 1; k <= s->num_patches; k++)
            patch_borders[k] = patch_borders[k - 1] + s->patch_num[k - 1];
        memcpy(s->f_lim, s->f_low, sizeof(int) * (size_t)(s->n[0] + 1));
        if (s->num_patches > 1)
            memcpy(s->f_lim + s->n[0] + 1, patch_borders + 1,
                   sizeof(int) * (size_t)(s->num_patches - 1));
        qsort(s->f_lim, (size_t)(s->num_patches + s->n[0]), sizeof(int),
              cmp_int);
        s->n_lim = s->n[0] + s->num_patches - 1;
        int *in = s->f_lim + 1, *out = s->f_lim;
        while (out < s->f_lim + s->n_lim) {
            if ((float)*in >= (float)*out * lim) {
                *++out = *in++;
            } else if (*in == *out ||
                       !in_table(patch_borders, s->num_patches, *in)) {
                in++;
                s->n_lim--;
            } else if (!in_table(patch_borders, s->num_patches, *out)) {
                *out = *in++;
                s->n_lim--;
            } else {
                *++out = *in++;
            }
        }
    } else {
        s->f_lim[0] = s->f_low[0];
        s->f_lim[1] = s->f_low[s->n[0]];
        s->n_lim = 1;
    }
}

static int read_header(struct sbr *s, struct abr *b)
{
    int old[6], old_limiter = s->limiter_bands;
    memcpy(old, s->params, sizeof old);
    s->start = 1;
    s->ready = 0;
    s->amp_res_header = abr_bit(b);
    s->params[0] = (int)abr_bits(b, 4);
    s->params[1] = (int)abr_bits(b, 4);
    s->params[2] = (int)abr_bits(b, 3);
    abr_bits(b, 2);
    int extra1 = abr_bit(b), extra2 = abr_bit(b);
    if (extra1) {
        s->params[3] = (int)abr_bits(b, 2);
        s->params[4] = abr_bit(b);
        s->params[5] = (int)abr_bits(b, 2);
    } else {
        s->params[3] = 2;
        s->params[4] = 1;
        s->params[5] = 2;
    }
    if (memcmp(old, s->params, sizeof old))
        s->reset = 1;
    if (extra2) {
        s->limiter_bands = (int)abr_bits(b, 2);
        s->limiter_gains = (int)abr_bits(b, 2);
        s->interpol_freq = abr_bit(b);
        s->smoothing = abr_bit(b);
    } else {
        s->limiter_bands = 2;
        s->limiter_gains = 2;
        s->interpol_freq = 1;
        s->smoothing = 1;
    }
    if (s->limiter_bands != old_limiter && !s->reset)
        make_f_lim(s);
    return 0;
}

static int make_f_master(struct sbr *s)
{
    const int8_t *offset;
    switch (s->rate) {
    case 16000:
        offset = SBR_OFFSET;
        break;
    case 22050:
        offset = SBR_OFFSET + 16;
        break;
    case 24000:
        offset = SBR_OFFSET + 32;
        break;
    case 32000:
        offset = SBR_OFFSET + 48;
        break;
    case 44100:
    case 48000:
    case 64000:
        offset = SBR_OFFSET + 64;
        break;
    case 88200:
    case 96000:
    case 128000:
    case 176400:
    case 192000:
        offset = SBR_OFFSET + 80;
        break;
    default:
        return -1;
    }
    unsigned temp = s->rate < 32000 ? 3000 : s->rate < 64000 ? 4000 : 5000;
    unsigned rate = (unsigned)s->rate;
    int start_min = (int)(((temp << 7) + (rate >> 1)) / rate);
    int stop_min = (int)(((temp << 8) + (rate >> 1)) / rate);
    s->k[0] = start_min + offset[s->params[0]];
    if (s->params[1] < 14) {
        int stop_dk[13];
        s->k[2] = stop_min;
        make_bands(stop_dk, stop_min, 64, 13);
        qsort(stop_dk, 13, sizeof(int), cmp_int);
        for (int k = 0; k < s->params[1]; k++)
            s->k[2] += stop_dk[k];
    } else if (s->params[1] == 14) {
        s->k[2] = 2 * s->k[0];
    } else {
        s->k[2] = 3 * s->k[0];
    }
    if (s->k[2] > 64)
        s->k[2] = 64;
    int max_qmf = s->rate <= 32000 ? 48 : s->rate == 44100 ? 35 : 32;
    if (s->k[2] - s->k[0] > max_qmf || s->k[0] <= 0)
        return -1;
    int xover = s->params[2];
    if (!s->params[3]) {
        int dk = s->params[4] + 1;
        s->n_master = ((s->k[2] - s->k[0] + (dk & 2)) >> dk) << 1;
        if (s->n_master <= 0 || xover >= s->n_master || s->n_master > 48)
            return -1;
        for (int k = 1; k <= s->n_master; k++)
            s->f_master[k] = dk;
        int k2diff = s->k[2] - s->k[0] - s->n_master * dk;
        if (k2diff < 0) {
            s->f_master[1]--;
            s->f_master[2] -= (k2diff < -1);
        } else if (k2diff) {
            s->f_master[s->n_master]++;
        }
        s->f_master[0] = s->k[0];
        for (int k = 1; k <= s->n_master; k++)
            s->f_master[k] += s->f_master[k - 1];
    } else {
        int half_bands = 7 - s->params[3];
        int two_regions, vk0[49];
        if (49 * s->k[2] > 110 * s->k[0]) {
            two_regions = 1;
            s->k[1] = 2 * s->k[0];
        } else {
            two_regions = 0;
            s->k[1] = s->k[2];
        }
        int num_bands_0 = (int)lrintf((float)half_bands *
                                      log2f((float)s->k[1] / (float)s->k[0])) *
                          2;
        if (num_bands_0 <= 0 || num_bands_0 > 48)
            return -1;
        vk0[0] = 0;
        make_bands(vk0 + 1, s->k[0], s->k[1], num_bands_0);
        qsort(vk0 + 1, (size_t)num_bands_0, sizeof(int), cmp_int);
        int vdk0_max = vk0[num_bands_0];
        vk0[0] = s->k[0];
        for (int k = 1; k <= num_bands_0; k++) {
            if (vk0[k] <= 0)
                return -1;
            vk0[k] += vk0[k - 1];
        }
        if (two_regions) {
            int vk1[49];
            float invwarp = s->params[4] ? 0.76923076923076923077f : 1.0f;
            int num_bands_1 =
                (int)lrintf((float)half_bands * invwarp *
                            log2f((float)s->k[2] / (float)s->k[1])) *
                2;
            if (num_bands_1 <= 0 || num_bands_0 + num_bands_1 > 48)
                return -1;
            make_bands(vk1 + 1, s->k[1], s->k[2], num_bands_1);
            int vdk1_min = vk1[1];
            for (int i = 2; i <= num_bands_1; i++)
                if (vk1[i] < vdk1_min)
                    vdk1_min = vk1[i];
            if (vdk1_min < vdk0_max) {
                qsort(vk1 + 1, (size_t)num_bands_1, sizeof(int), cmp_int);
                int change = vdk0_max - vk1[1];
                int half = (vk1[num_bands_1] - vk1[1]) >> 1;
                if (half < change)
                    change = half;
                vk1[1] += change;
                vk1[num_bands_1] -= change;
            }
            qsort(vk1 + 1, (size_t)num_bands_1, sizeof(int), cmp_int);
            vk1[0] = s->k[1];
            for (int k = 1; k <= num_bands_1; k++) {
                if (vk1[k] <= 0)
                    return -1;
                vk1[k] += vk1[k - 1];
            }
            s->n_master = num_bands_0 + num_bands_1;
            if (xover >= s->n_master)
                return -1;
            memcpy(s->f_master, vk0, sizeof(int) * (size_t)(num_bands_0 + 1));
            memcpy(s->f_master + num_bands_0 + 1, vk1 + 1,
                   sizeof(int) * (size_t)num_bands_1);
        } else {
            s->n_master = num_bands_0;
            if (xover >= s->n_master)
                return -1;
            memcpy(s->f_master, vk0, sizeof(int) * (size_t)(num_bands_0 + 1));
        }
    }
    return 0;
}

static int calc_patches(struct sbr *s)
{
    int last_k = -1, last_msb = -1, sb = 0, k;
    int msb = s->k[0], usb = s->kx[1];
    int goal_sb =
        (int)(((1000u << 11) + ((unsigned)s->rate >> 1)) / (unsigned)s->rate);
    s->num_patches = 0;
    if (goal_sb < s->kx[1] + s->m[1]) {
        for (k = 0; s->f_master[k] < goal_sb; k++)
            ;
    } else {
        k = s->n_master;
    }
    do {
        int odd = 0;
        if (k == last_k && msb == last_msb)
            return -1;
        last_k = k;
        last_msb = msb;
        for (int i = k; i == k || sb > (s->k[0] - 1 + msb - odd); i--) {
            if (i < 0)
                return -1;
            sb = s->f_master[i];
            odd = (sb + s->k[0]) & 1;
        }
        if (s->num_patches > 5)
            return -1;
        s->patch_num[s->num_patches] = sb - usb > 0 ? sb - usb : 0;
        s->patch_start[s->num_patches] =
            s->k[0] - odd - s->patch_num[s->num_patches];
        if (s->patch_num[s->num_patches] > 0) {
            usb = sb;
            msb = sb;
            s->num_patches++;
        } else {
            msb = s->kx[1];
        }
        if (s->f_master[k] - sb < 3)
            k = s->n_master;
    } while (sb != s->kx[1] + s->m[1]);
    if (s->num_patches > 1 && s->patch_num[s->num_patches - 1] < 3)
        s->num_patches--;
    for (int i = 0; i < s->num_patches; i++)
        if (s->patch_start[i] < 0 || s->patch_start[i] + s->patch_num[i] > 32)
            return -1;
    return 0;
}

static int make_f_derived(struct sbr *s)
{
    int xover = s->params[2];
    s->n[1] = s->n_master - xover;
    s->n[0] = (s->n[1] + 1) >> 1;
    memcpy(s->f_high, s->f_master + xover, sizeof(int) * (size_t)(s->n[1] + 1));
    s->m[1] = s->f_high[s->n[1]] - s->f_high[0];
    s->kx[1] = s->f_high[0];
    if (s->kx[1] + s->m[1] > 64 || s->kx[1] > 32 || s->m[1] > 48 ||
        s->m[1] <= 0)
        return -1;
    s->f_low[0] = s->f_high[0];
    int temp = s->n[1] & 1;
    for (int k = 1; k <= s->n[0]; k++)
        s->f_low[k] = s->f_high[2 * k - temp];
    s->n_q = (int)lrintf((float)s->params[5] *
                         log2f((float)s->k[2] / (float)s->kx[1]));
    if (s->n_q < 1)
        s->n_q = 1;
    if (s->n_q > 5)
        return -1;
    s->f_noise[0] = s->f_low[0];
    temp = 0;
    for (int k = 1; k <= s->n_q; k++) {
        temp += (s->n[0] - temp) / (s->n_q + 1 - k);
        s->f_noise[k] = s->f_low[temp];
    }
    if (calc_patches(s) != 0)
        return -1;
    make_f_lim(s);
    s->c[0].index_noise = s->c[1].index_noise = 0;
    return 0;
}

static void sbr_reset(struct sbr *s)
{
    if (make_f_master(s) != 0 || make_f_derived(s) != 0)
        turnoff(s);
}

/* ---- the data ---- */

static int read_grid(struct sbr *s, struct abr *b, struct sbr_ch *c)
{
    static const int CEIL_LOG2[] = {0, 1, 2, 2, 3, 3};
    int pointer = 0, trail = 16, num_old = c->num_env;
    int t_env[6], num_env = 0, freq_res[6];
    memcpy(freq_res, c->freq_res, sizeof freq_res);
    freq_res[0] = c->freq_res[c->num_env];
    int amp_res = s->amp_res_header;
    int cls = (int)abr_bits(b, 2);
    switch (cls) {
    case FIXFIX: {
        num_env = 1 << abr_bits(b, 2);
        if (num_env > 5)
            return -1;
        if (num_env == 1)
            amp_res = 0;
        t_env[0] = 0;
        t_env[num_env] = trail;
        int step = (trail + (num_env >> 1)) / num_env;
        for (int i = 0; i < num_env - 1; i++)
            t_env[i + 1] = t_env[i] + step;
        freq_res[1] = abr_bit(b);
        for (int i = 1; i < num_env; i++)
            freq_res[i + 1] = freq_res[1];
        break;
    }
    case FIXVAR: {
        trail += (int)abr_bits(b, 2);
        int rel = (int)abr_bits(b, 2);
        num_env = rel + 1;
        t_env[0] = 0;
        t_env[num_env] = trail;
        for (int i = 0; i < rel; i++)
            t_env[num_env - 1 - i] =
                t_env[num_env - i] - 2 * (int)abr_bits(b, 2) - 2;
        pointer = (int)abr_bits(b, CEIL_LOG2[num_env]);
        for (int i = 0; i < num_env; i++)
            freq_res[num_env - i] = abr_bit(b);
        break;
    }
    case VARFIX: {
        t_env[0] = (int)abr_bits(b, 2);
        int rel = (int)abr_bits(b, 2);
        num_env = rel + 1;
        t_env[num_env] = trail;
        for (int i = 0; i < rel; i++)
            t_env[i + 1] = t_env[i] + 2 * (int)abr_bits(b, 2) + 2;
        pointer = (int)abr_bits(b, CEIL_LOG2[num_env]);
        for (int i = 0; i < num_env; i++)
            freq_res[i + 1] = abr_bit(b);
        break;
    }
    default: {
        t_env[0] = (int)abr_bits(b, 2);
        trail += (int)abr_bits(b, 2);
        int lead = (int)abr_bits(b, 2), rel = (int)abr_bits(b, 2);
        num_env = lead + rel + 1;
        if (num_env > 5)
            return -1;
        t_env[num_env] = trail;
        for (int i = 0; i < lead; i++)
            t_env[i + 1] = t_env[i] + 2 * (int)abr_bits(b, 2) + 2;
        for (int i = 0; i < rel; i++)
            t_env[num_env - 1 - i] =
                t_env[num_env - i] - 2 * (int)abr_bits(b, 2) - 2;
        pointer = (int)abr_bits(b, CEIL_LOG2[num_env]);
        for (int i = 0; i < num_env; i++)
            freq_res[i + 1] = abr_bit(b);
        break;
    }
    }
    if (pointer > num_env + 1)
        return -1;
    for (int i = 1; i <= num_env; i++)
        if (t_env[i - 1] >= t_env[i] || t_env[i - 1] < 0)
            return -1;
    /* accepted: into the channel */
    c->freq_res[0] = freq_res[0];
    for (int i = 1; i <= num_env; i++)
        c->freq_res[i] = freq_res[i];
    c->amp_res = amp_res;
    c->t_env_old = c->t_env[num_old];
    memcpy(c->t_env, t_env, sizeof(int) * (size_t)(num_env + 1));
    c->num_env = num_env;
    c->frame_class = cls;
    c->num_noise = (num_env > 1) + 1;
    c->t_q[0] = c->t_env[0];
    c->t_q[c->num_noise] = c->t_env[num_env];
    if (c->num_noise > 1) {
        int idx;
        if (cls == FIXFIX)
            idx = num_env >> 1;
        else if (cls & 1)
            idx = num_env - (pointer - 1 > 1 ? pointer - 1 : 1);
        else if (!pointer)
            idx = 1;
        else if (pointer == 1)
            idx = num_env - 1;
        else
            idx = pointer - 1;
        c->t_q[1] = c->t_env[idx];
    }
    c->e_a[0] = -(c->e_a[1] != num_old);
    c->e_a[1] = -1;
    if ((cls & 1) && pointer)
        c->e_a[1] = num_env + 1 - pointer;
    else if (cls == VARFIX && pointer > 1)
        c->e_a[1] = pointer - 1;
    return 0;
}

static void copy_grid(struct sbr_ch *d, const struct sbr_ch *src)
{
    d->freq_res[0] = d->freq_res[d->num_env];
    d->t_env_old = d->t_env[d->num_env];
    d->e_a[0] = -(d->e_a[1] != d->num_env);
    memcpy(d->freq_res + 1, src->freq_res + 1, sizeof(int) * 5);
    memcpy(d->t_env, src->t_env, sizeof d->t_env);
    memcpy(d->t_q, src->t_q, sizeof d->t_q);
    d->num_env = src->num_env;
    d->amp_res = src->amp_res;
    d->num_noise = src->num_noise;
    d->frame_class = src->frame_class;
    d->e_a[1] = src->e_a[1];
}

static void read_dtdf(struct abr *b, struct sbr_ch *c)
{
    for (int i = 0; i < c->num_env; i++)
        c->df_env[i] = abr_bit(b);
    for (int i = 0; i < c->num_noise; i++)
        c->df_noise[i] = abr_bit(b);
}

static void read_invf(struct sbr *s, struct abr *b, struct sbr_ch *c)
{
    memcpy(c->invf[1], c->invf[0], sizeof c->invf[0]);
    for (int i = 0; i < s->n_q; i++)
        c->invf[0][i] = (int)abr_bits(b, 2);
}

static int huff(struct sbr *s, struct abr *b, int t)
{
    int v = vlc_get(b, &s->vlc[t]);
    return v < 0 ? 1000 : v + SBR_HUFF_OFFSET[t];
}

static int read_envelope(struct sbr *s, struct abr *b, struct sbr_ch *c, int ch)
{
    int bits, t, f;
    int delta = (ch == 1 && s->coupling == 1) + 1;
    int odd = s->n[1] & 1;
    if (s->coupling && ch) {
        if (c->amp_res) {
            bits = 5;
            t = T_ENV_BAL_3_0;
            f = F_ENV_BAL_3_0;
        } else {
            bits = 6;
            t = T_ENV_BAL_1_5;
            f = F_ENV_BAL_1_5;
        }
    } else {
        if (c->amp_res) {
            bits = 6;
            t = T_ENV_3_0;
            f = F_ENV_3_0;
        } else {
            bits = 7;
            t = T_ENV_1_5;
            f = F_ENV_1_5;
        }
    }
    for (int i = 0; i < c->num_env; i++) {
        int nb = s->n[c->freq_res[i + 1]];
        if (c->df_env[i]) {
            for (int j = 0; j < nb; j++) {
                int k;
                if (c->freq_res[i + 1] == c->freq_res[i])
                    k = j;
                else if (c->freq_res[i + 1])
                    k = (j + odd) >> 1;
                else
                    k = j ? 2 * j - odd : 0;
                int v = c->env_q[i][k] + delta * huff(s, b, t);
                if ((unsigned)v > 127)
                    return -1;
                c->env_q[i + 1][j] = v;
            }
        } else {
            c->env_q[i + 1][0] = delta * (int)abr_bits(b, bits);
            for (int j = 1; j < nb; j++) {
                int v = c->env_q[i + 1][j - 1] + delta * huff(s, b, f);
                if ((unsigned)v > 127)
                    return -1;
                c->env_q[i + 1][j] = v;
            }
        }
    }
    memcpy(c->env_q[0], c->env_q[c->num_env], sizeof c->env_q[0]);
    return 0;
}

static int read_noise(struct sbr *s, struct abr *b, struct sbr_ch *c, int ch)
{
    int delta = (ch == 1 && s->coupling == 1) + 1, t, f;
    if (s->coupling && ch) {
        t = T_NOISE_BAL_3_0;
        f = F_ENV_BAL_3_0;
    } else {
        t = T_NOISE_3_0;
        f = F_ENV_3_0;
    }
    for (int i = 0; i < c->num_noise; i++) {
        if (c->df_noise[i]) {
            for (int j = 0; j < s->n_q; j++) {
                int v = c->noise_q[i][j] + delta * huff(s, b, t);
                if ((unsigned)v > 30)
                    return -1;
                c->noise_q[i + 1][j] = v;
            }
        } else {
            c->noise_q[i + 1][0] = delta * (int)abr_bits(b, 5);
            for (int j = 1; j < s->n_q; j++) {
                int v = c->noise_q[i + 1][j - 1] + delta * huff(s, b, f);
                if ((unsigned)v > 30)
                    return -1;
                c->noise_q[i + 1][j] = v;
            }
        }
    }
    memcpy(c->noise_q[0], c->noise_q[c->num_noise], sizeof c->noise_q[0]);
    return 0;
}

static void read_harmonics(struct sbr *s, struct abr *b, struct sbr_ch *c)
{
    if ((c->add_harm_flag = abr_bit(b)))
        for (int i = 0; i < s->n[1]; i++)
            c->add_harm[i] = abr_bit(b);
}

static int read_data(struct sbr *s, struct abr *b)
{
    struct sbr_ch *c0 = &s->c[0], *c1 = &s->c[1];
    s->ready = 1;
    if (s->ch == 1) {
        if (abr_bit(b))
            abr_bits(b, 4);
        if (read_grid(s, b, c0) != 0)
            return -1;
        read_dtdf(b, c0);
        read_invf(s, b, c0);
        if (read_envelope(s, b, c0, 0) != 0 || read_noise(s, b, c0, 0) != 0)
            return -1;
        read_harmonics(s, b, c0);
    } else {
        if (abr_bit(b))
            abr_bits(b, 8);
        if ((s->coupling = abr_bit(b))) {
            if (read_grid(s, b, c0) != 0)
                return -1;
            copy_grid(c1, c0);
            read_dtdf(b, c0);
            read_dtdf(b, c1);
            read_invf(s, b, c0);
            memcpy(c1->invf[1], c1->invf[0], sizeof c1->invf[0]);
            memcpy(c1->invf[0], c0->invf[0], sizeof c1->invf[0]);
            if (read_envelope(s, b, c0, 0) != 0 ||
                read_noise(s, b, c0, 0) != 0 ||
                read_envelope(s, b, c1, 1) != 0 || read_noise(s, b, c1, 1) != 0)
                return -1;
        } else {
            if (read_grid(s, b, c0) != 0 || read_grid(s, b, c1) != 0)
                return -1;
            read_dtdf(b, c0);
            read_dtdf(b, c1);
            read_invf(s, b, c0);
            read_invf(s, b, c1);
            if (read_envelope(s, b, c0, 0) != 0 ||
                read_envelope(s, b, c1, 1) != 0 ||
                read_noise(s, b, c0, 0) != 0 || read_noise(s, b, c1, 1) != 0)
                return -1;
        }
        read_harmonics(s, b, c0);
        read_harmonics(s, b, c1);
    }
    if (abr_bit(b)) { /* bs_extended_data */
        int left = (int)abr_bits(b, 4);
        if (left == 15)
            left += (int)abr_bits(b, 8);
        left *= 8;
        while (left > 7) {
            int id = (int)abr_bits(b, 2);
            left -= 2;
            if (id == 2 && s->ch == 1) { /* EXTENSION_ID_PS */
                if (!s->ps && !(s->ps = ps_new())) {
                    b->pos += (size_t)left;
                    left = 0;
                    break;
                }
                int used = ps_read(s->ps, b, left);
                if (used < 0) {
                    b->pos += (size_t)left;
                    left = 0;
                } else {
                    s->ps_found = 1;
                    left -= used;
                }
            } else {
                b->pos += (size_t)left;
                left = 0;
            }
        }
        if (left > 0)
            b->pos += (size_t)left;
    }
    return 0;
}

int sbr_read(struct sbr *s, struct abr *b, int crc, int cnt)
{
    size_t end = b->pos + 8 * (size_t)cnt - 4;
    s->reset = 0;
    if (crc)
        abr_bits(b, 10);
    s->kx[0] = s->kx[1];
    s->m[0] = s->m[1];
    s->pushed = 1;
    if (abr_bit(b))
        read_header(s, b);
    if (s->reset)
        sbr_reset(s);
    if (s->start && read_data(s, b) != 0)
        turnoff(s);
    if (b->pos > end)
        turnoff(s);
    b->pos = end;
    return 0;
}

int sbr_has_ps(const struct sbr *s)
{
    return s->ps_found;
}

/* ---- dequantization ---- */

static void dequant(struct sbr *s)
{
    static const float EXP2_TAB[2] = {1.0f, (float)M_SQRT2};
    if (s->ch == 2 && s->coupling) {
        struct sbr_ch *a = &s->c[0], *b = &s->c[1];
        int pan_offset = a->amp_res ? 12 : 24;
        for (int e = 1; e <= a->num_env; e++) {
            for (int k = 0; k < s->n[a->freq_res[e]]; k++) {
                float t1, t2;
                if (a->amp_res) {
                    t1 = exp2fi(a->env_q[e][k] + 7);
                    t2 = exp2fi(pan_offset - b->env_q[e][k]);
                } else {
                    t1 = exp2fi((a->env_q[e][k] >> 1) + 7) *
                         EXP2_TAB[a->env_q[e][k] & 1];
                    t2 = exp2fi((pan_offset - b->env_q[e][k]) >> 1) *
                         EXP2_TAB[(pan_offset - b->env_q[e][k]) & 1];
                }
                if (t1 > 1E20f)
                    t1 = 1;
                float fac = t1 / (1.0f + t2);
                a->env[e][k] = fac;
                b->env[e][k] = fac * t2;
            }
        }
        for (int e = 1; e <= a->num_noise; e++) {
            for (int k = 0; k < s->n_q; k++) {
                float t1 = exp2fi(NOISE_FLOOR_OFFSET - a->noise_q[e][k] + 1);
                float t2 = exp2fi(12 - b->noise_q[e][k]);
                float fac = t1 / (1.0f + t2);
                a->noise[e][k] = fac;
                b->noise[e][k] = fac * t2;
            }
        }
    } else {
        for (int ch = 0; ch < s->ch; ch++) {
            struct sbr_ch *c = &s->c[ch];
            for (int e = 1; e <= c->num_env; e++)
                for (int k = 0; k < s->n[c->freq_res[e]]; k++) {
                    float v;
                    if (c->amp_res)
                        v = exp2fi(c->env_q[e][k] + 6);
                    else
                        v = exp2fi((c->env_q[e][k] >> 1) + 6) *
                            EXP2_TAB[c->env_q[e][k] & 1];
                    c->env[e][k] = v > 1E20f ? 1 : v;
                }
            for (int e = 1; e <= c->num_noise; e++)
                for (int k = 0; k < s->n_q; k++)
                    c->noise[e][k] =
                        exp2fi(NOISE_FLOOR_OFFSET - c->noise_q[e][k]);
        }
    }
}

/* ---- the QMF banks ---- */

/* 32 slots of 32 new samples each into W[slot][band] (4.6.18.4.1) */
static void qmf_analysis(struct sbr *s, struct sbr_ch *c, const float *in,
                         float (*W)[32][2])
{
    float *x = c->x_ana;
    memmove(x, x + 1024, sizeof(float) * 288);
    for (int i = 0; i < 1024; i++)
        x[288 + i] = in[i] * 32768.0f;
    for (int l = 0; l < 32; l++) {
        const float *xs = x + 32 * l; /* x[319 - n] is the spec's x[n] */
        float u[64];
        for (int n = 0; n < 64; n++) {
            float sum = 0;
            for (int j = 0; j < 5; j++) {
                int i = n + 64 * j;
                sum += xs[319 - i] * SBR_QMF_WINDOW[2 * i];
            }
            u[n] = sum;
        }
        /* W[k] = 2 sum u[n] e^(i pi / 64 (k + 1/2)(2n - 1/2)): the
           halves of u through DCT-IVs (sines by reversed inputs), then
           each band turned by e^(-i 3 pi (k + 1/2) / 128) */
        float lo[32], hi[32], rlo[32], rhi[32], c1[32], s1[32], c2[32], s2[32];
        for (int j = 0; j < 32; j++) {
            lo[j] = u[j];
            hi[j] = u[32 + j];
            rlo[j] = u[31 - j];
            rhi[j] = u[63 - j];
        }
        aimdct_dct4(s->m64, lo, 1, c1);
        aimdct_dct4(s->m64, rlo, 1, s1);
        aimdct_dct4(s->m64, hi, 1, c2);
        aimdct_dct4(s->m64, rhi, 1, s2);
        for (int k = 0; k < 32; k++) {
            float sg = k & 1 ? -1.0f : 1.0f;
            float re = c1[k] - s2[k], im = sg * s1[k] + sg * c2[k];
            W[l][k][0] = re * s->rot[k][0] - im * s->rot[k][1];
            W[l][k][1] = re * s->rot[k][1] + im * s->rot[k][0];
        }
    }
}

/* 32 slots of 64 bands to 2048 samples (4.6.18.4.2) */
static void qmf_synthesis(struct sbr *s, struct sbr_ch *c, float (*X)[38][64],
                          float *out)
{
    float *v = c->v;
    for (int l = 0; l < 32; l++) {
        memmove(v + 128, v, sizeof(float) * (1280 - 128));
        /* v[n] = 1/64 sum Re(X[k] e^(i pi / 128 (k + 1/2)(2n - 255))):
           the real parts' inverse MDCT and the imaginary parts', reversed
           (the sines), shifted by 32 and signed */
        float rev[64], y1[128], y2[128];
        for (int k = 0; k < 64; k++)
            rev[k] = X[1][l][63 - k];
        aimdct_run(s->m128, X[0][l], y1);
        aimdct_run(s->m128, rev, y2);
        for (int n = 0; n < 128; n++) {
            int i = n >= 32 ? n - 32 : n + 96;
            float c = y1[i], sn = i & 1 ? -y2[i] : y2[i];
            v[n] = (n >= 32 ? sn - c : c - sn) * (1.0f / 64.0f);
        }
        float *o = out + 64 * l;
        for (int k = 0; k < 64; k++) {
            float sum = 0;
            for (int n = 0; n < 5; n++) {
                sum += v[256 * n + k] * SBR_QMF_WINDOW[128 * n + k];
                sum += v[256 * n + 192 + k] * SBR_QMF_WINDOW[128 * n + 64 + k];
            }
            o[k] = sum * (1.0f / 32768.0f);
        }
    }
}

/* the downsampled bank: 32 slots of the lower 32 bands to 1024 samples
   (4.6.18.4.3), its phase (2n - 127.5) the one the reference decoder's
   output and FFmpeg's agree on */
static void qmf_synthesis_ds(struct sbr *s, struct sbr_ch *c,
                             float (*X)[38][64], float *out)
{
    float *v = c->v;
    for (int l = 0; l < 32; l++) {
        memmove(v + 64, v, sizeof(float) * (640 - 64));
        for (int n = 0; n < 64; n++) {
            float sum = 0;
            for (int k = 0; k < 32; k++)
                sum +=
                    X[0][l][k] * s->ds_cos[n][k] - X[1][l][k] * s->ds_sin[n][k];
            v[n] = sum * (1.0f / 64.0f);
        }
        float *o = out + 32 * l;
        for (int k = 0; k < 32; k++) {
            float sum = 0;
            for (int n = 0; n < 5; n++) {
                sum += v[128 * n + k] * SBR_QMF_WINDOW[2 * (64 * n + k)];
                sum +=
                    v[128 * n + 96 + k] * SBR_QMF_WINDOW[2 * (64 * n + 32 + k)];
            }
            o[k] = sum * (1.0f / 32768.0f);
        }
    }
}

/* ---- the high band ---- */

static void lf_gen(struct sbr *s, struct sbr_ch *c)
{
    memset(s->X_low, 0, sizeof s->X_low);
    for (int k = 0; k < s->kx[1]; k++)
        for (int i = 8; i < 40; i++) {
            s->X_low[k][i][0] = c->W[c->ypos][i - 8][k][0];
            s->X_low[k][i][1] = c->W[c->ypos][i - 8][k][1];
        }
    int prev = 1 - c->ypos;
    for (int k = 0; k < s->kx[0]; k++)
        for (int i = 0; i < 8; i++) {
            s->X_low[k][i][0] = c->W[prev][i + 24][k][0];
            s->X_low[k][i][1] = c->W[prev][i + 24][k][1];
        }
}

static void autocorrelate(const float x[40][2], float phi[3][2][2])
{
    float real2 = x[0][0] * x[2][0] + x[0][1] * x[2][1];
    float imag2 = x[0][0] * x[2][1] - x[0][1] * x[2][0];
    float real1 = 0, imag1 = 0, real0 = 0;
    for (int i = 1; i < 38; i++) {
        real0 += x[i][0] * x[i][0] + x[i][1] * x[i][1];
        real1 += x[i][0] * x[i + 1][0] + x[i][1] * x[i + 1][1];
        imag1 += x[i][0] * x[i + 1][1] - x[i][1] * x[i + 1][0];
        real2 += x[i][0] * x[i + 2][0] + x[i][1] * x[i + 2][1];
        imag2 += x[i][0] * x[i + 2][1] - x[i][1] * x[i + 2][0];
    }
    phi[0][1][0] = real2;
    phi[0][1][1] = imag2;
    phi[2][1][0] = real0 + x[0][0] * x[0][0] + x[0][1] * x[0][1];
    phi[1][0][0] = real0 + x[38][0] * x[38][0] + x[38][1] * x[38][1];
    phi[1][1][0] = real1 + x[0][0] * x[1][0] + x[0][1] * x[1][1];
    phi[1][1][1] = imag1 + x[0][0] * x[1][1] - x[0][1] * x[1][0];
    phi[0][0][0] = real1 + x[38][0] * x[39][0] + x[38][1] * x[39][1];
    phi[0][0][1] = imag1 + x[38][0] * x[39][1] - x[38][1] * x[39][0];
}

static void inverse_filter(struct sbr *s)
{
    for (int k = 0; k < s->k[0]; k++) {
        float phi[3][2][2];
        float *a0 = s->alpha0[k], *a1 = s->alpha1[k];
        autocorrelate((const float(*)[2])s->X_low[k], phi);
        float dk = phi[2][1][0] * phi[1][0][0] -
                   (phi[1][1][0] * phi[1][1][0] + phi[1][1][1] * phi[1][1][1]) /
                       1.000001f;
        if (!dk) {
            a1[0] = a1[1] = 0;
        } else {
            float re = phi[0][0][0] * phi[1][1][0] -
                       phi[0][0][1] * phi[1][1][1] -
                       phi[0][1][0] * phi[1][0][0];
            float im = phi[0][0][0] * phi[1][1][1] +
                       phi[0][0][1] * phi[1][1][0] -
                       phi[0][1][1] * phi[1][0][0];
            a1[0] = re / dk;
            a1[1] = im / dk;
        }
        if (!phi[1][0][0]) {
            a0[0] = a0[1] = 0;
        } else {
            float re =
                phi[0][0][0] + a1[0] * phi[1][1][0] + a1[1] * phi[1][1][1];
            float im =
                phi[0][0][1] + a1[1] * phi[1][1][0] - a1[0] * phi[1][1][1];
            a0[0] = -re / phi[1][0][0];
            a0[1] = -im / phi[1][0][0];
        }
        if (a1[0] * a1[0] + a1[1] * a1[1] >= 16.0f ||
            a0[0] * a0[0] + a0[1] * a0[1] >= 16.0f)
            a0[0] = a0[1] = a1[0] = a1[1] = 0;
    }
}

static void chirp(struct sbr *s, struct sbr_ch *c)
{
    static const float BW[] = {0.0f, 0.75f, 0.9f, 0.98f};
    for (int i = 0; i < s->n_q; i++) {
        float nb;
        if (c->invf[0][i] + c->invf[1][i] == 1)
            nb = 0.6f;
        else
            nb = BW[c->invf[0][i]];
        if (nb < c->bw[i])
            nb = 0.75f * nb + 0.25f * c->bw[i];
        else
            nb = 0.90625f * nb + 0.09375f * c->bw[i];
        c->bw[i] = nb < 0.015625f ? 0.0f : nb;
    }
}

static int hf_gen(struct sbr *s, struct sbr_ch *c)
{
    int g = 0, k = s->kx[1];
    for (int j = 0; j < s->num_patches; j++) {
        for (int x = 0; x < s->patch_num[j]; x++, k++) {
            int p = s->patch_start[j] + x;
            while (g <= s->n_q && k >= s->f_noise[g])
                g++;
            g--;
            if (g < 0 || k >= 64)
                return -1;
            float bw = c->bw[g];
            float a[4] = {s->alpha1[p][0] * bw * bw, s->alpha1[p][1] * bw * bw,
                          s->alpha0[p][0] * bw, s->alpha0[p][1] * bw};
            float(*hi)[2] = s->X_high[k] + ENV_OFF;
            const float(*lo)[2] = (const float(*)[2])s->X_low[p] + ENV_OFF;
            for (int i = 2 * c->t_env[0]; i < 2 * c->t_env[c->num_env]; i++) {
                hi[i][0] = lo[i - 2][0] * a[0] - lo[i - 2][1] * a[1] +
                           lo[i - 1][0] * a[2] - lo[i - 1][1] * a[3] + lo[i][0];
                hi[i][1] = lo[i - 2][1] * a[0] + lo[i - 2][0] * a[1] +
                           lo[i - 1][1] * a[2] + lo[i - 1][0] * a[3] + lo[i][1];
            }
        }
    }
    if (k < s->m[1] + s->kx[1])
        memset(s->X_high + k, 0,
               sizeof(s->X_high[0]) * (size_t)(s->m[1] + s->kx[1] - k));
    return 0;
}

static int mapping(struct sbr *s, struct sbr_ch *c)
{
    int kx = s->kx[1];
    memset(c->s_index[1], 0, sizeof(c->s_index[1]) * 6);
    for (int e = 0; e < c->num_env; e++) {
        int ilim = s->n[c->freq_res[e + 1]];
        const int *table = c->freq_res[e + 1] ? s->f_high : s->f_low;
        if (kx != table[0])
            return -1;
        for (int i = 0; i < ilim; i++)
            for (int m = table[i]; m < table[i + 1]; m++)
                s->e_orig[e][m - kx] = c->env[e + 1][i];
        int k = c->num_noise > 1 && c->t_env[e] >= c->t_q[1];
        for (int i = 0; i < s->n_q; i++)
            for (int m = s->f_noise[i]; m < s->f_noise[i + 1]; m++)
                s->q_mapped[e][m - kx] = c->noise[k + 1][i];
        for (int i = 0; i < s->n[1]; i++) {
            if (c->add_harm_flag) {
                int mid = (s->f_high[i] + s->f_high[i + 1]) >> 1;
                c->s_index[e + 1][mid - kx] =
                    c->add_harm[i] *
                    (e >= c->e_a[1] || c->s_index[0][mid - kx] == 1);
            }
        }
        for (int i = 0; i < ilim; i++) {
            int present = 0;
            for (int m = table[i]; m < table[i + 1]; m++)
                if (c->s_index[e + 1][m - kx]) {
                    present = 1;
                    break;
                }
            for (int m = table[i]; m < table[i + 1]; m++)
                s->s_mapped[e][m - kx] = present;
        }
    }
    memcpy(c->s_index[0], c->s_index[c->num_env], sizeof c->s_index[0]);
    return 0;
}

static void env_estimate(struct sbr *s, struct sbr_ch *c)
{
    int kx = s->kx[1];
    for (int e = 0; e < c->num_env; e++) {
        int ilb = c->t_env[e] * 2 + ENV_OFF,
            iub = c->t_env[e + 1] * 2 + ENV_OFF;
        if (s->interpol_freq) {
            float recip = 0.5f / (float)(c->t_env[e + 1] - c->t_env[e]);
            for (int m = 0; m < s->m[1]; m++) {
                float sum = 0;
                for (int i = ilb; i < iub; i++)
                    sum += s->X_high[m + kx][i][0] * s->X_high[m + kx][i][0] +
                           s->X_high[m + kx][i][1] * s->X_high[m + kx][i][1];
                s->e_curr[e][m] = sum * recip;
            }
        } else {
            int size = 2 * (c->t_env[e + 1] - c->t_env[e]);
            const int *table = c->freq_res[e + 1] ? s->f_high : s->f_low;
            for (int p = 0; p < s->n[c->freq_res[e + 1]]; p++) {
                float sum = 0;
                int den = size * (table[p + 1] - table[p]);
                for (int k = table[p]; k < table[p + 1]; k++) {
                    float sq = 0;
                    for (int i = ilb; i < iub; i++)
                        sq += s->X_high[k][i][0] * s->X_high[k][i][0] +
                              s->X_high[k][i][1] * s->X_high[k][i][1];
                    sum += sq;
                }
                sum /= (float)den;
                for (int k = table[p]; k < table[p + 1]; k++)
                    s->e_curr[e][k - kx] = sum;
            }
        }
    }
}

static void gain_calc(struct sbr *s, struct sbr_ch *c)
{
    static const float LIMGAIN[4] = {0.70795f, 1.0f, 1.41254f, 10000000000.0f};
    int kx = s->kx[1];
    for (int e = 0; e < c->num_env; e++) {
        int delta = !(e == c->e_a[1] || e == c->e_a[0]);
        for (int k = 0; k < s->n_lim; k++) {
            int lo = s->f_lim[k] - kx, hi = s->f_lim[k + 1] - kx;
            float sum[2] = {0, 0};
            for (int m = lo; m < hi; m++) {
                float temp = s->e_orig[e][m] / (1.0f + s->q_mapped[e][m]);
                s->q_m[e][m] = sqrtf(temp * s->q_mapped[e][m]);
                s->s_m[e][m] = sqrtf(temp * (float)c->s_index[e + 1][m]);
                if (!s->s_mapped[e][m])
                    s->gain[e][m] =
                        sqrtf(s->e_orig[e][m] /
                              ((1.0f + s->e_curr[e][m]) *
                               (1.0f + s->q_mapped[e][m] * (float)delta)));
                else
                    s->gain[e][m] = sqrtf(s->e_orig[e][m] * s->q_mapped[e][m] /
                                          ((1.0f + s->e_curr[e][m]) *
                                           (1.0f + s->q_mapped[e][m])));
                s->gain[e][m] += FLT_MIN;
            }
            for (int m = lo; m < hi; m++) {
                sum[0] += s->e_orig[e][m];
                sum[1] += s->e_curr[e][m];
            }
            float gain_max =
                LIMGAIN[s->limiter_gains] *
                sqrtf((FLT_EPSILON + sum[0]) / (FLT_EPSILON + sum[1]));
            if (gain_max > 100000.f)
                gain_max = 100000.f;
            for (int m = lo; m < hi; m++) {
                float q_m_max = s->q_m[e][m] * gain_max / s->gain[e][m];
                if (q_m_max < s->q_m[e][m])
                    s->q_m[e][m] = q_m_max;
                if (gain_max < s->gain[e][m])
                    s->gain[e][m] = gain_max;
            }
            sum[0] = sum[1] = 0;
            for (int m = lo; m < hi; m++) {
                sum[0] += s->e_orig[e][m];
                sum[1] += s->e_curr[e][m] * s->gain[e][m] * s->gain[e][m] +
                          s->s_m[e][m] * s->s_m[e][m] +
                          (float)(delta && !s->s_m[e][m]) * s->q_m[e][m] *
                              s->q_m[e][m];
            }
            float boost =
                sqrtf((FLT_EPSILON + sum[0]) / (FLT_EPSILON + sum[1]));
            if (boost > 1.584893192f)
                boost = 1.584893192f;
            for (int m = lo; m < hi; m++) {
                s->gain[e][m] *= boost;
                s->q_m[e][m] *= boost;
                s->s_m[e][m] *= boost;
            }
        }
    }
}

static void hf_assemble(struct sbr *s, struct sbr_ch *c, float (*Y1)[64][2])
{
    static const float H_SMOOTH[5] = {0.33333333333333f, 0.30150283239582f,
                                      0.21816949906249f, 0.11516383427084f,
                                      0.03183050093751f};
    int h_sl = 4 * !s->smoothing;
    int kx = s->kx[1], m_max = s->m[1];
    float(*g_temp)[48] = c->g_temp, (*q_temp)[48] = c->q_temp;
    int index_noise = c->index_noise, index_sine = c->index_sine;
    if (s->reset) {
        for (int i = 0; i < h_sl; i++) {
            memcpy(g_temp[i + 2 * c->t_env[0]], s->gain[0],
                   sizeof(float) * (size_t)m_max);
            memcpy(q_temp[i + 2 * c->t_env[0]], s->q_m[0],
                   sizeof(float) * (size_t)m_max);
        }
    } else if (h_sl) {
        for (int i = 0; i < 4; i++) {
            memcpy(g_temp[i + 2 * c->t_env[0]], g_temp[i + 2 * c->t_env_old],
                   sizeof g_temp[0]);
            memcpy(q_temp[i + 2 * c->t_env[0]], q_temp[i + 2 * c->t_env_old],
                   sizeof q_temp[0]);
        }
    }
    for (int e = 0; e < c->num_env; e++)
        for (int i = 2 * c->t_env[e]; i < 2 * c->t_env[e + 1]; i++) {
            memcpy(g_temp[h_sl + i], s->gain[e], sizeof(float) * (size_t)m_max);
            memcpy(q_temp[h_sl + i], s->q_m[e], sizeof(float) * (size_t)m_max);
        }
    for (int e = 0; e < c->num_env; e++) {
        for (int i = 2 * c->t_env[e]; i < 2 * c->t_env[e + 1]; i++) {
            float g_tab[48], q_tab[48];
            const float *g_filt, *q_filt;
            if (h_sl && e != c->e_a[0] && e != c->e_a[1]) {
                for (int m = 0; m < m_max; m++) {
                    int idx1 = i + h_sl;
                    g_tab[m] = 0;
                    q_tab[m] = 0;
                    for (int j = 0; j <= h_sl; j++) {
                        g_tab[m] += g_temp[idx1 - j][m] * H_SMOOTH[j];
                        q_tab[m] += q_temp[idx1 - j][m] * H_SMOOTH[j];
                    }
                }
                g_filt = g_tab;
                q_filt = q_tab;
            } else {
                g_filt = g_temp[i + h_sl];
                q_filt = q_temp[i];
            }
            float(*Y)[2] = Y1[i] + kx;
            for (int m = 0; m < m_max; m++) {
                Y[m][0] = s->X_high[kx + m][i + ENV_OFF][0] * g_filt[m];
                Y[m][1] = s->X_high[kx + m][i + ENV_OFF][1] * g_filt[m];
            }
            const float *s_m = s->s_m[e];
            if (e != c->e_a[0] && e != c->e_a[1]) {
                float sign0, sign1;
                int ks = 1 - 2 * (kx & 1);
                switch (index_sine) {
                case 0:
                    sign0 = 1;
                    sign1 = 0;
                    break;
                case 1:
                    sign0 = 0;
                    sign1 = (float)ks;
                    break;
                case 2:
                    sign0 = -1;
                    sign1 = 0;
                    break;
                default:
                    sign0 = 0;
                    sign1 = (float)-ks;
                    break;
                }
                int noise = index_noise;
                for (int m = 0; m < m_max; m++) {
                    float y0 = Y[m][0], y1 = Y[m][1];
                    noise = (noise + 1) & 0x1ff;
                    if (s_m[m]) {
                        y0 += s_m[m] * sign0;
                        y1 += s_m[m] * sign1;
                    } else {
                        y0 += q_filt[m] * SBR_NOISE[2 * noise];
                        y1 += q_filt[m] * SBR_NOISE[2 * noise + 1];
                    }
                    Y[m][0] = y0;
                    Y[m][1] = y1;
                    sign1 = -sign1;
                }
            } else {
                int idx = index_sine & 1;
                int A = 1 - ((index_sine + (kx & 1)) & 2);
                int B = (A ^ (-idx)) + idx;
                for (int m = 0; m < m_max; m++)
                    Y[m][idx] += s_m[m] * (float)(m & 1 ? B : A);
            }
            index_noise = (index_noise + m_max) & 0x1ff;
            index_sine = (index_sine + 1) & 3;
        }
    }
    c->index_noise = index_noise;
    c->index_sine = index_sine;
}

/* the synthesis' input: the low band, the high band of the frame before
   that runs into this one, this frame's */
static void x_gen(struct sbr *s, struct sbr_ch *c, float (*X)[38][64])
{
    const float(*Y0)[64][2] = (const float(*)[64][2])c->Y[1 - c->ypos];
    const float(*Y1)[64][2] = (const float(*)[64][2])c->Y[c->ypos];
    int i_temp = 2 * c->t_env_old - 32;
    if (i_temp < 0)
        i_temp = 0;
    memset(X, 0, sizeof(float) * 2 * 38 * 64);
    int k;
    for (k = 0; k < s->kx[0]; k++)
        for (int i = 0; i < i_temp; i++) {
            X[0][i][k] = s->X_low[k][i + ENV_OFF][0];
            X[1][i][k] = s->X_low[k][i + ENV_OFF][1];
        }
    for (; k < s->kx[0] + s->m[0]; k++)
        for (int i = 0; i < i_temp; i++) {
            X[0][i][k] = Y0[i + 32][k][0];
            X[1][i][k] = Y0[i + 32][k][1];
        }
    for (k = 0; k < s->kx[1]; k++)
        for (int i = i_temp; i < 38; i++) {
            X[0][i][k] = s->X_low[k][i + ENV_OFF][0];
            X[1][i][k] = s->X_low[k][i + ENV_OFF][1];
        }
    for (; k < s->kx[1] + s->m[1]; k++)
        for (int i = i_temp; i < 32; i++) {
            X[0][i][k] = Y1[i][k][0];
            X[1][i][k] = Y1[i][k][1];
        }
}

void sbr_apply(struct sbr *s, int ch, float *const *in, float **out)
{
    if (ch != s->ch)
        turnoff(s);
    if (s->start && !s->ready)
        turnoff(s);
    if (!s->pushed) {
        s->kx[0] = s->kx[1];
        s->m[0] = s->m[1];
    } else {
        s->pushed = 0;
    }
    if (s->start) {
        dequant(s);
        s->ready = 0;
    }
    for (int k = 0; k < ch; k++) {
        struct sbr_ch *c = &s->c[k];
        qmf_analysis(s, c, in[k], c->W[c->ypos]);
        lf_gen(s, c);
        c->ypos ^= 1;
        if (s->start) {
            inverse_filter(s);
            chirp(s, c);
            if (c->num_env > 0 && hf_gen(s, c) == 0 && mapping(s, c) == 0) {
                env_estimate(s, c);
                gain_calc(s, c);
                hf_assemble(s, c, c->Y[c->ypos]);
            }
        }
        x_gen(s, c, s->X[k]);
    }
    int nch = ch;
    if (ch == 1 && s->ps_found) {
        ps_apply(s->ps, s->X[0], s->X[1], s->kx[1] + s->m[1]);
        nch = 2;
    }
    for (int k = 0; k < nch; k++) {
        if (s->down)
            qmf_synthesis_ds(s, &s->c[k], s->X[k], out[k]);
        else
            qmf_synthesis(s, &s->c[k], s->X[k], out[k]);
    }
}

/* ---- life ---- */

struct sbr *sbr_new(int rate, int ch, int down)
{
    struct sbr *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    s->rate = 2 * rate;
    s->ch = ch;
    s->down = down;
    s->kx[1] = 32;
    turnoff(s);
    for (int t = 0; t < 10; t++) {
        int at = SBR_HUFF_AT[t], n = SBR_HUFF_AT[t + 1] - at;
        if (vlc_build(&s->vlc[t], 9, SBR_HUFF_CODE + at, SBR_HUFF_LEN + at,
                      SBR_HUFF_SYM + at, n) != 0) {
            sbr_free(s);
            return NULL;
        }
    }
    s->m64 = aimdct_new(64);
    s->m128 = aimdct_new(128);
    if (!s->m64 || !s->m128) {
        sbr_free(s);
        return NULL;
    }
    for (int k = 0; k < 32; k++) {
        double a = -3 * M_PI * (k + 0.5) / 128;
        s->rot[k][0] = (float)(2 * cos(a));
        s->rot[k][1] = (float)(2 * sin(a));
    }
    for (int n = 0; n < 64; n++)
        for (int k = 0; k < 32; k++) {
            double a = M_PI / 64.0 * (k + 0.5) * (2 * n - 127.5);
            s->ds_cos[n][k] = (float)cos(a);
            s->ds_sin[n][k] = (float)sin(a);
        }
    return s;
}

void sbr_free(struct sbr *s)
{
    if (!s)
        return;
    for (int t = 0; t < 10; t++)
        vlc_free(&s->vlc[t]);
    ps_free(s->ps);
    aimdct_free(s->m64);
    aimdct_free(s->m128);
    free(s);
}
