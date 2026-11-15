/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac_ps.c - parametric stereo (ISO/IEC 14496-3, 8.6.4), HE-AAC
 * v2's two channels made from one: the ps_data in SBR's extension (the
 * intensity differences, coherences and, beyond the baseline, the phase
 * differences, in 10, 20 or 34 bands, Huffman-coded in frequency or in
 * time), the hybrid analysis that splits the lowest QMF bands, the
 * decorrelated signal (all-pass links and delays, transients attenuated),
 * the mixing matrices interpolated between the envelopes' borders, and the
 * hybrid synthesis back to the QMF bands that SBR's synthesis turns into
 * samples. Single precision, FFmpeg's order of operations.
 */
#include "audio_aac.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "audio_aac_tab.h"

#define MAX_ENV 5
#define MAX_PAR 34
#define MAX_SSB 91
#define MAX_AP_BANDS 50
#define SLOTS 32
#define MAX_DELAY 14
#define AP_LINKS 3
#define MAX_AP_DELAY 5

enum {
    IID_DF1,
    IID_DT1,
    IID_DF0,
    IID_DT0,
    ICC_DF,
    ICC_DT,
    IPD_DF,
    IPD_DT,
    OPD_DF,
    OPD_DT,
};

static const int NR_PAR_BANDS[2] = {20, 34};
static const int NR_IPDOPD_BANDS[2] = {11, 17};
static const int NR_BANDS[2] = {71, 91};
static const int DECAY_CUTOFF[2] = {10, 32};
static const int NR_ALLPASS_BANDS[2] = {30, 50};
static const int SHORT_DELAY_BAND[2] = {42, 62};

struct ps_tables {
    float pd_re[512], pd_im[512];
    float HA[46][8][4], HB[46][8][4];
    float f20_0_8[8][8][2], f34_0_12[12][8][2], f34_1_8[8][8][2],
        f34_2_4[4][8][2];
    float q_fract[2][50][3][2];
    float phi_fract[2][50][2];
};

struct ps {
    /* the bitstream's */
    int start;
    int enable_iid, iid_quant, nr_iid_par, nr_ipdopd_par;
    int enable_icc, icc_mode, nr_icc_par;
    int enable_ext, frame_class, num_env_old, num_env;
    int enable_ipdopd;
    int border[MAX_ENV + 1];
    int8_t iid[MAX_ENV][MAX_PAR], icc[MAX_ENV][MAX_PAR];
    int8_t ipd[MAX_ENV][MAX_PAR], opd[MAX_ENV][MAX_PAR];
    int is34, is34_old;
    /* the processing's */
    float in_buf[5][44][2];
    float delay[MAX_SSB][SLOTS + MAX_DELAY][2];
    float ap_delay[MAX_AP_BANDS][AP_LINKS][SLOTS + MAX_AP_DELAY][2];
    float peak_decay_nrg[34], power_smooth[34], peak_decay_diff_smooth[34];
    float H11[2][MAX_ENV + 1][MAX_PAR], H12[2][MAX_ENV + 1][MAX_PAR];
    float H21[2][MAX_ENV + 1][MAX_PAR], H22[2][MAX_ENV + 1][MAX_PAR];
    float Lbuf[91][32][2], Rbuf[91][32][2];
    int8_t opd_hist[MAX_PAR], ipd_hist[MAX_PAR];
    struct vlc vlc[10];
    struct ps_tables t;
};

/* ---- the tables (FFmpeg's ps_tableinit) ---- */

static void make_filters(float (*filter)[8][2], const float *proto, int bands)
{
    for (int q = 0; q < bands; q++)
        for (int n = 0; n < 7; n++) {
            double theta = 2 * M_PI * (q + 0.5) * (n - 6) / bands;
            filter[q][n][0] = (float)(proto[n] * cos(theta));
            filter[q][n][1] = (float)(proto[n] * -sin(theta));
        }
}

static void tables(struct ps_tables *t)
{
    static const float SIN[] = {0, (float)M_SQRT1_2,  1,  (float)M_SQRT1_2,
                                0, -(float)M_SQRT1_2, -1, -(float)M_SQRT1_2};
    static const float COS[] = {1,  (float)M_SQRT1_2,  0, -(float)M_SQRT1_2,
                                -1, -(float)M_SQRT1_2, 0, (float)M_SQRT1_2};
    static const float IID_DEQ[] = {
        0.05623413251903f,
        0.12589254117942f,
        0.19952623149689f,
        0.31622776601684f,
        0.44668359215096f,
        0.63095734448019f,
        0.79432823472428f,
        1,
        1.25892541179417f,
        1.58489319246111f,
        2.23872113856834f,
        3.16227766016838f,
        5.01187233627272f,
        7.94328234724282f,
        17.7827941003892f,
        0.00316227766017f,
        0.00562341325190f,
        0.01f,
        0.01778279410039f,
        0.03162277660168f,
        0.05623413251903f,
        0.07943282347243f,
        0.11220184543020f,
        0.15848931924611f,
        0.22387211385683f,
        0.31622776601684f,
        0.39810717055350f,
        0.50118723362727f,
        0.63095734448019f,
        0.79432823472428f,
        1,
        1.25892541179417f,
        1.58489319246111f,
        1.99526231496888f,
        2.51188643150958f,
        3.16227766016838f,
        4.46683592150963f,
        6.30957344480193f,
        8.91250938133745f,
        12.5892541179417f,
        17.7827941003892f,
        31.6227766016838f,
        56.2341325190349f,
        100,
        177.827941003892f,
        316.227766016837f,
    };
    static const float ICC_INVQ[] = {1,        0.937f, 0.84118f, 0.60092f,
                                     0.36764f, 0,      -0.589f,  -1};
    static const float ACOS_ICC_INVQ[] = {
        0,          0.35685527f,       0.57133466f, 0.92614472f,
        1.1943263f, (float)(M_PI / 2), 2.2006171f,  (float)M_PI};
    static const int8_t F_CENTER_20[] = {-3, -1, 1, 3, 5, 7, 10, 14, 18, 22};
    static const int8_t F_CENTER_34[] = {
        2,  6,  10, 14, 18, 22, 26, 30, 34,  -10, -6, -2, 51,  57,  15,  21,
        27, 33, 39, 45, 54, 66, 78, 42, 102, 66,  78, 90, 102, 114, 126, 90};
    static const float LINKS[] = {0.43f, 0.75f, 0.347f};
    const float gain = 0.39f;
    for (int p0 = 0; p0 < 8; p0++)
        for (int p1 = 0; p1 < 8; p1++)
            for (int p2 = 0; p2 < 8; p2++) {
                float re = 0.25f * COS[p0] + 0.5f * COS[p1] + COS[p2];
                float im = 0.25f * SIN[p0] + 0.5f * SIN[p1] + SIN[p2];
                float mag = (float)(1 / hypot(im, re));
                t->pd_re[p0 * 64 + p1 * 8 + p2] = re * mag;
                t->pd_im[p0 * 64 + p1 * 8 + p2] = im * mag;
            }
    for (int iid = 0; iid < 46; iid++) {
        float c = IID_DEQ[iid];
        float c1 = (float)M_SQRT2 / sqrtf(1.0f + c * c);
        float c2 = c * c1;
        for (int icc = 0; icc < 8; icc++) {
            float alpha = 0.5f * ACOS_ICC_INVQ[icc];
            float beta = alpha * (c1 - c2) * (float)M_SQRT1_2;
            t->HA[iid][icc][0] = c2 * cosf(beta + alpha);
            t->HA[iid][icc][1] = c1 * cosf(beta - alpha);
            t->HA[iid][icc][2] = c2 * sinf(beta + alpha);
            t->HA[iid][icc][3] = c1 * sinf(beta - alpha);
            float rho = ICC_INVQ[icc] > 0.05f ? ICC_INVQ[icc] : 0.05f;
            alpha = 0.5f * atan2f(2.0f * c * rho, c * c - 1.0f);
            float mu = c + 1.0f / c;
            mu = sqrtf(1 + (4 * rho * rho - 4) / (mu * mu));
            float gamma = atanf(sqrtf((1.0f - mu) / (1.0f + mu)));
            if (alpha < 0)
                alpha += (float)(M_PI / 2);
            float ac = cosf(alpha), as = sinf(alpha);
            float gc = cosf(gamma), gs = sinf(gamma);
            t->HB[iid][icc][0] = (float)(M_SQRT2 * ac * gc);
            t->HB[iid][icc][1] = (float)(M_SQRT2 * as * gc);
            t->HB[iid][icc][2] = (float)(-M_SQRT2 * as * gs);
            t->HB[iid][icc][3] = (float)(M_SQRT2 * ac * gs);
        }
    }
    for (int k = 0; k < 30; k++) {
        double fc = k < 10 ? F_CENTER_20[k] * 0.125 : k - 6.5f;
        for (int m = 0; m < AP_LINKS; m++) {
            double th = -M_PI * LINKS[m] * fc;
            t->q_fract[0][k][m][0] = (float)cos(th);
            t->q_fract[0][k][m][1] = (float)sin(th);
        }
        double th = -M_PI * gain * fc;
        t->phi_fract[0][k][0] = (float)cos(th);
        t->phi_fract[0][k][1] = (float)sin(th);
    }
    for (int k = 0; k < 50; k++) {
        double fc = k < 32 ? F_CENTER_34[k] / 24.0 : k - 26.5f;
        for (int m = 0; m < AP_LINKS; m++) {
            double th = -M_PI * LINKS[m] * fc;
            t->q_fract[1][k][m][0] = (float)cos(th);
            t->q_fract[1][k][m][1] = (float)sin(th);
        }
        double th = -M_PI * gain * fc;
        t->phi_fract[1][k][0] = (float)cos(th);
        t->phi_fract[1][k][1] = (float)sin(th);
    }
    make_filters(t->f20_0_8, PS_G0_Q8, 8);
    make_filters(t->f34_0_12, PS_G0_Q12, 12);
    make_filters(t->f34_1_8, PS_G1_Q8, 8);
    make_filters(t->f34_2_4, PS_G2_Q4, 4);
}

struct ps *ps_new(void)
{
    struct ps *p = calloc(1, sizeof *p);
    if (!p)
        return NULL;
    for (int t = 0; t < 10; t++) {
        int at = PS_HUFF_AT[t], n = PS_HUFF_AT[t + 1] - at;
        if (vlc_build(&p->vlc[t], 9, PS_HUFF_CODE + at, PS_HUFF_LEN + at,
                      PS_HUFF_SYM + at, n) != 0) {
            ps_free(p);
            return NULL;
        }
    }
    tables(&p->t);
    return p;
}

void ps_free(struct ps *p)
{
    if (!p)
        return;
    for (int t = 0; t < 10; t++)
        vlc_free(&p->vlc[t]);
    free(p);
}

/* ---- the data ---- */

static int huff(struct ps *p, struct abr *b, int t)
{
    int v = vlc_get(b, &p->vlc[t]);
    return v < 0 ? 1000 : v + PS_HUFF_OFFSET[t];
}

/* kind 0 iid, 1 icc, 2 ipd/opd: 0, or -1 */
static int read_par(struct ps *p, struct abr *b, int8_t (*par)[MAX_PAR],
                    int table, int e, int dt, int num, int kind)
{
    if (dt) {
        int ep = e ? e - 1 : p->num_env_old - 1;
        if (ep < 0)
            ep = 0;
        for (int k = 0; k < num; k++) {
            int v = par[ep][k] + huff(p, b, table);
            if (kind == 2)
                v &= 7;
            par[e][k] = (int8_t)v;
            if ((kind == 0 && abs(v) > 7 + 8 * p->iid_quant) ||
                (kind == 1 && (unsigned)v > 7))
                return -1;
        }
    } else {
        int v = 0;
        for (int k = 0; k < num; k++) {
            v += huff(p, b, table);
            if (kind == 2)
                v &= 7;
            par[e][k] = (int8_t)v;
            if ((kind == 0 && abs(v) > 7 + 8 * p->iid_quant) ||
                (kind == 1 && (unsigned)v > 7))
                return -1;
        }
    }
    return 0;
}

static int read_ext(struct ps *p, struct abr *b, int id)
{
    size_t start = b->pos;
    if (id)
        return 0;
    p->enable_ipdopd = abr_bit(b);
    if (p->enable_ipdopd) {
        for (int e = 0; e < p->num_env; e++) {
            int dt = abr_bit(b);
            read_par(p, b, p->ipd, dt ? IPD_DT : IPD_DF, e, dt,
                     p->nr_ipdopd_par, 2);
            dt = abr_bit(b);
            read_par(p, b, p->opd, dt ? OPD_DT : OPD_DF, e, dt,
                     p->nr_ipdopd_par, 2);
        }
    }
    abr_bit(b);
    return (int)(b->pos - start);
}

int ps_read(struct ps *p, struct abr *host, int bits)
{
    static const int8_t NUM_ENV[2][4] = {{0, 1, 2, 4}, {1, 2, 3, 4}};
    static const int8_t NR_IIDICC[] = {10, 20, 34, 10, 20, 34};
    static const int8_t NR_IPDOPD[] = {5, 11, 17, 5, 11, 17};
    struct abr g = *host, *b = &g;
    size_t start = b->pos;
    int header = abr_bit(b);
    if (header) {
        p->enable_iid = abr_bit(b);
        if (p->enable_iid) {
            int mode = (int)abr_bits(b, 3);
            if (mode > 5)
                goto err;
            p->nr_iid_par = NR_IIDICC[mode];
            p->iid_quant = mode > 2;
            p->nr_ipdopd_par = NR_IPDOPD[mode];
        }
        p->enable_icc = abr_bit(b);
        if (p->enable_icc) {
            p->icc_mode = (int)abr_bits(b, 3);
            if (p->icc_mode > 5)
                goto err;
            p->nr_icc_par = NR_IIDICC[p->icc_mode];
        }
        p->enable_ext = abr_bit(b);
    }
    p->frame_class = abr_bit(b);
    p->num_env_old = p->num_env;
    p->num_env = NUM_ENV[p->frame_class][abr_bits(b, 2)];
    p->border[0] = -1;
    if (p->frame_class) {
        for (int e = 1; e <= p->num_env; e++) {
            p->border[e] = (int)abr_bits(b, 5);
            if (p->border[e] < p->border[e - 1])
                goto err;
        }
    } else {
        static const int LOG2[5] = {0, 0, 1, 1, 2};
        for (int e = 1; e <= p->num_env; e++)
            p->border[e] = (e * SLOTS >> LOG2[p->num_env]) - 1;
    }
    if (p->enable_iid) {
        for (int e = 0; e < p->num_env; e++) {
            int dt = abr_bit(b);
            static const int HUFF_IID[4] = {IID_DF0, IID_DF1, IID_DT0, IID_DT1};
            if (read_par(p, b, p->iid, HUFF_IID[2 * dt + p->iid_quant], e, dt,
                         p->nr_iid_par, 0) != 0)
                goto err;
        }
    } else {
        memset(p->iid, 0, sizeof p->iid);
    }
    if (p->enable_icc) {
        for (int e = 0; e < p->num_env; e++) {
            int dt = abr_bit(b);
            if (read_par(p, b, p->icc, dt ? ICC_DT : ICC_DF, e, dt,
                         p->nr_icc_par, 1) != 0)
                goto err;
        }
    } else {
        memset(p->icc, 0, sizeof p->icc);
    }
    if (p->enable_ext) {
        int cnt = (int)abr_bits(b, 4);
        if (cnt == 15)
            cnt += (int)abr_bits(b, 8);
        cnt *= 8;
        while (cnt > 7) {
            int id = (int)abr_bits(b, 2);
            cnt -= 2 + read_ext(p, b, id);
        }
        if (cnt < 0)
            goto err;
        b->pos += (size_t)cnt;
    }
    /* an envelope that reaches the frame's end */
    if (!p->num_env || p->border[p->num_env] < SLOTS - 1) {
        int source = p->num_env ? p->num_env - 1 : p->num_env_old - 1;
        if (source >= 0 && source != p->num_env) {
            if (p->enable_iid)
                memcpy(p->iid[p->num_env], p->iid[source], sizeof p->iid[0]);
            if (p->enable_icc)
                memcpy(p->icc[p->num_env], p->icc[source], sizeof p->icc[0]);
            if (p->enable_ipdopd) {
                memcpy(p->ipd[p->num_env], p->ipd[source], sizeof p->ipd[0]);
                memcpy(p->opd[p->num_env], p->opd[source], sizeof p->opd[0]);
            }
        }
        if (p->enable_iid)
            for (int k = 0; k < p->nr_iid_par; k++)
                if (abs(p->iid[p->num_env][k]) > 7 + 8 * p->iid_quant)
                    goto err;
        if (p->enable_icc)
            for (int k = 0; k < p->nr_iid_par; k++)
                if ((unsigned)p->icc[p->num_env][k] > 7)
                    goto err;
        p->num_env++;
        p->border[p->num_env] = SLOTS - 1;
    }
    p->is34_old = p->is34;
    if (p->enable_iid || p->enable_icc)
        p->is34 = (p->enable_iid && p->nr_iid_par == 34) ||
                  (p->enable_icc && p->nr_icc_par == 34);
    if (!p->enable_ipdopd) {
        memset(p->ipd, 0, sizeof p->ipd);
        memset(p->opd, 0, sizeof p->opd);
    }
    if (header)
        p->start = 1;
    {
        int used = (int)(b->pos - start);
        if (used <= bits) {
            host->pos += (size_t)used;
            return used;
        }
    }
err:
    p->start = 0;
    host->pos += (size_t)bits;
    memset(p->iid, 0, sizeof p->iid);
    memset(p->icc, 0, sizeof p->icc);
    memset(p->ipd, 0, sizeof p->ipd);
    memset(p->opd, 0, sizeof p->opd);
    return bits;
}

/* ---- the processing ---- */

static void hybrid_filter(float (*out)[2], float (*in)[2],
                          const float (*filter)[8][2], ptrdiff_t stride, int n)
{
    float inre0[6], inre1[6], inim0[6], inim1[6];
    for (int j = 0; j < 6; j++) {
        inre0[j] = in[j][0] + in[12 - j][0];
        inre1[j] = in[j][1] - in[12 - j][1];
        inim0[j] = in[j][1] + in[12 - j][1];
        inim1[j] = in[j][0] - in[12 - j][0];
    }
    for (int i = 0; i < n; i++) {
        float re = filter[i][6][0] * in[6][0];
        float im = filter[i][6][0] * in[6][1];
        for (int j = 0; j < 6; j++) {
            re += filter[i][j][0] * inre0[j] - filter[i][j][1] * inre1[j];
            im += filter[i][j][0] * inim0[j] + filter[i][j][1] * inim1[j];
        }
        out[i * stride][0] = re;
        out[i * stride][1] = im;
    }
}

static void hybrid2_re(float (*in)[2], float (*out)[32][2], int len,
                       int reverse)
{
    static const float G1_Q2[] = {
        0.0f, 0.01899487526049f, 0.0f, -0.07293139167538f,
        0.0f, 0.30596630545168f, 0.5f};
    for (int i = 0; i < len; i++, in++) {
        float re_in = G1_Q2[6] * in[6][0], re_op = 0;
        float im_in = G1_Q2[6] * in[6][1], im_op = 0;
        for (int j = 0; j < 6; j += 2) {
            re_op += G1_Q2[j + 1] * (in[j + 1][0] + in[12 - j - 1][0]);
            im_op += G1_Q2[j + 1] * (in[j + 1][1] + in[12 - j - 1][1]);
        }
        out[reverse][i][0] = re_in + re_op;
        out[reverse][i][1] = im_in + im_op;
        out[!reverse][i][0] = re_in - re_op;
        out[!reverse][i][1] = im_in - im_op;
    }
}

static void hybrid_analysis(struct ps *p, float (*out)[32][2],
                            float (*L)[38][64], int is34)
{
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 38; j++) {
            p->in_buf[i][j + 6][0] = L[0][j][i];
            p->in_buf[i][j + 6][1] = L[1][j][i];
        }
    const int len = 32;
    if (is34) {
        for (int i = 0; i < len; i++) {
            hybrid_filter(out[0] + i, p->in_buf[0] + i,
                          (const float(*)[8][2])p->t.f34_0_12, 32, 12);
            hybrid_filter(out[12] + i, p->in_buf[1] + i,
                          (const float(*)[8][2])p->t.f34_1_8, 32, 8);
            hybrid_filter(out[20] + i, p->in_buf[2] + i,
                          (const float(*)[8][2])p->t.f34_2_4, 32, 4);
            hybrid_filter(out[24] + i, p->in_buf[3] + i,
                          (const float(*)[8][2])p->t.f34_2_4, 32, 4);
            hybrid_filter(out[28] + i, p->in_buf[4] + i,
                          (const float(*)[8][2])p->t.f34_2_4, 32, 4);
        }
        for (int i = 5; i < 64; i++)
            for (int j = 0; j < len; j++) {
                out[27 + i][j][0] = L[0][j][i];
                out[27 + i][j][1] = L[1][j][i];
            }
    } else {
        for (int i = 0; i < len; i++) {
            float temp[8][2];
            hybrid_filter(temp, p->in_buf[0] + i,
                          (const float(*)[8][2])p->t.f20_0_8, 1, 8);
            out[0][i][0] = temp[6][0];
            out[0][i][1] = temp[6][1];
            out[1][i][0] = temp[7][0];
            out[1][i][1] = temp[7][1];
            out[2][i][0] = temp[0][0];
            out[2][i][1] = temp[0][1];
            out[3][i][0] = temp[1][0];
            out[3][i][1] = temp[1][1];
            out[4][i][0] = temp[2][0] + temp[5][0];
            out[4][i][1] = temp[2][1] + temp[5][1];
            out[5][i][0] = temp[3][0] + temp[4][0];
            out[5][i][1] = temp[3][1] + temp[4][1];
        }
        hybrid2_re(p->in_buf[1], out + 6, len, 1);
        hybrid2_re(p->in_buf[2], out + 8, len, 0);
        for (int i = 3; i < 64; i++)
            for (int j = 0; j < len; j++) {
                out[7 + i][j][0] = L[0][j][i];
                out[7 + i][j][1] = L[1][j][i];
            }
    }
    for (int i = 0; i < 5; i++)
        memcpy(p->in_buf[i], p->in_buf[i] + 32, sizeof(p->in_buf[i][0]) * 6);
}

static void hybrid_synthesis(float (*out)[38][64], float (*in)[32][2], int is34)
{
    const int len = 32;
    if (is34) {
        for (int n = 0; n < len; n++) {
            for (int c = 0; c < 2; c++)
                for (int i = 0; i < 5; i++)
                    out[c][n][i] = 0;
            for (int i = 0; i < 12; i++) {
                out[0][n][0] += in[i][n][0];
                out[1][n][0] += in[i][n][1];
            }
            for (int i = 0; i < 8; i++) {
                out[0][n][1] += in[12 + i][n][0];
                out[1][n][1] += in[12 + i][n][1];
            }
            for (int i = 0; i < 4; i++) {
                out[0][n][2] += in[20 + i][n][0];
                out[1][n][2] += in[20 + i][n][1];
                out[0][n][3] += in[24 + i][n][0];
                out[1][n][3] += in[24 + i][n][1];
                out[0][n][4] += in[28 + i][n][0];
                out[1][n][4] += in[28 + i][n][1];
            }
        }
        for (int i = 5; i < 64; i++)
            for (int n = 0; n < len; n++) {
                out[0][n][i] = in[27 + i][n][0];
                out[1][n][i] = in[27 + i][n][1];
            }
    } else {
        for (int n = 0; n < len; n++) {
            out[0][n][0] = in[0][n][0] + in[1][n][0] + in[2][n][0] +
                           in[3][n][0] + in[4][n][0] + in[5][n][0];
            out[1][n][0] = in[0][n][1] + in[1][n][1] + in[2][n][1] +
                           in[3][n][1] + in[4][n][1] + in[5][n][1];
            out[0][n][1] = in[6][n][0] + in[7][n][0];
            out[1][n][1] = in[6][n][1] + in[7][n][1];
            out[0][n][2] = in[8][n][0] + in[9][n][0];
            out[1][n][2] = in[8][n][1] + in[9][n][1];
        }
        for (int i = 3; i < 64; i++)
            for (int n = 0; n < len; n++) {
                out[0][n][i] = in[7 + i][n][0];
                out[1][n][i] = in[7 + i][n][1];
            }
    }
}

static void decorrelate(float (*out)[2], float (*delay)[2],
                        float (*ap_delay)[SLOTS + MAX_AP_DELAY][2],
                        const float phi_fract[2], const float (*q_fract)[2],
                        const float *transient_gain, float g_decay_slope,
                        int len)
{
    static const float A[] = {0.65143905753106f, 0.56471812200776f,
                              0.48954165955695f};
    float ag[AP_LINKS];
    for (int m = 0; m < AP_LINKS; m++)
        ag[m] = A[m] * g_decay_slope;
    for (int n = 0; n < len; n++) {
        float in_re = delay[n][0] * phi_fract[0] - delay[n][1] * phi_fract[1];
        float in_im = delay[n][0] * phi_fract[1] + delay[n][1] * phi_fract[0];
        for (int m = 0; m < AP_LINKS; m++) {
            float a_re = ag[m] * in_re, a_im = ag[m] * in_im;
            float ld_re = ap_delay[m][n + 2 - m][0];
            float ld_im = ap_delay[m][n + 2 - m][1];
            float fd_re = q_fract[m][0], fd_im = q_fract[m][1];
            float apd_re = in_re, apd_im = in_im;
            in_re = ld_re * fd_re - ld_im * fd_im;
            in_re -= a_re;
            in_im = ld_re * fd_im + ld_im * fd_re;
            in_im -= a_im;
            ap_delay[m][n + 5][0] = apd_re + ag[m] * in_re;
            ap_delay[m][n + 5][1] = apd_im + ag[m] * in_im;
        }
        out[n][0] = transient_gain[n] * in_re;
        out[n][1] = transient_gain[n] * in_im;
    }
}

static void decorrelation(struct ps *p, float (*out)[32][2],
                          const float (*s)[32][2], int is34)
{
    float power[34][SLOTS], transient_gain[34][SLOTS];
    const int8_t *k_to_i = is34 ? PS_K_TO_I_34 : PS_K_TO_I_20;
    const float peak_decay_factor = 0.76592833836465f;
    const float transient_impact = 1.5f, a_smooth = 0.25f;
    memset(power, 0, sizeof power);
    if (is34 != p->is34_old) {
        memset(p->peak_decay_nrg, 0, sizeof p->peak_decay_nrg);
        memset(p->power_smooth, 0, sizeof p->power_smooth);
        memset(p->peak_decay_diff_smooth, 0, sizeof p->peak_decay_diff_smooth);
        memset(p->delay, 0, sizeof p->delay);
        memset(p->ap_delay, 0, sizeof p->ap_delay);
    }
    for (int k = 0; k < NR_BANDS[is34]; k++) {
        int i = k_to_i[k];
        for (int n = 0; n < SLOTS; n++)
            power[i][n] += s[k][n][0] * s[k][n][0] + s[k][n][1] * s[k][n][1];
    }
    for (int i = 0; i < NR_PAR_BANDS[is34]; i++) {
        for (int n = 0; n < SLOTS; n++) {
            float decayed = peak_decay_factor * p->peak_decay_nrg[i];
            p->peak_decay_nrg[i] =
                decayed > power[i][n] ? decayed : power[i][n];
            p->power_smooth[i] += a_smooth * (power[i][n] - p->power_smooth[i]);
            p->peak_decay_diff_smooth[i] +=
                a_smooth * (p->peak_decay_nrg[i] - power[i][n] -
                            p->peak_decay_diff_smooth[i]);
            float denom = transient_impact * p->peak_decay_diff_smooth[i];
            transient_gain[i][n] =
                denom > p->power_smooth[i] ? p->power_smooth[i] / denom : 1.0f;
        }
    }
    int k;
    for (k = 0; k < NR_ALLPASS_BANDS[is34]; k++) {
        int b = k_to_i[k];
        float slope = 1.f - 0.05f * (float)(k - DECAY_CUTOFF[is34]);
        slope = slope < 0 ? 0 : slope > 1 ? 1 : slope;
        memcpy(p->delay[k], p->delay[k] + SLOTS,
               sizeof(p->delay[k][0]) * MAX_DELAY);
        memcpy(p->delay[k] + MAX_DELAY, s[k], sizeof(p->delay[k][0]) * SLOTS);
        for (int m = 0; m < AP_LINKS; m++)
            memcpy(p->ap_delay[k][m], p->ap_delay[k][m] + SLOTS,
                   sizeof(p->ap_delay[k][m][0]) * 5);
        decorrelate(out[k], p->delay[k] + MAX_DELAY - 2, p->ap_delay[k],
                    p->t.phi_fract[is34][k],
                    (const float(*)[2])p->t.q_fract[is34][k], transient_gain[b],
                    slope, SLOTS);
    }
    for (; k < NR_BANDS[is34]; k++) {
        int i = k_to_i[k];
        int d = k < SHORT_DELAY_BAND[is34] ? 14 : 1;
        memcpy(p->delay[k], p->delay[k] + SLOTS,
               sizeof(p->delay[k][0]) * MAX_DELAY);
        memcpy(p->delay[k] + MAX_DELAY, s[k], sizeof(p->delay[k][0]) * SLOTS);
        const float(*src)[2] = (const float(*)[2])p->delay[k] + MAX_DELAY - d;
        for (int n = 0; n < SLOTS; n++) {
            out[k][n][0] = src[n][0] * transient_gain[i][n];
            out[k][n][1] = src[n][1] * transient_gain[i][n];
        }
    }
}

static void map_idx_10_to_20(int8_t *pm, const int8_t *par, int full)
{
    int b;
    if (full) {
        b = 9;
    } else {
        b = 4;
        pm[10] = 0;
    }
    for (; b >= 0; b--)
        pm[2 * b + 1] = pm[2 * b] = par[b];
}

static void map_idx_34_to_20(int8_t *pm, const int8_t *par, int full)
{
    pm[0] = (int8_t)((2 * par[0] + par[1]) / 3);
    pm[1] = (int8_t)((par[1] + 2 * par[2]) / 3);
    pm[2] = (int8_t)((2 * par[3] + par[4]) / 3);
    pm[3] = (int8_t)((par[4] + 2 * par[5]) / 3);
    pm[4] = (int8_t)((par[6] + par[7]) / 2);
    pm[5] = (int8_t)((par[8] + par[9]) / 2);
    pm[6] = par[10];
    pm[7] = par[11];
    pm[8] = (int8_t)((par[12] + par[13]) / 2);
    pm[9] = (int8_t)((par[14] + par[15]) / 2);
    pm[10] = par[16];
    if (full) {
        pm[11] = par[17];
        pm[12] = par[18];
        pm[13] = par[19];
        pm[14] = (int8_t)((par[20] + par[21]) / 2);
        pm[15] = (int8_t)((par[22] + par[23]) / 2);
        pm[16] = (int8_t)((par[24] + par[25]) / 2);
        pm[17] = (int8_t)((par[26] + par[27]) / 2);
        pm[18] = (int8_t)((par[28] + par[29] + par[30] + par[31]) / 4);
        pm[19] = (int8_t)((par[32] + par[33]) / 2);
    }
}

static void map_val_34_to_20(float *par)
{
    par[0] = (2 * par[0] + par[1]) * 0.33333333f;
    par[1] = (par[1] + 2 * par[2]) * 0.33333333f;
    par[2] = (2 * par[3] + par[4]) * 0.33333333f;
    par[3] = (par[4] + 2 * par[5]) * 0.33333333f;
    par[4] = (par[6] + par[7]) * 0.5f;
    par[5] = (par[8] + par[9]) * 0.5f;
    par[6] = par[10];
    par[7] = par[11];
    par[8] = (par[12] + par[13]) * 0.5f;
    par[9] = (par[14] + par[15]) * 0.5f;
    par[10] = par[16];
    par[11] = par[17];
    par[12] = par[18];
    par[13] = par[19];
    par[14] = (par[20] + par[21]) * 0.5f;
    par[15] = (par[22] + par[23]) * 0.5f;
    par[16] = (par[24] + par[25]) * 0.5f;
    par[17] = (par[26] + par[27]) * 0.5f;
    par[18] = (par[28] + par[29] + par[30] + par[31]) * 0.25f;
    par[19] = (par[32] + par[33]) * 0.5f;
}

static void map_idx_10_to_34(int8_t *pm, const int8_t *par, int full)
{
    static const int8_t FULL[34] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3,
                                    4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7,
                                    8, 8, 8, 8, 9, 9, 9, 9, 9, 9};
    int n = full ? 34 : 16;
    for (int i = 0; i < n; i++)
        pm[i] = par[FULL[i]];
    if (!full)
        pm[16] = 0;
}

static void map_idx_20_to_34(int8_t *pm, const int8_t *par, int full)
{
    if (full) {
        pm[33] = par[19];
        pm[32] = par[19];
        pm[31] = par[18];
        pm[30] = par[18];
        pm[29] = par[18];
        pm[28] = par[18];
        pm[27] = par[17];
        pm[26] = par[17];
        pm[25] = par[16];
        pm[24] = par[16];
        pm[23] = par[15];
        pm[22] = par[15];
        pm[21] = par[14];
        pm[20] = par[14];
        pm[19] = par[13];
        pm[18] = par[12];
        pm[17] = par[11];
    }
    pm[16] = par[10];
    pm[15] = par[9];
    pm[14] = par[9];
    pm[13] = par[8];
    pm[12] = par[8];
    pm[11] = par[7];
    pm[10] = par[6];
    pm[9] = par[5];
    pm[8] = par[5];
    pm[7] = par[4];
    pm[6] = par[4];
    pm[5] = par[3];
    pm[4] = (int8_t)((par[2] + par[3]) / 2);
    pm[3] = par[2];
    pm[2] = par[1];
    pm[1] = (int8_t)((par[0] + par[1]) / 2);
    pm[0] = par[0];
}

static void map_val_20_to_34(float *par)
{
    par[33] = par[19];
    par[32] = par[19];
    par[31] = par[18];
    par[30] = par[18];
    par[29] = par[18];
    par[28] = par[18];
    par[27] = par[17];
    par[26] = par[17];
    par[25] = par[16];
    par[24] = par[16];
    par[23] = par[15];
    par[22] = par[15];
    par[21] = par[14];
    par[20] = par[14];
    par[19] = par[13];
    par[18] = par[12];
    par[17] = par[11];
    par[16] = par[10];
    par[15] = par[9];
    par[14] = par[9];
    par[13] = par[8];
    par[12] = par[8];
    par[11] = par[7];
    par[10] = par[6];
    par[9] = par[5];
    par[8] = par[5];
    par[7] = par[4];
    par[6] = par[4];
    par[5] = par[3];
    par[4] = (par[2] + par[3]) * 0.5f;
    par[3] = par[2];
    par[2] = par[1];
    par[1] = (par[0] + par[1]) * 0.5f;
}

/* the parameters in the bands the processing runs in: par itself, or
   mapped into buf */
static int8_t (*remap(int8_t (*buf)[MAX_PAR], int8_t (*par)[MAX_PAR],
                      int num_par, int num_env, int full, int is34))[MAX_PAR]
{
    if (is34) {
        if (num_par == 20 || num_par == 11) {
            for (int e = 0; e < num_env; e++)
                map_idx_20_to_34(buf[e], par[e], full);
            return buf;
        }
        if (num_par == 10 || num_par == 5) {
            for (int e = 0; e < num_env; e++)
                map_idx_10_to_34(buf[e], par[e], full);
            return buf;
        }
    } else {
        if (num_par == 34 || num_par == 17) {
            for (int e = 0; e < num_env; e++)
                map_idx_34_to_20(buf[e], par[e], full);
            return buf;
        }
        if (num_par == 10 || num_par == 5) {
            for (int e = 0; e < num_env; e++)
                map_idx_10_to_20(buf[e], par[e], full);
            return buf;
        }
    }
    return par;
}

static void stereo_processing(struct ps *p, float (*l)[32][2],
                              float (*r)[32][2], int is34)
{
    int8_t iid_buf[MAX_ENV][MAX_PAR], icc_buf[MAX_ENV][MAX_PAR];
    int8_t ipd_buf[MAX_ENV][MAX_PAR], opd_buf[MAX_ENV][MAX_PAR];
    int8_t(*iid_m)[MAX_PAR] = iid_buf, (*icc_m)[MAX_PAR] = icc_buf;
    int8_t(*ipd_m)[MAX_PAR] = ipd_buf, (*opd_m)[MAX_PAR] = opd_buf;
    const int8_t *k_to_i = is34 ? PS_K_TO_I_34 : PS_K_TO_I_20;
    const float(*H_LUT)[8][4] = p->icc_mode < 3 ? (const float(*)[8][4])p->t.HA
                                                : (const float(*)[8][4])p->t.HB;
    float(*H[4])[MAX_ENV + 1][MAX_PAR] = {p->H11, p->H12, p->H21, p->H22};
    if (p->num_env_old)
        for (int j = 0; j < 4; j++)
            for (int c = 0; c < 2; c++)
                memcpy(H[j][c][0], H[j][c][p->num_env_old], sizeof(H[j][c][0]));
    iid_m = remap(iid_buf, p->iid, p->nr_iid_par, p->num_env, 1, is34);
    icc_m = remap(icc_buf, p->icc, p->nr_icc_par, p->num_env, 1, is34);
    if (p->enable_ipdopd) {
        ipd_m = remap(ipd_buf, p->ipd, p->nr_ipdopd_par, p->num_env, 0, is34);
        opd_m = remap(opd_buf, p->opd, p->nr_ipdopd_par, p->num_env, 0, is34);
    }
    if (is34 && !p->is34_old) {
        for (int j = 0; j < 4; j++)
            for (int c = 0; c < 2; c++)
                map_val_20_to_34(H[j][c][0]);
        memset(p->ipd_hist, 0, sizeof p->ipd_hist);
        memset(p->opd_hist, 0, sizeof p->opd_hist);
    } else if (!is34 && p->is34_old) {
        for (int j = 0; j < 4; j++)
            for (int c = 0; c < 2; c++)
                map_val_34_to_20(H[j][c][0]);
        memset(p->ipd_hist, 0, sizeof p->ipd_hist);
        memset(p->opd_hist, 0, sizeof p->opd_hist);
    }
    for (int e = 0; e < p->num_env; e++) {
        for (int b = 0; b < NR_PAR_BANDS[is34]; b++) {
            const float *h =
                H_LUT[iid_m[e][b] + 7 + 23 * p->iid_quant][icc_m[e][b]];
            float h11 = h[0], h12 = h[1], h21 = h[2], h22 = h[3];
            if (p->enable_ipdopd && b < NR_IPDOPD_BANDS[is34]) {
                int opd_idx = p->opd_hist[b] * 8 + opd_m[e][b];
                int ipd_idx = p->ipd_hist[b] * 8 + ipd_m[e][b];
                float opd_re = p->t.pd_re[opd_idx],
                      opd_im = p->t.pd_im[opd_idx];
                float ipd_re = p->t.pd_re[ipd_idx],
                      ipd_im = p->t.pd_im[ipd_idx];
                p->opd_hist[b] = (int8_t)(opd_idx & 0x3F);
                p->ipd_hist[b] = (int8_t)(ipd_idx & 0x3F);
                float adj_re = opd_re * ipd_re + opd_im * ipd_im;
                float adj_im = opd_im * ipd_re - opd_re * ipd_im;
                float h11i = h11 * opd_im;
                h11 = h11 * opd_re;
                float h12i = h12 * adj_im;
                h12 = h12 * adj_re;
                float h21i = h21 * opd_im;
                h21 = h21 * opd_re;
                float h22i = h22 * adj_im;
                h22 = h22 * adj_re;
                p->H11[1][e + 1][b] = h11i;
                p->H12[1][e + 1][b] = h12i;
                p->H21[1][e + 1][b] = h21i;
                p->H22[1][e + 1][b] = h22i;
            }
            p->H11[0][e + 1][b] = h11;
            p->H12[0][e + 1][b] = h12;
            p->H21[0][e + 1][b] = h21;
            p->H22[0][e + 1][b] = h22;
        }
        for (int k = 0; k < NR_BANDS[is34]; k++) {
            float h[2][4], hs[2][4];
            int start = p->border[e], stop = p->border[e + 1];
            float width = 1.0f / (float)(stop - start ? stop - start : 1);
            int b = k_to_i[k];
            for (int j = 0; j < 4; j++)
                h[0][j] = H[j][0][e][b];
            if (p->enable_ipdopd) {
                float sign = (is34 && k <= 13 && k >= 9) || (!is34 && k <= 1)
                                 ? -1.0f
                                 : 1.0f;
                for (int j = 0; j < 4; j++)
                    h[1][j] = sign * H[j][1][e][b];
            }
            for (int j = 0; j < 4; j++)
                hs[0][j] = (H[j][0][e + 1][b] - h[0][j]) * width;
            if (p->enable_ipdopd)
                for (int j = 0; j < 4; j++)
                    hs[1][j] = (H[j][1][e + 1][b] - h[1][j]) * width;
            if (stop - start <= 0)
                continue;
            float(*lp)[2] = l[k] + 1 + start, (*rp)[2] = r[k] + 1 + start;
            int len = stop - start;
            if (!p->enable_ipdopd) {
                float h0 = h[0][0], h1 = h[0][1], h2 = h[0][2], h3 = h[0][3];
                for (int n = 0; n < len; n++) {
                    float l_re = lp[n][0], l_im = lp[n][1];
                    float r_re = rp[n][0], r_im = rp[n][1];
                    h0 += hs[0][0];
                    h1 += hs[0][1];
                    h2 += hs[0][2];
                    h3 += hs[0][3];
                    lp[n][0] = h0 * l_re + h2 * r_re;
                    lp[n][1] = h0 * l_im + h2 * r_im;
                    rp[n][0] = h1 * l_re + h3 * r_re;
                    rp[n][1] = h1 * l_im + h3 * r_im;
                }
            } else {
                float h00 = h[0][0], h01 = h[0][1], h02 = h[0][2],
                      h03 = h[0][3];
                float h10 = h[1][0], h11 = h[1][1], h12 = h[1][2],
                      h13 = h[1][3];
                for (int n = 0; n < len; n++) {
                    float l_re = lp[n][0], l_im = lp[n][1];
                    float r_re = rp[n][0], r_im = rp[n][1];
                    h00 += hs[0][0];
                    h01 += hs[0][1];
                    h02 += hs[0][2];
                    h03 += hs[0][3];
                    h10 += hs[1][0];
                    h11 += hs[1][1];
                    h12 += hs[1][2];
                    h13 += hs[1][3];
                    lp[n][0] =
                        h00 * l_re + h02 * r_re - h10 * l_im - h12 * r_im;
                    lp[n][1] =
                        h00 * l_im + h02 * r_im + h10 * l_re + h12 * r_re;
                    rp[n][0] =
                        h01 * l_re + h03 * r_re - h11 * l_im - h13 * r_im;
                    rp[n][1] =
                        h01 * l_im + h03 * r_im + h11 * l_re + h13 * r_re;
                }
            }
        }
    }
}

void ps_apply(struct ps *p, float (*L)[38][64], float (*R)[38][64], int top)
{
    int is34 = p->is34;
    if (!p->start) {
        memcpy(R, L, sizeof(float) * 2 * 38 * 64);
        return;
    }
    top += NR_BANDS[is34] - 64;
    if (top < 0)
        top = 0;
    if (top < NR_BANDS[is34])
        memset(p->delay + top, 0,
               sizeof(p->delay[0]) * (size_t)(NR_BANDS[is34] - top));
    if (top < NR_ALLPASS_BANDS[is34])
        memset(p->ap_delay + top, 0,
               sizeof(p->ap_delay[0]) * (size_t)(NR_ALLPASS_BANDS[is34] - top));
    hybrid_analysis(p, p->Lbuf, L, is34);
    decorrelation(p, p->Rbuf, (const float(*)[32][2])p->Lbuf, is34);
    stereo_processing(p, p->Lbuf, p->Rbuf, is34);
    hybrid_synthesis(L, p->Lbuf, is34);
    hybrid_synthesis(R, p->Rbuf, is34);
}
