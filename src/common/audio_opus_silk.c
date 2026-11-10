/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_opus_silk.c - the SILK layer of Opus (RFC 6716 section 4.2;
 * audio_opus.h), in the fixed-point arithmetic the reference decoder
 * defines it by, to the bit: the voice activity and LBRR flags, the
 * stereo prediction weights, each frame's indices (signal type, gains,
 * the NLSF codebook path, interpolation, pitch lag and contour, the LTP
 * filters and their scaling, the seed), the excitation (pulse counts per
 * block, the shell-coded split of the pulses, their least significant bits
 * and signs), the parameters (gains dequantized, NLSFs from the two-stage
 * codebook, stabilized, to LPC coefficients checked for stability, pitch
 * lags), the synthesis (excitation with its offsets and random signs, the
 * long-term and the short-term predictors), the mid/side to left/right
 * mixing with its interpolated predictors, and the resampler to 48 kHz
 * (the 2x all-pass upsampler and the fractional FIR) that libopus uses;
 * the concealment of a lost frame and the comfort noise, which a change
 * of mode without redundancy fades from.
 */
#include "audio_opus.h"

#include <stdlib.h>
#include <string.h>

#include "audio_opus_tab.h"

#define MAX_ORDER 16
#define MAX_FRAME 320
#define MAX_SUBFR 80
#define LTP_ORDER 5
#define TYPE_NO_VOICE 0
#define TYPE_UNVOICED 1
#define TYPE_VOICED 2
#define CODE_INDEPENDENTLY 0
#define CODE_INDEPENDENTLY_NO_LTP_SCALING 1
#define CODE_CONDITIONALLY 2
#define N_LEVELS_QGAIN 64
#define MIN_DELTA_GAIN 4 /* -MIN_DELTA_GAIN_QUANT */
#define MAX_DELTA_GAIN 36
#define OFFSET_Q7 (2 * 128 / 6 + 16 * 128)
#define INV_SCALE_Q16 ((65536 * ((86 * 128) / 6)) / (N_LEVELS_QGAIN - 1))

/* ---- fixed-point arithmetic, as SigProc_FIX.h defines it ---- */

/* a left shift of a value that may be negative, as two's complement
   does it */
#define LSH(v, n) ((int32_t)((uint32_t)(int32_t)(v) << (n)))

static inline int32_t smulwb(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * (int16_t)b) >> 16);
}

static inline int32_t smlawb(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((int64_t)a + (((int64_t)b * (int16_t)c) >> 16));
}

static inline int32_t smulbb(int32_t a, int32_t b)
{
    return (int32_t)(int16_t)a * (int16_t)b;
}

static inline int32_t smlabb(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((uint32_t)a +
                     (uint32_t)((int32_t)(int16_t)b * (int16_t)c));
}

static inline int32_t smulww(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 16);
}

static inline int32_t smlaww(int32_t a, int32_t b, int32_t c)
{
    return (int32_t)((int64_t)a + (((int64_t)b * c) >> 16));
}

static inline int32_t smmul(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 32);
}

static inline int32_t smultt(int32_t a, int32_t b)
{
    return (a >> 16) * (b >> 16);
}

static inline int32_t rshift_round(int32_t a, int s)
{
    return s == 1 ? (a >> 1) + (a & 1) : ((a >> (s - 1)) + 1) >> 1;
}

static inline int64_t rshift_round64(int64_t a, int s)
{
    return s == 1 ? (a >> 1) + (a & 1) : ((a >> (s - 1)) + 1) >> 1;
}

static inline int32_t sat16(int32_t a)
{
    return a > 32767 ? 32767 : a < -32768 ? -32768 : a;
}

static inline int32_t sat32(int64_t a)
{
    return a > INT32_MAX ? INT32_MAX : a < INT32_MIN ? INT32_MIN : (int32_t)a;
}

static inline int32_t limit(int32_t a, int32_t l1, int32_t l2)
{
    if (l1 > l2)
        return a > l1 ? l1 : a < l2 ? l2 : a;
    return a > l2 ? l2 : a < l1 ? l1 : a;
}

static inline int32_t lshift_sat32(int32_t a, int s)
{
    return (int32_t)((uint32_t)limit(a, INT32_MIN >> s, INT32_MAX >> s) << s);
}

static inline int32_t add_sat32(int32_t a, int32_t b)
{
    return sat32((int64_t)a + b);
}

static inline int32_t sub_sat32(int32_t a, int32_t b)
{
    return sat32((int64_t)a - b);
}

static inline int32_t clz32(int32_t v)
{
    return v ? __builtin_clz((uint32_t)v) : 32;
}

static inline int32_t ror32(int32_t a, int rot)
{
    uint32_t x = (uint32_t)a, r = (uint32_t)rot, m = (uint32_t)-rot;
    if (rot == 0)
        return a;
    if (rot < 0)
        return (int32_t)((x << m) | (x >> (32 - m)));
    return (int32_t)((x << (32 - r)) | (x >> r));
}

static inline int32_t silk_rand(int32_t seed)
{
    return (int32_t)(907633515u + (uint32_t)seed * 196314165u);
}

static void clz_frac(int32_t in, int32_t *lz, int32_t *frac_q7)
{
    int32_t l = clz32(in);
    *lz = l;
    *frac_q7 = ror32(in, 24 - l) & 0x7f;
}

static int32_t sqrt_approx(int32_t x)
{
    int32_t lz, frac;
    if (x <= 0)
        return 0;
    clz_frac(x, &lz, &frac);
    int32_t y = lz & 1 ? 32768 : 46214;
    y >>= lz >> 1;
    return smlawb(y, y, smulbb(213, frac));
}

static int32_t div32_varq(int32_t a, int32_t b, int qres)
{
    int ah = clz32(a > 0 ? a : -a) - 1;
    int32_t an = (int32_t)((uint32_t)a << ah);
    int bh = clz32(b > 0 ? b : -b) - 1;
    int32_t bn = (int32_t)((uint32_t)b << bh);
    int32_t binv = (INT32_MAX >> 2) / (int16_t)(bn >> 16);
    int32_t result = smulwb(an, binv);
    an = (int32_t)((uint32_t)an - ((uint32_t)smmul(bn, result) << 3));
    result = smlawb(result, an, binv);
    int lshift = 29 + ah - bh - qres;
    if (lshift < 0)
        return lshift_sat32(result, -lshift);
    return lshift < 32 ? result >> lshift : 0;
}

static int32_t inverse32_varq(int32_t b, int qres)
{
    int bh = clz32(b > 0 ? b : -b) - 1;
    int32_t bn = (int32_t)((uint32_t)b << bh);
    int32_t binv = (INT32_MAX >> 2) / (int16_t)(bn >> 16);
    int32_t result = (int32_t)((uint32_t)binv << 16);
    int32_t err = (int32_t)((uint32_t)((1 << 29) - smulwb(bn, binv)) << 3);
    result = smlaww(result, err, binv);
    int lshift = 61 - bh - qres;
    if (lshift <= 0)
        return lshift_sat32(result, -lshift);
    return lshift < 32 ? result >> lshift : 0;
}

static int32_t log2lin(int32_t in)
{
    if (in < 0)
        return 0;
    if (in >= 3967)
        return INT32_MAX;
    int32_t out = 1 << (in >> 7), frac = in & 0x7F;
    if (in < 2048)
        return out +
               ((out * smlawb(frac, smulbb(frac, 128 - frac), -174)) >> 7);
    return out + (out >> 7) * smlawb(frac, smulbb(frac, 128 - frac), -174);
}

static void sum_sqr_shift(int32_t *energy, int *shift, const int16_t *x,
                          int len)
{
    int shft = 31 - clz32(len), i;
    int32_t nrg = len;
    for (i = 0; i < len - 1; i += 2) {
        uint32_t t = (uint32_t)smulbb(x[i], x[i]);
        t = (uint32_t)smlabb((int32_t)t, x[i + 1], x[i + 1]);
        nrg = (int32_t)((uint32_t)nrg + (t >> shft));
    }
    if (i < len)
        nrg = (int32_t)((uint32_t)nrg + ((uint32_t)smulbb(x[i], x[i]) >> shft));
    shft = shft + 3 - clz32(nrg);
    if (shft < 0)
        shft = 0;
    nrg = 0;
    for (i = 0; i < len - 1; i += 2) {
        uint32_t t = (uint32_t)smulbb(x[i], x[i]);
        t = (uint32_t)smlabb((int32_t)t, x[i + 1], x[i + 1]);
        nrg = (int32_t)((uint32_t)nrg + (t >> shft));
    }
    if (i < len)
        nrg = (int32_t)((uint32_t)nrg + ((uint32_t)smulbb(x[i], x[i]) >> shft));
    *shift = shft;
    *energy = nrg;
}

/* ---- the state ---- */

struct nlsf_cb {
    int vectors, order;
    int32_t step_q16;
    const uint8_t *cb1_q8;
    const int16_t *wght_q9;
    const uint8_t *cb1_icdf, *pred_q8, *ec_sel, *ec_icdf;
    const int16_t *delta_min;
};

static const struct nlsf_cb CB_NB_MB = {32,
                                        10,
                                        11796,
                                        SILK_NLSF_CB1_NB_MB_Q8,
                                        SILK_NLSF_CB1_WGHT_Q9,
                                        SILK_NLSF_CB1_ICDF_NB_MB,
                                        SILK_NLSF_PRED_NB_MB_Q8,
                                        SILK_NLSF_CB2_SELECT_NB_MB,
                                        SILK_NLSF_CB2_ICDF_NB_MB,
                                        SILK_NLSF_DELTA_MIN_NB_MB_Q15};
static const struct nlsf_cb CB_WB = {32,
                                     16,
                                     9830,
                                     SILK_NLSF_CB1_WB_Q8,
                                     SILK_NLSF_CB1_WB_WGHT_Q9,
                                     SILK_NLSF_CB1_ICDF_WB,
                                     SILK_NLSF_PRED_WB_Q8,
                                     SILK_NLSF_CB2_SELECT_WB,
                                     SILK_NLSF_CB2_ICDF_WB,
                                     SILK_NLSF_DELTA_MIN_WB_Q15};

struct resampler {
    int32_t sIIR[6];
    int16_t sFIR[8];
    int16_t delay[48];
    int fs_in, fs_out, batch, delay_n, copy, up2;
    int32_t inv_ratio_q16;
};

struct indices {
    int8_t gains[4], nlsf[MAX_ORDER + 1], ltp[4];
    int16_t lag;
    int8_t contour, signal_type, quant_offset, nlsf_interp, per, ltp_scale,
        seed;
};

struct cng {
    int32_t exc_buf[MAX_FRAME];
    int16_t smth_nlsf[MAX_ORDER];
    int32_t synth[MAX_ORDER], smth_gain, seed;
    int fs_khz;
};

struct plc {
    int32_t pitch_l_q8;
    int16_t ltp_coef[LTP_ORDER], prev_lpc[MAX_ORDER];
    int32_t last_frame_lost, seed, prev_gain[2], conc_energy;
    int16_t rand_scale, prev_ltp_scale;
    int conc_energy_shift, fs_khz, nb_subfr, subfr_length;
};

struct ctrl {
    int pitch_l[4];
    int32_t gains[4];
    int16_t pred[2][MAX_ORDER], ltp[LTP_ORDER * 4];
    int ltp_scale;
};

struct chan {
    int32_t prev_gain, exc[MAX_FRAME], slpc[MAX_ORDER];
    int16_t out[MAX_FRAME + 2 * MAX_SUBFR];
    int lag_prev;
    int8_t last_gain;
    int fs_khz, nb_subfr, frame_length, subfr_length, ltp_mem, order;
    int16_t prev_nlsf[MAX_ORDER];
    int first_after_reset;
    const uint8_t *lag_low_icdf, *contour_icdf;
    int frames_decoded, frames_per_packet, ec_prev_type;
    int16_t ec_prev_lag;
    int vad[3], lbrr, lbrr_flags[3];
    struct resampler rs;
    const struct nlsf_cb *cb;
    struct indices ind;
    struct cng cng;
    int loss_cnt, prev_type;
    struct plc plc;
};

struct silk_dec {
    struct chan ch[2];
    int16_t pred_prev[2], smid[2], sside[2];
    int api_ch, int_ch, prev_mid_only;
    int last_khz, last_ch; /* what the concealment goes on with */
    int api_hz;            /* the output's rate: 48000 unless set */
};

/* ---- the resampler (8, 12, 16 kHz up to 48 kHz) ---- */

static void resampler_init(struct resampler *s, int in_hz, int out_hz)
{
    static const int8_t DELAY[3][5] = {
        {4, 0, 2, 0, 0}, {0, 9, 4, 7, 4}, {0, 3, 12, 7, 7}};
    int rin = in_hz == 8000 ? 0 : in_hz == 12000 ? 1 : 2;
    int rout = out_hz == 8000    ? 0
               : out_hz == 12000 ? 1
               : out_hz == 16000 ? 2
               : out_hz == 24000 ? 3
                                 : 4;
    memset(s, 0, sizeof *s);
    s->delay_n = DELAY[rin][rout];
    s->fs_in = in_hz / 1000;
    s->fs_out = out_hz / 1000;
    s->batch = s->fs_in * 10;
    s->copy = out_hz == in_hz;
    s->up2 = out_hz == 2 * in_hz;
    int up2x = !s->copy && !s->up2; /* the rest: 2x then the FIR */
    s->inv_ratio_q16 = (int32_t)((uint32_t)(((uint32_t)in_hz << (14 + up2x)) /
                                            (uint32_t)out_hz)
                                 << 2);
    while (smulww(s->inv_ratio_q16, out_hz) < (in_hz << up2x))
        s->inv_ratio_q16++;
}

static void up2_hq(int32_t *S, int16_t *out, const int16_t *in, int len)
{
    static const int16_t H0[3] = {1746, 14986, 39083 - 65536};
    static const int16_t H1[3] = {6854, 25769, 55542 - 65536};
    for (int k = 0; k < len; k++) {
        int32_t in32 = LSH(in[k], 10), Y, X, o1, o2;
        Y = in32 - S[0];
        X = smulwb(Y, H0[0]);
        o1 = S[0] + X;
        S[0] = in32 + X;
        Y = o1 - S[1];
        X = smulwb(Y, H0[1]);
        o2 = S[1] + X;
        S[1] = o1 + X;
        Y = o2 - S[2];
        X = smlawb(Y, Y, H0[2]);
        o1 = S[2] + X;
        S[2] = o2 + X;
        out[2 * k] = (int16_t)sat16(rshift_round(o1, 10));
        Y = in32 - S[3];
        X = smulwb(Y, H1[0]);
        o1 = S[3] + X;
        S[3] = in32 + X;
        Y = o1 - S[4];
        X = smulwb(Y, H1[1]);
        o2 = S[4] + X;
        S[4] = o1 + X;
        Y = o2 - S[5];
        X = smlawb(Y, Y, H1[2]);
        o1 = S[5] + X;
        S[5] = o2 + X;
        out[2 * k + 1] = (int16_t)sat16(rshift_round(o1, 10));
    }
}

static int16_t *fir_interpol(int16_t *out, const int16_t *buf, int32_t max_q16,
                             int32_t inc_q16)
{
    for (int32_t ix = 0; ix < max_q16; ix += inc_q16) {
        int32_t t = smulwb(ix & 0xFFFF, 12);
        const int16_t *b = &buf[ix >> 16];
        const int16_t *c0 = SILK_FRAC_FIR_12 + 4 * t;
        const int16_t *c1 = SILK_FRAC_FIR_12 + 4 * (11 - t);
        int32_t r = smulbb(b[0], c0[0]);
        r = smlabb(r, b[1], c0[1]);
        r = smlabb(r, b[2], c0[2]);
        r = smlabb(r, b[3], c0[3]);
        r = smlabb(r, b[4], c1[3]);
        r = smlabb(r, b[5], c1[2]);
        r = smlabb(r, b[6], c1[1]);
        r = smlabb(r, b[7], c1[0]);
        *out++ = (int16_t)sat16(rshift_round(r, 15));
    }
    return out;
}

static void iir_fir(struct resampler *s, int16_t *out, const int16_t *in,
                    int len)
{
    int16_t buf[2 * 160 + 8];
    int n = 0;
    memcpy(buf, s->sFIR, sizeof s->sFIR);
    for (;;) {
        n = len < s->batch ? len : s->batch;
        up2_hq(s->sIIR, &buf[8], in, n);
        out = fir_interpol(out, buf, (int32_t)n << 17, s->inv_ratio_q16);
        in += n;
        len -= n;
        if (len > 0)
            memcpy(buf, &buf[n << 1], 8 * sizeof(int16_t));
        else
            break;
    }
    memcpy(s->sFIR, &buf[n << 1], 8 * sizeof(int16_t));
}

static void resample(struct resampler *s, int16_t *out, const int16_t *in,
                     int len)
{
    int n = s->fs_in - s->delay_n;
    memcpy(&s->delay[s->delay_n], in, sizeof(int16_t) * (size_t)n);
    if (s->copy) {
        memcpy(out, s->delay, sizeof(int16_t) * (size_t)s->fs_in);
        memcpy(&out[s->fs_out], &in[n],
               sizeof(int16_t) * (size_t)(len - s->fs_in));
    } else if (s->up2) {
        up2_hq(s->sIIR, out, s->delay, s->fs_in);
        up2_hq(s->sIIR, &out[s->fs_out], &in[n], len - s->fs_in);
    } else {
        iir_fir(s, out, s->delay, s->fs_in);
        iir_fir(s, &out[s->fs_out], &in[n], len - s->fs_in);
    }
    memcpy(s->delay, &in[len - s->delay_n],
           sizeof(int16_t) * (size_t)s->delay_n);
}

/* ---- channel state ---- */

static void cng_reset(struct chan *c)
{
    int32_t step = 32767 / (c->order + 1), acc = 0;
    for (int i = 0; i < c->order; i++) {
        acc += step;
        c->cng.smth_nlsf[i] = (int16_t)acc;
    }
    c->cng.smth_gain = 0;
    c->cng.seed = 3176576;
}

static void plc_reset(struct chan *c)
{
    c->plc.pitch_l_q8 = (int32_t)((uint32_t)c->frame_length << 7);
    c->plc.prev_gain[0] = c->plc.prev_gain[1] = 65536;
    c->plc.subfr_length = 20;
    c->plc.nb_subfr = 2;
}

static void chan_reset(struct chan *c)
{
    memset(c, 0, sizeof *c);
    c->first_after_reset = 1;
    c->prev_gain = 65536;
    cng_reset(c);
    plc_reset(c);
}

static void set_fs(struct chan *c, int fs_khz, int api_hz)
{
    c->subfr_length = 5 * fs_khz;
    int frame_length = c->nb_subfr * c->subfr_length;
    if (c->fs_khz != fs_khz || c->rs.fs_out * 1000 != api_hz)
        resampler_init(&c->rs, fs_khz * 1000, api_hz);
    if (c->fs_khz != fs_khz || frame_length != c->frame_length) {
        if (fs_khz == 8)
            c->contour_icdf = c->nb_subfr == 4
                                  ? SILK_PITCH_CONTOUR_NB_ICDF
                                  : SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
        else
            c->contour_icdf = c->nb_subfr == 4 ? SILK_PITCH_CONTOUR_ICDF
                                               : SILK_PITCH_CONTOUR_10_MS_ICDF;
        if (c->fs_khz != fs_khz) {
            c->ltp_mem = 20 * fs_khz;
            c->order = fs_khz == 16 ? 16 : 10;
            c->cb = fs_khz == 16 ? &CB_WB : &CB_NB_MB;
            c->lag_low_icdf = fs_khz == 16   ? SILK_UNIFORM8_ICDF
                              : fs_khz == 12 ? SILK_UNIFORM6_ICDF
                                             : SILK_UNIFORM4_ICDF;
            c->first_after_reset = 1;
            c->lag_prev = 100;
            c->last_gain = 10;
            c->prev_type = TYPE_NO_VOICE;
            memset(c->out, 0, sizeof c->out);
            memset(c->slpc, 0, sizeof c->slpc);
        }
        c->fs_khz = fs_khz;
        c->frame_length = frame_length;
    }
}

/* ---- indices ---- */

static void nlsf_unpack(int16_t *ec_ix, uint8_t *pred, const struct nlsf_cb *cb,
                        int cb1)
{
    const uint8_t *sel = &cb->ec_sel[cb1 * cb->order / 2];
    for (int i = 0; i < cb->order; i += 2) {
        uint8_t e = *sel++;
        ec_ix[i] = (int16_t)(((e >> 1) & 7) * 9);
        pred[i] = cb->pred_q8[i + (e & 1) * (cb->order - 1)];
        ec_ix[i + 1] = (int16_t)(((e >> 5) & 7) * 9);
        pred[i + 1] = cb->pred_q8[i + ((e >> 4) & 1) * (cb->order - 1) + 1];
    }
}

static void decode_indices(struct chan *c, struct rc *r, int frame, int lbrr,
                           int cond)
{
    struct indices *ix = &c->ind;
    int k;
    if (lbrr || c->vad[frame])
        k = rc_icdf(r, SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2;
    else
        k = rc_icdf(r, SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    ix->signal_type = (int8_t)(k >> 1);
    ix->quant_offset = (int8_t)(k & 1);
    if (cond == CODE_CONDITIONALLY) {
        ix->gains[0] = (int8_t)rc_icdf(r, SILK_DELTA_GAIN_ICDF, 8);
    } else {
        ix->gains[0] =
            (int8_t)(rc_icdf(r, SILK_GAIN_ICDF + 8 * ix->signal_type, 8) << 3);
        ix->gains[0] += (int8_t)rc_icdf(r, SILK_UNIFORM8_ICDF, 8);
    }
    for (int i = 1; i < c->nb_subfr; i++)
        ix->gains[i] = (int8_t)rc_icdf(r, SILK_DELTA_GAIN_ICDF, 8);
    ix->nlsf[0] = (int8_t)rc_icdf(
        r, &c->cb->cb1_icdf[(ix->signal_type >> 1) * c->cb->vectors], 8);
    int16_t ec_ix[MAX_ORDER];
    uint8_t pred[MAX_ORDER];
    nlsf_unpack(ec_ix, pred, c->cb, ix->nlsf[0]);
    for (int i = 0; i < c->cb->order; i++) {
        k = rc_icdf(r, &c->cb->ec_icdf[ec_ix[i]], 8);
        if (k == 0)
            k -= rc_icdf(r, SILK_NLSF_EXT_ICDF, 8);
        else if (k == 8)
            k += rc_icdf(r, SILK_NLSF_EXT_ICDF, 8);
        ix->nlsf[i + 1] = (int8_t)(k - 4);
    }
    ix->nlsf_interp =
        c->nb_subfr == 4
            ? (int8_t)rc_icdf(r, SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8)
            : 4;
    if (ix->signal_type == TYPE_VOICED) {
        int absolute = 1;
        if (cond == CODE_CONDITIONALLY && c->ec_prev_type == TYPE_VOICED) {
            int delta = rc_icdf(r, SILK_PITCH_DELTA_ICDF, 8);
            if (delta > 0) {
                ix->lag = (int16_t)(c->ec_prev_lag + delta - 9);
                absolute = 0;
            }
        }
        if (absolute) {
            ix->lag = (int16_t)(rc_icdf(r, SILK_PITCH_LAG_ICDF, 8) *
                                (c->fs_khz >> 1));
            ix->lag += (int16_t)rc_icdf(r, c->lag_low_icdf, 8);
        }
        c->ec_prev_lag = ix->lag;
        ix->contour = (int8_t)rc_icdf(r, c->contour_icdf, 8);
        ix->per = (int8_t)rc_icdf(r, SILK_LTP_PER_INDEX_ICDF, 8);
        const uint8_t *gi = ix->per == 0   ? SILK_LTP_GAIN_ICDF_0
                            : ix->per == 1 ? SILK_LTP_GAIN_ICDF_1
                                           : SILK_LTP_GAIN_ICDF_2;
        for (k = 0; k < c->nb_subfr; k++)
            ix->ltp[k] = (int8_t)rc_icdf(r, gi, 8);
        ix->ltp_scale = cond == CODE_INDEPENDENTLY
                            ? (int8_t)rc_icdf(r, SILK_LTPSCALE_ICDF, 8)
                            : 0;
    }
    c->ec_prev_type = ix->signal_type;
    ix->seed = (int8_t)rc_icdf(r, SILK_UNIFORM4_ICDF, 8);
}

/* ---- the excitation ---- */

static void decode_split(int16_t *a, int16_t *b, struct rc *r, int p,
                         const uint8_t *table)
{
    if (p > 0) {
        *a = (int16_t)rc_icdf(r, &table[SILK_SHELL_CODE_TABLE_OFFSETS[p]], 8);
        *b = (int16_t)(p - *a);
    } else {
        *a = *b = 0;
    }
}

static void shell_decoder(int16_t *p0, struct rc *r, int p4)
{
    int16_t p3[2], p2[4], p1[8];
    const uint8_t *T0 = SILK_SHELL_CODE_TABLE0, *T1 = SILK_SHELL_CODE_TABLE1;
    const uint8_t *T2 = SILK_SHELL_CODE_TABLE2, *T3 = SILK_SHELL_CODE_TABLE3;
    decode_split(&p3[0], &p3[1], r, p4, T3);
    decode_split(&p2[0], &p2[1], r, p3[0], T2);
    decode_split(&p1[0], &p1[1], r, p2[0], T1);
    decode_split(&p0[0], &p0[1], r, p1[0], T0);
    decode_split(&p0[2], &p0[3], r, p1[1], T0);
    decode_split(&p1[2], &p1[3], r, p2[1], T1);
    decode_split(&p0[4], &p0[5], r, p1[2], T0);
    decode_split(&p0[6], &p0[7], r, p1[3], T0);
    decode_split(&p2[2], &p2[3], r, p3[1], T2);
    decode_split(&p1[4], &p1[5], r, p2[2], T1);
    decode_split(&p0[8], &p0[9], r, p1[4], T0);
    decode_split(&p0[10], &p0[11], r, p1[5], T0);
    decode_split(&p1[6], &p1[7], r, p2[3], T1);
    decode_split(&p0[12], &p0[13], r, p1[6], T0);
    decode_split(&p0[14], &p0[15], r, p1[7], T0);
}

static void decode_pulses(struct rc *r, int16_t *pulses, int type, int offset,
                          int frame_length)
{
    int sum[MAX_FRAME / 16 + 1], nls[MAX_FRAME / 16 + 1];
    int rate = rc_icdf(r, SILK_RATE_LEVELS_ICDF + 9 * (type >> 1), 8);
    int iter = frame_length >> 4;
    if (iter * 16 < frame_length)
        iter++;
    const uint8_t *cdf = SILK_PULSES_PER_BLOCK_ICDF + 18 * rate;
    for (int i = 0; i < iter; i++) {
        nls[i] = 0;
        sum[i] = rc_icdf(r, cdf, 8);
        while (sum[i] == 17) {
            nls[i]++;
            sum[i] = rc_icdf(
                r, SILK_PULSES_PER_BLOCK_ICDF + 18 * 9 + (nls[i] == 10), 8);
        }
    }
    for (int i = 0; i < iter; i++) {
        if (sum[i] > 0)
            shell_decoder(&pulses[16 * i], r, sum[i]);
        else
            memset(&pulses[16 * i], 0, 16 * sizeof(int16_t));
    }
    for (int i = 0; i < iter; i++)
        if (nls[i] > 0) {
            int16_t *p = &pulses[16 * i];
            for (int k = 0; k < 16; k++) {
                int a = p[k];
                for (int j = 0; j < nls[i]; j++) {
                    a <<= 1;
                    a += rc_icdf(r, SILK_LSB_ICDF, 8);
                }
                p[k] = (int16_t)a;
            }
            sum[i] |= nls[i] << 5;
        }
    /* the signs */
    uint8_t icdf[2] = {0, 0};
    const uint8_t *sp = &SILK_SIGN_ICDF[7 * (offset + (type << 1))];
    int len = (frame_length + 8) >> 4;
    int16_t *q = pulses;
    for (int i = 0; i < len; i++) {
        int p = sum[i];
        if (p > 0) {
            icdf[0] = sp[(p & 0x1F) < 6 ? (p & 0x1F) : 6];
            for (int j = 0; j < 16; j++)
                if (q[j] > 0)
                    q[j] = (int16_t)(q[j] * ((rc_icdf(r, icdf, 8) << 1) - 1));
        }
        q += 16;
    }
}

/* ---- parameters ---- */

static void gains_dequant(int32_t *gain, const int8_t *ind, int8_t *prev,
                          int conditional, int nb)
{
    for (int k = 0; k < nb; k++) {
        if (k == 0 && !conditional) {
            *prev = (int8_t)(ind[k] > *prev - 16 ? ind[k] : *prev - 16);
        } else {
            int t = ind[k] - MIN_DELTA_GAIN;
            int thr = 2 * MAX_DELTA_GAIN - N_LEVELS_QGAIN + *prev;
            if (t > thr)
                *prev = (int8_t)(*prev + (t << 1) - thr);
            else
                *prev = (int8_t)(*prev + t);
        }
        *prev = (int8_t)limit(*prev, 0, N_LEVELS_QGAIN - 1);
        int32_t v = smulwb(INV_SCALE_Q16, *prev) + OFFSET_Q7;
        gain[k] = log2lin(v < 3967 ? v : 3967);
    }
}

static void nlsf_stabilize(int16_t *nlsf, const int16_t *dmin, int L)
{
    int loops, I = 0;
    for (loops = 0; loops < 20; loops++) {
        int32_t min_diff = nlsf[0] - dmin[0];
        I = 0;
        for (int i = 1; i <= L - 1; i++) {
            int32_t d = nlsf[i] - (nlsf[i - 1] + dmin[i]);
            if (d < min_diff) {
                min_diff = d;
                I = i;
            }
        }
        int32_t d = (1 << 15) - (nlsf[L - 1] + dmin[L]);
        if (d < min_diff) {
            min_diff = d;
            I = L;
        }
        if (min_diff >= 0)
            return;
        if (I == 0) {
            nlsf[0] = dmin[0];
        } else if (I == L) {
            nlsf[L - 1] = (int16_t)((1 << 15) - dmin[L]);
        } else {
            int32_t min_c = 0, max_c = 1 << 15;
            for (int k = 0; k < I; k++)
                min_c += dmin[k];
            min_c += dmin[I] >> 1;
            for (int k = L; k > I; k--)
                max_c -= dmin[k];
            max_c -= dmin[I] >> 1;
            int16_t center = (int16_t)limit(
                rshift_round((int32_t)nlsf[I - 1] + nlsf[I], 1), min_c, max_c);
            nlsf[I - 1] = (int16_t)(center - (dmin[I] >> 1));
            nlsf[I] = (int16_t)(nlsf[I - 1] + dmin[I]);
        }
    }
    for (int i = 1; i < L; i++) { /* sort, increasing */
        int16_t v = nlsf[i];
        int j;
        for (j = i - 1; j >= 0 && v < nlsf[j]; j--)
            nlsf[j + 1] = nlsf[j];
        nlsf[j + 1] = v;
    }
    nlsf[0] = (int16_t)(nlsf[0] > dmin[0] ? nlsf[0] : dmin[0]);
    for (int i = 1; i < L; i++) {
        int32_t lo = sat16((int32_t)nlsf[i - 1] + dmin[i]);
        nlsf[i] = (int16_t)(nlsf[i] > lo ? nlsf[i] : lo);
    }
    int32_t hi = (1 << 15) - dmin[L];
    nlsf[L - 1] = (int16_t)(nlsf[L - 1] < hi ? nlsf[L - 1] : hi);
    for (int i = L - 2; i >= 0; i--) {
        int32_t v = nlsf[i + 1] - dmin[i + 1];
        nlsf[i] = (int16_t)(nlsf[i] < v ? nlsf[i] : v);
    }
}

static void nlsf_decode(int16_t *nlsf, const int8_t *ind,
                        const struct nlsf_cb *cb)
{
    uint8_t pred[MAX_ORDER];
    int16_t ec_ix[MAX_ORDER], res[MAX_ORDER];
    nlsf_unpack(ec_ix, pred, cb, ind[0]);
    int32_t out = 0;
    for (int i = cb->order - 1; i >= 0; i--) {
        int32_t p = smulbb(out, (int16_t)pred[i]) >> 8;
        out = LSH(ind[i + 1], 10);
        if (out > 0)
            out = (int16_t)(out - 102);
        else if (out < 0)
            out = (int16_t)(out + 102);
        out = smlawb(p, out, cb->step_q16);
        res[i] = (int16_t)out;
    }
    const uint8_t *e = &cb->cb1_q8[ind[0] * cb->order];
    const int16_t *w = &cb->wght_q9[ind[0] * cb->order];
    for (int i = 0; i < cb->order; i++) {
        int32_t t = (int32_t)(LSH(res[i], 14) / w[i]) + LSH((int16_t)e[i], 7);
        nlsf[i] = (int16_t)limit(t, 0, 32767);
    }
    nlsf_stabilize(nlsf, cb->delta_min, cb->order);
}

static void bwexpander32(int32_t *ar, int d, int32_t chirp)
{
    int32_t m1 = chirp - 65536;
    for (int i = 0; i < d - 1; i++) {
        ar[i] = smulww(chirp, ar[i]);
        chirp += rshift_round(chirp * m1, 16);
    }
    ar[d - 1] = smulww(chirp, ar[d - 1]);
}

static void bwexpander16(int16_t *ar, int d, int32_t chirp)
{
    int32_t m1 = chirp - 65536;
    for (int i = 0; i < d - 1; i++) {
        ar[i] = (int16_t)rshift_round(chirp * ar[i], 16);
        chirp += rshift_round(chirp * m1, 16);
    }
    ar[d - 1] = (int16_t)rshift_round(chirp * ar[d - 1], 16);
}

static int32_t inv_pred_gain_qa(int32_t *A, int order)
{
    const int32_t LIMIT = 16773022; /* 0.99975 in Q24 */
    const int32_t MIN_INV_GAIN = 107374;
    int32_t inv_gain = 1 << 30;
    int k;
    for (k = order - 1; k > 0; k--) {
        if (A[k] > LIMIT || A[k] < -LIMIT)
            return 0;
        int32_t rc = -(int32_t)((uint32_t)A[k] << 7);
        int32_t rc1 = (1 << 30) - smmul(rc, rc);
        inv_gain = (int32_t)((uint32_t)smmul(inv_gain, rc1) << 2);
        if (inv_gain < MIN_INV_GAIN)
            return 0;
        int mult2q = 32 - clz32(rc1 > 0 ? rc1 : -rc1);
        int32_t rc2 = inverse32_varq(rc1, mult2q + 30);
        for (int n = 0; n < (k + 1) >> 1; n++) {
            int32_t t1 = A[n], t2 = A[k - n - 1];
            int64_t v = rshift_round64(
                (int64_t)sub_sat32(
                    t1, (int32_t)rshift_round64((int64_t)t2 * rc, 31)) *
                    rc2,
                mult2q);
            if (v > INT32_MAX || v < INT32_MIN)
                return 0;
            A[n] = (int32_t)v;
            v = rshift_round64(
                (int64_t)sub_sat32(
                    t2, (int32_t)rshift_round64((int64_t)t1 * rc, 31)) *
                    rc2,
                mult2q);
            if (v > INT32_MAX || v < INT32_MIN)
                return 0;
            A[k - n - 1] = (int32_t)v;
        }
    }
    if (A[k] > LIMIT || A[k] < -LIMIT)
        return 0;
    int32_t rc = -(int32_t)((uint32_t)A[0] << 7);
    int32_t rc1 = (1 << 30) - smmul(rc, rc);
    inv_gain = (int32_t)((uint32_t)smmul(inv_gain, rc1) << 2);
    return inv_gain < MIN_INV_GAIN ? 0 : inv_gain;
}

static int32_t inv_pred_gain(const int16_t *a, int order)
{
    int32_t A[MAX_ORDER], dc = 0;
    for (int k = 0; k < order; k++) {
        dc += a[k];
        A[k] = LSH(a[k], 12);
    }
    if (dc >= 4096)
        return 0;
    return inv_pred_gain_qa(A, order);
}

static void lpc_fit(int16_t *out, int32_t *in, int qout, int qin, int d)
{
    int i, idx = 0;
    for (i = 0; i < 10; i++) {
        int32_t maxabs = 0;
        for (int k = 0; k < d; k++) {
            int32_t a = in[k] > 0 ? in[k] : -in[k];
            if (a > maxabs) {
                maxabs = a;
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, qin - qout);
        if (maxabs > 32767) {
            maxabs = maxabs < 163838 ? maxabs : 163838;
            int32_t chirp =
                65470 - (int32_t)(((uint32_t)(maxabs - 32767) << 14) /
                                  (uint32_t)((maxabs * (idx + 1)) >> 2));
            bwexpander32(in, d, chirp);
        } else {
            break;
        }
    }
    if (i == 10) {
        for (int k = 0; k < d; k++) {
            out[k] = (int16_t)sat16(rshift_round(in[k], qin - qout));
            in[k] = (int32_t)((uint32_t)(int32_t)out[k] << (qin - qout));
        }
    } else {
        for (int k = 0; k < d; k++)
            out[k] = (int16_t)rshift_round(in[k], qin - qout);
    }
}

static void find_poly(int32_t *out, const int32_t *c, int dd)
{
    out[0] = 1 << 16;
    out[1] = -c[0];
    for (int k = 1; k < dd; k++) {
        int32_t f = c[2 * k];
        out[k + 1] = (int32_t)((uint32_t)out[k - 1] << 1) -
                     (int32_t)rshift_round64((int64_t)f * out[k], 16);
        for (int n = k; n > 1; n--)
            out[n] += out[n - 2] -
                      (int32_t)rshift_round64((int64_t)f * out[n - 1], 16);
        out[1] -= f;
    }
}

static void nlsf2a(int16_t *a, const int16_t *nlsf, int d)
{
    static const uint8_t ORD16[16] = {0, 15, 8,  7, 4, 11, 12, 3,
                                      2, 13, 10, 5, 6, 9,  14, 1};
    static const uint8_t ORD10[10] = {0, 9, 6, 3, 4, 5, 8, 1, 2, 7};
    const uint8_t *ord = d == 16 ? ORD16 : ORD10;
    int32_t cosv[MAX_ORDER] = {0}, P[MAX_ORDER / 2 + 1], Q[MAX_ORDER / 2 + 1];
    int32_t a32[MAX_ORDER];
    for (int k = 0; k < d; k++) {
        int32_t fi = nlsf[k] >> 8, ff = nlsf[k] - (fi << 8);
        int32_t cv = SILK_LSF_COS_Q12[fi],
                delta = SILK_LSF_COS_Q12[fi + 1] - cv;
        cosv[ord[k]] = rshift_round(LSH(cv, 8) + delta * ff, 20 - 16);
    }
    int dd = d >> 1;
    find_poly(P, &cosv[0], dd);
    find_poly(Q, &cosv[1], dd);
    for (int k = 0; k < dd; k++) {
        int32_t pt = P[k + 1] + P[k], qt = Q[k + 1] - Q[k];
        a32[k] = -qt - pt;
        a32[d - k - 1] = qt - pt;
    }
    lpc_fit(a, a32, 12, 17, d);
    for (int i = 0; inv_pred_gain(a, d) == 0 && i < 16; i++) {
        bwexpander32(a32, d, 65536 - (2 << i));
        for (int k = 0; k < d; k++)
            a[k] = (int16_t)rshift_round(a32[k], 17 - 12);
    }
}

static void decode_pitch(int lag_index, int contour, int *lags, int fs_khz,
                         int nb)
{
    const int8_t *cb;
    int size;
    if (fs_khz == 8) {
        cb = nb == 4 ? SILK_CB_LAGS_STAGE2 : SILK_CB_LAGS_STAGE2_10_MS;
        size = nb == 4 ? 11 : 3;
    } else {
        cb = nb == 4 ? SILK_CB_LAGS_STAGE3 : SILK_CB_LAGS_STAGE3_10_MS;
        size = nb == 4 ? 34 : 12;
    }
    int min_lag = 2 * fs_khz, max_lag = 18 * fs_khz, lag = min_lag + lag_index;
    for (int k = 0; k < nb; k++)
        lags[k] = limit(lag + cb[k * size + contour], min_lag, max_lag);
}

static void decode_parameters(struct chan *c, struct ctrl *ct, int cond)
{
    int16_t nlsf[MAX_ORDER], nlsf0[MAX_ORDER];
    gains_dequant(ct->gains, c->ind.gains, &c->last_gain,
                  cond == CODE_CONDITIONALLY, c->nb_subfr);
    nlsf_decode(nlsf, c->ind.nlsf, c->cb);
    nlsf2a(ct->pred[1], nlsf, c->order);
    if (c->first_after_reset == 1)
        c->ind.nlsf_interp = 4;
    if (c->ind.nlsf_interp < 4) {
        for (int i = 0; i < c->order; i++)
            nlsf0[i] =
                (int16_t)(c->prev_nlsf[i] +
                          ((c->ind.nlsf_interp * (nlsf[i] - c->prev_nlsf[i])) >>
                           2));
        nlsf2a(ct->pred[0], nlsf0, c->order);
    } else {
        memcpy(ct->pred[0], ct->pred[1], sizeof(int16_t) * (size_t)c->order);
    }
    memcpy(c->prev_nlsf, nlsf, sizeof(int16_t) * (size_t)c->order);
    if (c->loss_cnt) {
        bwexpander16(ct->pred[0], c->order, 63570);
        bwexpander16(ct->pred[1], c->order, 63570);
    }
    if (c->ind.signal_type == TYPE_VOICED) {
        decode_pitch(c->ind.lag, c->ind.contour, ct->pitch_l, c->fs_khz,
                     c->nb_subfr);
        const int8_t *cbk = c->ind.per == 0   ? SILK_LTP_GAIN_VQ_0
                            : c->ind.per == 1 ? SILK_LTP_GAIN_VQ_1
                                              : SILK_LTP_GAIN_VQ_2;
        for (int k = 0; k < c->nb_subfr; k++)
            for (int i = 0; i < LTP_ORDER; i++)
                ct->ltp[k * LTP_ORDER + i] =
                    (int16_t)(cbk[c->ind.ltp[k] * LTP_ORDER + i] * 128);
        static const int16_t SCALES[3] = {15565, 12288, 8192};
        ct->ltp_scale = SCALES[c->ind.ltp_scale];
    } else {
        memset(ct->pitch_l, 0, sizeof ct->pitch_l);
        memset(ct->ltp, 0, sizeof ct->ltp);
        c->ind.per = 0;
        ct->ltp_scale = 0;
    }
}

/* ---- synthesis ---- */

static void lpc_analysis(int16_t *out, const int16_t *in, const int16_t *B,
                         int len, int d)
{
    for (int ix = d; ix < len; ix++) {
        const int16_t *p = &in[ix - 1];
        int32_t o = smulbb(p[0], B[0]);
        for (int j = 1; j < d; j++)
            o = smlabb(o, p[-j], B[j]);
        o = (int32_t)(((uint32_t)p[1] << 12) - (uint32_t)o);
        out[ix] = (int16_t)sat16(rshift_round(o, 12));
    }
    memset(out, 0, sizeof(int16_t) * (size_t)d);
}

static void decode_core(struct chan *c, struct ctrl *ct, int16_t *xq,
                        const int16_t *pulses)
{
    static const int16_t OFFSETS[2][2] = {{100, 240}, {32, 100}};
    int16_t sLTP[MAX_FRAME + 2 * MAX_SUBFR], A_tmp[MAX_ORDER];
    int32_t sLTP_Q15[2 * MAX_FRAME + 2 * MAX_SUBFR], res[MAX_SUBFR];
    int32_t sLPC[MAX_SUBFR + MAX_ORDER];
    int lag = 0;
    int32_t offset = OFFSETS[c->ind.signal_type >> 1][c->ind.quant_offset];
    int interp_flag = c->ind.nlsf_interp < 4;
    int32_t seed = c->ind.seed;
    for (int i = 0; i < c->frame_length; i++) {
        seed = silk_rand(seed);
        c->exc[i] = LSH(pulses[i], 14);
        if (c->exc[i] > 0)
            c->exc[i] -= 80 << 4;
        else if (c->exc[i] < 0)
            c->exc[i] += 80 << 4;
        c->exc[i] += offset << 4;
        if (seed < 0)
            c->exc[i] = -c->exc[i];
        seed = (int32_t)((uint32_t)seed + (uint32_t)pulses[i]);
    }
    memcpy(sLPC, c->slpc, sizeof c->slpc);
    int32_t *pexc = c->exc;
    int16_t *pxq = xq;
    int buf_idx = c->ltp_mem;
    for (int k = 0; k < c->nb_subfr; k++) {
        int32_t *pres = res;
        int16_t *A = ct->pred[k >> 1];
        memcpy(A_tmp, A, sizeof(int16_t) * (size_t)c->order);
        int16_t *B = &ct->ltp[k * LTP_ORDER];
        int type = c->ind.signal_type;
        int32_t gain_q10 = ct->gains[k] >> 6;
        int32_t inv_gain = inverse32_varq(ct->gains[k], 47), gain_adj;
        if (ct->gains[k] != c->prev_gain) {
            gain_adj = div32_varq(c->prev_gain, ct->gains[k], 16);
            for (int i = 0; i < MAX_ORDER; i++)
                sLPC[i] = smulww(gain_adj, sLPC[i]);
        } else {
            gain_adj = 1 << 16;
        }
        c->prev_gain = ct->gains[k];
        if (c->loss_cnt && c->prev_type == TYPE_VOICED &&
            c->ind.signal_type != TYPE_VOICED && k < 2) {
            memset(B, 0, LTP_ORDER * sizeof(int16_t));
            B[LTP_ORDER / 2] = 4096;
            type = TYPE_VOICED;
            ct->pitch_l[k] = c->lag_prev;
        }
        if (type == TYPE_VOICED) {
            lag = ct->pitch_l[k];
            if (k == 0 || (k == 2 && interp_flag)) {
                int start = c->ltp_mem - lag - c->order - LTP_ORDER / 2;
                if (k == 2)
                    memcpy(&c->out[c->ltp_mem], xq,
                           2 * (size_t)c->subfr_length * sizeof(int16_t));
                lpc_analysis(&sLTP[start], &c->out[start + k * c->subfr_length],
                             A, c->ltp_mem - start, c->order);
                if (k == 0)
                    inv_gain =
                        (int32_t)((uint32_t)smulwb(inv_gain, ct->ltp_scale)
                                  << 2);
                for (int i = 0; i < lag + LTP_ORDER / 2; i++)
                    sLTP_Q15[buf_idx - i - 1] =
                        smulwb(inv_gain, sLTP[c->ltp_mem - i - 1]);
            } else if (gain_adj != 1 << 16) {
                for (int i = 0; i < lag + LTP_ORDER / 2; i++)
                    sLTP_Q15[buf_idx - i - 1] =
                        smulww(gain_adj, sLTP_Q15[buf_idx - i - 1]);
            }
        }
        if (type == TYPE_VOICED) {
            int32_t *pl = &sLTP_Q15[buf_idx - lag + LTP_ORDER / 2];
            for (int i = 0; i < c->subfr_length; i++) {
                int32_t p = 2;
                p = smlawb(p, pl[0], B[0]);
                p = smlawb(p, pl[-1], B[1]);
                p = smlawb(p, pl[-2], B[2]);
                p = smlawb(p, pl[-3], B[3]);
                p = smlawb(p, pl[-4], B[4]);
                pl++;
                pres[i] = (int32_t)((uint32_t)pexc[i] + ((uint32_t)p << 1));
                sLTP_Q15[buf_idx] = (int32_t)((uint32_t)pres[i] << 1);
                buf_idx++;
            }
        } else {
            pres = pexc;
        }
        for (int i = 0; i < c->subfr_length; i++) {
            int32_t p = c->order >> 1;
            for (int j = 0; j < c->order; j++)
                p = smlawb(p, sLPC[MAX_ORDER + i - j - 1], A_tmp[j]);
            sLPC[MAX_ORDER + i] = add_sat32(pres[i], lshift_sat32(p, 4));
            pxq[i] = (int16_t)sat16(
                rshift_round(smulww(sLPC[MAX_ORDER + i], gain_q10), 8));
        }
        memcpy(sLPC, &sLPC[c->subfr_length], MAX_ORDER * sizeof(int32_t));
        pexc += c->subfr_length;
        pxq += c->subfr_length;
    }
    memcpy(c->slpc, sLPC, sizeof c->slpc);
}

/* ---- concealment ---- */

static void plc_update(struct chan *c, const struct ctrl *ct)
{
    struct plc *p = &c->plc;
    int32_t gain = 0;
    c->prev_type = c->ind.signal_type;
    if (c->ind.signal_type == TYPE_VOICED) {
        for (int j = 0; j * c->subfr_length < ct->pitch_l[c->nb_subfr - 1];
             j++) {
            if (j == c->nb_subfr)
                break;
            int32_t t = 0;
            for (int i = 0; i < LTP_ORDER; i++)
                t += ct->ltp[(c->nb_subfr - 1 - j) * LTP_ORDER + i];
            if (t > gain) {
                gain = t;
                memcpy(p->ltp_coef, &ct->ltp[(c->nb_subfr - 1 - j) * LTP_ORDER],
                       sizeof p->ltp_coef);
                p->pitch_l_q8 =
                    (int32_t)((uint32_t)ct->pitch_l[c->nb_subfr - 1 - j] << 8);
            }
        }
        memset(p->ltp_coef, 0, sizeof p->ltp_coef);
        p->ltp_coef[LTP_ORDER / 2] = (int16_t)gain;
        if (gain < 11469) {
            int32_t scale = (11469 << 10) / (gain > 1 ? gain : 1);
            for (int i = 0; i < LTP_ORDER; i++)
                p->ltp_coef[i] = (int16_t)(smulbb(p->ltp_coef[i], scale) >> 10);
        } else if (gain > 15565) {
            int32_t scale = (15565 << 14) / (gain > 1 ? gain : 1);
            for (int i = 0; i < LTP_ORDER; i++)
                p->ltp_coef[i] = (int16_t)(smulbb(p->ltp_coef[i], scale) >> 14);
        }
    } else {
        p->pitch_l_q8 = (int32_t)((uint32_t)smulbb(c->fs_khz, 18) << 8);
        memset(p->ltp_coef, 0, sizeof p->ltp_coef);
    }
    memcpy(p->prev_lpc, ct->pred[1], sizeof(int16_t) * (size_t)c->order);
    p->prev_ltp_scale = (int16_t)ct->ltp_scale;
    memcpy(p->prev_gain, &ct->gains[c->nb_subfr - 2], 2 * sizeof(int32_t));
    p->subfr_length = c->subfr_length;
    p->nb_subfr = c->nb_subfr;
}

static void plc_conceal(struct chan *c, struct ctrl *ct, int16_t *frame)
{
    static const int16_t HARM_ATT[2] = {32440, 31130};
    static const int16_t RAND_ATT_V[2] = {31130, 26214};
    static const int16_t RAND_ATT_UV[2] = {32440, 29491};
    struct plc *p = &c->plc;
    int32_t sLTP_Q14[2 * MAX_FRAME + 2 * MAX_SUBFR];
    int16_t sLTP[MAX_FRAME + 2 * MAX_SUBFR], A[MAX_ORDER], ebuf[2 * MAX_SUBFR];
    int32_t prev_q10[2] = {p->prev_gain[0] >> 6, p->prev_gain[1] >> 6};
    if (c->first_after_reset)
        memset(p->prev_lpc, 0, sizeof p->prev_lpc);
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < c->subfr_length; i++)
            ebuf[k * c->subfr_length + i] = (int16_t)sat16(
                smulww(c->exc[i + (k + c->nb_subfr - 2) * c->subfr_length],
                       prev_q10[k]) >>
                8);
    int32_t e1, e2;
    int s1, s2;
    sum_sqr_shift(&e1, &s1, ebuf, c->subfr_length);
    sum_sqr_shift(&e2, &s2, &ebuf[c->subfr_length], c->subfr_length);
    int32_t *rand_ptr;
    if ((e1 >> s2) < (e2 >> s1)) {
        int at = (p->nb_subfr - 1) * p->subfr_length - 128;
        rand_ptr = &c->exc[at > 0 ? at : 0];
    } else {
        int at = p->nb_subfr * p->subfr_length - 128;
        rand_ptr = &c->exc[at > 0 ? at : 0];
    }
    int16_t *B = p->ltp_coef;
    int16_t rand_scale = p->rand_scale;
    int la = c->loss_cnt < 1 ? c->loss_cnt : 1;
    int32_t harm_gain = HARM_ATT[la];
    int32_t rand_gain =
        c->prev_type == TYPE_VOICED ? RAND_ATT_V[la] : RAND_ATT_UV[la];
    bwexpander16(p->prev_lpc, c->order, 64880);
    memcpy(A, p->prev_lpc, sizeof(int16_t) * (size_t)c->order);
    if (c->loss_cnt == 0) {
        rand_scale = 1 << 14;
        if (c->prev_type == TYPE_VOICED) {
            for (int i = 0; i < LTP_ORDER; i++)
                rand_scale = (int16_t)(rand_scale - B[i]);
            rand_scale = rand_scale > 3277 ? rand_scale : 3277;
            rand_scale = (int16_t)(smulbb(rand_scale, p->prev_ltp_scale) >> 14);
        } else {
            int32_t ig = inv_pred_gain(p->prev_lpc, c->order);
            int32_t ds = (1 << 30) >> 3;
            ds = ds < ig ? ds : ig;
            int32_t lo = (1 << 30) >> 8;
            ds = ds > lo ? ds : lo;
            ds = (int32_t)((uint32_t)ds << 3);
            rand_gain = smulwb(ds, rand_gain) >> 14;
        }
    }
    int32_t seed = p->seed;
    int lag = rshift_round(p->pitch_l_q8, 8), buf_idx = c->ltp_mem;
    int idx = c->ltp_mem - lag - c->order - LTP_ORDER / 2;
    lpc_analysis(&sLTP[idx], &c->out[idx], A, c->ltp_mem - idx, c->order);
    int32_t inv_gain = inverse32_varq(p->prev_gain[1], 46);
    if (inv_gain > INT32_MAX >> 1)
        inv_gain = INT32_MAX >> 1;
    for (int i = idx + c->order; i < c->ltp_mem; i++)
        sLTP_Q14[i] = smulwb(inv_gain, sLTP[i]);
    for (int k = 0; k < c->nb_subfr; k++) {
        int32_t *pl = &sLTP_Q14[buf_idx - lag + LTP_ORDER / 2];
        for (int i = 0; i < c->subfr_length; i++) {
            int32_t pr = 2;
            pr = smlawb(pr, pl[0], B[0]);
            pr = smlawb(pr, pl[-1], B[1]);
            pr = smlawb(pr, pl[-2], B[2]);
            pr = smlawb(pr, pl[-3], B[3]);
            pr = smlawb(pr, pl[-4], B[4]);
            pl++;
            seed = silk_rand(seed);
            int ri = (seed >> 25) & 127;
            sLTP_Q14[buf_idx] =
                (int32_t)((uint32_t)smlawb(pr, rand_ptr[ri], rand_scale) << 2);
            buf_idx++;
        }
        for (int j = 0; j < LTP_ORDER; j++)
            B[j] = (int16_t)(smulbb(harm_gain, B[j]) >> 15);
        rand_scale = (int16_t)(smulbb(rand_scale, rand_gain) >> 15);
        p->pitch_l_q8 = smlawb(p->pitch_l_q8, p->pitch_l_q8, 655);
        int32_t mx = (int32_t)((uint32_t)smulbb(18, c->fs_khz) << 8);
        p->pitch_l_q8 = p->pitch_l_q8 < mx ? p->pitch_l_q8 : mx;
        lag = rshift_round(p->pitch_l_q8, 8);
    }
    int32_t *sl = &sLTP_Q14[c->ltp_mem - MAX_ORDER];
    memcpy(sl, c->slpc, sizeof c->slpc);
    for (int i = 0; i < c->frame_length; i++) {
        int32_t pr = c->order >> 1;
        for (int j = 0; j < c->order; j++)
            pr = smlawb(pr, sl[MAX_ORDER + i - j - 1], A[j]);
        sl[MAX_ORDER + i] = add_sat32(sl[MAX_ORDER + i], lshift_sat32(pr, 4));
        frame[i] = (int16_t)sat16(
            rshift_round(smulww(sl[MAX_ORDER + i], prev_q10[1]), 8));
    }
    memcpy(c->slpc, &sl[c->frame_length], sizeof c->slpc);
    p->seed = seed;
    p->rand_scale = rand_scale;
    for (int i = 0; i < 4; i++)
        ct->pitch_l[i] = lag;
}

static void plc_glue(struct chan *c, int16_t *frame, int length)
{
    struct plc *p = &c->plc;
    if (c->loss_cnt) {
        sum_sqr_shift(&p->conc_energy, &p->conc_energy_shift, frame, length);
        p->last_frame_lost = 1;
        return;
    }
    if (p->last_frame_lost) {
        int32_t energy;
        int shift;
        sum_sqr_shift(&energy, &shift, frame, length);
        if (shift > p->conc_energy_shift)
            p->conc_energy >>= shift - p->conc_energy_shift;
        else if (shift < p->conc_energy_shift)
            energy >>= p->conc_energy_shift - shift;
        if (energy > p->conc_energy) {
            int32_t lz = clz32(p->conc_energy) - 1;
            p->conc_energy = (int32_t)((uint32_t)p->conc_energy << lz);
            energy >>= 24 - lz > 0 ? 24 - lz : 0;
            int32_t frac = p->conc_energy / (energy > 1 ? energy : 1);
            int32_t gain = (int32_t)((uint32_t)sqrt_approx(frac) << 4);
            int32_t slope = ((1 << 16) - gain) / length;
            slope = (int32_t)((uint32_t)slope << 2);
            for (int i = 0; i < length; i++) {
                frame[i] = (int16_t)smulwb(gain, frame[i]);
                gain += slope;
                if (gain > 1 << 16)
                    break;
            }
        }
    }
    p->last_frame_lost = 0;
}

static void cng(struct chan *c, const struct ctrl *ct, int16_t *frame,
                int length)
{
    struct cng *g = &c->cng;
    if (c->fs_khz != g->fs_khz) {
        cng_reset(c);
        g->fs_khz = c->fs_khz;
    }
    if (c->loss_cnt == 0 && c->prev_type == TYPE_NO_VOICE) {
        for (int i = 0; i < c->order; i++)
            g->smth_nlsf[i] =
                (int16_t)(g->smth_nlsf[i] +
                          smulwb((int32_t)c->prev_nlsf[i] - g->smth_nlsf[i],
                                 16348));
        int32_t max_gain = 0;
        int subfr = 0;
        for (int i = 0; i < c->nb_subfr; i++)
            if (ct->gains[i] > max_gain) {
                max_gain = ct->gains[i];
                subfr = i;
            }
        memmove(&g->exc_buf[c->subfr_length], g->exc_buf,
                (size_t)((c->nb_subfr - 1) * c->subfr_length) *
                    sizeof(int32_t));
        memcpy(g->exc_buf, &c->exc[subfr * c->subfr_length],
               (size_t)c->subfr_length * sizeof(int32_t));
        for (int i = 0; i < c->nb_subfr; i++) {
            g->smth_gain += smulwb(ct->gains[i] - g->smth_gain, 4634);
            if (smulww(g->smth_gain, 46396) > ct->gains[i])
                g->smth_gain = ct->gains[i];
        }
    }
    if (c->loss_cnt) {
        int32_t sig[MAX_FRAME + MAX_ORDER];
        int16_t A[MAX_ORDER];
        int32_t gain = smulww(c->plc.rand_scale, c->plc.prev_gain[1]);
        if (gain >= 1 << 21 || g->smth_gain > 1 << 23) {
            gain = smultt(gain, gain);
            gain = smultt(g->smth_gain, g->smth_gain) -
                   (int32_t)((uint32_t)gain << 5);
            gain = (int32_t)((uint32_t)sqrt_approx(gain) << 16);
        } else {
            gain = smulww(gain, gain);
            gain = smulww(g->smth_gain, g->smth_gain) -
                   (int32_t)((uint32_t)gain << 5);
            gain = (int32_t)((uint32_t)sqrt_approx(gain) << 8);
        }
        int32_t gain_q10 = gain >> 6;
        int mask = 255;
        while (mask > length)
            mask >>= 1;
        int32_t seed = g->seed;
        for (int i = 0; i < length; i++) {
            seed = silk_rand(seed);
            sig[MAX_ORDER + i] = g->exc_buf[(seed >> 24) & mask];
        }
        g->seed = seed;
        nlsf2a(A, g->smth_nlsf, c->order);
        memcpy(sig, g->synth, sizeof g->synth);
        for (int i = 0; i < length; i++) {
            int32_t pr = c->order >> 1;
            for (int j = 0; j < c->order; j++)
                pr = smlawb(pr, sig[MAX_ORDER + i - j - 1], A[j]);
            sig[MAX_ORDER + i] =
                add_sat32(sig[MAX_ORDER + i], lshift_sat32(pr, 4));
            frame[i] = (int16_t)sat16(
                frame[i] +
                sat16(rshift_round(smulww(sig[MAX_ORDER + i], gain_q10), 8)));
        }
        memcpy(g->synth, &sig[length], sizeof g->synth);
    } else {
        memset(g->synth, 0, (size_t)c->order * sizeof(int32_t));
    }
}

/* ---- frames ---- */

static void decode_frame(struct chan *c, struct rc *r, int16_t *out, int lost,
                         int cond)
{
    struct ctrl ct = {.ltp_scale = 0};
    int L = c->frame_length;
    if (!lost) {
        int16_t pulses[MAX_FRAME + 16];
        decode_indices(c, r, c->frames_decoded, 0, cond);
        decode_pulses(r, pulses, c->ind.signal_type, c->ind.quant_offset, L);
        decode_parameters(c, &ct, cond);
        decode_core(c, &ct, out, pulses);
        int mv = c->ltp_mem - L;
        memmove(c->out, &c->out[L], (size_t)mv * sizeof(int16_t));
        memcpy(&c->out[mv], out, (size_t)L * sizeof(int16_t));
        if (c->fs_khz != c->plc.fs_khz) {
            plc_reset(c);
            c->plc.fs_khz = c->fs_khz;
        }
        plc_update(c, &ct);
        c->loss_cnt = 0;
        c->prev_type = c->ind.signal_type;
        c->first_after_reset = 0;
    } else {
        if (c->fs_khz != c->plc.fs_khz) {
            plc_reset(c);
            c->plc.fs_khz = c->fs_khz;
        }
        plc_conceal(c, &ct, out);
        c->loss_cnt++;
        int mv = c->ltp_mem - L;
        memmove(c->out, &c->out[L], (size_t)mv * sizeof(int16_t));
        memcpy(&c->out[mv], out, (size_t)L * sizeof(int16_t));
    }
    cng(c, &ct, out, L);
    plc_glue(c, out, L);
    c->lag_prev = ct.pitch_l[c->nb_subfr - 1];
}

struct silk_dec *jsilk_new(void)
{
    struct silk_dec *d = calloc(1, sizeof *d);
    if (d) {
        d->api_hz = 48000;
        jsilk_reset(d);
    }
    return d;
}

void jsilk_set_rate(struct silk_dec *d, int hz)
{
    d->api_hz = hz;
}

void jsilk_free(struct silk_dec *d)
{
    free(d);
}

void jsilk_reset(struct silk_dec *d)
{
    for (int n = 0; n < 2; n++)
        chan_reset(&d->ch[n]);
    memset(d->pred_prev, 0, sizeof d->pred_prev);
    memset(d->smid, 0, sizeof d->smid);
    memset(d->sside, 0, sizeof d->sside);
    d->prev_mid_only = 0;
}

static void stereo_pred(struct rc *r, int32_t *pred)
{
    int ix[2][3];
    int n = rc_icdf(r, SILK_STEREO_PRED_JOINT_ICDF, 8);
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for (n = 0; n < 2; n++) {
        ix[n][0] = rc_icdf(r, SILK_UNIFORM3_ICDF, 8);
        ix[n][1] = rc_icdf(r, SILK_UNIFORM5_ICDF, 8);
    }
    for (n = 0; n < 2; n++) {
        ix[n][0] += 3 * ix[n][2];
        int32_t low = SILK_STEREO_PRED_QUANT_Q13[ix[n][0]];
        int32_t step =
            smulwb(SILK_STEREO_PRED_QUANT_Q13[ix[n][0] + 1] - low, 6554);
        pred[n] = smlabb(low, step, 2 * ix[n][1] + 1);
    }
    pred[0] -= pred[1];
}

static void ms_to_lr(struct silk_dec *d, int16_t *x1, int16_t *x2,
                     const int32_t *pred, int fs_khz, int len)
{
    memcpy(x1, d->smid, 2 * sizeof(int16_t));
    memcpy(x2, d->sside, 2 * sizeof(int16_t));
    memcpy(d->smid, &x1[len], 2 * sizeof(int16_t));
    memcpy(d->sside, &x2[len], 2 * sizeof(int16_t));
    int32_t p0 = d->pred_prev[0], p1 = d->pred_prev[1];
    int32_t denom = (1 << 16) / (8 * fs_khz);
    int32_t d0 = rshift_round(smulbb(pred[0] - d->pred_prev[0], denom), 16);
    int32_t d1 = rshift_round(smulbb(pred[1] - d->pred_prev[1], denom), 16);
    int n;
    for (n = 0; n < 8 * fs_khz; n++) {
        p0 += d0;
        p1 += d1;
        int32_t sum =
            (int32_t)((uint32_t)(x1[n] + (int32_t)x1[n + 2] + LSH(x1[n + 1], 1))
                      << 9);
        sum = smlawb(LSH(x2[n + 1], 8), sum, p0);
        sum = smlawb(sum, LSH(x1[n + 1], 11), p1);
        x2[n + 1] = (int16_t)sat16(rshift_round(sum, 8));
    }
    p0 = pred[0];
    p1 = pred[1];
    for (; n < len; n++) {
        int32_t sum =
            (int32_t)((uint32_t)(x1[n] + (int32_t)x1[n + 2] + LSH(x1[n + 1], 1))
                      << 9);
        sum = smlawb(LSH(x2[n + 1], 8), sum, p0);
        sum = smlawb(sum, LSH(x1[n + 1], 11), p1);
        x2[n + 1] = (int16_t)sat16(rshift_round(sum, 8));
    }
    d->pred_prev[0] = (int16_t)pred[0];
    d->pred_prev[1] = (int16_t)pred[1];
    for (n = 0; n < len; n++) {
        int32_t s = x1[n + 1] + (int32_t)x2[n + 1];
        int32_t df = x1[n + 1] - (int32_t)x2[n + 1];
        x1[n + 1] = (int16_t)sat16(s);
        x2[n + 1] = (int16_t)sat16(df);
    }
}

int jsilk_frame(struct silk_dec *d, struct rc *r, int ms, int khz,
                int stream_ch, int out_ch, float **out, int first)
{
    int lost = r == NULL, mid_only = 0;
    int32_t pred[2] = {0, 0};
    struct chan *ch = d->ch;
    if (lost) {
        khz = d->last_khz;
        stream_ch = d->last_ch;
    } else {
        d->last_khz = khz;
        d->last_ch = stream_ch;
    }
    if (!khz || !stream_ch)
        return -1;
    if (first)
        for (int n = 0; n < stream_ch; n++)
            ch[n].frames_decoded = 0;
    if (stream_ch > d->int_ch)
        chan_reset(&ch[1]);
    int stereo_to_mono =
        stream_ch == 1 && d->int_ch == 2 && khz * 1000 == 1000 * ch[0].fs_khz;
    if (ch[0].frames_decoded == 0)
        for (int n = 0; n < stream_ch; n++) {
            if (ms == 10) {
                ch[n].frames_per_packet = 1;
                ch[n].nb_subfr = 2;
            } else if (ms == 20) {
                ch[n].frames_per_packet = 1;
                ch[n].nb_subfr = 4;
            } else if (ms == 40 || ms == 60) {
                ch[n].frames_per_packet = ms / 20;
                ch[n].nb_subfr = 4;
            } else {
                return -1;
            }
            set_fs(&ch[n], khz, d->api_hz);
        }
    if (out_ch == 2 && stream_ch == 2 && (d->api_ch == 1 || d->int_ch == 1)) {
        memset(d->pred_prev, 0, sizeof d->pred_prev);
        memset(d->sside, 0, sizeof d->sside);
        ch[1].rs = ch[0].rs;
    }
    d->api_ch = out_ch;
    d->int_ch = stream_ch;
    if (!lost && ch[0].frames_decoded == 0) {
        for (int n = 0; n < stream_ch; n++) {
            for (int i = 0; i < ch[n].frames_per_packet; i++)
                ch[n].vad[i] = rc_bit_logp(r, 1);
            ch[n].lbrr = rc_bit_logp(r, 1);
        }
        for (int n = 0; n < stream_ch; n++) {
            memset(ch[n].lbrr_flags, 0, sizeof ch[n].lbrr_flags);
            if (ch[n].lbrr) {
                if (ch[n].frames_per_packet == 1) {
                    ch[n].lbrr_flags[0] = 1;
                } else {
                    int sym = rc_icdf(r,
                                      ch[n].frames_per_packet == 2
                                          ? SILK_LBRR_FLAGS_2_ICDF
                                          : SILK_LBRR_FLAGS_3_ICDF,
                                      8) +
                              1;
                    for (int i = 0; i < ch[n].frames_per_packet; i++)
                        ch[n].lbrr_flags[i] = (sym >> i) & 1;
                }
            }
        }
        /* the redundant (LBRR) frames: read past */
        for (int i = 0; i < ch[0].frames_per_packet; i++)
            for (int n = 0; n < stream_ch; n++)
                if (ch[n].lbrr_flags[i]) {
                    int16_t pulses[MAX_FRAME + 16];
                    if (stream_ch == 2 && n == 0) {
                        stereo_pred(r, pred);
                        if (ch[1].lbrr_flags[i] == 0)
                            mid_only =
                                rc_icdf(r, SILK_STEREO_ONLY_CODE_MID_ICDF, 8);
                    }
                    int cond = i > 0 && ch[n].lbrr_flags[i - 1]
                                   ? CODE_CONDITIONALLY
                                   : CODE_INDEPENDENTLY;
                    decode_indices(&ch[n], r, i, 1, cond);
                    decode_pulses(r, pulses, ch[n].ind.signal_type,
                                  ch[n].ind.quant_offset, ch[n].frame_length);
                }
    }
    if (stream_ch == 2) {
        if (!lost) {
            stereo_pred(r, pred);
            if (ch[1].vad[ch[0].frames_decoded] == 0)
                mid_only = rc_icdf(r, SILK_STEREO_ONLY_CODE_MID_ICDF, 8);
            else
                mid_only = 0;
        } else {
            pred[0] = d->pred_prev[0];
            pred[1] = d->pred_prev[1];
        }
    }
    if (stream_ch == 2 && mid_only == 0 && d->prev_mid_only == 1) {
        memset(ch[1].out, 0, sizeof ch[1].out);
        memset(ch[1].slpc, 0, sizeof ch[1].slpc);
        ch[1].lag_prev = 100;
        ch[1].last_gain = 10;
        ch[1].prev_type = TYPE_NO_VOICE;
        ch[1].first_after_reset = 1;
    }
    int16_t buf[2][MAX_FRAME + 2];
    int has_side = lost ? !d->prev_mid_only : !mid_only;
    int n_dec = ch[0].frame_length;
    for (int n = 0; n < stream_ch; n++) {
        if (n == 0 || has_side) {
            int fi = ch[0].frames_decoded - n, cond;
            if (fi <= 0)
                cond = CODE_INDEPENDENTLY;
            else if (n > 0 && d->prev_mid_only)
                cond = CODE_INDEPENDENTLY_NO_LTP_SCALING;
            else
                cond = CODE_CONDITIONALLY;
            decode_frame(&ch[n], r, &buf[n][2], lost, cond);
            n_dec = ch[n].frame_length;
        } else {
            memset(&buf[n][2], 0, (size_t)n_dec * sizeof(int16_t));
        }
        ch[n].frames_decoded++;
    }
    if (out_ch == 2 && stream_ch == 2) {
        ms_to_lr(d, buf[0], buf[1], pred, ch[0].fs_khz, n_dec);
    } else {
        memcpy(buf[0], d->smid, 2 * sizeof(int16_t));
        memcpy(d->smid, &buf[0][n_dec], 2 * sizeof(int16_t));
    }
    int n_out = n_dec * (d->api_hz / 1000) / ch[0].fs_khz;
    int16_t res[2880];
    int nch = out_ch < stream_ch ? out_ch : stream_ch;
    for (int n = 0; n < nch; n++) {
        resample(&ch[n].rs, res, &buf[n][1], n_dec);
        for (int i = 0; i < n_out; i++)
            out[n][i] = (1.f / 32768) * res[i];
    }
    if (out_ch == 2 && stream_ch == 1) {
        if (stereo_to_mono) {
            resample(&ch[1].rs, res, &buf[0][1], n_dec);
            for (int i = 0; i < n_out; i++)
                out[1][i] = (1.f / 32768) * res[i];
        } else {
            memcpy(out[1], out[0], sizeof(float) * (size_t)n_out);
        }
    }
    if (lost)
        for (int i = 0; i < d->int_ch; i++)
            ch[i].last_gain = 10;
    else
        d->prev_mid_only = mid_only;
    return n_out;
}
