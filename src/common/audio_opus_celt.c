/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_opus_celt.c - the CELT layer of Opus (RFC 6716 section 4.3;
 * audio_opus.h), as the reference decoder computes it in floating point:
 * silence, the post-filter's parameters, transients, the coarse energy
 * (Laplace-coded, predicted in time and frequency), the time-frequency
 * resolution changes, the spreading, the dynamic allocation boosts and the
 * allocation trim, the bit allocation (interpolated between the allocation
 * vectors, bands skipped from the top, intensity and dual stereo, fine
 * energy bits), the fine energy, the shapes of the bands (pyramid vector
 * quantization: theta splits, mid/side stereo, folding from lower bands,
 * the pulses' combinatorial index, spreading rotations, Haar and Hadamard
 * reshuffles), the final energy bits, anti-collapse, the bands scaled by
 * their energies, the inverse MDCT with its short overlap (a DCT-IV on
 * audio_fft.c, mirrored for the overlap), the pitch post-filter and the
 * de-emphasis.
 */
#include "audio_opus.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "audio_fft.h"
#include "audio_opus_tab.h"

#define NB 21    /* bands */
#define OV 120   /* overlap */
#define DBS 2048 /* the decoder's memory a channel */
#define BITRES 3
#define MAX_FINE_BITS 8
#define FINE_OFFSET 21
#define QTHETA_OFFSET 4
#define QTHETA_OFFSET_TWOPHASE 16
#define SPREAD_NONE 0
#define SPREAD_NORMAL 2
#define SPREAD_AGGRESSIVE 3
#define PVQ_N 180
#define PVQ_K 132

struct celt_dec {
    int start, end, cc;
    float mem[2][DBS + OV];
    float old_e[2 * NB], old_loge[2 * NB], old_loge2[2 * NB], bg_loge[2 * NB];
    float preemph[2];
    int pf_period, pf_period_old, pf_tapset, pf_tapset_old;
    float pf_gain, pf_gain_old;
    uint32_t rng;
    struct aimdct *mdct[4]; /* 240, 480, 960, 1920 points */
    uint32_t *pvq;          /* U(a, b), a <= b: pvq[a * PVQ_N + b] */
    float x[2 * 960], freq[960], freq2[960], u[960];
    /* concealment: frames concealed since the last decoded one (in 2.5 ms
       units), whether the pitch-based kind is not to be used yet, its last
       pitch, the overlap still to be folded with the next frame, the LPC
       of the concealed signal */
    int loss_duration, skip_plc, last_pitch, prefilter_and_fold;
    float lpc[2][24];
    float norm[2 * 8 * 100]; /* the bands as folded from */
};

/* ---- small helpers ---- */

static inline int imin(int a, int b)
{
    return a < b ? a : b;
}

static inline int imax(int a, int b)
{
    return a > b ? a : b;
}

static int frac_mul16(int a, int b)
{
    return (16384 + (int32_t)(int16_t)a * (int16_t)b) >> 15;
}

static int bitexact_cos(int x)
{
    int32_t tmp = (4096 + (int32_t)x * x) >> 13;
    int x2 = tmp;
    x2 =
        (32767 - x2) +
        frac_mul16(x2, (-7651 + frac_mul16(x2, (8277 + frac_mul16(-626, x2)))));
    return 1 + x2;
}

static int bitexact_log2tan(int isin, int icos)
{
    int lc = rc_ilog((uint32_t)icos), ls = rc_ilog((uint32_t)isin);
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    return (ls - lc) * (1 << 11) +
           frac_mul16(isin, frac_mul16(isin, -2597) + 7932) -
           frac_mul16(icos, frac_mul16(icos, -2597) + 7932);
}

static uint32_t lcg(uint32_t s)
{
    return 1664525 * s + 1013904223;
}

static unsigned isqrt32(uint32_t v)
{
    unsigned r = (unsigned)sqrt((double)v);
    while ((uint64_t)r * r > v)
        r--;
    while ((uint64_t)(r + 1) * (r + 1) <= v)
        r++;
    return r;
}

static uint32_t pvq_u(const struct celt_dec *d, int a, int b)
{
    if (a > b) {
        int t = a;
        a = b;
        b = t;
    }
    return d->pvq[a * PVQ_N + b];
}

/* ---- the decoder ---- */

struct celt_dec *jcelt_new(int channels)
{
    struct celt_dec *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->cc = channels;
    d->end = NB;
    for (int i = 0; i < 4; i++)
        if (!(d->mdct[i] = aimdct_new(240 << i))) {
            jcelt_free(d);
            return NULL;
        }
    d->pvq = calloc((size_t)PVQ_K * PVQ_N, sizeof *d->pvq);
    if (!d->pvq) {
        jcelt_free(d);
        return NULL;
    }
    /* U(n, k) = U(n - 1, k) + U(n, k - 1) + U(n - 1, k - 1), U(0, 0) = 1 */
    uint64_t row[2][PVQ_N + 2];
    uint64_t *prev = row[0], *cur = row[1];
    for (int k = 0; k < PVQ_N; k++)
        prev[k] = k == 0;
    for (int a = 0; a < PVQ_K; a++) {
        for (int b = a; b < PVQ_N; b++)
            d->pvq[a * PVQ_N + b] =
                prev[b] > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)prev[b];
        /* the next row from this one */
        cur[0] = 0;
        for (int k = 1; k < PVQ_N; k++) {
            uint64_t v = prev[k] + cur[k - 1] + prev[k - 1];
            cur[k] = v > (1ull << 40) ? (1ull << 40) : v;
        }
        uint64_t *t = prev;
        prev = cur;
        cur = t;
    }
    jcelt_reset(d);
    return d;
}

void jcelt_free(struct celt_dec *d)
{
    if (!d)
        return;
    for (int i = 0; i < 4; i++)
        aimdct_free(d->mdct[i]);
    free(d->pvq);
    free(d);
}

void jcelt_reset(struct celt_dec *d)
{
    memset(d->mem, 0, sizeof d->mem);
    memset(d->old_e, 0, sizeof d->old_e);
    memset(d->bg_loge, 0, sizeof d->bg_loge);
    for (int i = 0; i < 2 * NB; i++)
        d->old_loge[i] = d->old_loge2[i] = -28.f;
    d->preemph[0] = d->preemph[1] = 0;
    d->pf_period = d->pf_period_old = d->pf_tapset = d->pf_tapset_old = 0;
    d->pf_gain = d->pf_gain_old = 0;
    d->rng = 0;
    d->loss_duration = 0;
    d->skip_plc = 1;
    d->last_pitch = 0;
    d->prefilter_and_fold = 0;
    memset(d->lpc, 0, sizeof d->lpc);
}

void jcelt_bands(struct celt_dec *d, int start, int end)
{
    d->start = start;
    d->end = end;
}

uint32_t jcelt_rng(const struct celt_dec *d)
{
    return d->rng;
}

/* ---- energies ---- */

static int laplace(struct rc *r, unsigned fs, int decay)
{
    int val = 0;
    unsigned fl = 0, fm = rc_decode_bin(r, 15);
    if (fm >= fs) {
        val++;
        fl = fs;
        fs = ((32768 - 2 * 16 - fs) * (unsigned)(16384 - decay) >> 15) + 1;
        while (fs > 1 && fm >= fl + 2 * fs) {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2) * (unsigned)decay >> 15) + 1;
            val++;
        }
        if (fs <= 1) {
            int di = (int)((fm - fl) >> 1);
            val += di;
            fl += 2 * (unsigned)di;
        }
        if (fm < fl + fs)
            val = -val;
        else
            fl += fs;
    }
    rc_update(r, fl, fl + fs < 32768 ? fl + fs : 32768, 32768);
    return val;
}

static void coarse_energy(struct celt_dec *d, struct rc *r, int intra, int C,
                          int LM)
{
    static const float PRED[4] = {29440 / 32768.f, 26112 / 32768.f,
                                  21248 / 32768.f, 16384 / 32768.f};
    static const float BETA[4] = {30147 / 32768.f, 22282 / 32768.f,
                                  12124 / 32768.f, 6554 / 32768.f};
    static const uint8_t SMALL[3] = {2, 1, 0};
    const uint8_t *prob = CELT_EPROB + (LM * 2 + intra) * 42;
    float prev[2] = {0, 0};
    float coef = intra ? 0 : PRED[LM], beta = intra ? 4915 / 32768.f : BETA[LM];
    int budget = (int)r->storage * 8;
    for (int i = d->start; i < d->end; i++)
        for (int c = 0; c < C; c++) {
            int tell = rc_tell(r), qi;
            if (budget - tell >= 15) {
                int pi = 2 * imin(i, 20);
                qi = laplace(r, (unsigned)prob[pi] << 7, prob[pi + 1] << 6);
            } else if (budget - tell >= 2) {
                qi = rc_icdf(r, SMALL, 2);
                qi = (qi >> 1) ^ -(qi & 1);
            } else if (budget - tell >= 1) {
                qi = -rc_bit_logp(r, 1);
            } else {
                qi = -1;
            }
            float q = (float)qi, *e = &d->old_e[i + c * NB];
            *e = *e > -9.f ? *e : -9.f;
            *e = coef * *e + prev[c] + q;
            prev[c] = prev[c] + q - beta * q;
        }
}

static void tf_decode(struct rc *r, int start, int end, int transient, int *tf,
                      int LM)
{
    uint32_t budget = r->storage * 8, tell = (uint32_t)rc_tell(r);
    int logp = transient ? 2 : 4, changed = 0, curr = 0;
    int rsv = LM > 0 && tell + (uint32_t)logp + 1 <= budget;
    budget -= (uint32_t)rsv;
    for (int i = start; i < end; i++) {
        if (tell + (uint32_t)logp <= budget) {
            curr ^= rc_bit_logp(r, (unsigned)logp);
            tell = (uint32_t)rc_tell(r);
            changed |= curr;
        }
        tf[i] = curr;
        logp = transient ? 4 : 5;
    }
    int sel = 0;
    const int8_t *t = CELT_TF_SELECT + 8 * LM;
    if (rsv && t[4 * transient + changed] != t[4 * transient + 2 + changed])
        sel = rc_bit_logp(r, 1);
    for (int i = start; i < end; i++)
        tf[i] = t[4 * transient + 2 * sel + tf[i]];
}

/* ---- the allocation ---- */

static const uint8_t LOG2_FRAC[24] = {0,  8,  13, 16, 19, 21, 23, 24,
                                      26, 27, 28, 29, 30, 31, 32, 32,
                                      33, 34, 34, 35, 36, 36, 37, 37};

static const uint8_t *cache_of(int LM, int band)
{
    return CELT_CACHE_BITS + CELT_CACHE_INDEX[(LM + 1) * NB + band];
}

static int bits2pulses(int band, int LM, int bits)
{
    const uint8_t *cache = cache_of(LM, band);
    int lo = 0, hi = cache[0];
    bits--;
    for (int i = 0; i < 6; i++) {
        int mid = (lo + hi + 1) >> 1;
        if ((int)cache[mid] >= bits)
            hi = mid;
        else
            lo = mid;
    }
    if (bits - (lo == 0 ? -1 : (int)cache[lo]) <= (int)cache[hi] - bits)
        return lo;
    return hi;
}

static int pulses2bits(int band, int LM, int pulses)
{
    return pulses == 0 ? 0 : cache_of(LM, band)[pulses] + 1;
}

static int get_pulses(int i)
{
    return i < 8 ? i : (8 + (i & 7)) << ((i >> 3) - 1);
}

static int band_w(int j)
{
    return CELT_EBANDS[j + 1] - CELT_EBANDS[j];
}

static int interp_bits2pulses(struct rc *r, int start, int end, int skip_start,
                              const int *bits1, const int *bits2,
                              const int *thresh, const int *cap, int32_t total,
                              int32_t *balance_out, int skip_rsv,
                              int *intensity, int intensity_rsv,
                              int *dual_stereo, int dual_stereo_rsv, int *bits,
                              int *ebits, int *fine_priority, int C, int LM)
{
    int alloc_floor = C << BITRES, stereo = C > 1, logM = LM << BITRES;
    int lo = 0, hi = 1 << 6, coded;
    int32_t psum;
    for (int i = 0; i < 6; i++) {
        int mid = (lo + hi) >> 1, done = 0;
        psum = 0;
        for (int j = end; j-- > start;) {
            int tmp = bits1[j] + (mid * bits2[j] >> 6);
            if (tmp >= thresh[j] || done) {
                done = 1;
                psum += imin(tmp, cap[j]);
            } else if (tmp >= alloc_floor) {
                psum += alloc_floor;
            }
        }
        if (psum > total)
            hi = mid;
        else
            lo = mid;
    }
    psum = 0;
    int done = 0;
    for (int j = end; j-- > start;) {
        int tmp = bits1[j] + (lo * bits2[j] >> 6);
        if (tmp < thresh[j] && !done)
            tmp = tmp >= alloc_floor ? alloc_floor : 0;
        else
            done = 1;
        tmp = imin(tmp, cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }
    for (coded = end;; coded--) {
        int j = coded - 1;
        if (j <= skip_start) {
            total += skip_rsv;
            break;
        }
        int32_t left = total - psum;
        int span = CELT_EBANDS[coded] - CELT_EBANDS[start];
        int32_t percoeff = left / span;
        left -= span * percoeff;
        int rem = imax((int)left - (CELT_EBANDS[j] - CELT_EBANDS[start]), 0);
        int bw = CELT_EBANDS[coded] - CELT_EBANDS[j];
        int band_bits = (int)(bits[j] + percoeff * bw + rem);
        if (band_bits >= imax(thresh[j], alloc_floor + (1 << BITRES))) {
            if (rc_bit_logp(r, 1))
                break;
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if (intensity_rsv > 0)
            intensity_rsv = LOG2_FRAC[j - start];
        psum += intensity_rsv;
        if (band_bits >= alloc_floor) {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
    }
    if (intensity_rsv > 0)
        *intensity = start + (int)rc_uint(r, (uint32_t)(coded + 1 - start));
    else
        *intensity = 0;
    if (*intensity <= start) {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    *dual_stereo = dual_stereo_rsv > 0 ? rc_bit_logp(r, 1) : 0;
    int32_t left = total - psum;
    int span = CELT_EBANDS[coded] - CELT_EBANDS[start];
    int32_t percoeff = left / span;
    left -= span * percoeff;
    for (int j = start; j < coded; j++)
        bits[j] += (int)percoeff * band_w(j);
    for (int j = start; j < coded; j++) {
        int tmp = (int)(left < band_w(j) ? left : band_w(j));
        bits[j] += tmp;
        left -= tmp;
    }
    int32_t balance = 0;
    int j;
    for (j = start; j < coded; j++) {
        int N0 = band_w(j), N = N0 << LM;
        int32_t bit = bits[j] + balance, excess;
        if (N > 1) {
            excess = bit - cap[j] > 0 ? bit - cap[j] : 0;
            bits[j] = (int)(bit - excess);
            int den =
                C * N + (C == 2 && N > 2 && !*dual_stereo && j < *intensity);
            int nclogn = den * (CELT_LOGN[j] + logM);
            int offset = (nclogn >> 1) - den * FINE_OFFSET;
            if (N == 2)
                offset += den << BITRES >> 2;
            if (bits[j] + offset < den * 2 << BITRES)
                offset += nclogn >> 2;
            else if (bits[j] + offset < den * 3 << BITRES)
                offset += nclogn >> 3;
            ebits[j] = imax(0, bits[j] + offset + (den << (BITRES - 1)));
            ebits[j] = (ebits[j] / den) >> BITRES;
            if (C * ebits[j] > (bits[j] >> BITRES))
                ebits[j] = bits[j] >> stereo >> BITRES;
            ebits[j] = imin(ebits[j], MAX_FINE_BITS);
            fine_priority[j] = ebits[j] * (den << BITRES) >= bits[j] + offset;
            bits[j] -= C * ebits[j] << BITRES;
        } else {
            excess = bit - (C << BITRES) > 0 ? bit - (C << BITRES) : 0;
            bits[j] = (int)(bit - excess);
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        if (excess > 0) {
            int extra_fine = imin((int)(excess >> (stereo + BITRES)),
                                  MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            int extra_bits = extra_fine * C << BITRES;
            fine_priority[j] = extra_bits >= excess - balance;
            excess -= extra_bits;
        }
        balance = excess;
    }
    *balance_out = balance;
    for (; j < end; j++) {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = ebits[j] < 1;
    }
    return coded;
}

static int allocation(struct rc *r, int start, int end, const int *offsets,
                      const int *cap, int trim, int *intensity, int *dual,
                      int32_t total, int32_t *balance, int *pulses, int *ebits,
                      int *fine_priority, int C, int LM)
{
    int bits1[NB], bits2[NB], thresh[NB], trim_off[NB];
    total = total > 0 ? total : 0;
    int skip_start = start;
    int skip_rsv = total >= 1 << BITRES ? 1 << BITRES : 0;
    total -= skip_rsv;
    int intensity_rsv = 0, dual_rsv = 0;
    if (C == 2) {
        intensity_rsv = LOG2_FRAC[end - start];
        if (intensity_rsv > total) {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_rsv = total >= 1 << BITRES ? 1 << BITRES : 0;
            total -= dual_rsv;
        }
    }
    for (int j = start; j < end; j++) {
        thresh[j] = imax(C << BITRES, (3 * band_w(j) << LM << BITRES) >> 4);
        trim_off[j] = C * band_w(j) * (trim - 5 - LM) * (end - j - 1) *
                          (1 << (LM + BITRES)) >>
                      6;
        if (band_w(j) << LM == 1)
            trim_off[j] -= C << BITRES;
    }
    int lo = 1, hi = 11 - 1;
    do {
        int done = 0, psum = 0, mid = (lo + hi) >> 1;
        for (int j = end; j-- > start;) {
            int b = C * band_w(j) * CELT_ALLOC[mid * NB + j] << LM >> 2;
            if (b > 0)
                b = imax(0, b + trim_off[j]);
            b += offsets[j];
            if (b >= thresh[j] || done) {
                done = 1;
                psum += imin(b, cap[j]);
            } else if (b >= C << BITRES) {
                psum += C << BITRES;
            }
        }
        if (psum > total)
            hi = mid - 1;
        else
            lo = mid + 1;
    } while (lo <= hi);
    hi = lo--;
    for (int j = start; j < end; j++) {
        int b1 = C * band_w(j) * CELT_ALLOC[lo * NB + j] << LM >> 2;
        int b2 = hi >= 11 ? cap[j]
                          : C * band_w(j) * CELT_ALLOC[hi * NB + j] << LM >> 2;
        if (b1 > 0)
            b1 = imax(0, b1 + trim_off[j]);
        if (b2 > 0)
            b2 = imax(0, b2 + trim_off[j]);
        if (lo > 0)
            b1 += offsets[j];
        b2 += offsets[j];
        if (offsets[j] > 0)
            skip_start = j;
        b2 = imax(0, b2 - b1);
        bits1[j] = b1;
        bits2[j] = b2;
    }
    return interp_bits2pulses(r, start, end, skip_start, bits1, bits2, thresh,
                              cap, total, balance, skip_rsv, intensity,
                              intensity_rsv, dual, dual_rsv, pulses, ebits,
                              fine_priority, C, LM);
}

/* ---- the shapes ---- */

static void exp_rotation1(float *X, int len, int stride, float c, float s)
{
    float ms = -s, *x = X;
    for (int i = 0; i < len - stride; i++) {
        float x1 = x[0], x2 = x[stride];
        x[stride] = c * x2 + s * x1;
        *x++ = c * x1 + ms * x2;
    }
    x = &X[len - 2 * stride - 1];
    for (int i = len - 2 * stride - 1; i >= 0; i--) {
        float x1 = x[0], x2 = x[stride];
        x[stride] = c * x2 + s * x1;
        *x-- = c * x1 + ms * x2;
    }
}

static void exp_rotation(float *X, int len, int stride, int K, int spread)
{
    static const int FACTOR[3] = {15, 10, 5};
    if (2 * K >= len || spread == SPREAD_NONE)
        return;
    int factor = FACTOR[spread - 1], stride2 = 0;
    float gain = (float)len / (float)(len + factor * K);
    float theta = .5f * gain * gain;
    float c = cosf(.5f * (float)M_PI * theta);
    float s = cosf(.5f * (float)M_PI * (1.f - theta));
    if (len >= 8 * stride) {
        stride2 = 1;
        while ((stride2 * stride2 + stride2) * stride + (stride >> 2) < len)
            stride2++;
    }
    len /= stride;
    for (int i = 0; i < stride; i++) {
        if (stride2)
            exp_rotation1(X + i * len, len, stride2, s, c);
        exp_rotation1(X + i * len, len, 1, c, s);
    }
}

static void renormalise(float *X, int N, float gain)
{
    float E = 1e-15f;
    for (int i = 0; i < N; i++)
        E += X[i] * X[i];
    float g = gain * (1.f / sqrtf(E));
    for (int i = 0; i < N; i++)
        X[i] *= g;
}

/* the vector of pulses with index i among those of n dimensions and k
   pulses (the reference's cwrsi), into y: their squared norm */
static float cwrsi(const struct celt_dec *d, int n, int k, uint32_t i, int *y)
{
    float yy = 0;
    while (n > 2) {
        uint32_t p, q;
        int s, k0, val;
        if (k >= n) {
            p = pvq_u(d, n, k + 1);
            s = -(i >= p);
            i -= p & (uint32_t)s;
            k0 = k;
            q = pvq_u(d, n, n);
            if (q > i) {
                k = n;
                do
                    p = pvq_u(d, --k, n);
                while (p > i);
            } else {
                for (p = pvq_u(d, n, k); p > i; p = pvq_u(d, n, k))
                    k--;
            }
            i -= p;
            val = (k0 - k + s) ^ s;
        } else {
            p = pvq_u(d, k, n);
            q = pvq_u(d, k + 1, n);
            if (p <= i && i < q) {
                i -= p;
                val = 0;
            } else {
                s = -(i >= q);
                i -= q & (uint32_t)s;
                k0 = k;
                do
                    p = pvq_u(d, --k, n);
                while (p > i);
                i -= p;
                val = (k0 - k + s) ^ s;
            }
        }
        *y++ = val;
        yy += (float)val * val;
        n--;
    }
    uint32_t p = 2 * (uint32_t)k + 1;
    int s = -(i >= p);
    i -= p & (uint32_t)s;
    int k0 = k;
    k = (int)((i + 1) >> 1);
    if (k)
        i -= 2 * (uint32_t)k - 1;
    int val = (k0 - k + s) ^ s;
    *y++ = val;
    yy += (float)val * val;
    s = -(int)i;
    val = (k + s) ^ s;
    *y = val;
    yy += (float)val * val;
    return yy;
}

static unsigned alg_unquant(const struct celt_dec *d, float *X, int N, int K,
                            int spread, int B, struct rc *r, float gain)
{
    int iy[176];
    uint32_t idx = rc_uint(r, pvq_u(d, N, K) + pvq_u(d, N, K + 1));
    float ryy = cwrsi(d, N, K, idx, iy);
    float g = gain * (1.f / sqrtf(ryy));
    for (int i = 0; i < N; i++)
        X[i] = g * (float)iy[i];
    exp_rotation(X, N, B, K, spread);
    if (B <= 1)
        return 1;
    int N0 = N / B;
    unsigned mask = 0;
    for (int i = 0; i < B; i++) {
        int t = 0;
        for (int j = 0; j < N0; j++)
            t |= iy[i * N0 + j];
        mask |= (unsigned)(t != 0) << i;
    }
    return mask;
}

static void haar1(float *X, int N0, int stride)
{
    N0 >>= 1;
    for (int i = 0; i < stride; i++)
        for (int j = 0; j < N0; j++) {
            float a = .70710678f * X[stride * 2 * j + i];
            float b = .70710678f * X[stride * (2 * j + 1) + i];
            X[stride * 2 * j + i] = a + b;
            X[stride * (2 * j + 1) + i] = a - b;
        }
}

static void deinterleave(float *X, int N0, int stride, int hadamard)
{
    float tmp[960];
    int N = N0 * stride;
    for (int i = 0; i < stride; i++)
        for (int j = 0; j < N0; j++) {
            int to = hadamard ? CELT_ORDERY[stride - 2 + i] : i;
            tmp[to * N0 + j] = X[j * stride + i];
        }
    memcpy(X, tmp, sizeof(float) * (size_t)N);
}

static void interleave(float *X, int N0, int stride, int hadamard)
{
    float tmp[960];
    int N = N0 * stride;
    for (int i = 0; i < stride; i++)
        for (int j = 0; j < N0; j++) {
            int from = hadamard ? CELT_ORDERY[stride - 2 + i] : i;
            tmp[j * stride + i] = X[from * N0 + j];
        }
    memcpy(X, tmp, sizeof(float) * (size_t)N);
}

static int compute_qn(int N, int b, int offset, int pulse_cap, int stereo)
{
    static const int16_t EXP2[8] = {16384, 17866, 19483, 21247,
                                    23170, 25267, 27554, 30048};
    int N2 = 2 * N - 1;
    if (stereo && N == 2)
        N2--;
    int qb = (b + N2 * offset) / N2;
    qb = imin(b - pulse_cap - (4 << BITRES), qb);
    qb = imin(8 << BITRES, qb);
    if (qb < (1 << BITRES >> 1))
        return 1;
    int qn = EXP2[qb & 7] >> (14 - (qb >> BITRES));
    return (qn + 1) >> 1 << 1;
}

struct bctx {
    struct celt_dec *d;
    struct rc *r;
    int i, intensity, spread, tf_change, avoid_split_noise;
    int32_t remaining;
    uint32_t seed;
};

struct split {
    int inv, imid, iside, delta, itheta, qalloc;
};

static void compute_theta(struct bctx *c, struct split *s, int N, int *b, int B,
                          int B0, int LM, int stereo, int *fill)
{
    struct rc *r = c->r;
    int i = c->i, itheta = 0, inv = 0;
    int pulse_cap = CELT_LOGN[i] + LM * (1 << BITRES);
    int offset = (pulse_cap >> 1) -
                 (stereo && N == 2 ? QTHETA_OFFSET_TWOPHASE : QTHETA_OFFSET);
    int qn = compute_qn(N, *b, offset, pulse_cap, stereo);
    if (stereo && i >= c->intensity)
        qn = 1;
    uint32_t tell = rc_tell_frac(r);
    if (qn != 1) {
        if (stereo && N > 2) {
            int p0 = 3, x0 = qn / 2, ft = p0 * (x0 + 1) + x0, x;
            int fs = (int)rc_decode(r, (unsigned)ft);
            if (fs < (x0 + 1) * p0)
                x = fs / p0;
            else
                x = x0 + 1 + (fs - (x0 + 1) * p0);
            rc_update(
                r, (unsigned)(x <= x0 ? p0 * x : (x - 1 - x0) + (x0 + 1) * p0),
                (unsigned)(x <= x0 ? p0 * (x + 1) : (x - x0) + (x0 + 1) * p0),
                (unsigned)ft);
            itheta = x;
        } else if (B0 > 1 || stereo) {
            itheta = (int)rc_uint(r, (uint32_t)qn + 1);
        } else {
            int ft = ((qn >> 1) + 1) * ((qn >> 1) + 1), fs, fl;
            int fm = (int)rc_decode(r, (unsigned)ft);
            if (fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1)) {
                itheta = (int)(isqrt32(8 * (uint32_t)fm + 1) - 1) >> 1;
                fs = itheta + 1;
                fl = itheta * (itheta + 1) >> 1;
            } else {
                itheta = (int)(2 * (qn + 1) -
                               (int)isqrt32(8 * (uint32_t)(ft - fm - 1) + 1)) >>
                         1;
                fs = qn + 1 - itheta;
                fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
            }
            rc_update(r, (unsigned)fl, (unsigned)(fl + fs), (unsigned)ft);
        }
        itheta = itheta * 16384 / qn;
    } else if (stereo) {
        inv = *b > 2 << BITRES && c->remaining > 2 << BITRES ? rc_bit_logp(r, 2)
                                                             : 0;
        itheta = 0;
    }
    int qalloc = (int)(rc_tell_frac(r) - tell);
    *b -= qalloc;
    int imid, iside, delta;
    if (itheta == 0) {
        imid = 32767;
        iside = 0;
        *fill &= (1 << B) - 1;
        delta = -16384;
    } else if (itheta == 16384) {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << B) - 1) << B;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta);
        iside = bitexact_cos(16384 - itheta);
        delta = frac_mul16((N - 1) << 7, bitexact_log2tan(iside, imid));
    }
    s->inv = inv;
    s->imid = imid;
    s->iside = iside;
    s->delta = delta;
    s->itheta = itheta;
    s->qalloc = qalloc;
}

static unsigned band_n1(struct bctx *c, float *X, float *Y, float *lowband_out)
{
    float *x = X;
    for (int k = 0; k < 1 + (Y != NULL); k++) {
        int sign = 0;
        if (c->remaining >= 1 << BITRES) {
            sign = (int)rc_bits(c->r, 1);
            c->remaining -= 1 << BITRES;
        }
        x[0] = sign ? -1.f : 1.f;
        x = Y;
    }
    if (lowband_out)
        lowband_out[0] = X[0];
    return 1;
}

static unsigned partition(struct bctx *c, float *X, int N, int b, int B,
                          float *lowband, int LM, float gain, int fill)
{
    int i = c->i, B0 = B;
    unsigned cm = 0;
    const uint8_t *cache = cache_of(LM, i);
    if (LM != -1 && b > cache[cache[0]] + 12 && N > 2) {
        struct split s;
        float *Y, *next_lowband2 = NULL;
        N >>= 1;
        Y = X + N;
        LM -= 1;
        if (B == 1)
            fill = (fill & 1) | (fill << 1);
        B = (B + 1) >> 1;
        compute_theta(c, &s, N, &b, B, B0, LM, 0, &fill);
        float mid = (1.f / 32768) * s.imid, side = (1.f / 32768) * s.iside;
        int delta = s.delta;
        if (B0 > 1 && (s.itheta & 0x3fff)) {
            if (s.itheta > 8192)
                delta -= delta >> (4 - LM);
            else
                delta = imin(0, delta + (N << BITRES >> (5 - LM)));
        }
        int mbits = imax(0, imin(b, (b - delta) / 2)), sbits = b - mbits;
        c->remaining -= s.qalloc;
        if (lowband)
            next_lowband2 = lowband + N;
        int32_t rebalance = c->remaining;
        if (mbits >= sbits) {
            cm = partition(c, X, N, mbits, B, lowband, LM, gain * mid, fill);
            rebalance = mbits - (rebalance - c->remaining);
            if (rebalance > 3 << BITRES && s.itheta != 0)
                sbits += (int)rebalance - (3 << BITRES);
            cm |= partition(c, Y, N, sbits, B, next_lowband2, LM, gain * side,
                            fill >> B)
                  << (B0 >> 1);
        } else {
            cm = partition(c, Y, N, sbits, B, next_lowband2, LM, gain * side,
                           fill >> B)
                 << (B0 >> 1);
            rebalance = sbits - (rebalance - c->remaining);
            if (rebalance > 3 << BITRES && s.itheta != 16384)
                mbits += (int)rebalance - (3 << BITRES);
            cm |= partition(c, X, N, mbits, B, lowband, LM, gain * mid, fill);
        }
        return cm;
    }
    int q = bits2pulses(i, LM, b), curr = pulses2bits(i, LM, q);
    c->remaining -= curr;
    while (c->remaining < 0 && q > 0) {
        c->remaining += curr;
        q--;
        curr = pulses2bits(i, LM, q);
        c->remaining -= curr;
    }
    if (q != 0)
        return alg_unquant(c->d, X, N, get_pulses(q), c->spread, B, c->r, gain);
    unsigned mask = (unsigned)(1UL << B) - 1;
    fill &= (int)mask;
    if (!fill) {
        memset(X, 0, sizeof(float) * (size_t)N);
        return 0;
    }
    if (!lowband) {
        for (int j = 0; j < N; j++) {
            c->seed = lcg(c->seed);
            X[j] = (float)((int32_t)c->seed >> 20);
        }
        cm = mask;
    } else {
        for (int j = 0; j < N; j++) {
            c->seed = lcg(c->seed);
            X[j] = lowband[j] + (c->seed & 0x8000 ? 1.f / 256 : -1.f / 256);
        }
        cm = (unsigned)fill;
    }
    renormalise(X, N, gain);
    return cm;
}

static unsigned quant_band(struct bctx *c, float *X, int N, int b, int B,
                           float *lowband, int LM, float *lowband_out,
                           float gain, float *scratch, int fill)
{
    int N0 = N, N_B = N, B0 = B, time_divide = 0, recombine = 0;
    int long_blocks = B0 == 1, tf_change = c->tf_change;
    N_B /= B;
    if (N == 1)
        return band_n1(c, X, NULL, lowband_out);
    if (tf_change > 0)
        recombine = tf_change;
    if (scratch && lowband &&
        (recombine || ((N_B & 1) == 0 && tf_change < 0) || B0 > 1)) {
        memcpy(scratch, lowband, sizeof(float) * (size_t)N);
        lowband = scratch;
    }
    for (int k = 0; k < recombine; k++) {
        static const uint8_t T[16] = {0, 1, 1, 1, 2, 3, 3, 3,
                                      2, 3, 3, 3, 2, 3, 3, 3};
        if (lowband)
            haar1(lowband, N >> k, 1 << k);
        fill = T[fill & 0xF] | T[fill >> 4] << 2;
    }
    B >>= recombine;
    N_B <<= recombine;
    while ((N_B & 1) == 0 && tf_change < 0) {
        if (lowband)
            haar1(lowband, N_B, B);
        fill |= fill << B;
        B <<= 1;
        N_B >>= 1;
        time_divide++;
        tf_change++;
    }
    B0 = B;
    int N_B0 = N_B;
    if (B0 > 1 && lowband)
        deinterleave(lowband, N_B >> recombine, B0 << recombine, long_blocks);
    unsigned cm = partition(c, X, N, b, B, lowband, LM, gain, fill);
    if (B0 > 1)
        interleave(X, N_B >> recombine, B0 << recombine, long_blocks);
    N_B = N_B0;
    B = B0;
    for (int k = 0; k < time_divide; k++) {
        B >>= 1;
        N_B <<= 1;
        cm |= cm >> B;
        haar1(X, N_B, B);
    }
    for (int k = 0; k < recombine; k++) {
        static const uint8_t T[16] = {0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33,
                                      0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF,
                                      0xF0, 0xF3, 0xFC, 0xFF};
        cm = T[cm];
        haar1(X, N0 >> k, 1 << k);
    }
    B <<= recombine;
    if (lowband_out) {
        float n = sqrtf((float)N0);
        for (int j = 0; j < N0; j++)
            lowband_out[j] = n * X[j];
    }
    return cm & ((1u << B) - 1);
}

static void stereo_merge(float *X, float *Y, float mid, int N)
{
    float xp = 0, side = 0;
    for (int j = 0; j < N; j++) {
        xp += Y[j] * X[j];
        side += Y[j] * Y[j];
    }
    xp *= mid;
    float El = mid * mid + side - 2 * xp, Er = mid * mid + side + 2 * xp;
    if (Er < 6e-4f || El < 6e-4f) {
        memcpy(Y, X, sizeof(float) * (size_t)N);
        return;
    }
    float lgain = 1.f / sqrtf(El), rgain = 1.f / sqrtf(Er);
    for (int j = 0; j < N; j++) {
        float l = mid * X[j], r = Y[j];
        X[j] = lgain * (l - r);
        Y[j] = rgain * (l + r);
    }
}

static unsigned band_stereo(struct bctx *c, float *X, float *Y, int N, int b,
                            int B, float *lowband, int LM, float *lowband_out,
                            float *scratch, int fill)
{
    if (N == 1)
        return band_n1(c, X, Y, lowband_out);
    int orig_fill = fill;
    struct split s;
    unsigned cm;
    compute_theta(c, &s, N, &b, B, B, LM, 1, &fill);
    float mid = (1.f / 32768) * s.imid, side = (1.f / 32768) * s.iside;
    if (N == 2) {
        int mbits = b, sbits = 0, sign = 0;
        if (s.itheta != 0 && s.itheta != 16384)
            sbits = 1 << BITRES;
        mbits -= sbits;
        int ch = s.itheta > 8192;
        c->remaining -= s.qalloc + sbits;
        float *x2 = ch ? Y : X, *y2 = ch ? X : Y;
        if (sbits)
            sign = (int)rc_bits(c->r, 1);
        sign = 1 - 2 * sign;
        cm = quant_band(c, x2, N, mbits, B, lowband, LM, lowband_out, 1.f,
                        scratch, orig_fill);
        y2[0] = (float)-sign * x2[1];
        y2[1] = (float)sign * x2[0];
        X[0] *= mid;
        X[1] *= mid;
        Y[0] *= side;
        Y[1] *= side;
        float t = X[0];
        X[0] = t - Y[0];
        Y[0] = t + Y[0];
        t = X[1];
        X[1] = t - Y[1];
        Y[1] = t + Y[1];
    } else {
        int mbits = imax(0, imin(b, (b - s.delta) / 2)), sbits = b - mbits;
        c->remaining -= s.qalloc;
        int32_t rebalance = c->remaining;
        if (mbits >= sbits) {
            cm = quant_band(c, X, N, mbits, B, lowband, LM, lowband_out, 1.f,
                            scratch, fill);
            rebalance = mbits - (rebalance - c->remaining);
            if (rebalance > 3 << BITRES && s.itheta != 0)
                sbits += (int)rebalance - (3 << BITRES);
            cm |= quant_band(c, Y, N, sbits, B, NULL, LM, NULL, side, NULL,
                             fill >> B);
        } else {
            cm = quant_band(c, Y, N, sbits, B, NULL, LM, NULL, side, NULL,
                            fill >> B);
            rebalance = sbits - (rebalance - c->remaining);
            if (rebalance > 3 << BITRES && s.itheta != 16384)
                mbits += (int)rebalance - (3 << BITRES);
            cm |= quant_band(c, X, N, mbits, B, lowband, LM, lowband_out, 1.f,
                             scratch, fill);
        }
    }
    if (N != 2)
        stereo_merge(X, Y, mid, N);
    if (s.inv)
        for (int j = 0; j < N; j++)
            Y[j] = -Y[j];
    return cm;
}

static void all_bands(struct celt_dec *d, struct rc *r, int start, int end,
                      float *X_, float *Y_, uint8_t *collapse, int *pulses,
                      int short_blocks, int spread, int dual_stereo,
                      int intensity, int *tf_res, int32_t total_bits,
                      int32_t balance, int LM, int coded_bands)
{
    int M = 1 << LM, B = short_blocks ? M : 1, C = Y_ ? 2 : 1;
    int norm_off = M * CELT_EBANDS[start], update_lowband = 1;
    int lowband_offset = 0;
    float *norm = d->norm, *norm2 = norm + M * CELT_EBANDS[NB - 1] - norm_off;
    float *scratch = X_ + M * CELT_EBANDS[NB - 1];
    struct bctx c = {.d = d,
                     .r = r,
                     .intensity = intensity,
                     .spread = spread,
                     .seed = d->rng,
                     .avoid_split_noise = B > 1};
    for (int i = start; i < end; i++) {
        int last = i == end - 1, effective_lowband = -1, b;
        float *X = X_ + M * CELT_EBANDS[i],
              *Y = Y_ ? Y_ + M * CELT_EBANDS[i] : NULL;
        int N = M * CELT_EBANDS[i + 1] - M * CELT_EBANDS[i];
        unsigned x_cm, y_cm;
        c.i = i;
        int32_t tell = (int32_t)rc_tell_frac(r);
        if (i != start)
            balance -= tell;
        int32_t remaining = total_bits - tell - 1;
        c.remaining = remaining;
        if (i <= coded_bands - 1) {
            int32_t curr_balance = balance / imin(3, coded_bands - i);
            b = imax(0, imin(16383, (int)imin((int)(remaining + 1),
                                              pulses[i] + (int)curr_balance)));
        } else {
            b = 0;
        }
        if ((M * CELT_EBANDS[i] - N >= M * CELT_EBANDS[start] ||
             i == start + 1) &&
            (update_lowband || lowband_offset == 0))
            lowband_offset = i;
        if (i == start + 1) { /* the folding of the second band, in hybrid */
            int n1 = M * band_w(start), n2 = M * band_w(start + 1);
            if (n2 > n1) {
                memmove(&norm[n1], &norm[2 * n1 - n2],
                        sizeof(float) * (size_t)(n2 - n1));
                if (dual_stereo)
                    memmove(&norm2[n1], &norm2[2 * n1 - n2],
                            sizeof(float) * (size_t)(n2 - n1));
            }
        }
        c.tf_change = tf_res[i];
        float *sc = scratch;
        if (last)
            sc = NULL;
        if (lowband_offset != 0 &&
            (spread != SPREAD_AGGRESSIVE || B > 1 || c.tf_change < 0)) {
            effective_lowband =
                imax(0, M * CELT_EBANDS[lowband_offset] - norm_off - N);
            int fold_start = lowband_offset;
            while (M * CELT_EBANDS[--fold_start] > effective_lowband + norm_off)
                ;
            int fold_end = lowband_offset - 1;
            while (++fold_end < i &&
                   M * CELT_EBANDS[fold_end] < effective_lowband + norm_off + N)
                ;
            x_cm = y_cm = 0;
            int f = fold_start;
            do {
                x_cm |= collapse[f * C + 0];
                y_cm |= collapse[f * C + C - 1];
            } while (++f < fold_end);
        } else {
            x_cm = y_cm = (1u << B) - 1;
        }
        if (dual_stereo && i == intensity) {
            dual_stereo = 0;
            for (int j = 0; j < M * CELT_EBANDS[i] - norm_off; j++)
                norm[j] = .5f * (norm[j] + norm2[j]);
        }
        float *lo1 = effective_lowband != -1 ? norm + effective_lowband : NULL;
        float *out1 = last ? NULL : norm + M * CELT_EBANDS[i] - norm_off;
        if (dual_stereo) {
            float *lo2 =
                effective_lowband != -1 ? norm2 + effective_lowband : NULL;
            float *out2 = last ? NULL : norm2 + M * CELT_EBANDS[i] - norm_off;
            x_cm = quant_band(&c, X, N, b / 2, B, lo1, LM, out1, 1.f, sc,
                              (int)x_cm);
            y_cm = quant_band(&c, Y, N, b / 2, B, lo2, LM, out2, 1.f, sc,
                              (int)y_cm);
        } else {
            if (Y)
                x_cm = band_stereo(&c, X, Y, N, b, B, lo1, LM, out1, sc,
                                   (int)(x_cm | y_cm));
            else
                x_cm = quant_band(&c, X, N, b, B, lo1, LM, out1, 1.f, sc,
                                  (int)(x_cm | y_cm));
            y_cm = x_cm;
        }
        collapse[i * C + 0] = (uint8_t)x_cm;
        collapse[i * C + C - 1] = (uint8_t)y_cm;
        balance += pulses[i] + tell;
        update_lowband = b > (N << BITRES);
        c.avoid_split_noise = 0;
    }
    d->rng = c.seed;
}

static void anti_collapse(struct celt_dec *d, float *X_, const uint8_t *masks,
                          int LM, int C, int size, int start, int end,
                          const int *pulses)
{
    uint32_t seed = d->rng;
    for (int i = start; i < end; i++) {
        int N0 = band_w(i);
        int depth = (1 + pulses[i]) / N0 >> LM;
        float thresh = .5f * exp2f(-.125f * (float)depth);
        float sqrt_1 = 1.f / sqrtf((float)(N0 << LM));
        for (int c = 0; c < C; c++) {
            float prev1 = d->old_loge[c * NB + i],
                  prev2 = d->old_loge2[c * NB + i];
            if (C == 1) {
                prev1 =
                    prev1 > d->old_loge[NB + i] ? prev1 : d->old_loge[NB + i];
                prev2 =
                    prev2 > d->old_loge2[NB + i] ? prev2 : d->old_loge2[NB + i];
            }
            float ediff =
                d->old_e[c * NB + i] - (prev1 < prev2 ? prev1 : prev2);
            ediff = ediff > 0 ? ediff : 0;
            float rr = 2.f * exp2f(-ediff);
            if (LM == 3)
                rr *= 1.41421356f;
            rr = rr < thresh ? rr : thresh;
            rr *= sqrt_1;
            float *X = X_ + c * size + (CELT_EBANDS[i] << LM);
            int renorm = 0;
            for (int k = 0; k < 1 << LM; k++)
                if (!(masks[i * C + c] & 1 << k)) {
                    for (int j = 0; j < N0; j++) {
                        seed = lcg(seed);
                        X[(j << LM) + k] = seed & 0x8000 ? rr : -rr;
                    }
                    renorm = 1;
                }
            if (renorm)
                renormalise(X, N0 << LM, 1.f);
        }
    }
}

/* ---- synthesis ---- */

static void denormalise(const float *X, float *freq, const float *loge,
                        int start, int end, int M, int N, int silence)
{
    int bound = M * CELT_EBANDS[end];
    if (silence) {
        bound = 0;
        start = end = 0;
    }
    memset(freq, 0, sizeof(float) * (size_t)N);
    for (int i = start; i < end; i++) {
        float lg = loge[i] + CELT_EMEANS[i];
        float g = exp2f(lg < 32.f ? lg : 32.f);
        for (int j = M * CELT_EBANDS[i]; j < M * CELT_EBANDS[i + 1]; j++)
            freq[j] = X[j] * g;
    }
    (void)bound;
}

/* the inverse MDCT of a block of n / 2 coefficients (every stride-th of
   in) into out[OV / 2 ...], then its overlap mirrored with what out[0 ..
   OV) held */
static void imdct_block(struct celt_dec *d, int which, const float *in,
                        int stride, float *out)
{
    struct aimdct *m = d->mdct[which];
    int N2 = 120 << which;
    aimdct_dct4(m, in, stride, d->u);
    for (int k = 0; k < N2; k++)
        out[OV / 2 + k] = -d->u[N2 - 1 - k];
    const float *w = CELT_WINDOW;
    for (int i = 0; i < OV / 2; i++) {
        float x1 = out[OV - 1 - i], x2 = out[i];
        out[i] = w[OV - 1 - i] * x2 - w[i] * x1;
        out[OV - 1 - i] = w[i] * x2 + w[OV - 1 - i] * x1;
    }
}

static void synthesis(struct celt_dec *d, float *X, float **out_syn, int start,
                      int end, int C, int CC, int transient, int LM,
                      int silence)
{
    int M = 1 << LM, N = 120 << LM, B, NB_, which;
    if (transient) {
        B = M;
        NB_ = 120;
        which = 0;
    } else {
        B = 1;
        NB_ = N;
        which = LM;
    }
    float *freq = d->freq;
    if (CC == 2 && C == 1) {
        denormalise(X, freq, d->old_e, start, end, M, N, silence);
        for (int c = 0; c < 2; c++)
            for (int b = 0; b < B; b++)
                imdct_block(d, which, freq + b, B, out_syn[c] + NB_ * b);
    } else if (CC == 1 && C == 2) {
        float *f2 = d->freq2;
        denormalise(X, freq, d->old_e, start, end, M, N, silence);
        denormalise(X + N, f2, d->old_e + NB, start, end, M, N, silence);
        for (int i = 0; i < N; i++)
            freq[i] = .5f * freq[i] + .5f * f2[i];
        for (int b = 0; b < B; b++)
            imdct_block(d, which, freq + b, B, out_syn[0] + NB_ * b);
    } else {
        for (int c = 0; c < CC; c++) {
            denormalise(X + c * N, freq, d->old_e + c * NB, start, end, M, N,
                        silence);
            for (int b = 0; b < B; b++)
                imdct_block(d, which, freq + b, B, out_syn[c] + NB_ * b);
        }
    }
}

/* the pitch post-filter (pre-filter with negative gains): y from x, which
   may be the same buffer, its history before it */
static void comb_filter(float *y, const float *x, int T0, int T1, int N,
                        float g0, float g1, int tap0, int tap1, int overlap)
{
    static const float G[3][3] = {{0.3066406250f, 0.2170410156f, 0.1296386719f},
                                  {0.4638671875f, 0.2680664062f, 0.f},
                                  {0.7998046875f, 0.1000976562f, 0.f}};
    if (g0 == 0 && g1 == 0) {
        if (x != y)
            memmove(y, x, sizeof(float) * (size_t)N);
        return;
    }
    T0 = imax(T0, 15);
    T1 = imax(T1, 15);
    float g00 = g0 * G[tap0][0], g01 = g0 * G[tap0][1], g02 = g0 * G[tap0][2];
    float g10 = g1 * G[tap1][0], g11 = g1 * G[tap1][1], g12 = g1 * G[tap1][2];
    float x1 = x[-T1 + 1], x2 = x[-T1], x3 = x[-T1 - 1], x4 = x[-T1 - 2];
    if (g0 == g1 && T0 == T1 && tap0 == tap1)
        overlap = 0;
    int i;
    for (i = 0; i < overlap; i++) {
        float x0 = x[i - T1 + 2];
        float f = CELT_WINDOW[i] * CELT_WINDOW[i];
        y[i] = x[i] + (1.f - f) * g00 * x[i - T0] +
               (1.f - f) * g01 * (x[i - T0 + 1] + x[i - T0 - 1]) +
               (1.f - f) * g02 * (x[i - T0 + 2] + x[i - T0 - 2]) +
               f * g10 * x2 + f * g11 * (x1 + x3) + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
    if (g1 == 0) {
        if (x != y)
            memmove(y + overlap, x + overlap,
                    sizeof(float) * (size_t)(N - overlap));
        return;
    }
    x4 = x[i - T1 - 2];
    x3 = x[i - T1 - 1];
    x2 = x[i - T1];
    x1 = x[i - T1 + 1];
    for (; i < N; i++) {
        float x0 = x[i - T1 + 2];
        y[i] = x[i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/* the concealed overlap pre-filtered and folded as an MDCT would leave
   it, so the next decoded frame blends with it */
static void prefilter_and_fold(struct celt_dec *d, int N)
{
    float etmp[OV];
    for (int c = 0; c < d->cc; c++) {
        comb_filter(etmp, d->mem[c] + DBS - N, d->pf_period_old, d->pf_period,
                    OV, -d->pf_gain_old, -d->pf_gain, d->pf_tapset_old,
                    d->pf_tapset, 0);
        for (int i = 0; i < OV / 2; i++)
            d->mem[c][DBS - N + i] = CELT_WINDOW[i] * etmp[OV - 1 - i] +
                                     CELT_WINDOW[OV - i - 1] * etmp[i];
    }
}

int jcelt_frame(struct celt_dec *d, struct rc *r, int len, int n, int stream_ch,
                float **out)
{
    static const uint8_t SPREAD_ICDF[4] = {25, 23, 2, 0};
    static const uint8_t TRIM_ICDF[11] = {126, 124, 119, 109, 87, 41,
                                          19,  9,   4,   2,   0};
    static const uint8_t TAPSET_ICDF[3] = {2, 1, 0};
    int LM, C = stream_ch, CC = d->cc, start = d->start, end = d->end;
    for (LM = 0; LM <= 3 && 120 << LM != n; LM++)
        ;
    if (LM > 3 || len < 0 || len > 1275)
        return -1;
    int M = 1 << LM, N = n;
    float *out_syn[2];
    for (int c = 0; c < CC; c++)
        out_syn[c] = d->mem[c] + DBS - N;
    int eff_end = end > NB ? NB : end;
    if (C == 1)
        for (int i = 0; i < NB; i++)
            d->old_e[i] =
                d->old_e[i] > d->old_e[NB + i] ? d->old_e[i] : d->old_e[NB + i];
    if (d->loss_duration == 0)
        d->skip_plc = 0;
    int32_t total_bits = len * 8;
    int tell = rc_tell(r), silence;
    if (tell >= total_bits)
        silence = 1;
    else if (tell == 1)
        silence = rc_bit_logp(r, 15);
    else
        silence = 0;
    if (silence) {
        tell = len * 8;
        r->nbits_total += tell - rc_tell(r);
    }
    float pf_gain = 0;
    int pf_pitch = 0, pf_tapset = 0;
    if (start == 0 && tell + 16 <= total_bits) {
        if (rc_bit_logp(r, 1)) {
            int octave = (int)rc_uint(r, 6);
            pf_pitch =
                (16 << octave) + (int)rc_bits(r, 4 + (unsigned)octave) - 1;
            int qg = (int)rc_bits(r, 3);
            if (rc_tell(r) + 2 <= total_bits)
                pf_tapset = rc_icdf(r, TAPSET_ICDF, 2);
            pf_gain = .09375f * (float)(qg + 1);
        }
        tell = rc_tell(r);
    }
    int transient = 0;
    if (LM > 0 && tell + 3 <= total_bits) {
        transient = rc_bit_logp(r, 3);
        tell = rc_tell(r);
    }
    int short_blocks = transient ? M : 0;
    int intra = tell + 3 <= total_bits ? rc_bit_logp(r, 3) : 0;
    if (!intra && d->loss_duration != 0) {
        /* after concealment, a safe start for the energy's prediction */
        int missing = imin(10, d->loss_duration >> LM);
        float safety = LM == 0 ? 1.5f : LM == 1 ? .5f : 0;
        for (int c = 0; c < 2; c++)
            for (int i = start; i < end; i++) {
                float *e = &d->old_e[c * NB + i];
                float e1 = d->old_loge[c * NB + i],
                      e2 = d->old_loge2[c * NB + i];
                if (*e < (e1 > e2 ? e1 : e2)) {
                    float slope =
                        e1 - *e > .5f * (e2 - *e) ? e1 - *e : .5f * (e2 - *e);
                    float t = (1 + missing) * slope;
                    float v = *e - (t > 0 ? t : 0);
                    *e = v > -20.f ? v : -20.f;
                } else {
                    float m = *e < e1 ? *e : e1;
                    *e = m < e2 ? m : e2;
                }
                *e -= safety;
            }
    }
    coarse_energy(d, r, intra, C, LM);
    int tf_res[NB];
    tf_decode(r, start, end, transient, tf_res, LM);
    tell = rc_tell(r);
    int spread = SPREAD_NORMAL;
    if (tell + 4 <= total_bits)
        spread = rc_icdf(r, SPREAD_ICDF, 5);
    int cap[NB], offsets[NB];
    for (int i = 0; i < NB; i++) {
        int w = band_w(i) << LM;
        cap[i] = (CELT_CACHE_CAPS[NB * (2 * LM + C - 1) + i] + 64) * C * w >> 2;
    }
    int dynalloc_logp = 6;
    total_bits <<= BITRES;
    uint32_t tf = rc_tell_frac(r);
    for (int i = start; i < end; i++) {
        int width = C * band_w(i) << LM;
        int quanta = imin(width << BITRES, imax(6 << BITRES, width));
        int loop_logp = dynalloc_logp, boost = 0;
        while ((int32_t)tf + (loop_logp << BITRES) < total_bits &&
               boost < cap[i]) {
            int flag = rc_bit_logp(r, (unsigned)loop_logp);
            tf = rc_tell_frac(r);
            if (!flag)
                break;
            boost += quanta;
            total_bits -= quanta;
            loop_logp = 1;
        }
        offsets[i] = boost;
        if (boost > 0)
            dynalloc_logp = imax(2, dynalloc_logp - 1);
    }
    for (int i = 0; i < start; i++)
        offsets[i] = 0;
    int trim = (int32_t)tf + (6 << BITRES) <= total_bits
                   ? rc_icdf(r, TRIM_ICDF, 7)
                   : 5;
    int32_t bits =
        (((int32_t)len * 8) << BITRES) - (int32_t)rc_tell_frac(r) - 1;
    int anti_rsv =
        transient && LM >= 2 && bits >= ((LM + 2) << BITRES) ? 1 << BITRES : 0;
    bits -= anti_rsv;
    int pulses[NB], fine_quant[NB], fine_prio[NB], intensity = 0, dual = 0;
    int32_t balance = 0;
    int coded =
        allocation(r, start, end, offsets, cap, trim, &intensity, &dual, bits,
                   &balance, pulses, fine_quant, fine_prio, C, LM);
    for (int i = start; i < end; i++) {
        if (fine_quant[i] <= 0)
            continue;
        for (int c = 0; c < C; c++) {
            int q2 = (int)rc_bits(r, (unsigned)fine_quant[i]);
            d->old_e[i + c * NB] += ((float)q2 + .5f) *
                                        (float)(1 << (14 - fine_quant[i])) *
                                        (1.f / 16384) -
                                    .5f;
        }
    }
    for (int c = 0; c < CC; c++)
        memmove(d->mem[c], d->mem[c] + N,
                sizeof(float) * (size_t)(DBS - N + OV));
    uint8_t collapse[2 * NB];
    float *X = d->x;
    memset(X, 0, sizeof(float) * (size_t)(C * N));
    all_bands(d, r, start, end, X, C == 2 ? X + N : NULL, collapse, pulses,
              short_blocks, spread, dual, intensity, tf_res,
              len * (8 << BITRES) - anti_rsv, balance, LM, coded);
    int anti_on = anti_rsv > 0 ? (int)rc_bits(r, 1) : 0;
    int bits_left = len * 8 - rc_tell(r);
    for (int prio = 0; prio < 2; prio++)
        for (int i = start; i < end && bits_left >= C; i++) {
            if (fine_quant[i] >= MAX_FINE_BITS || fine_prio[i] != prio)
                continue;
            for (int c = 0; c < C; c++) {
                int q2 = (int)rc_bits(r, 1);
                d->old_e[i + c * NB] += ((float)q2 - .5f) *
                                        (float)(1 << (14 - fine_quant[i] - 1)) *
                                        (1.f / 16384);
                bits_left--;
            }
        }
    if (anti_on)
        anti_collapse(d, X, collapse, LM, C, N, start, end, pulses);
    if (silence)
        for (int i = 0; i < C * NB; i++)
            d->old_e[i] = -28.f;
    if (d->prefilter_and_fold)
        prefilter_and_fold(d, N);
    synthesis(d, X, out_syn, start, eff_end, C, CC, transient, LM, silence);
    for (int c = 0; c < CC; c++) {
        d->pf_period = imax(d->pf_period, 15);
        d->pf_period_old = imax(d->pf_period_old, 15);
        comb_filter(out_syn[c], out_syn[c], d->pf_period_old, d->pf_period, 120,
                    d->pf_gain_old, d->pf_gain, d->pf_tapset_old, d->pf_tapset,
                    OV);
        if (LM != 0)
            comb_filter(out_syn[c] + 120, out_syn[c] + 120, d->pf_period,
                        pf_pitch, N - 120, d->pf_gain, pf_gain, d->pf_tapset,
                        pf_tapset, OV);
    }
    d->pf_period_old = d->pf_period;
    d->pf_gain_old = d->pf_gain;
    d->pf_tapset_old = d->pf_tapset;
    d->pf_period = pf_pitch;
    d->pf_gain = pf_gain;
    d->pf_tapset = pf_tapset;
    if (LM != 0) {
        d->pf_period_old = d->pf_period;
        d->pf_gain_old = d->pf_gain;
        d->pf_tapset_old = d->pf_tapset;
    }
    if (C == 1)
        memcpy(&d->old_e[NB], d->old_e, sizeof(float) * NB);
    if (!transient) {
        memcpy(d->old_loge2, d->old_loge, sizeof d->old_loge);
        memcpy(d->old_loge, d->old_e, sizeof d->old_e);
    } else {
        for (int i = 0; i < 2 * NB; i++)
            d->old_loge[i] =
                d->old_loge[i] < d->old_e[i] ? d->old_loge[i] : d->old_e[i];
    }
    float max_bg = (float)imin(160, d->loss_duration + M) * 0.001f;
    for (int i = 0; i < 2 * NB; i++)
        d->bg_loge[i] = d->bg_loge[i] + max_bg < d->old_e[i]
                            ? d->bg_loge[i] + max_bg
                            : d->old_e[i];
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < start; i++) {
            d->old_e[c * NB + i] = 0;
            d->old_loge[c * NB + i] = d->old_loge2[c * NB + i] = -28.f;
        }
        for (int i = end; i < NB; i++) {
            d->old_e[c * NB + i] = 0;
            d->old_loge[c * NB + i] = d->old_loge2[c * NB + i] = -28.f;
        }
    }
    d->rng = r->rng;
    d->loss_duration = 0;
    d->prefilter_and_fold = 0;
    for (int c = 0; c < CC; c++) {
        float m = d->preemph[c];
        for (int j = 0; j < N; j++) {
            float tmp = out_syn[c][j] + 1e-30f + m;
            m = 0.85000610f * tmp;
            out[c][j] = tmp * (1.f / 32768);
        }
        d->preemph[c] = m;
    }
    return rc_tell(r) > 8 * len ? -1 : 0;
}

/* ---- concealment ---- */

static void autocorr(const float *x, float *ac, const float *window,
                     int overlap, int lag, int n)
{
    float xx[1024];
    const float *xp = x;
    if (overlap) {
        memcpy(xx, x, sizeof(float) * (size_t)n);
        for (int i = 0; i < overlap; i++) {
            xx[i] = x[i] * window[i];
            xx[n - i - 1] = x[n - i - 1] * window[i];
        }
        xp = xx;
    }
    for (int k = 0; k <= lag; k++) {
        float s = 0;
        for (int i = k; i < n; i++)
            s += xp[i] * xp[i - k];
        ac[k] = s;
    }
}

static void celt_lpc(float *lpc, const float *ac, int p)
{
    float err = ac[0];
    memset(lpc, 0, sizeof(float) * (size_t)p);
    if (ac[0] <= 1e-10f)
        return;
    for (int i = 0; i < p; i++) {
        float rr = 0;
        for (int j = 0; j < i; j++)
            rr += lpc[j] * ac[i - j];
        rr += ac[i + 1];
        float r = -rr / err;
        lpc[i] = r;
        for (int j = 0; j < (i + 1) >> 1; j++) {
            float t1 = lpc[j], t2 = lpc[i - 1 - j];
            lpc[j] = t1 + r * t2;
            lpc[i - 1 - j] = t2 + r * t1;
        }
        err = err - r * r * err;
        if (err <= .001f * ac[0])
            break;
    }
}

static void find_best_pitch(const float *xcorr, const float *y, int len,
                            int max_pitch, int *best)
{
    float Syy = 1, bnum[2] = {-1, -1}, bden[2] = {0, 0};
    best[0] = 0;
    best[1] = 1;
    for (int j = 0; j < len; j++)
        Syy += y[j] * y[j];
    for (int i = 0; i < max_pitch; i++) {
        if (xcorr[i] > 0) {
            float x16 = xcorr[i] * 1e-12f, num = x16 * x16;
            if (num * bden[1] > bnum[1] * Syy) {
                if (num * bden[0] > bnum[0] * Syy) {
                    bnum[1] = bnum[0];
                    bden[1] = bden[0];
                    best[1] = best[0];
                    bnum[0] = num;
                    bden[0] = Syy;
                    best[0] = i;
                } else {
                    bnum[1] = num;
                    bden[1] = Syy;
                    best[1] = i;
                }
            }
        }
        Syy += y[i + len] * y[i + len] - y[i] * y[i];
        Syy = Syy > 1 ? Syy : 1;
    }
}

static int plc_pitch(struct celt_dec *d)
{
    float lp[DBS >> 1];
    float ac[5], lpc[4], lpc2[5], tmp = 1.f;
    int len = DBS, C = d->cc;
    for (int i = 1; i < len >> 1; i++) {
        lp[i] = .25f * d->mem[0][2 * i - 1] + .25f * d->mem[0][2 * i + 1] +
                .5f * d->mem[0][2 * i];
        if (C == 2)
            lp[i] += .25f * d->mem[1][2 * i - 1] + .25f * d->mem[1][2 * i + 1] +
                     .5f * d->mem[1][2 * i];
    }
    lp[0] = .25f * d->mem[0][1] + .5f * d->mem[0][0];
    if (C == 2)
        lp[0] += .25f * d->mem[1][1] + .5f * d->mem[1][0];
    autocorr(lp, ac, NULL, 0, 4, len >> 1);
    ac[0] *= 1.0001f;
    for (int i = 1; i <= 4; i++)
        ac[i] -= ac[i] * (.008f * i) * (.008f * i);
    celt_lpc(lpc, ac, 4);
    for (int i = 0; i < 4; i++) {
        tmp *= .9f;
        lpc[i] *= tmp;
    }
    lpc2[0] = lpc[0] + .8f;
    lpc2[1] = lpc[1] + .8f * lpc[0];
    lpc2[2] = lpc[2] + .8f * lpc[1];
    lpc2[3] = lpc[3] + .8f * lpc[2];
    lpc2[4] = .8f * lpc[3];
    float m[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < len >> 1; i++) {
        float s = lp[i] + lpc2[0] * m[0] + lpc2[1] * m[1] + lpc2[2] * m[2] +
                  lpc2[3] * m[3] + lpc2[4] * m[4];
        m[4] = m[3];
        m[3] = m[2];
        m[2] = m[1];
        m[1] = m[0];
        m[0] = lp[i];
        lp[i] = s;
    }
    /* the search: x the last part, y all of it */
    const float *x = lp + (720 >> 1), *y = lp;
    int slen = DBS - 720, max_pitch = 720 - 100, lag = slen + max_pitch;
    float x4[(DBS - 720) >> 2], y4[(DBS + 620) >> 2], xcorr[(720 - 100) >> 1];
    int best[2] = {0, 0};
    for (int j = 0; j < slen >> 2; j++)
        x4[j] = x[2 * j];
    for (int j = 0; j < lag >> 2; j++)
        y4[j] = y[2 * j];
    for (int i = 0; i < max_pitch >> 2; i++) {
        float s = 0;
        for (int j = 0; j < slen >> 2; j++)
            s += x4[j] * y4[i + j];
        xcorr[i] = s;
    }
    find_best_pitch(xcorr, y4, slen >> 2, max_pitch >> 2, best);
    for (int i = 0; i < max_pitch >> 1; i++) {
        xcorr[i] = 0;
        if (abs(i - 2 * best[0]) > 2 && abs(i - 2 * best[1]) > 2)
            continue;
        float s = 0;
        for (int j = 0; j < slen >> 1; j++)
            s += x[j] * y[i + j];
        xcorr[i] = s > -1 ? s : -1;
    }
    find_best_pitch(xcorr, y, slen >> 1, max_pitch >> 1, best);
    int offset = 0;
    if (best[0] > 0 && best[0] < (max_pitch >> 1) - 1) {
        float a = xcorr[best[0] - 1], b = xcorr[best[0]],
              c = xcorr[best[0] + 1];
        if ((c - a) > .7f * (b - a))
            offset = 1;
        else if ((a - c) > .7f * (b - c))
            offset = -1;
    }
    return 720 - (2 * best[0] - offset);
}

void jcelt_plc(struct celt_dec *d, int n, float **out)
{
    int LM, N = n, C = d->cc, start = d->start;
    for (LM = 0; LM < 3 && 120 << LM != n; LM++)
        ;
    if (120 << LM != n) { /* not a CELT frame's size: silence */
        for (int c = 0; c < C; c++)
            if (out[c] && n > 0)
                memset(out[c], 0, sizeof(float) * (size_t)n);
        return;
    }
    float *out_syn[2];
    for (int c = 0; c < C; c++)
        out_syn[c] = d->mem[c] + DBS - N;
    int noise = d->loss_duration >= 40 || start != 0 || d->skip_plc;
    if (noise) {
        int end = d->end, eff_end = imax(start, imin(end, NB));
        for (int c = 0; c < C; c++)
            memmove(d->mem[c], d->mem[c] + N,
                    sizeof(float) * (size_t)(DBS - N + OV));
        if (d->prefilter_and_fold)
            prefilter_and_fold(d, N);
        float decay = d->loss_duration == 0 ? 1.5f : .5f;
        for (int c = 0; c < C; c++)
            for (int i = start; i < end; i++) {
                float *e = &d->old_e[c * NB + i], v = *e - decay;
                *e = d->bg_loge[c * NB + i] > v ? d->bg_loge[c * NB + i] : v;
            }
        uint32_t seed = d->rng;
        float *X = d->x;
        for (int c = 0; c < C; c++)
            for (int i = start; i < eff_end; i++) {
                int off = N * c + (CELT_EBANDS[i] << LM);
                int blen = band_w(i) << LM;
                for (int j = 0; j < blen; j++) {
                    seed = lcg(seed);
                    X[off + j] = (float)((int32_t)seed >> 20);
                }
                renormalise(X + off, blen, 1.f);
            }
        d->rng = seed;
        synthesis(d, X, out_syn, start, eff_end, C, C, 0, LM, 0);
        d->prefilter_and_fold = 0;
        d->skip_plc = 1;
    } else {
        float fade = 1.f;
        int pitch;
        if (d->loss_duration == 0) {
            d->last_pitch = pitch = plc_pitch(d);
        } else {
            pitch = d->last_pitch;
            fade = .8f;
        }
        int exc_len = imin(2 * pitch, 1024);
        float exc_buf[1024 + 24], fir_tmp[1024], *exc = exc_buf + 24;
        for (int c = 0; c < C; c++) {
            float *buf = d->mem[c], S1 = 0;
            for (int i = 0; i < 1024 + 24; i++)
                exc[i - 24] = buf[DBS - 1024 - 24 + i];
            if (d->loss_duration == 0) {
                float ac[25];
                autocorr(exc, ac, CELT_WINDOW, OV, 24, 1024);
                ac[0] *= 1.0001f;
                for (int i = 1; i <= 24; i++)
                    ac[i] -= ac[i] * (0.008f * 0.008f) * i * i;
                celt_lpc(d->lpc[c], ac, 24);
            }
            for (int i = 0; i < exc_len; i++) {
                float s = exc[1024 - exc_len + i];
                for (int k = 0; k < 24; k++)
                    s += d->lpc[c][k] * exc[1024 - exc_len + i - k - 1];
                fir_tmp[i] = s;
            }
            memcpy(exc + 1024 - exc_len, fir_tmp,
                   sizeof(float) * (size_t)exc_len);
            float E1 = 1, E2 = 1;
            int dl = exc_len >> 1;
            for (int i = 0; i < dl; i++) {
                float e = exc[1024 - dl + i];
                E1 += e * e;
                e = exc[1024 - 2 * dl + i];
                E2 += e * e;
            }
            E1 = E1 < E2 ? E1 : E2;
            float decay = sqrtf((E1 * .5f) / E2);
            memmove(buf, buf + N, sizeof(float) * (size_t)(DBS - N));
            int ex_off = 1024 - pitch, ex_len = N + OV;
            float att = fade * decay;
            for (int i = 0, j = 0; i < ex_len; i++, j++) {
                if (j >= pitch) {
                    j -= pitch;
                    att *= decay;
                }
                buf[DBS - N + i] = att * exc[ex_off + j];
                float t = buf[DBS - 1024 - N + ex_off + j];
                S1 += t * t;
            }
            float mem[24];
            for (int i = 0; i < 24; i++)
                mem[i] = buf[DBS - N - 1 - i];
            for (int i = 0; i < ex_len; i++) {
                float s = buf[DBS - N + i];
                for (int k = 0; k < 24; k++)
                    s -= d->lpc[c][k] * mem[k];
                memmove(mem + 1, mem, 23 * sizeof(float));
                mem[0] = s;
                buf[DBS - N + i] = s;
            }
            float S2 = 0;
            for (int i = 0; i < ex_len; i++)
                S2 += buf[DBS - N + i] * buf[DBS - N + i];
            if (!(S1 > 0.2f * S2)) {
                for (int i = 0; i < ex_len; i++)
                    buf[DBS - N + i] = 0;
            } else if (S1 < S2) {
                float ratio = sqrtf((S1 * .5f + 1) / (S2 + 1));
                for (int i = 0; i < OV; i++)
                    buf[DBS - N + i] *= 1.f - CELT_WINDOW[i] * (1.f - ratio);
                for (int i = OV; i < ex_len; i++)
                    buf[DBS - N + i] *= ratio;
            }
        }
        d->prefilter_and_fold = 1;
    }
    d->loss_duration = imin(10000, d->loss_duration + (1 << LM));
    for (int c = 0; c < C; c++) {
        float m = d->preemph[c];
        for (int j = 0; j < N; j++) {
            float tmp = out_syn[c][j] + 1e-30f + m;
            m = 0.85000610f * tmp;
            out[c][j] = tmp * (1.f / 32768);
        }
        d->preemph[c] = m;
    }
}
