/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_mp2.c - MPEG audio Layers I and II (audio_mpa.h), as ISO/IEC
 * 11172-3 and 13818-3 describe them: the bit allocation (Layer II by the
 * table the bit rate, the rate and the channels choose, the subbands above
 * the joint stereo bound shared), the scale factors (Layer II's three
 * with their selection information), the samples (Layer II's of 3, 5 and 9
 * levels grouped three to a code), requantized to (2m - (N - 1)) / N times
 * 2^(1 - i / 3), and the polyphase filter bank.
 */
#include <math.h>
#include <string.h>

#include "audio_mpa.h"
#include "audio_mpa_tab.h"

static float level(uint32_t m, uint32_t steps, int sf)
{
    return (float)((2.0 * m - (steps - 1)) / steps * exp2(1 - sf / 3.0));
}

static int layer1(struct mpa_br *b, const struct mpa_hdr *h,
                  struct mpa_synth *s, float (*pcm)[1152])
{
    int nch = h->ch, bound = 32;
    if (h->mode == MPA_MODE_JOINT)
        bound = (h->mode_ext + 1) * 4;
    int alloc[2][32] = {{0}}, sf[2][32] = {{0}};
    for (int sb = 0; sb < 32; sb++)
        for (int ch = 0; ch < nch; ch++) {
            if (sb >= bound && ch == 1) {
                alloc[1][sb] = alloc[0][sb];
                continue;
            }
            alloc[ch][sb] = (int)mpa_bits(b, 4);
            if (alloc[ch][sb] == 15)
                return -1;
        }
    for (int sb = 0; sb < 32; sb++)
        for (int ch = 0; ch < nch; ch++)
            if (alloc[ch][sb])
                sf[ch][sb] = (int)mpa_bits(b, 6);
    for (int t = 0; t < 12; t++) {
        float sbv[2][32];
        for (int sb = 0; sb < 32; sb++) {
            uint32_t m = 0;
            for (int ch = 0; ch < nch; ch++) {
                int a = alloc[ch][sb];
                if (!a) {
                    sbv[ch][sb] = 0;
                    continue;
                }
                if (sb < bound || ch == 0)
                    m = mpa_bits(b, a + 1);
                sbv[ch][sb] = level(m, (1u << (a + 1)) - 1, sf[ch][sb]);
            }
        }
        for (int ch = 0; ch < nch; ch++)
            mpa_synth(s, ch, sbv[ch], pcm[ch] + 32 * t);
    }
    return 384;
}

static int l2_table(const struct mpa_hdr *h)
{
    int chr = h->bitrate / h->ch;
    if (h->lsf)
        return 4;
    if ((h->rate == 48000 && chr >= 56) || (chr >= 56 && chr <= 80))
        return 0;
    if (h->rate != 48000 && chr >= 96)
        return 1;
    if (h->rate != 32000 && chr <= 48)
        return 2;
    return 3;
}

static int layer2(struct mpa_br *b, const struct mpa_hdr *h,
                  struct mpa_synth *s, float (*pcm)[1152])
{
    int table = l2_table(h), nch = h->ch;
    int sblimit = MPA_SBLIMIT[table];
    const uint8_t *tab = table < 2   ? MPA_ALLOC_1
                         : table < 4 ? MPA_ALLOC_3
                                     : MPA_ALLOC_4;
    int bound = sblimit;
    if (h->mode == MPA_MODE_JOINT && (h->mode_ext + 1) * 4 < sblimit)
        bound = (h->mode_ext + 1) * 4;
    int qidx[2][32] = {{0}}, scfsi[2][32] = {{0}}, sf[2][32][3];
    const uint8_t *t = tab;
    for (int sb = 0; sb < sblimit; sb++) {
        int nb = t[0];
        for (int ch = 0; ch < nch; ch++) {
            if (sb >= bound && ch == 1) {
                qidx[1][sb] = qidx[0][sb];
                continue;
            }
            int a = (int)mpa_bits(b, nb);
            qidx[ch][sb] = a ? t[a] + 1 : 0; /* 0: no samples */
        }
        t += 1 << nb; /* the row: its width, then the classes */
    }
    for (int sb = 0; sb < sblimit; sb++)
        for (int ch = 0; ch < nch; ch++)
            if (qidx[ch][sb])
                scfsi[ch][sb] = (int)mpa_bits(b, 2);
    for (int sb = 0; sb < sblimit; sb++)
        for (int ch = 0; ch < nch; ch++) {
            int *f = sf[ch][sb];
            if (!qidx[ch][sb])
                continue;
            switch (scfsi[ch][sb]) {
            case 0:
                f[0] = (int)mpa_bits(b, 6);
                f[1] = (int)mpa_bits(b, 6);
                f[2] = (int)mpa_bits(b, 6);
                break;
            case 1:
                f[0] = f[1] = (int)mpa_bits(b, 6);
                f[2] = (int)mpa_bits(b, 6);
                break;
            case 2:
                f[0] = f[1] = f[2] = (int)mpa_bits(b, 6);
                break;
            default:
                f[0] = (int)mpa_bits(b, 6);
                f[1] = f[2] = (int)mpa_bits(b, 6);
                break;
            }
        }
    for (int gr = 0; gr < 12; gr++) {
        float sbv[2][3][32];
        memset(sbv, 0, sizeof sbv);
        for (int sb = 0; sb < sblimit; sb++) {
            uint32_t m[3] = {0, 0, 0};
            for (int ch = 0; ch < nch; ch++) {
                int q = qidx[ch][sb];
                if (!q)
                    continue;
                uint32_t steps = MPA_QSTEPS[q - 1];
                int bits = MPA_QBITS[q - 1];
                if (sb < bound || ch == 0) {
                    if (bits < 0) { /* three to a code */
                        uint32_t c = mpa_bits(b, -bits);
                        m[0] = c % steps;
                        c /= steps;
                        m[1] = c % steps;
                        m[2] = c / steps;
                    } else {
                        for (int k = 0; k < 3; k++)
                            m[k] = mpa_bits(b, bits);
                    }
                }
                for (int k = 0; k < 3; k++)
                    sbv[ch][k][sb] = level(m[k] < steps ? m[k] : steps - 1,
                                           steps, sf[ch][sb][gr / 4]);
            }
        }
        for (int k = 0; k < 3; k++)
            for (int ch = 0; ch < nch; ch++)
                mpa_synth(s, ch, sbv[ch][k], pcm[ch] + 32 * (3 * gr + k));
    }
    return 1152;
}

int mpa_layer12(const uint8_t *f, size_t n, const struct mpa_hdr *h,
                struct mpa_synth *s, float (*pcm)[1152])
{
    size_t at = 4 + (h->crc ? 2 : 0);
    if (n <= at)
        return -1;
    struct mpa_br b = {.p = f + at, .n = n - at};
    return h->layer == 1 ? layer1(&b, h, s, pcm) : layer2(&b, h, s, pcm);
}
