/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_opus.h - what the parts of the Opus decoder share (RFC 6716): the
 * range decoder (section 4.1), and the CELT (audio_opus_celt.c) and SILK
 * (audio_opus_silk.c) layers as audio_opus.c drives them. Inside
 * src/common.
 */
#ifndef JANAS_COMMON_AUDIO_OPUS_H
#define JANAS_COMMON_AUDIO_OPUS_H

#include <stddef.h>
#include <stdint.h>

/* ---- the range decoder ---- */

struct rc {
    const uint8_t *buf;
    uint32_t storage, offs, end_offs;
    uint32_t end_window;
    int nend_bits, nbits_total;
    uint32_t rng, val, ext;
    int rem, error;
};

#define RC_CODE_BOT (1u << 23)
#define RC_CODE_TOP (1u << 31)

static inline int rc_ilog(uint32_t v)
{
    return v ? 32 - __builtin_clz(v) : 0;
}

static inline int rc_byte(struct rc *r)
{
    return r->offs < r->storage ? r->buf[r->offs++] : 0;
}

static inline int rc_byte_end(struct rc *r)
{
    return r->end_offs < r->storage ? r->buf[r->storage - ++r->end_offs] : 0;
}

static inline void rc_normalize(struct rc *r)
{
    while (r->rng <= RC_CODE_BOT) {
        r->nbits_total += 8;
        r->rng <<= 8;
        int sym = r->rem;
        r->rem = rc_byte(r);
        sym = (sym << 8 | r->rem) >> 1;
        r->val = ((r->val << 8) + (255 & ~(uint32_t)sym)) & (RC_CODE_TOP - 1);
    }
}

static inline void rc_init(struct rc *r, const uint8_t *buf, uint32_t n)
{
    r->buf = buf;
    r->storage = n;
    r->offs = r->end_offs = 0;
    r->end_window = 0;
    r->nend_bits = 0;
    r->nbits_total = 33 - 24;
    r->rng = 1u << 7;
    r->rem = rc_byte(r);
    r->val = r->rng - 1 - (uint32_t)(r->rem >> 1);
    r->error = 0;
    rc_normalize(r);
}

static inline unsigned rc_decode(struct rc *r, unsigned ft)
{
    r->ext = r->rng / ft;
    unsigned s = r->val / r->ext;
    return ft - (s + 1 < ft ? s + 1 : ft);
}

static inline unsigned rc_decode_bin(struct rc *r, unsigned bits)
{
    r->ext = r->rng >> bits;
    unsigned s = r->val / r->ext;
    return (1u << bits) - (s + 1 < (1u << bits) ? s + 1 : (1u << bits));
}

static inline void rc_update(struct rc *r, unsigned fl, unsigned fh,
                             unsigned ft)
{
    uint32_t s = r->ext * (ft - fh);
    r->val -= s;
    r->rng = fl > 0 ? r->ext * (fh - fl) : r->rng - s;
    rc_normalize(r);
}

/* a one with probability 1 / 2^logp */
static inline int rc_bit_logp(struct rc *r, unsigned logp)
{
    uint32_t s = r->rng >> logp;
    int ret = r->val < s;
    if (!ret)
        r->val -= s;
    r->rng = ret ? s : r->rng - s;
    rc_normalize(r);
    return ret;
}

static inline int rc_icdf(struct rc *r, const uint8_t *icdf, unsigned ftb)
{
    uint32_t s = r->rng, d = r->val, q = s >> ftb, t;
    int ret = -1;
    do {
        t = s;
        s = q * icdf[++ret];
    } while (d < s);
    r->val = d - s;
    r->rng = t - s;
    rc_normalize(r);
    return ret;
}

/* raw bits from the end of the frame */
static inline uint32_t rc_bits(struct rc *r, unsigned bits)
{
    uint32_t window = r->end_window;
    int avail = r->nend_bits;
    if ((unsigned)avail < bits) {
        do {
            window |= (uint32_t)rc_byte_end(r) << avail;
            avail += 8;
        } while (avail <= 32 - 8);
    }
    uint32_t ret = window & ((bits < 32 ? 1u << bits : 0) - 1u);
    window = bits < 32 ? window >> bits : 0;
    avail -= (int)bits;
    r->end_window = window;
    r->nend_bits = avail;
    r->nbits_total += (int)bits;
    return ret;
}

static inline uint32_t rc_uint(struct rc *r, uint32_t ft)
{
    ft--;
    int ftb = rc_ilog(ft);
    if (ftb > 8) {
        ftb -= 8;
        unsigned f = (unsigned)(ft >> ftb) + 1;
        unsigned s = rc_decode(r, f);
        rc_update(r, s, s + 1, f);
        uint32_t t = (uint32_t)s << ftb | rc_bits(r, (unsigned)ftb);
        if (t <= ft)
            return t;
        r->error = 1;
        return ft;
    }
    ft++;
    unsigned s = rc_decode(r, (unsigned)ft);
    rc_update(r, s, s + 1, (unsigned)ft);
    return s;
}

static inline int rc_tell(const struct rc *r)
{
    return r->nbits_total - rc_ilog(r->rng);
}

/* in eighths of a bit */
static inline uint32_t rc_tell_frac(const struct rc *r)
{
    static const unsigned CORR[8] = {35733, 38967, 42495, 46340,
                                     50535, 55109, 60097, 65535};
    uint32_t nbits = (uint32_t)r->nbits_total << 3;
    int l = rc_ilog(r->rng);
    uint32_t q = r->rng >> (l - 16);
    unsigned b = (q >> 12) - 8;
    b += q > CORR[b];
    return nbits - (uint32_t)((l << 3) + (int)b);
}

/* ---- the layers ---- */

struct celt_dec;
/* a CELT decoder of channels channels (1, 2): NULL when out of memory */
struct celt_dec *jcelt_new(int channels);
void jcelt_free(struct celt_dec *d);
void jcelt_reset(struct celt_dec *d);
/* the bands decoded: start 0 (17 in hybrid frames), end by bandwidth */
void jcelt_bands(struct celt_dec *d, int start, int end);
/* a frame of n samples (120, 240, 480, 960 at 48 kHz) from r (len bytes
   of the packet's frame) into out[ch][n]; stream_ch the channels coded;
   0 or -1 */
int jcelt_frame(struct celt_dec *d, struct rc *r, int len, int n, int stream_ch,
                float **out);
/* the decoder's range after the frame (the bitstream check) */
uint32_t jcelt_rng(const struct celt_dec *d);
/* n samples concealed, as when a packet is lost: what a change of mode
   without a redundant frame fades from */
void jcelt_plc(struct celt_dec *d, int n, float **out);

struct silk_dec;
struct silk_dec *jsilk_new(void);
void jsilk_free(struct silk_dec *d);
void jsilk_reset(struct silk_dec *d);
/* the rate of the output: 8, 12, 16, 24 or 48 kHz (48 by default) */
void jsilk_set_rate(struct silk_dec *d, int hz);
/* the SILK layer of a frame: one call a SILK frame of up to 20 ms of a
   packet's frame of ms milliseconds (10, 20, 40, 60) at its internal rate
   (bandwidth: 8, 12, 16 kHz), stereo or mono, into out[ch] at 48 kHz
   (resampled as libopus does); first: the first call of the packet's
   frame; r NULL: a lost frame concealed. The samples a channel, or -1 */
int jsilk_frame(struct silk_dec *d, struct rc *r, int ms, int khz,
                int stream_ch, int out_ch, float **out, int first);

#endif
