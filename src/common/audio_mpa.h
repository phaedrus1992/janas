/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_mpa.h - what the MPEG audio decoders share (audio_mp3.c for Layer
 * III and the file, audio_mp2.c for Layers I and II): a frame's header, a
 * reader of bits, the polyphase synthesis filter bank of ISO/IEC 11172-3.
 * Inside src/common.
 */
#ifndef JANAS_COMMON_AUDIO_MPA_H
#define JANAS_COMMON_AUDIO_MPA_H

#include <stddef.h>
#include <stdint.h>

#define MPA_MODE_STEREO 0
#define MPA_MODE_JOINT 1
#define MPA_MODE_DUAL 2
#define MPA_MODE_MONO 3

struct mpa_hdr {
    int lsf;        /* MPEG-2 or 2.5: half rates, one granule */
    int layer;      /* 1, 2, 3 */
    int crc;        /* a CRC follows the header */
    int bitrate;    /* kbit/s; 0: free format */
    int rate;       /* samples a second */
    int rate_index; /* 0 .. 8: 44.1, 48, 32, 22.05, 24, 16, 11.025, 12, 8 */
    int pad, mode, mode_ext, ch;
    size_t size; /* the frame's bytes (free format: 0 until found) */
};

/* the header at p (4 bytes): 0, or -1 when it is not one */
int mpa_header(const uint8_t *p, struct mpa_hdr *h);

struct mpa_br {
    const uint8_t *p;
    size_t n;   /* bytes */
    size_t pos; /* bits */
};

/* k bits, 0 to 32; past the end the bits read as zeros */
static inline uint32_t mpa_bits(struct mpa_br *b, int k)
{
    if (!k)
        return 0;
    size_t at = b->pos >> 3;
    uint64_t w = 0;
    for (int i = 0; i < 8; i++)
        w = w << 8 | (at + (size_t)i < b->n ? b->p[at + (size_t)i] : 0);
    b->pos += (size_t)k;
    return (uint32_t)((w << ((b->pos - (size_t)k) & 7)) >> (64 - k));
}

/* the synthesis filter bank of each channel */
struct mpa_synth {
    float v[2][1024];
    int off[2];
    float n[64][32]; /* the matrixing: cos((16 + i)(2k + 1) pi / 64) */
    float d[512];    /* the window D */
};

void mpa_synth_init(struct mpa_synth *s);
/* 32 subband samples of channel ch into 32 PCM samples */
void mpa_synth(struct mpa_synth *s, int ch, const float *sb, float *pcm);

/* audio_mp2.c: a Layer I or II frame (its bytes from the header on) into
   pcm[ch][384 or 1152]: how many samples a channel, or -1 */
int mpa_layer12(const uint8_t *f, size_t n, const struct mpa_hdr *h,
                struct mpa_synth *s, float (*pcm)[1152]);

#endif
