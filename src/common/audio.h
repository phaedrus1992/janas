/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio.h - sound read from its files into mono float samples in [-1, 1],
 * as the speech models take it: the channels averaged, then resampled to
 * the model's rate. Janas's own decoders (audio_wav.c, audio_flac.c,
 * audio_mp3.c and audio_mp2.c, audio_ogg.c with audio_vorbis.c and
 * audio_opus*.c), with bounds on sizes so a file cannot make them take
 * more memory than its sound.
 */
#ifndef JANAS_COMMON_AUDIO_H
#define JANAS_COMMON_AUDIO_H

#include <stddef.h>
#include <stdint.h>

/* at most four hours at 48 kHz, after mixing */
#define JANAS_AUDIO_MAX_SAMPLES ((size_t)48000 * 3600 * 4)

struct janas_audio {
    float *pcm; /* n samples, mono */
    size_t n;
    uint32_t rate; /* samples a second */
};

/* n bytes of a sound file (WAV, FLAC, MP3 and MPEG audio, Ogg with
   Vorbis, Opus or FLAC), the kind told by its bytes: 0 with *a filled (free it
   with janas_audio_free), -1 with the reason in why. */
int janas_audio_decode(const void *data, size_t n, struct janas_audio *a,
                       char *why, size_t wn);
/* the sound resampled to rate, in place (a windowed sinc, its cut-off
   below the lower of the two Nyquist frequencies): 0, or -1 when out of
   memory */
int janas_audio_resample(struct janas_audio *a, uint32_t rate);
void janas_audio_free(struct janas_audio *a);

/* audio_wav.c: RIFF WAVE, PCM of 8 to 32 bits or float of 32 or 64,
   plain or extensible; 1 when the bytes are not a WAV file */
int janas_audio_wav(const uint8_t *d, size_t n, struct janas_audio *a,
                    char *why, size_t wn);
/* audio_mp3.c (and audio_mp2.c): MPEG-1, 2 and 2.5 audio, Layers I, II
   and III; 1 when no frames are found in the first 64 KiB */
int janas_audio_mpeg(const uint8_t *d, size_t n, struct janas_audio *a,
                     char *why, size_t wn);
/* audio_ogg.c: Ogg with Vorbis, Opus or FLAC; 1 when not Ogg */
int janas_audio_ogg(const uint8_t *d, size_t n, struct janas_audio *a,
                    char *why, size_t wn);
/* audio_flac.c: FLAC, every kind RFC 9639 describes; 1 when not FLAC */
int janas_audio_flac(const uint8_t *d, size_t n, struct janas_audio *a,
                     char *why, size_t wn);
/* audio_mp4.c: MP4, M4A, MOV with AAC, ALAC, MP3, Opus, FLAC or PCM; 1
   when not MP4 */
int janas_audio_mp4(const uint8_t *d, size_t n, struct janas_audio *a,
                    char *why, size_t wn);
/* audio_mkv.c: Matroska and WebM with Opus, Vorbis, FLAC, AAC, MPEG
   audio, ALAC or PCM; 1 when not EBML */
int janas_audio_mkv(const uint8_t *d, size_t n, struct janas_audio *a,
                    char *why, size_t wn);
/* audio_aac.c: AAC in ADTS frames (.aac); 1 when not ADTS */
int janas_audio_adts(const uint8_t *d, size_t n, struct janas_audio *a,
                     char *why, size_t wn);

#endif
