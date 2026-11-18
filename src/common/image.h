/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * image.h - pictures read from their files (PNG, JPEG) into 8-bit RGB, as
 * the vision models take them: the pixels Pillow's convert("RGB") gives
 * (an alpha channel dropped, 16-bit samples their high byte, a 16-bit gray
 * clipped at 255), which is what the models' processors gave them. Janas's
 * own decoders (image_png.c, image_jpeg.c), with bounds on sizes so a file
 * cannot make them take more memory than the picture.
 */
#ifndef JANAS_COMMON_IMAGE_H
#define JANAS_COMMON_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_IMAGE_MAX_SIDE 16384
#define JANAS_IMAGE_MAX_PIXELS ((size_t)64 << 20)

struct janas_image {
    int w, h;
    uint8_t *rgb; /* w * h * 3 bytes, rows top to bottom */
};

/* n bytes of a PNG or a JPEG: 0 with *im filled (free it with
   janas_image_free), -1 with the reason in why. */
int janas_image_decode(const void *data, size_t n, struct janas_image *im,
                       char *why, size_t wn);
void janas_image_free(struct janas_image *im);

/* in resized to w x h into *out (image_resize.c) as Pillow's
   resize(BICUBIC) does - the filter widened when shrinking, so every source
   pixel counts - which is what the vision processors use: 0, or -1 on
   memory or a size of 0. */
int janas_image_resize(const struct janas_image *in, int w, int h,
                       struct janas_image *out);

/* the formats, called by janas_image_decode */
int janas_png_decode(const uint8_t *p, size_t n, struct janas_image *im,
                     char *why, size_t wn);
int janas_jpeg_decode(const uint8_t *p, size_t n, struct janas_image *im,
                      char *why, size_t wn);

#endif
