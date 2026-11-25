/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * image.c - a picture's format told by its first bytes, and the decoder
 * of that format (image.h).
 */
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int janas_image_decode(const void *data, size_t n, struct janas_image *im,
                       char *why, size_t wn)
{
    const uint8_t *p = data;
    memset(im, 0, sizeof(*im));
    if (n >= 8 && !memcmp(p, "\x89PNG\r\n\x1a\n", 8))
        return janas_png_decode(p, n, im, why, wn);
    if (n >= 3 && p[0] == 0xff && p[1] == 0xd8 && p[2] == 0xff)
        return janas_jpeg_decode(p, n, im, why, wn);
    if (n >= 12 && !memcmp(p, "RIFF", 4) && !memcmp(p + 8, "WEBP", 4))
        snprintf(why, wn, "WebP pictures are not read yet: send PNG or JPEG");
    else
        snprintf(why, wn, "not a picture this reads (PNG or JPEG)");
    return -1;
}

void janas_image_free(struct janas_image *im)
{
    free(im->rgb);
    memset(im, 0, sizeof(*im));
}
