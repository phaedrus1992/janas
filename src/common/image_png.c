/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * image_png.c - PNG (ISO/IEC 15948) into 8-bit RGB (image.h): every color
 * type and bit depth, palettes, Adam7 interlacing. The chunks are walked
 * with their lengths checked, the pixel data inflated into a buffer of
 * exactly the size the header says (inflate.c), the rows unfiltered, the
 * samples made RGB; transparency (tRNS, alpha) is dropped. The result is
 Pillow's convert("RGB"), pixel for pixel.
 */
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inflate.h"

static uint32_t be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 |
           p[3];
}

static int paeth(int a, int b, int c)
{
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* rows of rb bytes (each after its filter byte), bpp bytes a pixel for the
   filters: unfiltered in place, 0 or -1 */
static int unfilter(uint8_t *d, int rows, size_t rb, int bpp)
{
    uint8_t *prev = NULL;
    for (int y = 0; y < rows; y++) {
        uint8_t f = d[0], *r = d + 1;
        for (size_t i = 0; i < rb; i++) {
            int a = i >= (size_t)bpp ? r[i - (size_t)bpp] : 0;
            int b = prev ? prev[i] : 0;
            int c = prev && i >= (size_t)bpp ? prev[i - (size_t)bpp] : 0;
            switch (f) {
            case 0:
                break;
            case 1:
                r[i] = (uint8_t)(r[i] + a);
                break;
            case 2:
                r[i] = (uint8_t)(r[i] + b);
                break;
            case 3:
                r[i] = (uint8_t)(r[i] + ((a + b) >> 1));
                break;
            case 4:
                r[i] = (uint8_t)(r[i] + paeth(a, b, c));
                break;
            default:
                return -1;
            }
        }
        prev = r;
        d += rb + 1;
    }
    return 0;
}

struct png {
    int w, h, depth, ctype, channels;
    uint8_t pal[256][3];
    int npal;
};

/* sample k of a row (depth bits each), scaled to 8 bits unless it is a
   palette index */
static int sample(const struct png *g, const uint8_t *row, size_t k)
{
    switch (g->depth) {
    case 16:
        /* the high byte; a lone gray channel clipped at 255 instead, as
           Pillow reads it (mode I;16), so the vision models get what their
           processors gave them in training */
        if (g->ctype == 0) {
            int v = row[2 * k] << 8 | row[2 * k + 1];
            return v > 255 ? 255 : v;
        }
        return row[2 * k];
    case 8:
        return row[k];
    default: {
        size_t bit = k * (size_t)g->depth;
        int v = (row[bit >> 3] >> (8 - g->depth - (int)(bit & 7))) &
                ((1 << g->depth) - 1);
        if (g->ctype == 3)
            return v;
        return v * 255 / ((1 << g->depth) - 1);
    }
    }
}

/* row y of a pass of pw pixels into the picture, at x0 + x * dx */
static void put_row(const struct png *g, const uint8_t *row, int pw,
                    uint8_t *dst, int x0, int dx)
{
    for (int x = 0; x < pw; x++) {
        uint8_t *o = dst + (size_t)(x0 + x * dx) * 3;
        size_t s = (size_t)x * (size_t)g->channels;
        switch (g->ctype) {
        case 0:
        case 4: { /* gray, gray + alpha */
            int v = sample(g, row, s);
            o[0] = o[1] = o[2] = (uint8_t)v;
            break;
        }
        case 2:
        case 6: /* RGB, RGBA */
            o[0] = (uint8_t)sample(g, row, s);
            o[1] = (uint8_t)sample(g, row, s + 1);
            o[2] = (uint8_t)sample(g, row, s + 2);
            break;
        default: { /* palette */
            int i = sample(g, row, s);
            if (i < g->npal)
                memcpy(o, g->pal[i], 3);
            else
                o[0] = o[1] = o[2] = 0;
        }
        }
    }
}

static size_t row_bytes(const struct png *g, int w)
{
    return ((size_t)w * (size_t)g->channels * (size_t)g->depth + 7) / 8;
}

int janas_png_decode(const uint8_t *p, size_t n, struct janas_image *im,
                     char *why, size_t wn)
{
    static const uint8_t SIG[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    memset(im, 0, sizeof(*im));
    if (n < 8 || memcmp(p, SIG, 8)) {
        snprintf(why, wn, "not a PNG");
        return -1;
    }
    struct png g = {0};
    uint8_t *idat = NULL;
    size_t nidat = 0;
    int have_hdr = 0, interlace = 0, ended = 0;
    size_t pos = 8;
    const char *err = NULL;
    while (!err && !ended) {
        if (n - pos < 12) {
            err = "the file ends inside a chunk";
            break;
        }
        uint32_t len = be32(p + pos);
        const uint8_t *type = p + pos + 4, *d = p + pos + 8;
        if (len > n - pos - 12) {
            err = "a chunk runs past the end of the file";
            break;
        }
        if (!memcmp(type, "IHDR", 4)) {
            if (len != 13 || have_hdr) {
                err = "a bad header";
                break;
            }
            g.w = (int)be32(d) > 0 ? (int)be32(d) : 0;
            g.h = (int)be32(d + 4) > 0 ? (int)be32(d + 4) : 0;
            g.depth = d[8];
            g.ctype = d[9];
            interlace = d[12];
            static const int CH[7] = {1, 0, 3, 1, 2, 0, 4};
            g.channels = g.ctype <= 6 ? CH[g.ctype] : 0;
            int dep = g.depth;
            int ok =
                g.channels && d[10] == 0 && d[11] == 0 && interlace < 2 &&
                (g.ctype == 0
                     ? dep == 1 || dep == 2 || dep == 4 || dep == 8 || dep == 16
                 : g.ctype == 3 ? dep == 1 || dep == 2 || dep == 4 || dep == 8
                                : dep == 8 || dep == 16);
            if (!ok)
                err = "a color type and depth PNG does not have";
            else if (!g.w || !g.h || g.w > JANAS_IMAGE_MAX_SIDE ||
                     g.h > JANAS_IMAGE_MAX_SIDE ||
                     (size_t)g.w * (size_t)g.h > JANAS_IMAGE_MAX_PIXELS)
                err = "a picture larger than allowed (16384 a side, 64 "
                      "Mpixel)";
            have_hdr = 1;
        } else if (!memcmp(type, "PLTE", 4)) {
            if (len % 3 || len > 768) {
                err = "a bad palette";
                break;
            }
            g.npal = (int)len / 3;
            memcpy(g.pal, d, len);
        } else if (!memcmp(type, "IDAT", 4)) {
            uint8_t *t = realloc(idat, nidat + len + 1);
            if (!t) {
                err = "out of memory";
                break;
            }
            idat = t;
            memcpy(idat + nidat, d, len);
            nidat += len;
        } else if (!memcmp(type, "IEND", 4)) {
            ended = 1;
        } else if (!(type[0] & 0x20)) {
            err = "a chunk a reader must know and this one does not";
        }
        pos += 12 + (size_t)len;
    }
    if (!err && (!have_hdr || !nidat))
        err = "no picture in the file";
    if (!err && g.ctype == 3 && !g.npal)
        err = "a palette picture without its palette";
    /* the inflated data: each pass's rows, each with its filter byte */
    static const int AX[7] = {0, 4, 0, 2, 0, 1, 0},
                     AY[7] = {0, 0, 4, 0, 2, 0, 1};
    static const int DX[7] = {8, 8, 4, 4, 2, 2, 1},
                     DY[7] = {8, 8, 8, 4, 4, 2, 2};
    int passes = interlace ? 7 : 1;
    size_t raw = 0;
    for (int i = 0; !err && i < passes; i++) {
        int pw = interlace ? (g.w - AX[i] + DX[i] - 1) / DX[i] : g.w;
        int ph = interlace ? (g.h - AY[i] + DY[i] - 1) / DY[i] : g.h;
        if (pw > 0 && ph > 0)
            raw += (size_t)ph * (row_bytes(&g, pw) + 1);
    }
    /* DEFLATE packs at most about 1032 bytes into one: a header claiming
       more pixels than the data can hold is refused before the memory is
       taken */
    if (!err && raw / 1032 > nidat + 64)
        err = "a picture larger than its data";
    uint8_t *px = err ? NULL : malloc(raw ? raw : 1);
    im->rgb = err ? NULL : malloc((size_t)g.w * (size_t)g.h * 3);
    if (!err && (!px || !im->rgb))
        err = "out of memory";
    char zwhy[128];
    if (!err && janas_zlib_inflate(idat, nidat, px, raw, zwhy, sizeof zwhy)) {
        snprintf(why, wn, "%s", zwhy);
        err = why;
    }
    int bpp = (g.channels * g.depth + 7) / 8;
    uint8_t *d = px;
    for (int i = 0; !err && i < passes; i++) {
        int pw = interlace ? (g.w - AX[i] + DX[i] - 1) / DX[i] : g.w;
        int ph = interlace ? (g.h - AY[i] + DY[i] - 1) / DY[i] : g.h;
        if (pw <= 0 || ph <= 0)
            continue;
        size_t rb = row_bytes(&g, pw);
        if (unfilter(d, ph, rb, bpp) != 0) {
            err = "a row with an unknown filter";
            break;
        }
        for (int y = 0; y < ph; y++) {
            int yy = interlace ? AY[i] + y * DY[i] : y;
            put_row(&g, d + (size_t)y * (rb + 1) + 1, pw,
                    im->rgb + (size_t)yy * (size_t)g.w * 3,
                    interlace ? AX[i] : 0, interlace ? DX[i] : 1);
        }
        d += (size_t)ph * (rb + 1);
    }
    free(idat);
    free(px);
    if (err) {
        if (err != why)
            snprintf(why, wn, "PNG: %s", err);
        free(im->rgb);
        memset(im, 0, sizeof(*im));
        return -1;
    }
    im->w = g.w;
    im->h = g.h;
    return 0;
}
