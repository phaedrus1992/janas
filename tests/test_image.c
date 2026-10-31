/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_image.c - the picture decoders (src/common/image*.c): small PNGs
 * and JPEGs of every kind (palette with transparency, 16-bit gray, RGBA,
 * Adam7; baseline and progressive, 4:2:0, 4:2:2, 4:4:0, gray, restart
 * intervals) give the pixels Pillow gives, by the hash of their RGB, and
 * so does the bicubic resize, up and down; then
 * every truncation and thousands of corrupted copies of each must be read
 * or refused, never crash (the sanitizer builds watch).
 */
#include "common/image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_image_data.h"

static int failures;

static unsigned long long fnv(const unsigned char *p, size_t n)
{
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

static unsigned long long rng = 0x9e3779b97f4a7c15ULL;

static unsigned next(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (unsigned)(rng >> 32);
}

int main(void)
{
    const int n = (int)(sizeof PICS / sizeof PICS[0]);
    char why[200];
    for (int i = 0; i < n; i++) {
        const struct pic *p = &PICS[i];
        struct janas_image im;
        if (janas_image_decode(p->data, p->n, &im, why, sizeof why) != 0) {
            fprintf(stderr, "test_image: %s: %s\n", p->name, why);
            failures++;
            continue;
        }
        unsigned long long h = fnv(im.rgb, (size_t)im.w * (size_t)im.h * 3);
        if (im.w != p->w || im.h != p->h || h != p->fnv) {
            fprintf(stderr,
                    "test_image: %s: %dx%d, not the pixels Pillow "
                    "gives\n",
                    p->name, im.w, im.h);
            failures++;
        }
        janas_image_free(&im);
    }
    /* resized as Pillow's resize(BICUBIC): up and down */
    static const struct {
        int pic, w, h;
        unsigned long long fnv;
    } RS[] = {{3, 13, 9, 0x22722aaa28b4a0cfULL},
              {4, 37, 21, 0xaf2f59852ea157f3ULL}};
    for (size_t i = 0; i < sizeof RS / sizeof RS[0]; i++) {
        const struct pic *p = &PICS[RS[i].pic];
        struct janas_image im, out;
        if (janas_image_decode(p->data, p->n, &im, why, sizeof why) != 0 ||
            janas_image_resize(&im, RS[i].w, RS[i].h, &out) != 0) {
            fprintf(stderr, "test_image: %s not resized\n", p->name);
            failures++;
            continue;
        }
        if (fnv(out.rgb, (size_t)RS[i].w * (size_t)RS[i].h * 3) != RS[i].fnv) {
            fprintf(stderr, "test_image: %s to %dx%d: not Pillow's pixels\n",
                    p->name, RS[i].w, RS[i].h);
            failures++;
        }
        janas_image_free(&im);
        janas_image_free(&out);
    }
    /* broken files: read or refused, never more */
    long refused = 0, read = 0;
    for (int i = 0; i < n; i++) {
        const struct pic *p = &PICS[i];
        unsigned char *b = malloc(p->n);
        if (!b)
            return 1;
        for (unsigned cut = 0; cut < p->n; cut += 1 + p->n / 300) {
            struct janas_image im;
            if (janas_image_decode(p->data, cut, &im, why, sizeof why) == 0) {
                read++;
                janas_image_free(&im);
            } else {
                refused++;
            }
        }
        for (int k = 0; k < 1500; k++) {
            memcpy(b, p->data, p->n);
            int flips = 1 + (int)(next() % 8);
            for (int f = 0; f < flips; f++)
                b[next() % p->n] = (unsigned char)next();
            struct janas_image im;
            if (janas_image_decode(b, p->n, &im, why, sizeof why) == 0) {
                read++;
                janas_image_free(&im);
            } else {
                refused++;
            }
        }
        free(b);
    }
    if (failures) {
        fprintf(stderr, "test_image: %d failures\n", failures);
        return 1;
    }
    printf("test_image: ok (%d pictures as Pillow; broken copies: %ld read, "
           "%ld refused)\n",
           n, read, refused);
    return 0;
}
