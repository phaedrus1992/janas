/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision.h - a model's vision encoder (vision.c), from the JNS file
 * tools/hf2jns_vision.c writes: a picture in, the embeddings of its tokens
 * out, as wide as the language model's.
 */
#ifndef JANAS_LLM_VISION_H
#define JANAS_LLM_VISION_H

#include <stddef.h>
#include <stdint.h>

struct janas_image;
struct janas_pool;
struct janas_vision;

/* NULL with the reason in err. */
struct janas_vision *janas_vision_open(const char *path, char *err,
                                       size_t err_len);
void janas_vision_close(struct janas_vision *v);

/* The width of an embedding, and the tokens that frame a picture. */
uint32_t janas_vision_dim(const struct janas_vision *v);
/* 1 when a picture's tokens attend to one another both ways in the
   language model (Gemma 4 12B and 26B) */
int janas_vision_both_ways(const struct janas_vision *v);
void janas_vision_tokens(const struct janas_vision *v, int32_t *image,
                         int32_t *start, int32_t *end);

/* The size a picture of w x h is resized to (multiples of the patch and the
   merge, its pixels within the model's bounds and max_tokens tokens when
   that is not 0), and the grid of tokens it gives. */
void janas_vision_size(const struct janas_vision *v, int w, int h,
                       int max_tokens, int *rw, int *rh, int *nx, int *ny);

/* The picture's embeddings: *out (malloc'd) nx * ny rows of
   janas_vision_dim floats, row-major over the token grid. 0, or -1 with
   the reason. pool may be NULL (one thread). */
int janas_vision_encode(struct janas_vision *v, const struct janas_image *im,
                        int max_tokens, struct janas_pool *pool, float **out,
                        int *nx, int *ny, char *err, size_t err_len);

#endif
