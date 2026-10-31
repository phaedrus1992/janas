/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deflate.h - the size of a zlib stream (deflate.c).
 */
#ifndef JANAS_COMMON_DEFLATE_H
#define JANAS_COMMON_DEFLATE_H

#include <stddef.h>
#include <stdint.h>

/* the bytes zlib's compress() makes of n bytes at its default level, to
   the byte; -1 when out of memory or longer than one window (32 KiB less
   262 bytes) */
long janas_deflate_size(const uint8_t *p, size_t n);

#endif
