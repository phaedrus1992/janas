/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * md5.h - MD5 (RFC 1321), only to check an upload against the checksum a
 * client sends with it, as OpenAI's Uploads API asks: not for security.
 */
#ifndef JANAS_COMMON_MD5_H
#define JANAS_COMMON_MD5_H

#include <stddef.h>
#include <stdint.h>

struct janas_md5 {
    uint32_t h[4];
    uint64_t len; /* bytes hashed */
    uint8_t buf[64];
    size_t n; /* bytes waiting in buf */
};

void janas_md5_init(struct janas_md5 *s);
void janas_md5_update(struct janas_md5 *s, const void *data, size_t n);
/* The digest as 32 lowercase hex digits and a NUL. */
void janas_md5_hex(struct janas_md5 *s, char out[33]);

#endif
