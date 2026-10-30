/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * onnx_pb.h - the protobuf wire format, as the converters of ONNX models
 * (silero2jns, wespeaker2jns) read it: fields one after another, their
 * numbers, wire types and bytes.
 */
#ifndef JANAS_TOOLS_ONNX_PB_H
#define JANAS_TOOLS_ONNX_PB_H

#include <stdint.h>
#include <string.h>

struct pb {
    const uint8_t *p, *end;
};

static inline int varint(struct pb *b, uint64_t *v)
{
    *v = 0;
    for (int s = 0; s < 64 && b->p < b->end; s += 7) {
        uint8_t c = *b->p++;
        *v |= (uint64_t)(c & 0x7F) << s;
        if (!(c & 0x80))
            return 0;
    }
    return -1;
}

/* the next field: its number, wire type, and for length-delimited ones
   its bytes in sub; 0 at the end, -1 broken */
static inline int field(struct pb *b, int *num, int *wt, uint64_t *v,
                        struct pb *sub)
{
    if (b->p >= b->end)
        return 0;
    uint64_t key;
    if (varint(b, &key) != 0)
        return -1;
    *num = (int)(key >> 3);
    *wt = (int)(key & 7);
    switch (*wt) {
    case 0:
        return varint(b, v) == 0 ? 1 : -1;
    case 1:
        if (b->end - b->p < 8)
            return -1;
        b->p += 8;
        return 1;
    case 5:
        if (b->end - b->p < 4)
            return -1;
        b->p += 4;
        return 1;
    case 2:
        if (varint(b, v) != 0 || *v > (uint64_t)(b->end - b->p))
            return -1;
        sub->p = b->p;
        sub->end = b->p + *v;
        b->p += *v;
        return 1;
    default:
        return -1;
    }
}

static inline int str_eq(const struct pb *s, const char *c)
{
    size_t n = strlen(c);
    return (size_t)(s->end - s->p) == n && !memcmp(s->p, c, n);
}

#endif
