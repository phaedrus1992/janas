/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * multipart.h - a multipart/form-data body (an upload) taken apart
 * (multipart.c).
 */
#ifndef JANAS_SERVER_MULTIPART_H
#define JANAS_SERVER_MULTIPART_H

#include <stddef.h>

struct srv_part {
    char name[64];      /* the form field */
    char filename[256]; /* as the client sent it: never a path here */
    int has_filename;
    char type[128];   /* its Content-Type, or "" */
    const char *data; /* into the body */
    size_t n;
};

/* The parts of body (ctype: the request's Content-Type, its boundary in
   it), at most max: how many, or -1 with the reason in err. */
int srv_multipart_parse(const char *ctype, const char *body, size_t n,
                        struct srv_part *parts, int max, char *err,
                        size_t err_len);
/* The part of that name, or NULL. */
const struct srv_part *srv_multipart_get(const struct srv_part *parts, int n,
                                         const char *name);

#endif
