/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * multipart.c - a multipart/form-data body taken apart (see multipart.h).
 * What a client sends is input from outside: every length is checked
 * against the body, and the parts point into it, never past it.
 */
#include "multipart.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

static const char *mp_find(const char *p, const char *end, const char *s,
                           size_t n)
{
    for (; p && (size_t)(end - p) >= n; p++) {
        p = memchr(p, s[0], (size_t)(end - p) - n + 1);
        if (!p)
            return NULL;
        if (memcmp(p, s, n) == 0)
            return p;
    }
    return NULL;
}

/* The value of param (boundary, name, filename) in a header's text: quoted
   or not; 0 with it in out, -1 absent or too long. */
static int param(const char *h, size_t hn, const char *key, char *out,
                 size_t cap)
{
    size_t kn = strlen(key);
    for (size_t i = 0; i + kn + 1 <= hn; i++) {
        /* at the start or after ';' and spaces, key= */
        if (strncasecmp(h + i, key, kn) != 0 || h[i + kn] != '=')
            continue;
        size_t j = i;
        while (j > 0 && (h[j - 1] == ' ' || h[j - 1] == '\t'))
            j--;
        if (j > 0 && h[j - 1] != ';')
            continue;
        const char *v = h + i + kn + 1, *e = h + hn;
        size_t n = 0;
        if (v < e && *v == '"') {
            for (v++; v < e && *v != '"'; v++) {
                if (*v == '\\' && v + 1 < e)
                    v++;
                if (n + 1 >= cap)
                    return -1;
                out[n++] = *v;
            }
            if (v >= e)
                return -1;
        } else {
            for (; v < e && *v != ';' && *v != ' ' && *v != '\t'; v++) {
                if (n + 1 >= cap)
                    return -1;
                out[n++] = *v;
            }
        }
        out[n] = 0;
        return 0;
    }
    return -1;
}

int srv_multipart_parse(const char *ctype, const char *body, size_t n,
                        struct srv_part *parts, int max, char *err,
                        size_t err_len)
{
    char bnd[80] = "--";
    if (!ctype || strncasecmp(ctype, "multipart/form-data", 19) != 0 ||
        param(ctype, strlen(ctype), "boundary", bnd + 2, sizeof(bnd) - 2) !=
            0 ||
        !bnd[2]) {
        snprintf(err, err_len,
                 "the body is multipart/form-data, with its boundary");
        return -1;
    }
    size_t bl = strlen(bnd);
    const char *end = body + n;
    const char *p = mp_find(body, end, bnd, bl);
    int count = 0;
    while (p) {
        p += bl;
        if (end - p >= 2 && p[0] == '-' && p[1] == '-')
            return count; /* the closing boundary */
        if (end - p < 2 || p[0] != '\r' || p[1] != '\n')
            break;
        p += 2;
        const char *he = mp_find(p, end, "\r\n\r\n", 4);
        if (!he)
            break;
        struct srv_part pt = {0};
        int named = 0;
        for (const char *l = p; l < he;) {
            const char *le = mp_find(l, he + 2, "\r\n", 2);
            if (!le)
                le = he;
            size_t ln = (size_t)(le - l);
            if (ln > 20 && strncasecmp(l, "Content-Disposition:", 20) == 0) {
                named = param(l, ln, "name", pt.name, sizeof(pt.name)) == 0;
                pt.has_filename = param(l, ln, "filename", pt.filename,
                                        sizeof(pt.filename)) == 0;
            } else if (ln > 13 && strncasecmp(l, "Content-Type:", 13) == 0) {
                const char *v = l + 13;
                while (v < le && *v == ' ')
                    v++;
                size_t vn = (size_t)(le - v);
                if (vn >= sizeof(pt.type))
                    vn = sizeof(pt.type) - 1;
                memcpy(pt.type, v, vn);
                pt.type[vn] = 0;
            }
            l = le + 2;
        }
        const char *data = he + 4;
        /* the part ends before CRLF and the next boundary */
        char next[84];
        snprintf(next, sizeof(next), "\r\n%s", bnd);
        const char *q = mp_find(data, end, next, bl + 2);
        if (!q || !named)
            break;
        pt.data = data;
        pt.n = (size_t)(q - data);
        if (count == max) {
            snprintf(err, err_len, "more than %d parts", max);
            return -1;
        }
        parts[count++] = pt;
        p = q + 2;
    }
    snprintf(err, err_len, "the multipart body is cut or malformed");
    return -1;
}

const struct srv_part *srv_multipart_get(const struct srv_part *parts, int n,
                                         const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(parts[i].name, name) == 0)
            return &parts[i];
    return NULL;
}
