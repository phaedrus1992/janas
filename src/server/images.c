/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * images.c - the pictures of a request's messages: chat's image_url parts
 * ({"url": a data: URL or http(s)}), the Responses' input_image parts (an
 * image_url, or a file_id of the Files API). The parsers keep each one's
 * reference (srv_image_part); the handler, which has the server, turns
 * them into bytes before the job goes to the engine (srv_images_fetch): a
 * data: URL decoded from base64, an http(s) one fetched at a public address
 * only (the MCP policy: --mcp-allow names the exceptions), a file read from
 * the store. The engine gives them to the library (janas_llm_chat_image),
 * which reads PNG and JPEG.
 */
#include "server.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "files_store.h"
#include "services/common/https_get.h"

#define MAX_IMAGE_BYTES ((size_t)16 << 20)
#define MAX_IMAGES 16

void srv_imgs_free(struct srv_img *imgs, int n)
{
    for (int i = 0; i < n; i++) {
        free(imgs[i].ref);
        free(imgs[i].data);
    }
    free(imgs);
}

int srv_image_part(yyjson_val *part, const char *type, struct srv_img **imgs,
                   int *n_imgs, char *err, size_t el)
{
    const char *ref = NULL;
    int is_file = 0;
    if (type && !strcmp(type, "image_url")) { /* chat */
        yyjson_val *iu = yyjson_obj_get(part, "image_url");
        ref = yyjson_is_str(iu) ? yyjson_get_str(iu)
                                : yyjson_get_str(yyjson_obj_get(iu, "url"));
    } else if (type && !strcmp(type, "input_image")) { /* Responses */
        ref = yyjson_get_str(yyjson_obj_get(part, "image_url"));
        if (!ref) {
            ref = yyjson_get_str(yyjson_obj_get(part, "file_id"));
            is_file = ref != NULL;
        }
    } else {
        return 0;
    }
    if (!ref || !*ref) {
        snprintf(err, el, "a picture part has no image_url (or file_id)");
        return -1;
    }
    if (*n_imgs >= MAX_IMAGES) {
        snprintf(err, el, "at most %d pictures a request", MAX_IMAGES);
        return -1;
    }
    struct srv_img *t = realloc(*imgs, sizeof(*t) * (size_t)(*n_imgs + 1));
    char *r = strdup(ref);
    if (!t || !r) {
        free(r);
        if (t)
            *imgs = t;
        snprintf(err, el, "out of memory");
        return -1;
    }
    *imgs = t;
    t[(*n_imgs)++] = (struct srv_img){.ref = r, .is_file = is_file};
    return 1;
}

static int b64v(int c)
{
    return c >= 'A' && c <= 'Z'   ? c - 'A'
           : c >= 'a' && c <= 'z' ? c - 'a' + 26
           : c >= '0' && c <= '9' ? c - '0' + 52
           : c == '+' || c == '-' ? 62
           : c == '/' || c == '_' ? 63
                                  : -1;
}

int srv_data_url(const char *url, size_t len, size_t max, unsigned char **data,
                 size_t *n_data, char *err, size_t el)
{
    const char *end = url + len;
    const char *comma = memchr(url, ',', len);
    int b64 = 0;
    for (const char *q = url; comma && q + 7 <= comma; q++)
        b64 |= !memcmp(q, ";base64", 7);
    if (len < 5 || memcmp(url, "data:", 5) || !comma || !b64) {
        snprintf(err, el, "a data: URL in base64 is expected");
        return -1;
    }
    const char *p = comma + 1;
    size_t n = (size_t)(end - p);
    if (n / 4 * 3 > max) {
        snprintf(err, el, "a data: URL of more than %zu MiB", max >> 20);
        return -1;
    }
    unsigned char *out = malloc(n / 4 * 3 + 3);
    if (!out) {
        snprintf(err, el, "out of memory");
        return -1;
    }
    size_t w = 0;
    unsigned acc = 0;
    int bits = 0;
    for (; p < end; p++) {
        if (*p == '=' || *p == '\n' || *p == '\r' || *p == ' ')
            continue;
        int v = b64v((unsigned char)*p);
        if (v < 0) {
            free(out);
            snprintf(err, el, "a data: URL that is not base64");
            return -1;
        }
        acc = acc << 6 | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[w++] = (unsigned char)(acc >> bits);
        }
    }
    *data = out;
    *n_data = w;
    return 0;
}

/* data:[type][;base64],<data> */
static int from_data_url(struct srv_img *im, char *err, size_t el)
{
    return srv_data_url(im->ref, strlen(im->ref), MAX_IMAGE_BYTES, &im->data,
                        &im->n, err, el);
}

static int from_file(struct srv *s, struct srv_img *im, char *err, size_t el)
{
    struct srv_file f;
    if (srv_files_get(s->files, im->ref, &f) != 0) {
        snprintf(err, el, "no file %s here", im->ref);
        return -1;
    }
    if (f.bytes > MAX_IMAGE_BYTES) {
        snprintf(err, el, "a picture of more than 16 MiB");
        return -1;
    }
    int fd = srv_files_open_content(s->files, im->ref);
    unsigned char *b = fd >= 0 ? malloc(f.bytes ? f.bytes : 1) : NULL;
    size_t got = 0;
    while (b && got < f.bytes) {
        ssize_t k = read(fd, b + got, f.bytes - got);
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    if (fd >= 0)
        close(fd);
    if (!b || got != f.bytes) {
        free(b);
        snprintf(err, el, "the file %s cannot be read", im->ref);
        return -1;
    }
    im->data = b;
    im->n = got;
    return 0;
}

static int from_url(struct srv *s, struct srv_img *im, char *err, size_t el)
{
    struct janas_buf b = {0};
    char why[300];
    int st = janas_https_get_policy(im->ref, &s->cfg.mcp_policy, 15000, &b, why,
                                    sizeof why);
    if (st != 200) {
        janas_buf_free(&b);
        if (st < 0)
            snprintf(err, el, "the picture %.200s cannot be fetched: %s",
                     im->ref, why);
        else
            snprintf(err, el, "the picture %.200s answered HTTP %d", im->ref,
                     st);
        return -1;
    }
    im->data = (unsigned char *)b.p;
    im->n = b.n;
    return 0;
}

int srv_images_fetch(struct srv *s, struct srv_job *j, char *err, size_t el)
{
    for (int32_t i = 0; i < j->n_msg; i++) {
        struct srv_msg *m = &j->msg[i];
        for (int k = 0; k < m->n_imgs; k++) {
            struct srv_img *im = &m->imgs[k];
            if (im->data)
                continue;
            int rc = im->is_file ? from_file(s, im, err, el)
                     : !strncmp(im->ref, "data:", 5)
                         ? from_data_url(im, err, el)
                     : !strncmp(im->ref, "http://", 7) ||
                             !strncmp(im->ref, "https://", 8)
                         ? from_url(s, im, err, el)
                         : (snprintf(err, el,
                                     "a picture is a data: URL, an http(s) "
                                     "URL or a file_id"),
                            -1);
            if (rc != 0)
                return -1;
        }
    }
    return 0;
}
