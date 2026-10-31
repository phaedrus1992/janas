/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * files.c - OpenAI's Files API: a file uploaded (multipart/form-data, as
 * the SDKs send it), listed, read, its bytes downloaded, deleted. The files
 * themselves are files_store.c's; what a batch or a vector store does with
 * them comes later.
 */
#include "server.h"

#include <microhttpd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

#include "files.h"
#include "files_store.h"
#include "multipart.h"

static const char *const PURPOSES[] = {"assistants", "batch",     "fine-tune",
                                       "vision",     "user_data", "evals"};

static int purpose_ok(const char *p)
{
    for (size_t i = 0; i < sizeof(PURPOSES) / sizeof(PURPOSES[0]); i++)
        if (strcmp(p, PURPOSES[i]) == 0)
            return 1;
    return 0;
}

yyjson_mut_val *srv_file_json(yyjson_mut_doc *d, const struct srv_file *x)
{
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", x->id);
    yyjson_mut_obj_add_str(d, o, "object", "file");
    yyjson_mut_obj_add_uint(d, o, "bytes", x->bytes);
    yyjson_mut_obj_add_int(d, o, "created_at", x->created_at);
    if (x->expires_at)
        yyjson_mut_obj_add_int(d, o, "expires_at", x->expires_at);
    else
        yyjson_mut_obj_add_null(d, o, "expires_at");
    yyjson_mut_obj_add_strcpy(d, o, "filename", x->filename);
    yyjson_mut_obj_add_strcpy(d, o, "purpose", x->purpose);
    /* deprecated at OpenAI, still read by older clients */
    yyjson_mut_obj_add_str(d, o, "status", "processed");
    yyjson_mut_obj_add_null(d, o, "status_details");
    return o;
}

static int reply_doc(struct srv_req *r, yyjson_mut_doc *d, yyjson_mut_val *o)
{
    yyjson_mut_doc_set_root(d, o);
    size_t n;
    char *out = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, n);
}

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

static int no_file(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No such File object: %s", r->param[0]);
}

/* The text of a form field, NUL-terminated into buf: 0, or -1. */
static int field(const struct srv_part *p, char *buf, size_t cap)
{
    if (!p || p->n >= cap || p->has_filename)
        return -1;
    memcpy(buf, p->data, p->n);
    buf[p->n] = 0;
    return 0;
}

int srv_files_create(struct srv_req *r)
{
    struct srv_part parts[8];
    char err[200];
    const char *ct =
        MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "Content-Type");
    int n =
        srv_multipart_parse(ct, r->body, r->n_body, parts, 8, err, sizeof err);
    if (n < 0)
        return bad(r, err);
    const struct srv_part *file = srv_multipart_get(parts, n, "file");
    char purpose[32], anchor[32] = "", secs[32] = "";
    if (!file || !file->has_filename)
        return bad(r, "'file' is required: the file's bytes, with a name.");
    if (field(srv_multipart_get(parts, n, "purpose"), purpose,
              sizeof purpose) != 0 ||
        !purpose_ok(purpose))
        return bad(r, "'purpose' is required: assistants, batch, fine-tune, "
                      "vision, user_data or evals.");
    const struct srv_part *pa =
        srv_multipart_get(parts, n, "expires_after[anchor]");
    const struct srv_part *ps =
        srv_multipart_get(parts, n, "expires_after[seconds]");
    long long after = 0;
    if (pa || ps) {
        char *end = NULL;
        if (field(pa, anchor, sizeof anchor) != 0 ||
            strcmp(anchor, "created_at") != 0 ||
            field(ps, secs, sizeof secs) != 0 ||
            (after = strtoll(secs, &end, 10), *end) || after < 3600 ||
            after > 2592000)
            return bad(r, "'expires_after' is anchor created_at and seconds "
                          "from 3600 to 2592000.");
    } else if (strcmp(purpose, "batch") == 0) {
        after = 2592000; /* as OpenAI: a batch's files go after 30 days */
    }
    struct srv_file x;
    int rc = srv_files_add(r->srv->files, file->filename, purpose, file->data,
                           file->n, after, &x, err, sizeof err);
    if (rc == -2)
        return srv_reply_error(r, 413, "invalid_request_error",
                               "files_limit_reached", "%s", err);
    if (rc != 0)
        return srv_reply_error(r, 500, "server_error", NULL, "%s", err);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    return reply_doc(r, d, srv_file_json(d, &x));
}

int srv_files_list_req(struct srv_req *r)
{
    const char *purpose = srv_query(r, "purpose"),
               *after = srv_query(r, "after");
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    long limit = 10000;
    if (l) {
        char *end;
        limit = strtol(l, &end, 10);
        if (*end || limit < 1 || limit > 10000)
            return bad(r, "'limit' is a number from 1 to 10000.");
    }
    if (o && strcmp(o, "asc") != 0 && strcmp(o, "desc") != 0)
        return bad(r, "'order' is asc or desc.");
    struct srv_file *all;
    size_t n =
        srv_files_list(r->srv->files, purpose && *purpose ? purpose : NULL,
                       o && !strcmp(o, "asc"), after, &all);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    size_t shown = n < (size_t)limit ? n : (size_t)limit;
    for (size_t i = 0; i < shown; i++)
        yyjson_mut_arr_append(data, srv_file_json(d, &all[i]));
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_obj_add_val(d, root, "data", data);
    if (shown) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", all[0].id);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", all[shown - 1].id);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", n > shown);
    free(all);
    return reply_doc(r, d, root);
}

int srv_files_retrieve(struct srv_req *r)
{
    struct srv_file x;
    if (srv_files_get(r->srv->files, r->param[0], &x) != 0)
        return no_file(r);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    return reply_doc(r, d, srv_file_json(d, &x));
}

int srv_files_delete_req(struct srv_req *r)
{
    if (srv_files_delete(r->srv->files, r->param[0]) != 0)
        return no_file(r);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", r->param[0]);
    yyjson_mut_obj_add_str(d, o, "object", "file");
    yyjson_mut_obj_add_bool(d, o, "deleted", true);
    return reply_doc(r, d, o);
}

int srv_files_content(struct srv_req *r)
{
    struct srv_file x;
    if (srv_files_get(r->srv->files, r->param[0], &x) != 0)
        return no_file(r);
    int fd = srv_files_open_content(r->srv->files, r->param[0]);
    if (fd < 0)
        return no_file(r);
    return srv_reply_fd(r, fd, x.bytes, "application/octet-stream");
}
