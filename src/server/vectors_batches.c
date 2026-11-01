/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vectors_batches.c - a batch of files attached to a vector store at once
 * (OpenAI's file_batches): kept as the list of its files in the store's
 * batches/ directory, its status and counts read from theirs when asked;
 * cancelled, the files still waiting are cancelled.
 */
#include "server.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <yyjson.h>

#include "files_store.h"
#include "store_disk.h"
#include "vectors.h"

#define BATCH_FILES 2000

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

static int not_found(struct srv_req *r, const char *what, const char *id)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No %s found with id '%s'.", what, id);
}

static void batch_path(char *out, size_t cap, const char *vs, const char *id)
{
    srv_vectors_dir(out, cap, vs);
    size_t n = strlen(out);
    snprintf(out + n, cap - n, "/batches/%s.json", id);
}

static yyjson_doc *read_batch(const char *vs, const char *id)
{
    if (!srv_disk_name_ok(vs) || !srv_disk_name_ok(id))
        return NULL;
    char p[4800];
    batch_path(p, sizeof p, vs, id);
    FILE *f = fopen(p, "rb");
    if (!f)
        return NULL;
    char *t = NULL;
    long n = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    if (n >= 0 && !fseek(f, 0, SEEK_SET) && (t = malloc((size_t)n + 1)) &&
        fread(t, 1, (size_t)n, f) != (size_t)n) {
        free(t);
        t = NULL;
    }
    fclose(f);
    yyjson_doc *d = t ? yyjson_read(t, (size_t)n, 0) : NULL;
    free(t);
    return d;
}

/* A file's status in the store ("" when detached). */
static void file_status(const char *vs, const char *id, char *out, size_t cap)
{
    char p[4800];
    srv_vectors_dir(p, sizeof p, vs);
    size_t n = strlen(p);
    snprintf(p + n, sizeof p - n, "/files/%s.json", id);
    out[0] = 0;
    srv_vectors_lock();
    FILE *f = fopen(p, "rb");
    char buf[4096];
    size_t len = f ? fread(buf, 1, sizeof buf - 1, f) : 0;
    if (f)
        fclose(f);
    srv_vectors_unlock();
    buf[len] = 0;
    yyjson_doc *d = len ? yyjson_read(buf, len, 0) : NULL;
    const char *s =
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(d), "status"));
    snprintf(out, cap, "%s", s ? s : "");
    yyjson_doc_free(d);
}

/* The batch object, its counts and status from its files'. */
static int reply_batch(struct srv_req *r, const char *vs, const char *id)
{
    yyjson_doc *bd = read_batch(vs, id);
    yyjson_val *b = yyjson_doc_get_root(bd);
    if (!b)
        return not_found(r, "vector store file batch", id);
    long c[4] = {0, 0, 0, 0};
    static const char *const S[] = {"in_progress", "completed", "failed",
                                    "cancelled"};
    size_t idx, mx;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_obj_get(b, "file_ids"), idx, mx, v)
    {
        char st[32];
        file_status(vs, yyjson_get_str(v), st, sizeof st);
        for (int i = 0; i < 4; i++)
            if (!strcmp(st, S[i]))
                c[i]++;
    }
    int cancelled = yyjson_get_bool(yyjson_obj_get(b, "cancelled"));
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.files_batch");
    yyjson_mut_obj_add_int(d, o, "created_at",
                           yyjson_get_sint(yyjson_obj_get(b, "created_at")));
    yyjson_mut_obj_add_strcpy(d, o, "vector_store_id", vs);
    yyjson_mut_obj_add_str(d, o, "status",
                           cancelled ? "cancelled"
                           : c[0]    ? "in_progress"
                           : c[1]    ? "completed"
                                     : "failed");
    yyjson_mut_val *fc = yyjson_mut_obj_add_obj(d, o, "file_counts");
    yyjson_mut_obj_add_int(d, fc, "in_progress", c[0]);
    yyjson_mut_obj_add_int(d, fc, "completed", c[1]);
    yyjson_mut_obj_add_int(d, fc, "failed", c[2]);
    yyjson_mut_obj_add_int(d, fc, "cancelled", c[3]);
    yyjson_mut_obj_add_int(
        d, fc, "total",
        (int64_t)yyjson_arr_size(yyjson_obj_get(b, "file_ids")));
    yyjson_doc_free(bd);
    yyjson_mut_doc_set_root(d, o);
    size_t n;
    char *out = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, n);
}

static int store_ok(const char *vs)
{
    char p[4700];
    if (!srv_disk_name_ok(vs))
        return 0;
    srv_vectors_dir(p, sizeof p, vs);
    size_t n = strlen(p);
    snprintf(p + n, sizeof p - n, "/meta.json");
    return access(p, F_OK) == 0;
}

int srv_vs_batch_create(struct srv_req *r)
{
    const char *vs = r->param[0];
    if (!store_ok(vs))
        return not_found(r, "vector store", vs);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *ids = yyjson_obj_get(root, "file_ids");
    yyjson_val *files = yyjson_obj_get(root, "files");
    yyjson_val *cs = yyjson_obj_get(root, "chunking_strategy");
    yyjson_val *attrs = yyjson_obj_get(root, "attributes");
    const char *why = NULL;
    if ((ids != NULL) == (files != NULL))
        why = "Give either 'file_ids' or 'files'.";
    else if (yyjson_arr_size(ids ? ids : files) < 1 ||
             yyjson_arr_size(ids ? ids : files) > BATCH_FILES)
        why = "A batch holds from 1 to 2,000 files.";
    /* every file there before any is attached */
    size_t idx, mx;
    yyjson_val *v;
    yyjson_arr_foreach(ids ? ids : files, idx, mx, v)
    {
        const char *fid = ids ? yyjson_get_str(v)
                              : yyjson_get_str(yyjson_obj_get(v, "file_id"));
        struct srv_file f;
        if (!why && (!fid || srv_files_get(r->srv->files, fid, &f) != 0))
            why = "A file_id names no file uploaded here.";
    }
    char id[48], p[4800];
    srv_new_id(id, sizeof id, "vsfb_");
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d), *list = yyjson_mut_arr(d);
    yyjson_arr_foreach(ids ? ids : files, idx, mx, v)
    {
        if (why)
            break;
        const char *fid = ids ? yyjson_get_str(v)
                              : yyjson_get_str(yyjson_obj_get(v, "file_id"));
        if (srv_vs_attach(r->srv, vs, fid,
                          ids ? cs : yyjson_obj_get(v, "chunking_strategy"),
                          ids ? attrs : yyjson_obj_get(v, "attributes"),
                          &why) != 0 &&
            !why)
            why = "A file could not be attached.";
        yyjson_mut_arr_add_strcpy(d, list, fid);
    }
    int ok = !why;
    if (ok) {
        yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
        yyjson_mut_obj_add_val(d, o, "file_ids", list);
        yyjson_mut_obj_add_bool(d, o, "cancelled", false);
        srv_vectors_dir(p, sizeof p, vs);
        strcat(p, "/batches");
        mkdir(p, 0700);
        batch_path(p, sizeof p, vs, id);
        size_t n;
        char *t = yyjson_mut_val_write(o, 0, &n);
        srv_vectors_lock();
        ok = t && srv_vectors_write(p, t, n) == 0;
        srv_vectors_unlock();
        free(t);
    }
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    if (why)
        return bad(r, why);
    if (!ok)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the batch");
    return reply_batch(r, vs, id);
}

int srv_vs_batch_get(struct srv_req *r)
{
    if (!store_ok(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    return reply_batch(r, r->param[0], r->param[1]);
}

int srv_vs_batch_cancel(struct srv_req *r)
{
    const char *vs = r->param[0], *id = r->param[1];
    if (!store_ok(vs))
        return not_found(r, "vector store", vs);
    yyjson_doc *bd = read_batch(vs, id);
    if (!bd)
        return not_found(r, "vector store file batch", id);
    yyjson_mut_doc *d = yyjson_doc_mut_copy(bd, NULL);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    yyjson_mut_obj_remove_key(o, "cancelled");
    yyjson_mut_obj_add_bool(d, o, "cancelled", true);
    size_t idx, mx;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(bd), "file_ids"), idx,
                       mx, v)
    {
        char st[32];
        file_status(vs, yyjson_get_str(v), st, sizeof st);
        if (!strcmp(st, "in_progress"))
            srv_vectors_set_status(vs, yyjson_get_str(v), "cancelled", NULL,
                                   NULL, -1);
    }
    char p[4800];
    batch_path(p, sizeof p, vs, id);
    size_t n;
    char *t = yyjson_mut_val_write(o, 0, &n);
    srv_vectors_lock();
    if (t)
        srv_vectors_write(p, t, n);
    srv_vectors_unlock();
    free(t);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(bd);
    return reply_batch(r, vs, id);
}

int srv_vs_batch_files(struct srv_req *r)
{
    if (!store_ok(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    yyjson_doc *bd = read_batch(r->param[0], r->param[1]);
    if (!bd)
        return not_found(r, "vector store file batch", r->param[1]);
    yyjson_val *ids = yyjson_obj_get(yyjson_doc_get_root(bd), "file_ids");
    size_t n = yyjson_arr_size(ids), idx, mx;
    const char **list = calloc(n ? n : 1, sizeof(*list));
    yyjson_val *v;
    yyjson_arr_foreach(ids, idx, mx, v) if (list) list[idx] = yyjson_get_str(v);
    int ret =
        list ? srv_vs_files_page(r, r->param[0], list, n)
             : srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    free(list);
    yyjson_doc_free(bd);
    return ret;
}
