/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vectors_api.c - OpenAI's Vector stores API (vectors.c holds the data):
 * stores made, listed, read, changed and deleted; files attached to them,
 * listed, read, their attributes changed, detached, their text read back;
 * the search. A store's counts and status are its files', read when asked.
 */
#include "server.h"

#include <dirent.h>
#include <errno.h>
#include <ftw.h>
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

/* ---- helpers ---- */

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

static int reply(struct srv_req *r, yyjson_mut_doc *d, yyjson_mut_val *o)
{
    yyjson_mut_doc_set_root(d, o);
    size_t n;
    char *out = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, n);
}

static char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *b = NULL;
    long len = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    if (len >= 0 && !fseek(f, 0, SEEK_SET) && (b = malloc((size_t)len + 1)) &&
        fread(b, 1, (size_t)len, f) != (size_t)len) {
        free(b);
        b = NULL;
    }
    fclose(f);
    if (b) {
        b[len] = 0;
        *n = (size_t)len;
    }
    return b;
}

static yyjson_doc *read_json(const char *path)
{
    size_t n = 0;
    srv_vectors_lock();
    char *t = slurp(path, &n);
    srv_vectors_unlock();
    yyjson_doc *d = t ? yyjson_read(t, n, 0) : NULL;
    free(t);
    return d;
}

static int write_json(const char *path, yyjson_mut_doc *d, yyjson_mut_val *o)
{
    size_t n;
    char *t = yyjson_mut_val_write(o, 0, &n);
    (void)d;
    srv_vectors_lock();
    int rc = t ? srv_vectors_write(path, t, n) : -1;
    srv_vectors_unlock();
    free(t);
    return rc;
}

static int store_exists(const char *vs)
{
    char p[4500];
    if (!srv_disk_name_ok(vs))
        return 0;
    srv_vectors_dir(p, sizeof p, vs);
    strcat(p, "/meta.json");
    return access(p, F_OK) == 0;
}

/* A chunking strategy as a request gives it into a static one; NULL when
   it is not one (why says). */
static yyjson_mut_val *chunking(yyjson_mut_doc *d, yyjson_val *cs,
                                const char **why)
{
    long size = 800, over = 400;
    const char *type = yyjson_get_str(yyjson_obj_get(cs, "type"));
    if (cs && type && !strcmp(type, "static")) {
        yyjson_val *st = yyjson_obj_get(cs, "static");
        yyjson_val *ms = yyjson_obj_get(st, "max_chunk_size_tokens");
        yyjson_val *co = yyjson_obj_get(st, "chunk_overlap_tokens");
        if (!yyjson_is_int(ms) || !yyjson_is_int(co) ||
            yyjson_get_sint(ms) < 100 || yyjson_get_sint(ms) > 4096 ||
            yyjson_get_sint(co) < 0 ||
            yyjson_get_sint(co) > yyjson_get_sint(ms) / 2) {
            *why = "A static chunking strategy has max_chunk_size_tokens "
                   "from 100 to 4096 and chunk_overlap_tokens at most half of "
                   "it.";
            return NULL;
        }
        size = (long)yyjson_get_sint(ms);
        over = (long)yyjson_get_sint(co);
    } else if (cs && !yyjson_is_null(cs) && (!type || strcmp(type, "auto"))) {
        *why = "'chunking_strategy' is auto or static.";
        return NULL;
    }
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, o, "type", "static");
    yyjson_mut_val *s = yyjson_mut_obj_add_obj(d, o, "static");
    yyjson_mut_obj_add_int(d, s, "max_chunk_size_tokens", size);
    yyjson_mut_obj_add_int(d, s, "chunk_overlap_tokens", over);
    return o;
}

/* Attributes: up to 16 keys of strings up to 512, numbers or booleans. */
static int attributes_ok(yyjson_val *a)
{
    if (!a || yyjson_is_null(a))
        return 1;
    if (!yyjson_is_obj(a) || yyjson_obj_size(a) > 16)
        return 0;
    size_t i, m;
    yyjson_val *k, *v;
    yyjson_obj_foreach(a, i, m, k, v)
    {
        if (yyjson_get_len(k) > 64 ||
            !(yyjson_is_num(v) || yyjson_is_bool(v) ||
              (yyjson_is_str(v) && yyjson_get_len(v) <= 512)))
            return 0;
    }
    return 1;
}

/* A file attached: its JSON written, the indexer woken. 0, -1 no such
   file (why), -2 the disk. */
static int attach(const char *vs, const char *file_id, yyjson_val *cs,
                  yyjson_val *attrs, struct srv *srv, const char **why)
{
    struct srv_file f;
    if (srv_files_get(srv->files, file_id, &f) != 0) {
        *why = "a file_id names no file uploaded here";
        return -1;
    }
    if (!attributes_ok(attrs)) {
        *why = "'attributes' holds at most 16 keys of strings (512 at "
               "most), numbers or booleans";
        return -1;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *cv = chunking(d, cs, why);
    if (!cv) {
        yyjson_mut_doc_free(d);
        return -1;
    }
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", file_id);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.file");
    yyjson_mut_obj_add_int(d, o, "usage_bytes", 0);
    yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
    yyjson_mut_obj_add_strcpy(d, o, "vector_store_id", vs);
    yyjson_mut_obj_add_str(d, o, "status", "in_progress");
    yyjson_mut_obj_add_null(d, o, "last_error");
    yyjson_mut_obj_add_val(d, o, "chunking_strategy", cv);
    if (attrs && yyjson_is_obj(attrs))
        yyjson_mut_obj_add_val(d, o, "attributes",
                               yyjson_val_mut_copy(d, attrs));
    else
        yyjson_mut_obj_add_obj(d, o, "attributes");
    char p[4600];
    srv_vectors_dir(p, sizeof p, vs);
    snprintf(p + strlen(p), sizeof p - strlen(p), "/files/%s.json", file_id);
    int rc = write_json(p, d, o) == 0 ? 0 : -2;
    yyjson_mut_doc_free(d);
    srv_vectors_wake();
    return rc;
}

int srv_vs_attach(struct srv *srv, const char *vs, const char *file_id,
                  yyjson_val *cs, yyjson_val *attrs, const char **why)
{
    return attach(vs, file_id, cs, attrs, srv, why);
}

/* A store's object with its counts and status, made from its files. */
static yyjson_mut_val *store_object(yyjson_mut_doc *d, const char *vs)
{
    char dir[4500], p[4700];
    srv_vectors_dir(dir, sizeof dir, vs);
    snprintf(p, sizeof p, "%s/meta.json", dir);
    yyjson_doc *md = read_json(p);
    if (!md)
        return NULL;
    yyjson_mut_val *o = yyjson_val_mut_copy(d, yyjson_doc_get_root(md));
    yyjson_doc_free(md);
    long c[4] = {0, 0, 0, 0}; /* in_progress, completed, failed, cancelled */
    long long usage = 0;
    snprintf(p, sizeof p, "%s/files", dir);
    DIR *fd = opendir(p);
    struct dirent *e;
    while (fd && (e = readdir(fd))) {
        size_t n = strlen(e->d_name);
        if (n < 6 || strcmp(e->d_name + n - 5, ".json") != 0)
            continue;
        char fp[5000];
        snprintf(fp, sizeof fp, "%s/%.255s", p, e->d_name);
        yyjson_doc *fdoc = read_json(fp);
        yyjson_val *fo = yyjson_doc_get_root(fdoc);
        const char *st = yyjson_get_str(yyjson_obj_get(fo, "status"));
        static const char *const S[] = {"in_progress", "completed", "failed",
                                        "cancelled"};
        for (int i = 0; st && i < 4; i++)
            if (!strcmp(st, S[i]))
                c[i]++;
        usage += yyjson_get_sint(yyjson_obj_get(fo, "usage_bytes"));
        yyjson_doc_free(fdoc);
    }
    if (fd)
        closedir(fd);
    yyjson_mut_obj_remove_key(o, "usage_bytes");
    yyjson_mut_obj_add_int(d, o, "usage_bytes", usage);
    yyjson_mut_val *fc = yyjson_mut_obj(d);
    yyjson_mut_obj_add_int(d, fc, "in_progress", c[0]);
    yyjson_mut_obj_add_int(d, fc, "completed", c[1]);
    yyjson_mut_obj_add_int(d, fc, "failed", c[2]);
    yyjson_mut_obj_add_int(d, fc, "cancelled", c[3]);
    yyjson_mut_obj_add_int(d, fc, "total", c[0] + c[1] + c[2] + c[3]);
    yyjson_mut_obj_remove_key(o, "file_counts");
    yyjson_mut_obj_add_val(d, o, "file_counts", fc);
    int64_t exp = yyjson_mut_get_sint(yyjson_mut_obj_get(o, "expires_at"));
    const char *status = exp && (int64_t)time(NULL) >= exp ? "expired"
                         : c[0]                            ? "in_progress"
                                                           : "completed";
    yyjson_mut_obj_remove_key(o, "status");
    yyjson_mut_obj_add_str(d, o, "status", status);
    return o;
}

/* expires_after {anchor: last_active_at, days 1-365} into the object, with
   expires_at from last_active_at. 0, or -1 (why). */
static int set_expiry(yyjson_mut_doc *d, yyjson_mut_val *o, yyjson_val *ea,
                      const char **why)
{
    yyjson_mut_obj_remove_key(o, "expires_after");
    yyjson_mut_obj_remove_key(o, "expires_at");
    if (!ea || yyjson_is_null(ea)) {
        yyjson_mut_obj_add_null(d, o, "expires_after");
        yyjson_mut_obj_add_null(d, o, "expires_at");
        return 0;
    }
    yyjson_val *days = yyjson_obj_get(ea, "days");
    if (!yyjson_equals_str(yyjson_obj_get(ea, "anchor"), "last_active_at") ||
        !yyjson_is_int(days) || yyjson_get_sint(days) < 1 ||
        yyjson_get_sint(days) > 365) {
        *why = "'expires_after' is anchor last_active_at and days from 1 to "
               "365.";
        return -1;
    }
    yyjson_mut_obj_add_val(d, o, "expires_after", yyjson_val_mut_copy(d, ea));
    int64_t last = yyjson_mut_get_sint(yyjson_mut_obj_get(o, "last_active_at"));
    yyjson_mut_obj_add_int(d, o, "expires_at",
                           last + yyjson_get_sint(days) * 86400);
    return 0;
}

static int metadata_ok(yyjson_val *m)
{
    if (!m || yyjson_is_null(m))
        return 1;
    if (!yyjson_is_obj(m) || yyjson_obj_size(m) > 16)
        return 0;
    size_t i, n;
    yyjson_val *k, *v;
    yyjson_obj_foreach(m, i, n, k, v) if (yyjson_get_len(k) > 64 ||
                                          !yyjson_is_str(v) ||
                                          yyjson_get_len(v) > 512) return 0;
    return 1;
}

/* ---- stores ---- */

int srv_vs_create(struct srv_req *r)
{
    yyjson_doc *doc = r->n_body ? yyjson_read(r->body, r->n_body, 0) : NULL;
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *why = NULL;
    yyjson_val *ids = yyjson_obj_get(root, "file_ids");
    yyjson_val *name = yyjson_obj_get(root, "name");
    yyjson_val *desc = yyjson_obj_get(root, "description");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    if (r->n_body && !yyjson_is_obj(root))
        why = "The body is a JSON object.";
    else if (ids && (!yyjson_is_arr(ids) || yyjson_arr_size(ids) > 500))
        why = "'file_ids' is a list of up to 500 ids.";
    else if (!metadata_ok(md))
        why = "'metadata' holds at most 16 strings.";
    /* every file there before anything is made */
    size_t idx, max;
    yyjson_val *v;
    yyjson_arr_foreach(ids, idx, max, v)
    {
        struct srv_file f;
        if (!why && (!yyjson_is_str(v) ||
                     srv_files_get(r->srv->files, yyjson_get_str(v), &f) != 0))
            why = "A file_id names no file uploaded here.";
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    if (!why) /* a strategy that is not one: said here, not per file */
        chunking(d, yyjson_obj_get(root, "chunking_strategy"), &why);
    char id[48];
    srv_new_id(id, sizeof id, "vs_");
    int64_t now = (int64_t)time(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store");
    yyjson_mut_obj_add_int(d, o, "created_at", now);
    if (yyjson_is_str(name))
        yyjson_mut_obj_add_strcpy(d, o, "name", yyjson_get_str(name));
    else
        yyjson_mut_obj_add_null(d, o, "name");
    if (yyjson_is_str(desc))
        yyjson_mut_obj_add_strcpy(d, o, "description", yyjson_get_str(desc));
    yyjson_mut_obj_add_int(d, o, "last_active_at", now);
    if (!why)
        set_expiry(d, o, yyjson_obj_get(root, "expires_after"), &why);
    if (md && yyjson_is_obj(md))
        yyjson_mut_obj_add_val(d, o, "metadata", yyjson_val_mut_copy(d, md));
    else
        yyjson_mut_obj_add_obj(d, o, "metadata");
    if (why) {
        yyjson_mut_doc_free(d);
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    char dir[4500], p[4700];
    srv_vectors_dir(dir, sizeof dir, id);
    snprintf(p, sizeof p, "%s/files", dir);
    int ok = mkdir(dir, 0700) == 0 && mkdir(p, 0700) == 0;
    snprintf(p, sizeof p, "%s/meta.json", dir);
    ok = ok && write_json(p, d, o) == 0;
    yyjson_arr_foreach(ids, idx, max, v) if (ok) ok =
        attach(id, yyjson_get_str(v), yyjson_obj_get(root, "chunking_strategy"),
               NULL, r->srv, &why) == 0;
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    if (!ok)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the vector store");
    d = yyjson_mut_doc_new(NULL);
    return reply(r, d, store_object(d, id));
}

int srv_vs_get(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = store_object(d, r->param[0]);
    if (!o) {
        yyjson_mut_doc_free(d);
        return not_found(r, "vector store", r->param[0]);
    }
    return reply(r, d, o);
}

int srv_vs_modify(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *why = NULL;
    char p[4700];
    srv_vectors_dir(p, sizeof p, r->param[0]);
    strcat(p, "/meta.json");
    yyjson_doc *md = read_json(p);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o =
        md ? yyjson_val_mut_copy(d, yyjson_doc_get_root(md)) : NULL;
    yyjson_doc_free(md);
    yyjson_val *name = yyjson_obj_get(root, "name");
    yyjson_val *meta = yyjson_obj_get(root, "metadata");
    if (!yyjson_is_obj(root) || !o)
        why = "The body is a JSON object.";
    else if (!metadata_ok(meta))
        why = "'metadata' holds at most 16 strings.";
    if (!why && name) {
        yyjson_mut_obj_remove_key(o, "name");
        if (yyjson_is_str(name))
            yyjson_mut_obj_add_strcpy(d, o, "name", yyjson_get_str(name));
        else
            yyjson_mut_obj_add_null(d, o, "name");
    }
    if (!why && meta) {
        yyjson_mut_obj_remove_key(o, "metadata");
        yyjson_mut_obj_add_val(d, o, "metadata",
                               yyjson_is_obj(meta)
                                   ? yyjson_val_mut_copy(d, meta)
                                   : yyjson_mut_obj(d));
    }
    if (!why && yyjson_obj_get(root, "expires_after"))
        set_expiry(d, o, yyjson_obj_get(root, "expires_after"), &why);
    int ok = !why && write_json(p, d, o) == 0;
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    if (why)
        return bad(r, why);
    if (!ok)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the vector store");
    d = yyjson_mut_doc_new(NULL);
    return reply(r, d, store_object(d, r->param[0]));
}

static int rm_one(const char *p, const struct stat *s, int f, struct FTW *w)
{
    (void)s;
    (void)f;
    (void)w;
    return remove(p);
}

int srv_vs_delete(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    char dir[4500];
    srv_vectors_dir(dir, sizeof dir, r->param[0]);
    srv_vectors_lock();
    nftw(dir, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    srv_vectors_unlock();
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", r->param[0]);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.deleted");
    yyjson_mut_obj_add_bool(d, o, "deleted", true);
    return reply(r, d, o);
}

/* ---- lists ---- */

struct entry {
    char id[48];
    int64_t at;
};

static int by_at(const void *a, const void *b)
{
    const struct entry *x = a, *y = b;
    if (x->at != y->at)
        return x->at < y->at ? -1 : 1;
    return strcmp(x->id, y->id);
}

/* A page of a list (limit, order, after, before) of entries in creation
   order, each made into an object by make. */
static int page(struct srv_req *r, struct entry *e, size_t n,
                yyjson_mut_val *(*make)(yyjson_mut_doc *, const char *,
                                        const void *),
                const void *ctx)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after"), *before = srv_query(r, "before");
    long limit = 20;
    if (l) {
        char *end;
        limit = strtol(l, &end, 10);
        if (*end || limit < 1 || limit > 100) {
            free(e);
            return bad(r, "'limit' is a number from 1 to 100.");
        }
    }
    int asc = o && !strcmp(o, "asc");
    if (n)
        qsort(e, n, sizeof(*e), by_at);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    long shown = 0;
    int more = 0, past = after == NULL;
    char first[48] = "", last[48] = "";
    for (size_t k = 0; k < n; k++) {
        const struct entry *x = &e[asc ? k : n - 1 - k];
        if (!past) {
            past = !strcmp(x->id, after);
            continue;
        }
        if (before && !strcmp(x->id, before))
            break;
        if (shown == limit) {
            more = 1;
            break;
        }
        yyjson_mut_val *v = make(d, x->id, ctx);
        if (!v)
            continue;
        yyjson_mut_arr_append(data, v);
        if (!shown)
            snprintf(first, sizeof first, "%s", x->id);
        snprintf(last, sizeof last, "%s", x->id);
        shown++;
    }
    free(e);
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_obj_add_val(d, root, "data", data);
    if (shown) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    return reply(r, d, root);
}

static yyjson_mut_val *make_store(yyjson_mut_doc *d, const char *id,
                                  const void *ctx)
{
    (void)ctx;
    return store_object(d, id);
}

int srv_vs_list(struct srv_req *r)
{
    char dir[4500];
    srv_vectors_dir(dir, sizeof dir, NULL);
    struct entry *e = NULL;
    size_t n = 0, cap = 0;
    DIR *dd = opendir(dir);
    struct dirent *x;
    while (dd && (x = readdir(dd))) {
        if (!srv_disk_name_ok(x->d_name) || strlen(x->d_name) >= 48 ||
            !store_exists(x->d_name))
            continue;
        if (n == cap) {
            cap = cap ? 2 * cap : 32;
            struct entry *t = realloc(e, cap * sizeof(*t));
            if (!t)
                break;
            e = t;
        }
        char p[4800];
        snprintf(p, sizeof p, "%s/%.47s/meta.json", dir, x->d_name);
        yyjson_doc *md = read_json(p);
        memcpy(e[n].id, x->d_name, strlen(x->d_name) + 1); /* < 48 */
        e[n].at = yyjson_get_sint(
            yyjson_obj_get(yyjson_doc_get_root(md), "created_at"));
        yyjson_doc_free(md);
        n++;
    }
    if (dd)
        closedir(dd);
    return page(r, e, n, make_store, NULL);
}

/* ---- files of a store ---- */

static void vfile_path(char *out, size_t cap, const char *vs, const char *id)
{
    srv_vectors_dir(out, cap, vs);
    snprintf(out + strlen(out), cap - strlen(out), "/files/%s.json", id);
}

static yyjson_mut_val *vfile_object(yyjson_mut_doc *d, const char *vs,
                                    const char *id)
{
    char p[4800];
    if (!srv_disk_name_ok(id))
        return NULL;
    vfile_path(p, sizeof p, vs, id);
    yyjson_doc *fd = read_json(p);
    if (!fd)
        return NULL;
    yyjson_mut_val *o = yyjson_val_mut_copy(d, yyjson_doc_get_root(fd));
    yyjson_doc_free(fd);
    return o;
}

int srv_vs_file_create(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *fs = yyjson_get_str(yyjson_obj_get(root, "file_id"));
    char fid_buf[64];
    snprintf(fid_buf, sizeof fid_buf, "%s", fs ? fs : "");
    const char *fid = fs ? fid_buf : NULL; /* kept past the body */
    const char *why = "'file_id' is required.";
    int rc = fid ? attach(r->param[0], fid,
                          yyjson_obj_get(root, "chunking_strategy"),
                          yyjson_obj_get(root, "attributes"), r->srv, &why)
                 : -1;
    yyjson_doc_free(doc);
    if (rc == -1)
        return bad(r, why);
    if (rc != 0)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the file");
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    return reply(r, d, vfile_object(d, r->param[0], fid));
}

int srv_vs_file_get(struct srv_req *r)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = store_exists(r->param[0])
                            ? vfile_object(d, r->param[0], r->param[1])
                            : NULL;
    if (!o) {
        yyjson_mut_doc_free(d);
        return not_found(r, "vector store file", r->param[1]);
    }
    return reply(r, d, o);
}

int srv_vs_file_update(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *attrs = yyjson_obj_get(yyjson_doc_get_root(doc), "attributes");
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = store_exists(r->param[0])
                            ? vfile_object(d, r->param[0], r->param[1])
                            : NULL;
    int ret;
    if (!o) {
        yyjson_mut_doc_free(d);
        ret = not_found(r, "vector store file", r->param[1]);
    } else if (!attrs || !attributes_ok(attrs)) {
        yyjson_mut_doc_free(d);
        ret = bad(r, "'attributes' holds at most 16 keys of strings (512 at "
                     "most), numbers or booleans.");
    } else {
        yyjson_mut_obj_remove_key(o, "attributes");
        yyjson_mut_obj_add_val(d, o, "attributes",
                               yyjson_is_obj(attrs)
                                   ? yyjson_val_mut_copy(d, attrs)
                                   : yyjson_mut_obj(d));
        char p[4800];
        vfile_path(p, sizeof p, r->param[0], r->param[1]);
        if (write_json(p, d, o) != 0) {
            yyjson_mut_doc_free(d);
            ret = srv_reply_error(r, 500, "server_error", NULL,
                                  "cannot keep the file");
        } else {
            ret = reply(r, d, o);
        }
    }
    yyjson_doc_free(doc);
    return ret;
}

int srv_vs_file_delete(struct srv_req *r)
{
    char p[4800];
    if (!store_exists(r->param[0]) || !srv_disk_name_ok(r->param[1]))
        return not_found(r, "vector store file", r->param[1]);
    vfile_path(p, sizeof p, r->param[0], r->param[1]);
    if (access(p, F_OK) != 0)
        return not_found(r, "vector store file", r->param[1]);
    srv_vectors_lock();
    unlink(p);
    srv_vectors_dir(p, sizeof p, r->param[0]);
    size_t at = strlen(p);
    snprintf(p + at, sizeof p - at, "/%s.chunks", r->param[1]);
    unlink(p);
    snprintf(p + at, sizeof p - at, "/%s.vec", r->param[1]);
    unlink(p);
    srv_vectors_unlock();
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", r->param[1]);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.file.deleted");
    yyjson_mut_obj_add_bool(d, o, "deleted", true);
    return reply(r, d, o);
}

struct vfile_ctx {
    const char *vs;
    const char *status; /* filter: NULL any */
};

static yyjson_mut_val *make_vfile(yyjson_mut_doc *d, const char *id,
                                  const void *ctx)
{
    const struct vfile_ctx *c = ctx;
    yyjson_mut_val *o = vfile_object(d, c->vs, id);
    const char *st =
        o ? yyjson_mut_get_str(yyjson_mut_obj_get(o, "status")) : NULL;
    return o && (!c->status || (st && !strcmp(st, c->status))) ? o : NULL;
}

/* The files of a store, or (ids not NULL) those of a batch of them. */
int srv_vs_files_page(struct srv_req *r, const char *vs, const char *const *ids,
                      size_t n_ids)
{
    char dir[4600];
    srv_vectors_dir(dir, sizeof dir, vs);
    strcat(dir, "/files");
    struct entry *e = NULL;
    size_t n = 0, cap = 0;
    DIR *dd = opendir(dir);
    struct dirent *x;
    while (dd && (x = readdir(dd))) {
        size_t nl = strlen(x->d_name);
        if (nl < 6 || nl - 5 >= 48 || strcmp(x->d_name + nl - 5, ".json"))
            continue;
        char id[48];
        memcpy(id, x->d_name, nl - 5);
        id[nl - 5] = 0;
        int in = !ids;
        for (size_t k = 0; ids && k < n_ids && !in; k++)
            in = !strcmp(ids[k], id);
        if (!in)
            continue;
        if (n == cap) {
            cap = cap ? 2 * cap : 32;
            struct entry *t = realloc(e, cap * sizeof(*t));
            if (!t)
                break;
            e = t;
        }
        char p[4900];
        snprintf(p, sizeof p, "%s/%s", dir, x->d_name);
        yyjson_doc *fd = read_json(p);
        snprintf(e[n].id, sizeof e[n].id, "%s", id);
        e[n].at = yyjson_get_sint(
            yyjson_obj_get(yyjson_doc_get_root(fd), "created_at"));
        yyjson_doc_free(fd);
        n++;
    }
    if (dd)
        closedir(dd);
    const char *f = srv_query(r, "filter");
    if (f && strcmp(f, "in_progress") && strcmp(f, "completed") &&
        strcmp(f, "failed") && strcmp(f, "cancelled")) {
        free(e);
        return bad(r, "'filter' is in_progress, completed, failed or "
                      "cancelled.");
    }
    struct vfile_ctx c = {vs, f};
    return page(r, e, n, make_vfile, &c);
}

int srv_vs_files_list(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    return srv_vs_files_page(r, r->param[0], NULL, 0);
}

int srv_vs_file_content(struct srv_req *r)
{
    char p[4800];
    if (!store_exists(r->param[0]) || !srv_disk_name_ok(r->param[1]))
        return not_found(r, "vector store file", r->param[1]);
    vfile_path(p, sizeof p, r->param[0], r->param[1]);
    if (access(p, F_OK) != 0)
        return not_found(r, "vector store file", r->param[1]);
    /* the text the file was read as: its chunks without their overlap is
       not kept, so the file itself, when it is text */
    struct srv_file f;
    int fd = srv_files_get(r->srv->files, r->param[1], &f) == 0
                 ? srv_files_open_content(r->srv->files, r->param[1])
                 : -1;
    char *t =
        fd >= 0 && f.bytes <= ((size_t)32 << 20) ? malloc(f.bytes + 1) : NULL;
    size_t got = 0;
    while (t && got < f.bytes) {
        ssize_t k = read(fd, t + got, f.bytes - got);
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    if (fd >= 0)
        close(fd);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.file_content.page");
    if (t) {
        yyjson_mut_val *c = yyjson_mut_arr_add_obj(d, data);
        yyjson_mut_obj_add_str(d, c, "type", "text");
        yyjson_mut_obj_add_strncpy(d, c, "text", t, got);
    }
    free(t);
    yyjson_mut_obj_add_val(d, o, "data", data);
    yyjson_mut_obj_add_bool(d, o, "has_more", false);
    yyjson_mut_obj_add_null(d, o, "next_page");
    return reply(r, d, o);
}

/* ---- search ---- */

int srv_vs_search(struct srv_req *r)
{
    if (!store_exists(r->param[0]))
        return not_found(r, "vector store", r->param[0]);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *q = yyjson_obj_get(root, "query");
    yyjson_val *mx = yyjson_obj_get(root, "max_num_results");
    yyjson_val *th = yyjson_obj_get(yyjson_obj_get(root, "ranking_options"),
                                    "score_threshold");
    const char *qs[16];
    int nq = 0;
    if (yyjson_is_str(q) && yyjson_get_len(q))
        qs[nq++] = yyjson_get_str(q);
    size_t idx, m;
    yyjson_val *v;
    if (yyjson_is_arr(q))
        yyjson_arr_foreach(q, idx, m, v) if (nq < 16 && yyjson_is_str(v))
            qs[nq++] = yyjson_get_str(v);
    long max = mx ? (long)yyjson_get_sint(mx) : 10;
    double thr = th ? yyjson_get_num(th) : 0;
    const char *why = NULL;
    if (!nq)
        why = "'query' is a string or a list of them.";
    else if (mx && (!yyjson_is_int(mx) || max < 1 || max > 50))
        why = "'max_num_results' is from 1 to 50.";
    else if (th && (!yyjson_is_num(th) || thr < 0 || thr > 1))
        why = "'score_threshold' is from 0 to 1.";
    if (why) {
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    struct srv_vhit *hits;
    char err[200];
    int n =
        srv_vectors_search(r->param[0], qs, nq, yyjson_obj_get(root, "filters"),
                           (int)max, thr, &hits, err, sizeof err);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    yyjson_mut_obj_add_str(d, o, "object", "vector_store.search_results.page");
    yyjson_mut_val *sq = yyjson_mut_obj_add_arr(d, o, "search_query");
    for (int i = 0; i < nq; i++)
        yyjson_mut_arr_add_strcpy(d, sq, qs[i]);
    for (int i = 0; i < n; i++) {
        yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, data);
        yyjson_mut_obj_add_strcpy(d, it, "file_id", hits[i].file_id);
        yyjson_mut_obj_add_strcpy(d, it, "filename", hits[i].filename);
        yyjson_mut_obj_add_real(d, it, "score", hits[i].score);
        yyjson_doc *ad =
            hits[i].attributes
                ? yyjson_read(hits[i].attributes, strlen(hits[i].attributes), 0)
                : NULL;
        yyjson_mut_obj_add_val(
            d, it, "attributes",
            ad ? yyjson_val_mut_copy(d, yyjson_doc_get_root(ad))
               : yyjson_mut_obj(d));
        yyjson_doc_free(ad);
        yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, it, "content");
        yyjson_mut_val *t = yyjson_mut_arr_add_obj(d, c);
        yyjson_mut_obj_add_str(d, t, "type", "text");
        yyjson_mut_obj_add_strcpy(d, t, "text", hits[i].text);
    }
    srv_vhits_free(hits, n > 0 ? n : 0);
    yyjson_doc_free(doc);
    yyjson_mut_obj_add_val(d, o, "data", data);
    yyjson_mut_obj_add_bool(d, o, "has_more", false);
    yyjson_mut_obj_add_null(d, o, "next_page");
    return reply(r, d, o);
}
