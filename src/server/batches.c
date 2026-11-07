/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * batches.c - OpenAI's Batch API: a JSONL file of requests to one endpoint
 * (/v1/chat/completions, /v1/responses, /v1/completions, /v1/embeddings,
 * /v1/moderations), run by the server one at a time on a thread of its own,
 * each behind every request a client is waiting for (srv_engine_low), and
 * answered in two files: the answers, and the requests that failed.
 *
 * A line runs through the same handler as the HTTP request it stands for,
 * its answer kept instead of sent (struct srv_capture). A batch is a
 * document of the store, saved after every line with how far it got and
 * how long its two files are, so a server stopped halfway goes on from
 * there when it starts again; the files themselves wait beside the
 * uploaded ones (.batches) until the batch ends and they become Files.
 */
#include "server.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <yyjson.h>

#include "files.h"
#include "files_store.h"

#define KIND "batch"
#define MAX_LINES 50000
#define BATCH_INPUT_MAX ((uint64_t)200 << 20)
#define WINDOW 86400 /* "24h" */
#define MAX_ERRORS 100

enum {
    VALIDATING,
    FAILED,
    IN_PROGRESS,
    FINALIZING,
    COMPLETED,
    EXPIRED,
    CANCELLING,
    CANCELLED
};
static const char *const STATUS[] = {"validating", "failed",    "in_progress",
                                     "finalizing", "completed", "expired",
                                     "cancelling", "cancelled"};

static const char *const ENDPOINTS[] = {"/v1/chat/completions", "/v1/responses",
                                        "/v1/completions", "/v1/embeddings",
                                        "/v1/moderations"};

struct batch {
    char id[48], endpoint[32], input[48], output[48], error[48];
    int status;
    int64_t created, in_progress, expires, finalizing, completed, failed,
        expired, cancelling, cancelled;
    long total, done_ok, done_bad, next;
    long long out_bytes, err_bytes; /* of the files being written */
    long long in_tokens, out_tokens;
    int64_t output_after; /* the outputs' expiry, seconds; 0: 30 days */
    char *metadata;       /* JSON text, or NULL */
    char *errors;         /* JSON array text of the validation's, or NULL */
};

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static pthread_t worker;
static int running, stopping;
static struct srv *server;
static char work_dir[4200];

/* ---- the document ---- */

static void add_time(yyjson_mut_doc *d, yyjson_mut_val *o, const char *k,
                     int64_t t)
{
    if (t)
        yyjson_mut_obj_add_int(d, o, k, t);
    else
        yyjson_mut_obj_add_null(d, o, k);
}

/* The Batch object as OpenAI writes it (aux: what only the server needs). */
static char *batch_json(const struct batch *b, char **aux)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "id", b->id);
    yyjson_mut_obj_add_str(d, o, "object", "batch");
    yyjson_mut_obj_add_strcpy(d, o, "endpoint", b->endpoint);
    yyjson_mut_obj_add_strcpy(d, o, "model", server->cfg.model_id);
    yyjson_doc *ed =
        b->errors ? yyjson_read(b->errors, strlen(b->errors), 0) : NULL;
    if (ed) {
        yyjson_mut_val *e = yyjson_mut_obj(d);
        yyjson_mut_obj_add_str(d, e, "object", "list");
        yyjson_mut_obj_add_val(d, e, "data",
                               yyjson_val_mut_copy(d, yyjson_doc_get_root(ed)));
        yyjson_mut_obj_add_val(d, o, "errors", e);
    } else {
        yyjson_mut_obj_add_null(d, o, "errors");
    }
    yyjson_mut_obj_add_strcpy(d, o, "input_file_id", b->input);
    yyjson_mut_obj_add_str(d, o, "completion_window", "24h");
    yyjson_mut_obj_add_str(d, o, "status", STATUS[b->status]);
    if (b->output[0])
        yyjson_mut_obj_add_strcpy(d, o, "output_file_id", b->output);
    else
        yyjson_mut_obj_add_null(d, o, "output_file_id");
    if (b->error[0])
        yyjson_mut_obj_add_strcpy(d, o, "error_file_id", b->error);
    else
        yyjson_mut_obj_add_null(d, o, "error_file_id");
    yyjson_mut_obj_add_int(d, o, "created_at", b->created);
    add_time(d, o, "in_progress_at", b->in_progress);
    add_time(d, o, "expires_at", b->expires);
    add_time(d, o, "finalizing_at", b->finalizing);
    add_time(d, o, "completed_at", b->completed);
    add_time(d, o, "failed_at", b->failed);
    add_time(d, o, "expired_at", b->expired);
    add_time(d, o, "cancelling_at", b->cancelling);
    add_time(d, o, "cancelled_at", b->cancelled);
    yyjson_mut_val *rc = yyjson_mut_obj(d);
    yyjson_mut_obj_add_int(d, rc, "total", b->total);
    yyjson_mut_obj_add_int(d, rc, "completed", b->done_ok);
    yyjson_mut_obj_add_int(d, rc, "failed", b->done_bad);
    yyjson_mut_obj_add_val(d, o, "request_counts", rc);
    yyjson_mut_val *u = yyjson_mut_obj(d);
    yyjson_mut_obj_add_int(d, u, "input_tokens", b->in_tokens);
    yyjson_mut_obj_add_int(d, u, "output_tokens", b->out_tokens);
    yyjson_mut_obj_add_int(d, u, "total_tokens", b->in_tokens + b->out_tokens);
    yyjson_mut_obj_add_val(d, o, "usage", u);
    yyjson_doc *md =
        b->metadata ? yyjson_read(b->metadata, strlen(b->metadata), 0) : NULL;
    if (md)
        yyjson_mut_obj_add_val(d, o, "metadata",
                               yyjson_val_mut_copy(d, yyjson_doc_get_root(md)));
    else
        yyjson_mut_obj_add_null(d, o, "metadata");
    char *out = yyjson_mut_write(d, 0, NULL);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(ed);
    yyjson_doc_free(md);
    if (aux) {
        char tmp[512];
        snprintf(tmp, sizeof tmp,
                 "{\"next\":%ld,\"out_bytes\":%lld,\"err_bytes\":%lld,"
                 "\"output_after\":%lld}",
                 b->next, b->out_bytes, b->err_bytes,
                 (long long)b->output_after);
        *aux = strdup(tmp);
    }
    return out;
}

static int64_t num(yyjson_val *o, const char *k)
{
    return yyjson_get_sint(yyjson_obj_get(o, k));
}

static void str(char *dst, size_t cap, yyjson_val *o, const char *k)
{
    const char *s = yyjson_get_str(yyjson_obj_get(o, k));
    snprintf(dst, cap, "%s", s ? s : "");
}

/* A batch read back from the store: 0, or -1. */
static int batch_read(const char *json, const char *aux, struct batch *b)
{
    memset(b, 0, sizeof(*b));
    yyjson_doc *d = yyjson_read(json, strlen(json), 0);
    yyjson_val *o = yyjson_doc_get_root(d);
    if (!yyjson_is_obj(o)) {
        yyjson_doc_free(d);
        return -1;
    }
    str(b->id, sizeof b->id, o, "id");
    str(b->endpoint, sizeof b->endpoint, o, "endpoint");
    str(b->input, sizeof b->input, o, "input_file_id");
    str(b->output, sizeof b->output, o, "output_file_id");
    str(b->error, sizeof b->error, o, "error_file_id");
    const char *st = yyjson_get_str(yyjson_obj_get(o, "status"));
    for (int i = 0; st && i < 8; i++)
        if (!strcmp(st, STATUS[i]))
            b->status = i;
    b->created = num(o, "created_at");
    b->in_progress = num(o, "in_progress_at");
    b->expires = num(o, "expires_at");
    b->finalizing = num(o, "finalizing_at");
    b->completed = num(o, "completed_at");
    b->failed = num(o, "failed_at");
    b->expired = num(o, "expired_at");
    b->cancelling = num(o, "cancelling_at");
    b->cancelled = num(o, "cancelled_at");
    yyjson_val *rc = yyjson_obj_get(o, "request_counts");
    b->total = (long)num(rc, "total");
    b->done_ok = (long)num(rc, "completed");
    b->done_bad = (long)num(rc, "failed");
    yyjson_val *u = yyjson_obj_get(o, "usage");
    b->in_tokens = num(u, "input_tokens");
    b->out_tokens = num(u, "output_tokens");
    yyjson_val *md = yyjson_obj_get(o, "metadata");
    if (yyjson_is_obj(md))
        b->metadata = yyjson_val_write(md, 0, NULL);
    yyjson_val *er = yyjson_obj_get(yyjson_obj_get(o, "errors"), "data");
    if (yyjson_is_arr(er))
        b->errors = yyjson_val_write(er, 0, NULL);
    yyjson_doc_free(d);
    yyjson_doc *a = aux ? yyjson_read(aux, strlen(aux), 0) : NULL;
    yyjson_val *ar = yyjson_doc_get_root(a);
    b->next = (long)num(ar, "next");
    b->out_bytes = num(ar, "out_bytes");
    b->err_bytes = num(ar, "err_bytes");
    b->output_after = num(ar, "output_after");
    yyjson_doc_free(a);
    return 0;
}

static void batch_free(struct batch *b)
{
    free(b->metadata);
    free(b->errors);
    b->metadata = b->errors = NULL;
}

static int save(const struct batch *b, int fresh)
{
    char *aux = NULL, *json = batch_json(b, &aux);
    int rc = -1;
    if (json && aux)
        rc = fresh ? srv_store_put(server->store, KIND, b->id, b->created, json,
                                   aux)
                   : srv_store_update(server->store, KIND, b->id, json, aux);
    free(json);
    free(aux);
    return rc;
}

static int load(const char *id, struct batch *b)
{
    char *json = NULL, *aux = NULL;
    int rc = srv_store_get(server->store, KIND, id, &json, &aux) == 0 && json
                 ? batch_read(json, aux, b)
                 : -1;
    free(json);
    free(aux);
    return rc;
}

/* ---- running a batch ---- */

static void work_path(char *out, size_t cap, const char *id, const char *suf)
{
    snprintf(out, cap, "%s/%s%s", work_dir, id, suf);
}

/* The input file whole, its lines' starts into *starts: how many, or -1
   with the reason in why. */
static long read_input(const struct batch *b, char **text, long **starts,
                       char *why, size_t wl)
{
    struct srv_file f;
    *text = NULL;
    *starts = NULL;
    if (srv_files_get(server->files, b->input, &f) != 0) {
        snprintf(why, wl, "the input file %s is not there", b->input);
        return -1;
    }
    if (f.bytes > BATCH_INPUT_MAX) {
        snprintf(why, wl, "the input file is larger than 200 MB");
        return -1;
    }
    int fd = srv_files_open_content(server->files, b->input);
    char *t = fd >= 0 ? malloc(f.bytes + 1) : NULL;
    size_t got = 0;
    while (t && got < f.bytes) {
        ssize_t r = read(fd, t + got, f.bytes - got);
        if (r <= 0)
            break;
        got += (size_t)r;
    }
    if (fd >= 0)
        close(fd);
    if (!t || got != f.bytes) {
        free(t);
        snprintf(why, wl, "the input file cannot be read");
        return -1;
    }
    t[got] = 0;
    long n = 0, cap = 0;
    long *s = NULL;
    for (size_t i = 0; i < got;) {
        size_t e = i;
        while (e < got && t[e] != '\n')
            e++;
        size_t k = i;
        while (k < e && (t[k] == ' ' || t[k] == '\t' || t[k] == '\r'))
            k++;
        if (k < e) { /* not a blank line */
            if (n == cap) {
                cap = cap ? 2 * cap : 1024;
                long *x = realloc(s, (size_t)cap * sizeof(long));
                if (!x) {
                    free(s);
                    free(t);
                    snprintf(why, wl, "out of memory");
                    return -1;
                }
                s = x;
            }
            s[n++] = (long)i;
        }
        t[e < got ? e : got] = 0; /* lines as strings */
        i = e + 1;
    }
    *text = t;
    *starts = s;
    return n;
}

static void add_error(yyjson_mut_doc *d, yyjson_mut_val *arr, const char *code,
                      const char *msg, long line, int *count)
{
    if (++*count > MAX_ERRORS)
        return;
    yyjson_mut_val *e = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, e, "code", code);
    yyjson_mut_obj_add_strcpy(d, e, "message", msg);
    yyjson_mut_obj_add_null(d, e, "param");
    yyjson_mut_obj_add_int(d, e, "line", line);
    yyjson_mut_arr_append(arr, e);
}

/* Every line a request to the batch's endpoint with its own custom_id: the
   errors as a JSON array text (NULL: none). */
static char *validate(const struct batch *b, char *text, const long *starts,
                      long n)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    int count = 0;
    char **ids = calloc((size_t)(n ? n : 1), sizeof(char *));
    if (n > MAX_LINES)
        add_error(d, arr, "too_many_requests",
                  "A batch holds at most 50,000 requests.", 0, &count);
    for (long i = 0; ids && i < n && count <= MAX_ERRORS; i++) {
        const char *line = text + starts[i];
        yyjson_doc *ld = yyjson_read(line, strlen(line), 0);
        yyjson_val *o = yyjson_doc_get_root(ld);
        const char *cid = yyjson_get_str(yyjson_obj_get(o, "custom_id"));
        const char *method = yyjson_get_str(yyjson_obj_get(o, "method"));
        const char *url = yyjson_get_str(yyjson_obj_get(o, "url"));
        if (!yyjson_is_obj(o))
            add_error(d, arr, "invalid_json_line", "Not a JSON object.", i + 1,
                      &count);
        else if (!cid || !*cid)
            add_error(d, arr, "missing_required_parameter",
                      "Each request needs a custom_id.", i + 1, &count);
        else if (!method || strcmp(method, "POST") != 0)
            add_error(d, arr, "invalid_request",
                      "The method of a request is POST.", i + 1, &count);
        else if (!url || strcmp(url, b->endpoint) != 0)
            add_error(d, arr, "mismatched_endpoint",
                      "The url of a request is the batch's endpoint.", i + 1,
                      &count);
        else if (!yyjson_is_obj(yyjson_obj_get(o, "body")))
            add_error(d, arr, "invalid_request",
                      "The body of a request is a JSON object.", i + 1, &count);
        else {
            for (long k = 0; k < i; k++)
                if (ids[k] && !strcmp(ids[k], cid)) {
                    add_error(d, arr, "duplicate_custom_id",
                              "The custom_id is used twice.", i + 1, &count);
                    break;
                }
            ids[i] = strdup(cid);
        }
        yyjson_doc_free(ld);
    }
    for (long i = 0; ids && i < n; i++)
        free(ids[i]);
    free(ids);
    char *out = count ? yyjson_mut_val_write(arr, 0, NULL) : NULL;
    yyjson_mut_doc_free(d);
    return out;
}

/* bytes appended to a work file: 0, or -1 */
static int append(const char *path, const char *s, size_t n)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0)
        return -1;
    size_t w = 0;
    while (w < n) {
        ssize_t k = write(fd, s + w, n - w);
        if (k <= 0)
            break;
        w += (size_t)k;
    }
    return close(fd) == 0 && w == n ? 0 : -1;
}

/* One line run: its answer into the output or the error file. */
static void run_line(struct batch *b, const char *line)
{
    yyjson_doc *ld = yyjson_read(line, strlen(line), 0);
    yyjson_val *o = yyjson_doc_get_root(ld);
    yyjson_val *body = yyjson_obj_get(o, "body");
    const char *cid = yyjson_get_str(yyjson_obj_get(o, "custom_id"));
    struct srv_capture cap = {0};
    yyjson_val *stream = yyjson_obj_get(body, "stream");
    if (yyjson_is_true(stream)) {
        cap.status = 400;
        cap.body = strdup("{\"error\":{\"message\":\"A batch's requests are "
                          "not streamed.\",\"type\":"
                          "\"invalid_request_error\",\"param\":\"stream\","
                          "\"code\":null}}");
    } else {
        size_t bn = 0;
        char *bt = yyjson_val_write(body, 0, &bn);
        srv_handler h = srv_route_find("POST", b->endpoint + 3);
        struct srv_req r = {.srv = server,
                            .cap = &cap,
                            .method = "POST",
                            .path = b->endpoint + 3,
                            .body = bt ? bt : "{}",
                            .n_body = bt ? bn : 2};
        if (h)
            h(&r);
        free(bt);
        if (!cap.body) {
            cap.status = 500;
            cap.body = strdup("{\"error\":{\"message\":\"no answer\","
                              "\"type\":\"server_error\",\"param\":null,"
                              "\"code\":null}}");
        }
    }
    /* the answer's line */
    yyjson_doc *ad =
        cap.body ? yyjson_read(cap.body, strlen(cap.body), 0) : NULL;
    yyjson_val *ar = yyjson_doc_get_root(ad);
    yyjson_val *usage = yyjson_obj_get(ar, "usage");
    if (usage) {
        int64_t in = num(usage, "prompt_tokens") + num(usage, "input_tokens");
        int64_t out =
            num(usage, "completion_tokens") + num(usage, "output_tokens");
        b->in_tokens += in;
        b->out_tokens += out;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *x = yyjson_mut_obj(d);
    char rid[48], qid[48];
    srv_new_id(rid, sizeof rid, "batch_req_");
    srv_new_id(qid, sizeof qid, "req_");
    yyjson_mut_obj_add_strcpy(d, x, "id", rid);
    yyjson_mut_obj_add_strcpy(d, x, "custom_id", cid ? cid : "");
    yyjson_mut_val *resp = yyjson_mut_obj(d);
    yyjson_mut_obj_add_uint(d, resp, "status_code", cap.status);
    yyjson_mut_obj_add_strcpy(d, resp, "request_id", qid);
    if (ar)
        yyjson_mut_obj_add_val(d, resp, "body", yyjson_val_mut_copy(d, ar));
    else
        yyjson_mut_obj_add_null(d, resp, "body");
    yyjson_mut_obj_add_val(d, x, "response", resp);
    yyjson_mut_obj_add_null(d, x, "error");
    size_t n = 0;
    char *text = yyjson_mut_val_write(x, 0, &n);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(ad);
    yyjson_doc_free(ld);
    free(cap.body);
    int ok = cap.status == 200;
    char path[4300];
    work_path(path, sizeof path, b->id, ok ? ".out" : ".err");
    if (text && append(path, text, n) == 0 && append(path, "\n", 1) == 0) {
        if (ok) {
            b->out_bytes += (long long)n + 1;
            b->done_ok++;
        } else {
            b->err_bytes += (long long)n + 1;
            b->done_bad++;
        }
    } else {
        b->done_bad++; /* not written: counted failed */
    }
    free(text);
}

/* A work file cut back to what the last save counted (a line half written
   when the server stopped), or removed when nothing was. */
static void trim_work(const struct batch *b, const char *suf, long long n)
{
    char path[4300];
    work_path(path, sizeof path, b->id, suf);
    if (n == 0)
        unlink(path);
    else if (truncate(path, (off_t)n) != 0 && errno != ENOENT)
        fprintf(stderr, "janas-server: batch %s: %s cannot be cut back\n",
                b->id, path);
}

/* The work files as Files of the batch. */
static void finalize(struct batch *b, int status)
{
    int64_t now = (int64_t)time(NULL);
    b->status = FINALIZING;
    b->finalizing = now;
    save(b, 0);
    int64_t after = b->output_after ? b->output_after : 2592000;
    const char *suf[2] = {".out", ".err"};
    long long bytes[2] = {b->out_bytes, b->err_bytes};
    char *ids[2] = {b->output, b->error};
    for (int k = 0; k < 2; k++) {
        char path[4300], name[96], err[200];
        work_path(path, sizeof path, b->id, suf[k]);
        if (bytes[k] > 0) {
            snprintf(name, sizeof name, "%s_%s.jsonl", b->id,
                     k ? "error" : "output");
            const char *paths[1] = {path};
            struct srv_file f;
            if (srv_files_add_parts(server->files, name, "batch_output", paths,
                                    1, (uint64_t)bytes[k], NULL, after, &f, err,
                                    sizeof err) == 0)
                snprintf(ids[k], 48, "%s", f.id);
            else
                fprintf(stderr, "janas-server: batch %s: %s\n", b->id, err);
        }
        unlink(path);
    }
    b->status = status;
    now = (int64_t)time(NULL);
    if (status == COMPLETED)
        b->completed = now;
    else if (status == CANCELLED)
        b->cancelled = now;
    else
        b->expired = now;
    save(b, 0);
}

static int status_of(const char *id)
{
    struct batch b;
    int s = load(id, &b) == 0 ? b.status : -1;
    batch_free(&b);
    return s;
}

static void run_batch(const char *id)
{
    struct batch b;
    if (load(id, &b) != 0)
        return;
    char *text = NULL, why[200];
    long *starts = NULL, n = 0;
    if (b.status == VALIDATING || b.status == IN_PROGRESS ||
        b.status == CANCELLING) {
        n = read_input(&b, &text, &starts, why, sizeof why);
        if (n < 0 && b.status != CANCELLING) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *arr = yyjson_mut_arr(d);
            int c = 0;
            add_error(d, arr, "invalid_input_file", why, 0, &c);
            free(b.errors);
            b.errors = yyjson_mut_val_write(arr, 0, NULL);
            yyjson_mut_doc_free(d);
            b.status = FAILED;
            b.failed = (int64_t)time(NULL);
            save(&b, 0);
            batch_free(&b);
            return;
        }
    }
    if (b.status == VALIDATING) {
        char *errs = n > 0 ? validate(&b, text, starts, n) : NULL;
        if (n == 0)
            errs = strdup("[{\"code\":\"empty_file\",\"message\":\"The input "
                          "file holds no request.\",\"param\":null,"
                          "\"line\":null}]");
        if (errs) {
            free(b.errors);
            b.errors = errs;
            b.status = FAILED;
            b.failed = (int64_t)time(NULL);
            save(&b, 0);
            free(text);
            free(starts);
            batch_free(&b);
            return;
        }
        b.status = IN_PROGRESS;
        b.in_progress = (int64_t)time(NULL);
        b.total = n;
        b.next = 0;
        save(&b, 0);
    }
    /* a run stopped halfway: the files back to what was counted */
    trim_work(&b, ".out", b.out_bytes);
    trim_work(&b, ".err", b.err_bytes);
    int end = COMPLETED, stop = 0;
    srv_engine_low = 1;
    while (b.status == IN_PROGRESS && b.next < n) {
        /* a cancel is saved under mu, and so is every line's progress:
           neither overwrites the other */
        pthread_mutex_lock(&mu);
        stop = stopping;
        int s = status_of(b.id);
        pthread_mutex_unlock(&mu);
        if (stop)
            break;
        if (s == CANCELLING) {
            b.status = CANCELLING;
            break;
        }
        if ((int64_t)time(NULL) >= b.expires) {
            end = EXPIRED;
            break;
        }
        run_line(&b, text + starts[b.next]);
        b.next++;
        pthread_mutex_lock(&mu);
        if (status_of(b.id) == CANCELLING)
            b.status = CANCELLING;
        save(&b, 0);
        pthread_mutex_unlock(&mu);
    }
    if (b.status == CANCELLING)
        end = CANCELLED;
    if (!stop && (b.status == IN_PROGRESS || b.status == CANCELLING ||
                  b.status == FINALIZING))
        finalize(&b, end);
    free(text);
    free(starts);
    batch_free(&b);
}

/* The oldest batch with work left, into id: 1, or 0. */
static int next_batch(char *id, size_t cap)
{
    char **ids;
    size_t n = srv_store_list(server->store, KIND, &ids);
    int found = 0;
    for (size_t i = 0; i < n && !found; i++) {
        int s = status_of(ids[i]);
        if (s == VALIDATING || s == IN_PROGRESS || s == CANCELLING ||
            s == FINALIZING) {
            snprintf(id, cap, "%s", ids[i]);
            found = 1;
        }
    }
    srv_store_list_free(ids, n);
    return found;
}

static void *worker_main(void *arg)
{
    (void)arg;
    char id[64];
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!stopping && !next_batch(id, sizeof id))
            pthread_cond_wait(&cv, &mu);
        int stop = stopping;
        pthread_mutex_unlock(&mu);
        if (stop)
            break;
        run_batch(id);
    }
    return NULL;
}

int srv_batches_start(struct srv *s)
{
    server = s;
    snprintf(work_dir, sizeof work_dir, "%s/.batches", srv_files_dir(s->files));
    if (mkdir(work_dir, 0700) != 0 && errno != EEXIST)
        return -1;
    stopping = 0;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0)
        return -1;
    running = 1;
    return 0;
}

void srv_batches_stop(void)
{
    if (!running)
        return;
    pthread_mutex_lock(&mu);
    stopping = 1;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
    pthread_join(worker, NULL);
    running = 0;
}

/* ---- the API ---- */

static int reply_batch(struct srv_req *r, const struct batch *b)
{
    char *json = batch_json(b, NULL);
    return srv_reply_json(r, 200, json, json ? strlen(json) : 0);
}

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

static int no_batch(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No batch found with id '%s'.", r->param[0]);
}

int srv_batches_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *input = yyjson_get_str(yyjson_obj_get(root, "input_file_id"));
    const char *ep = yyjson_get_str(yyjson_obj_get(root, "endpoint"));
    const char *win = yyjson_get_str(yyjson_obj_get(root, "completion_window"));
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    yyjson_val *oe = yyjson_obj_get(root, "output_expires_after");
    int ep_ok = 0;
    for (size_t i = 0; ep && i < sizeof(ENDPOINTS) / sizeof(ENDPOINTS[0]); i++)
        ep_ok |= !strcmp(ep, ENDPOINTS[i]);
    struct srv_file f;
    const char *why = NULL;
    int64_t out_after = 0;
    if (!yyjson_is_obj(root))
        why = "The body is a JSON object: input_file_id, endpoint, "
              "completion_window.";
    else if (!input || srv_files_get(r->srv->files, input, &f) != 0)
        why = "'input_file_id' names no file uploaded here.";
    else if (strcmp(f.purpose, "batch") != 0)
        why = "The input file must be uploaded with purpose batch.";
    else if (!ep_ok)
        why = "'endpoint' is /v1/chat/completions, /v1/responses, "
              "/v1/completions, /v1/embeddings or /v1/moderations here.";
    else if (!win || strcmp(win, "24h") != 0)
        why = "'completion_window' is 24h.";
    else if (md && !yyjson_is_null(md) && !yyjson_is_obj(md))
        why = "'metadata' is an object.";
    else if (oe) {
        yyjson_val *se = yyjson_obj_get(oe, "seconds");
        if (!yyjson_equals_str(yyjson_obj_get(oe, "anchor"), "created_at") ||
            !yyjson_is_int(se) || yyjson_get_sint(se) < 3600 ||
            yyjson_get_sint(se) > 2592000)
            why = "'output_expires_after' is anchor created_at and seconds "
                  "from 3600 to 2592000.";
        else
            out_after = yyjson_get_sint(se);
    }
    if (why) {
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    struct batch b = {0};
    srv_new_id(b.id, sizeof b.id, "batch_");
    snprintf(b.endpoint, sizeof b.endpoint, "%s", ep);
    snprintf(b.input, sizeof b.input, "%s", input);
    b.status = VALIDATING;
    b.created = (int64_t)time(NULL);
    b.expires = b.created + WINDOW;
    b.output_after = out_after;
    if (yyjson_is_obj(md))
        b.metadata = yyjson_val_write(md, 0, NULL);
    yyjson_doc_free(doc);
    int rc = save(&b, 1);
    if (rc == 0) {
        pthread_mutex_lock(&mu);
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
    }
    int ret = rc == 0 ? reply_batch(r, &b)
                      : srv_reply_error(r, 500, "server_error", NULL,
                                        "cannot keep the batch");
    batch_free(&b);
    return ret;
}

int srv_batches_get(struct srv_req *r)
{
    struct batch b;
    if (load(r->param[0], &b) != 0)
        return no_batch(r);
    int ret = reply_batch(r, &b);
    batch_free(&b);
    return ret;
}

int srv_batches_cancel(struct srv_req *r)
{
    struct batch b;
    if (load(r->param[0], &b) != 0)
        return no_batch(r);
    if (b.status == VALIDATING || b.status == IN_PROGRESS) {
        pthread_mutex_lock(&mu);
        batch_free(&b);
        if (load(r->param[0], &b) == 0 &&
            (b.status == VALIDATING || b.status == IN_PROGRESS)) {
            b.status = CANCELLING;
            b.cancelling = (int64_t)time(NULL);
            save(&b, 0);
        }
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
    } else if (b.status != CANCELLING) {
        batch_free(&b);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_state",
                               "The batch %s is %s: it cannot be cancelled.",
                               r->param[0], STATUS[b.status]);
    }
    int ret = reply_batch(r, &b);
    batch_free(&b);
    return ret;
}

int srv_batches_list(struct srv_req *r)
{
    const char *l = srv_query(r, "limit"), *after = srv_query(r, "after");
    long limit = 20;
    if (l) {
        char *end;
        limit = strtol(l, &end, 10);
        if (*end || limit < 1 || limit > 100)
            return bad(r, "'limit' is a number from 1 to 100.");
    }
    char **ids;
    size_t n = srv_store_list(r->srv->store, KIND, &ids);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    long shown = 0;
    int past = after == NULL, more = 0;
    char first[64] = "", last[64] = "";
    for (size_t k = n; k-- > 0;) { /* newest first */
        if (!past) {
            past = !strcmp(ids[k], after);
            continue;
        }
        if (shown == limit) {
            more = 1;
            break;
        }
        struct batch b;
        if (load(ids[k], &b) != 0)
            continue;
        char *json = batch_json(&b, NULL);
        yyjson_doc *bd = json ? yyjson_read(json, strlen(json), 0) : NULL;
        if (bd)
            yyjson_mut_arr_append(
                data, yyjson_val_mut_copy(d, yyjson_doc_get_root(bd)));
        yyjson_doc_free(bd);
        free(json);
        if (!shown)
            snprintf(first, sizeof first, "%s", b.id);
        snprintf(last, sizeof last, "%s", b.id);
        shown++;
        batch_free(&b);
    }
    srv_store_list_free(ids, n);
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
    yyjson_mut_doc_set_root(d, root);
    size_t len;
    char *out = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, len);
}
