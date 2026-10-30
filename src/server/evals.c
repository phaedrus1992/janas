/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * evals.c - OpenAI's Evals API: an eval is a data schema and graders; a
 * run of it takes rows of data (given in the request, or a JSONL file),
 * has the model write a sample for each where the run asks (a template of
 * messages filled from the row), and grades each row: string_check,
 * text_similarity (evals_metrics.c), label_model and score_model, the
 * model itself answering through a JSON schema that holds it to a label or
 * a score. A run goes on a thread of its own, behind every client's
 * request, each row through the same handlers as /v1/chat/completions and
 * /v1/embeddings; its rows' results are written as they come, beside the
 * files (.evals), and the run saved after each, so a stopped server goes
 * on from there.
 */
#include "server.h"

#include <ctype.h>
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

#include "evals_metrics.h"
#include "files_store.h"
#include "llm/json.h"
#include "store_disk.h"

#define EVAL "eval"
#define RUN "eval.run"
#define MAX_ROWS 10000

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static pthread_t worker;
static int running, stopping;
static struct srv *server;
static char work[4200];

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

static int reply_text(struct srv_req *r, char *json)
{
    return srv_reply_json(r, 200, json, json ? strlen(json) : 0);
}

static char *store_get(const char *kind, const char *id, char **aux)
{
    char *j = NULL;
    if (srv_store_get(server->store, kind, id, &j, aux) != 0)
        return NULL;
    return j;
}

/* A document of the store as a mutable JSON object (NULL: none). */
static yyjson_mut_val *load(yyjson_mut_doc *d, const char *kind, const char *id)
{
    char *t = store_get(kind, id, NULL);
    yyjson_doc *x = t ? yyjson_read(t, strlen(t), 0) : NULL;
    free(t);
    yyjson_mut_val *o =
        x ? yyjson_val_mut_copy(d, yyjson_doc_get_root(x)) : NULL;
    yyjson_doc_free(x);
    return o;
}

static int save(const char *kind, const char *id, yyjson_mut_val *o,
                const char *aux, int fresh)
{
    char *t = yyjson_mut_val_write(o, 0, NULL);
    int rc = -1;
    if (t)
        rc = fresh ? srv_store_put(server->store, kind, id, (int64_t)time(NULL),
                                   t, aux)
                   : srv_store_update(server->store, kind, id, t, aux);
    free(t);
    return rc;
}

static int metadata_ok(yyjson_val *m)
{
    if (!m || yyjson_is_null(m))
        return 1;
    if (!yyjson_is_obj(m) || yyjson_obj_size(m) > 16)
        return 0;
    size_t i, n;
    yyjson_val *k, *v;
    yyjson_obj_foreach(m, i, n, k, v) if (!yyjson_is_str(v) ||
                                          yyjson_get_len(v) > 512) return 0;
    return 1;
}

/* ---- templates: {{item.a.b}}, {{sample.output_text}} ---- */

static yyjson_val *path(yyjson_val *root, const char *p, size_t n)
{
    yyjson_val *v = root;
    while (v && n) {
        size_t k = 0;
        while (k < n && p[k] != '.')
            k++;
        char key[128];
        if (k >= sizeof key)
            return NULL;
        memcpy(key, p, k);
        key[k] = 0;
        if (yyjson_is_arr(v) && isdigit((unsigned char)key[0]))
            v = yyjson_arr_get(v, (size_t)atol(key));
        else
            v = yyjson_obj_get(v, key);
        p += k;
        n -= k;
        if (n && *p == '.')
            p++, n--;
    }
    return v;
}

/* A template filled from the namespaces item and sample (malloc'd). */
static char *render(const char *t, yyjson_val *item, yyjson_val *sample)
{
    struct janas_buf b = {0};
    for (const char *p = t; *p;) {
        const char *open = strstr(p, "{{");
        if (!open) {
            janas_buf_put(&b, p, strlen(p));
            break;
        }
        janas_buf_put(&b, p, (size_t)(open - p));
        const char *close = strstr(open + 2, "}}");
        if (!close) {
            janas_buf_put(&b, open, strlen(open));
            break;
        }
        const char *a = open + 2, *e = close;
        while (a < e && *a == ' ')
            a++;
        while (e > a && e[-1] == ' ')
            e--;
        yyjson_val *v = NULL;
        if ((size_t)(e - a) >= 5 && !strncmp(a, "item.", 5))
            v = path(item, a + 5, (size_t)(e - a) - 5);
        else if ((size_t)(e - a) == 4 && !strncmp(a, "item", 4))
            v = item;
        else if ((size_t)(e - a) >= 7 && !strncmp(a, "sample.", 7))
            v = path(sample, a + 7, (size_t)(e - a) - 7);
        if (yyjson_is_str(v))
            janas_buf_put(&b, yyjson_get_str(v), yyjson_get_len(v));
        else if (v) {
            size_t n;
            char *j = yyjson_val_write(v, 0, &n);
            if (j)
                janas_buf_put(&b, j, n);
            free(j);
        }
        p = close + 2;
    }
    janas_buf_put(&b, "", 1);
    if (b.oom) {
        janas_buf_free(&b);
        return NULL;
    }
    return b.p;
}

/* A message's text: content a string, or {type: input_text, text}, or a
   list of those. */
static const char *message_text(yyjson_val *m)
{
    yyjson_val *c = yyjson_obj_get(m, "content");
    if (yyjson_is_str(c))
        return yyjson_get_str(c);
    if (yyjson_is_obj(c))
        return yyjson_get_str(yyjson_obj_get(c, "text"));
    if (yyjson_is_arr(c))
        return yyjson_get_str(yyjson_obj_get(yyjson_arr_get_first(c), "text"));
    return NULL;
}

/* ---- the model, asked from inside ---- */

/* A chat completion of messages (a JSON array text) with the run's
   sampling, and an optional response_format (JSON text): the status, the
   answer's body in *out (malloc'd). */
static unsigned chat(const char *messages, yyjson_val *sampling,
                     const char *format, char **out)
{
    *out = NULL;
    srv_handler h = srv_route_find("POST", "/chat/completions");
    if (!h)
        return 500;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_doc *md = yyjson_read(messages, strlen(messages), 0);
    yyjson_mut_obj_add_val(d, o, "messages",
                           yyjson_val_mut_copy(d, yyjson_doc_get_root(md)));
    static const char *const keys[] = {"temperature", "top_p", "seed"};
    for (int i = 0; i < 3; i++) {
        yyjson_val *v = yyjson_obj_get(sampling, keys[i]);
        if (yyjson_is_num(v))
            yyjson_mut_obj_add_val(d, o, keys[i], yyjson_val_mut_copy(d, v));
    }
    yyjson_val *mx = yyjson_obj_get(sampling, "max_completion_tokens");
    if (!mx)
        mx = yyjson_obj_get(sampling, "max_completions_tokens");
    yyjson_mut_obj_add_int(d, o, "max_completion_tokens",
                           yyjson_is_int(mx) ? yyjson_get_sint(mx) : 1024);
    /* graders and samples answer, they do not muse at length */
    yyjson_val *re = yyjson_obj_get(sampling, "reasoning_effort");
    yyjson_mut_obj_add_strcpy(d, o, "reasoning_effort",
                              yyjson_is_str(re) ? yyjson_get_str(re) : "none");
    yyjson_doc *fd = format ? yyjson_read(format, strlen(format), 0) : NULL;
    if (fd)
        yyjson_mut_obj_add_val(d, o, "response_format",
                               yyjson_val_mut_copy(d, yyjson_doc_get_root(fd)));
    size_t bn;
    char *body = yyjson_mut_write(d, 0, &bn);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(md);
    yyjson_doc_free(fd);
    struct srv_capture cap = {0};
    struct srv_req r = {.srv = server,
                        .cap = &cap,
                        .method = "POST",
                        .path = "/chat/completions",
                        .body = body ? body : "{}",
                        .n_body = body ? bn : 2};
    h(&r);
    free(body);
    *out = cap.body;
    return cap.status;
}

/* The messages of a template filled from a row, as a JSON array text. */
static char *fill_messages(yyjson_val *tmpl, yyjson_val *item,
                           yyjson_val *sample)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    size_t i, n;
    yyjson_val *m;
    yyjson_arr_foreach(tmpl, i, n, m)
    {
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        const char *text = message_text(m);
        char *t = render(text ? text : "", item, sample);
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_strcpy(d, o, "role",
                                  role && !strcmp(role, "developer") ? "system"
                                  : role                             ? role
                                                                     : "user");
        yyjson_mut_obj_add_strcpy(d, o, "content", t ? t : "");
        free(t);
    }
    char *out = yyjson_mut_val_write(arr, 0, NULL);
    yyjson_mut_doc_free(d);
    return out;
}

/* ---- graders ---- */

/* One grader on a row: its result object into results; 1 passed, 0
   failed, -1 an error. */
static int grade(yyjson_mut_doc *d, yyjson_mut_val *results, yyjson_val *g,
                 yyjson_val *item, yyjson_val *sample)
{
    const char *type = yyjson_get_str(yyjson_obj_get(g, "type"));
    const char *name = yyjson_get_str(yyjson_obj_get(g, "name"));
    yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, results);
    yyjson_mut_obj_add_strcpy(d, o, "name", name ? name : "");
    yyjson_mut_obj_add_strcpy(d, o, "type", type ? type : "");
    double score = 0;
    int passed = 0, err = 0;
    if (type && !strcmp(type, "string_check")) {
        char *in =
            render(yyjson_get_str(yyjson_obj_get(g, "input")), item, sample);
        char *ref = render(yyjson_get_str(yyjson_obj_get(g, "reference")), item,
                           sample);
        const char *op = yyjson_get_str(yyjson_obj_get(g, "operation"));
        if (in && ref && op) {
            if (!strcmp(op, "eq"))
                passed = !strcmp(in, ref);
            else if (!strcmp(op, "ne"))
                passed = strcmp(in, ref) != 0;
            else if (!strcmp(op, "like"))
                passed = strstr(in, ref) != NULL;
            else if (!strcmp(op, "ilike"))
                passed = strcasestr(in, ref) != NULL;
        } else {
            err = 1;
        }
        score = passed;
        free(in);
        free(ref);
    } else if (type && !strcmp(type, "text_similarity")) {
        char *in =
            render(yyjson_get_str(yyjson_obj_get(g, "input")), item, sample);
        char *ref = render(yyjson_get_str(yyjson_obj_get(g, "reference")), item,
                           sample);
        const char *m = yyjson_get_str(yyjson_obj_get(g, "evaluation_metric"));
        if (in && ref && m)
            score = srv_eval_similarity(m, in, ref);
        else
            err = 1;
        yyjson_val *th = yyjson_obj_get(g, "pass_threshold");
        passed = !err && score >= (th ? yyjson_get_num(th) : 1);
        free(in);
        free(ref);
    } else if (type &&
               (!strcmp(type, "label_model") || !strcmp(type, "score_model"))) {
        int label = type[0] == 'l';
        char *msgs = fill_messages(yyjson_obj_get(g, "input"), item, sample);
        /* the answer held to a label of the list, or a number */
        yyjson_mut_doc *fd = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *f = yyjson_mut_obj(fd);
        yyjson_mut_doc_set_root(fd, f);
        yyjson_mut_obj_add_str(fd, f, "type", "json_schema");
        yyjson_mut_val *js = yyjson_mut_obj_add_obj(fd, f, "json_schema");
        yyjson_mut_obj_add_str(fd, js, "name", "grade");
        yyjson_mut_val *sc = yyjson_mut_obj_add_obj(fd, js, "schema");
        yyjson_mut_obj_add_str(fd, sc, "type", "object");
        yyjson_mut_val *pr = yyjson_mut_obj_add_obj(fd, sc, "properties");
        yyjson_mut_val *res = yyjson_mut_obj_add_obj(fd, pr, "result");
        yyjson_val *range = yyjson_obj_get(g, "range");
        double lo = 0, hi = 1;
        if (yyjson_arr_size(range) == 2) {
            lo = yyjson_get_num(yyjson_arr_get(range, 0));
            hi = yyjson_get_num(yyjson_arr_get(range, 1));
        }
        if (label) {
            yyjson_mut_obj_add_str(fd, res, "type", "string");
            yyjson_mut_obj_add_val(
                fd, res, "enum",
                yyjson_val_mut_copy(fd, yyjson_obj_get(g, "labels")));
        } else {
            yyjson_mut_obj_add_str(fd, res, "type", "number");
            yyjson_mut_obj_add_real(fd, res, "minimum", lo);
            yyjson_mut_obj_add_real(fd, res, "maximum", hi);
        }
        yyjson_mut_val *rq = yyjson_mut_obj_add_arr(fd, sc, "required");
        yyjson_mut_arr_add_str(fd, rq, "result");
        yyjson_mut_obj_add_bool(fd, sc, "additionalProperties", false);
        char *fmt = yyjson_mut_write(fd, 0, NULL);
        yyjson_mut_doc_free(fd);
        char *body = NULL;
        unsigned st =
            msgs && fmt
                ? chat(msgs, yyjson_obj_get(g, "sampling_params"), fmt, &body)
                : 500;
        free(msgs);
        free(fmt);
        yyjson_doc *bd = body ? yyjson_read(body, strlen(body), 0) : NULL;
        const char *content = yyjson_get_str(yyjson_obj_get(
            yyjson_obj_get(yyjson_arr_get_first(yyjson_obj_get(
                               yyjson_doc_get_root(bd), "choices")),
                           "message"),
            "content"));
        yyjson_doc *cd =
            content ? yyjson_read(content, strlen(content), 0) : NULL;
        yyjson_val *rv = yyjson_obj_get(yyjson_doc_get_root(cd), "result");
        if (st != 200 || !rv) {
            err = 1;
        } else if (label) {
            const char *l = yyjson_get_str(rv);
            size_t i, n;
            yyjson_val *x;
            yyjson_arr_foreach(yyjson_obj_get(g, "passing_labels"), i, n, x)
                passed |= l && yyjson_equals_str(x, l);
            score = passed;
            yyjson_mut_val *smp = yyjson_mut_obj_add_obj(d, o, "sample");
            yyjson_mut_obj_add_strcpy(d, smp, "label", l ? l : "");
        } else {
            score = yyjson_get_num(rv);
            yyjson_val *th = yyjson_obj_get(g, "pass_threshold");
            passed = score >= (th ? yyjson_get_num(th) : (lo + hi) / 2);
        }
        yyjson_doc_free(cd);
        yyjson_doc_free(bd);
        free(body);
    } else {
        err = 1;
    }
    yyjson_mut_obj_add_real(d, o, "score", score);
    yyjson_mut_obj_add_bool(d, o, "passed", passed && !err);
    if (err)
        yyjson_mut_obj_add_str(d, o, "error", "the grader could not run");
    return err ? -1 : passed;
}

/* ---- a run, row by row ---- */

static void out_path(char *p, size_t cap, const char *run)
{
    snprintf(p, cap, "%s/%s.jsonl", work, run);
}

/* The rows of a run's data source: a JSON array text of {item, sample?}
   (malloc'd), or NULL with the reason. */
static char *rows_of(yyjson_val *ds, char *why, size_t wl)
{
    yyjson_val *src = yyjson_obj_get(ds, "source");
    const char *st = yyjson_get_str(yyjson_obj_get(src, "type"));
    if (st && !strcmp(st, "file_content")) {
        yyjson_val *c = yyjson_obj_get(src, "content");
        if (!yyjson_is_arr(c)) {
            snprintf(why, wl, "the source's content is a list of rows");
            return NULL;
        }
        return yyjson_val_write(c, 0, NULL);
    }
    if (st && !strcmp(st, "file_id")) {
        const char *id = yyjson_get_str(yyjson_obj_get(src, "id"));
        struct srv_file f;
        if (!id || srv_files_get(server->files, id, &f) != 0 ||
            f.bytes > ((size_t)64 << 20)) {
            snprintf(why, wl, "the source's file is not here");
            return NULL;
        }
        int fd = srv_files_open_content(server->files, id);
        char *t = fd >= 0 ? malloc(f.bytes + 1) : NULL;
        size_t got = 0;
        while (t && got < f.bytes) {
            ssize_t k = read(fd, t + got, f.bytes - got);
            if (k <= 0)
                break;
            got += (size_t)k;
        }
        if (fd >= 0)
            close(fd);
        if (!t) {
            snprintf(why, wl, "the source's file cannot be read");
            return NULL;
        }
        t[got] = 0;
        /* its JSON lines into an array */
        struct janas_buf b = {0};
        janas_buf_put(&b, "[", 1);
        int first = 1;
        for (char *l = t; *l;) {
            char *e = strchr(l, '\n');
            size_t n = e ? (size_t)(e - l) : strlen(l);
            size_t k = 0;
            while (k < n && isspace((unsigned char)l[k]))
                k++;
            if (k < n) {
                if (!first)
                    janas_buf_put(&b, ",", 1);
                janas_buf_put(&b, l, n);
                first = 0;
            }
            l = e ? e + 1 : l + n;
        }
        janas_buf_put(&b, "]", 2);
        free(t);
        if (b.oom || !yyjson_read(b.p, b.n - 1, 0)) {
            janas_buf_free(&b);
            snprintf(why, wl, "the source's file is not JSON lines");
            return NULL;
        }
        return b.p;
    }
    snprintf(why, wl, "the source is file_content or file_id");
    return NULL;
}

static void run_one(const char *run_id)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    char *aux = NULL;
    char *rt = store_get(RUN, run_id, &aux);
    yyjson_doc *rd = rt ? yyjson_read(rt, strlen(rt), 0) : NULL;
    free(rt);
    yyjson_val *run = yyjson_doc_get_root(rd);
    const char *eval_id = yyjson_get_str(yyjson_obj_get(run, "eval_id"));
    char *et = eval_id ? store_get(EVAL, eval_id, NULL) : NULL;
    yyjson_doc *ed = et ? yyjson_read(et, strlen(et), 0) : NULL;
    free(et);
    yyjson_val *criteria =
        yyjson_obj_get(yyjson_doc_get_root(ed), "testing_criteria");
    yyjson_val *ds = yyjson_obj_get(run, "data_source");
    char why[200] = "";
    char *rows = run && criteria ? rows_of(ds, why, sizeof why) : NULL;
    yyjson_doc *rowsd = rows ? yyjson_read(rows, strlen(rows), 0) : NULL;
    free(rows);
    yyjson_val *all = yyjson_doc_get_root(rowsd);
    yyjson_mut_val *o = run ? yyjson_val_mut_copy(d, run) : NULL;
    long next = aux ? atol(aux) : 0;
    free(aux);
    if (!o) {
        yyjson_doc_free(rd);
        yyjson_doc_free(ed);
        yyjson_doc_free(rowsd);
        yyjson_mut_doc_free(d);
        return;
    }
    if (!all || yyjson_arr_size(all) > MAX_ROWS) {
        yyjson_mut_obj_remove_key(o, "status");
        yyjson_mut_obj_add_str(d, o, "status", "failed");
        yyjson_mut_obj_remove_key(o, "error");
        yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, o, "error");
        yyjson_mut_obj_add_str(d, e, "code", "invalid_data_source");
        yyjson_mut_obj_add_strcpy(d, e, "message",
                                  all ? "more than 10,000 rows" : why);
        save(RUN, run_id, o, "0", 0);
        yyjson_doc_free(rd);
        yyjson_doc_free(ed);
        yyjson_doc_free(rowsd);
        yyjson_mut_doc_free(d);
        return;
    }
    yyjson_mut_obj_remove_key(o, "status");
    yyjson_mut_obj_add_str(d, o, "status", "in_progress");
    const char *dtype = yyjson_get_str(yyjson_obj_get(ds, "type"));
    int sampled = dtype && strcmp(dtype, "jsonl") != 0;
    yyjson_val *tmpl =
        yyjson_obj_get(yyjson_obj_get(ds, "input_messages"), "template");
    const char *iref = yyjson_get_str(
        yyjson_obj_get(yyjson_obj_get(ds, "input_messages"), "item_reference"));
    yyjson_val *sampling = yyjson_obj_get(ds, "sampling_params");
    size_t total = yyjson_arr_size(all), nc = yyjson_arr_size(criteria);
    char op[4600];
    out_path(op, sizeof op, run_id);
    for (size_t i = (size_t)next; i < total; i++) {
        pthread_mutex_lock(&mu);
        int stop = stopping;
        pthread_mutex_unlock(&mu);
        /* a cancel saved meanwhile */
        char *now = store_get(RUN, run_id, NULL);
        int cancelled = now && strstr(now, "\"status\":\"canceled\"");
        free(now);
        if (stop || cancelled)
            break;
        yyjson_val *row = yyjson_arr_get(all, i);
        yyjson_val *item = yyjson_obj_get(row, "item");
        if (!item)
            item = row;
        yyjson_mut_doc *od = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *oi = yyjson_mut_obj(od);
        char oid[48];
        srv_new_id(oid, sizeof oid, "outputitem_");
        yyjson_mut_obj_add_str(od, oi, "object", "eval.run.output_item");
        yyjson_mut_obj_add_strcpy(od, oi, "id", oid);
        yyjson_mut_obj_add_strcpy(od, oi, "run_id", run_id);
        yyjson_mut_obj_add_strcpy(od, oi, "eval_id", eval_id);
        yyjson_mut_obj_add_int(od, oi, "created_at", (int64_t)time(NULL));
        yyjson_mut_obj_add_int(od, oi, "datasource_item_id", (int64_t)i);
        yyjson_mut_obj_add_val(od, oi, "datasource_item",
                               yyjson_val_mut_copy(od, item));
        /* the sample: the row's, or the model's answer to the template */
        yyjson_doc *sd = NULL;
        int errored = 0;
        yyjson_mut_val *smp = yyjson_mut_obj(od);
        if (sampled) {
            char *msgs = NULL;
            if (tmpl)
                msgs = fill_messages(tmpl, item, NULL);
            else if (iref) {
                const char *p = !strncmp(iref, "item.", 5) ? iref + 5 : iref;
                yyjson_val *ms = path(item, p, strlen(p));
                msgs = ms ? fill_messages(ms, item, NULL) : NULL;
            }
            char *body = NULL;
            unsigned st = msgs ? chat(msgs, sampling, NULL, &body) : 400;
            yyjson_doc *bd = body ? yyjson_read(body, strlen(body), 0) : NULL;
            yyjson_val *ch0 = yyjson_arr_get_first(
                yyjson_obj_get(yyjson_doc_get_root(bd), "choices"));
            const char *text = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(ch0, "message"), "content"));
            yyjson_doc *md = msgs ? yyjson_read(msgs, strlen(msgs), 0) : NULL;
            yyjson_mut_obj_add_val(
                od, smp, "input",
                md ? yyjson_val_mut_copy(od, yyjson_doc_get_root(md))
                   : yyjson_mut_arr(od));
            yyjson_mut_val *outs = yyjson_mut_obj_add_arr(od, smp, "output");
            yyjson_mut_val *om = yyjson_mut_arr_add_obj(od, outs);
            yyjson_mut_obj_add_str(od, om, "role", "assistant");
            yyjson_mut_obj_add_strcpy(od, om, "content", text ? text : "");
            const char *fr =
                yyjson_get_str(yyjson_obj_get(ch0, "finish_reason"));
            yyjson_mut_obj_add_strcpy(od, smp, "finish_reason",
                                      fr ? fr : "error");
            yyjson_mut_obj_add_strcpy(od, smp, "model", server->cfg.model_id);
            yyjson_val *u = yyjson_obj_get(yyjson_doc_get_root(bd), "usage");
            if (u)
                yyjson_mut_obj_add_val(od, smp, "usage",
                                       yyjson_val_mut_copy(od, u));
            yyjson_mut_obj_add_strcpy(od, smp, "output_text", text ? text : "");
            if (st != 200) {
                errored = 1;
                yyjson_mut_val *e = yyjson_mut_obj_add_obj(od, smp, "error");
                yyjson_mut_obj_add_str(od, e, "code", "sample_failed");
                yyjson_mut_obj_add_strcpy(od, e, "message",
                                          body ? body : "no answer");
            }
            yyjson_doc_free(md);
            yyjson_doc_free(bd);
            free(body);
            free(msgs);
        } else if (yyjson_obj_get(row, "sample")) {
            yyjson_val *rs = yyjson_obj_get(row, "sample");
            smp = yyjson_val_mut_copy(od, rs);
        }
        /* the graders see the sample as written */
        char *st = yyjson_mut_val_write(smp, 0, NULL);
        sd = st ? yyjson_read(st, strlen(st), 0) : NULL;
        free(st);
        yyjson_mut_val *results = yyjson_mut_arr(od);
        int all_pass = !errored;
        yyjson_val *g;
        size_t gi, gm;
        yyjson_arr_foreach(criteria, gi, gm, g)
        {
            int p = errored
                        ? -1
                        : grade(od, results, g, item, yyjson_doc_get_root(sd));
            if (p < 0)
                errored = 1;
            if (p != 1)
                all_pass = 0;
        }
        yyjson_doc_free(sd);
        yyjson_mut_obj_add_val(od, oi, "results", results);
        yyjson_mut_obj_add_val(od, oi, "sample", smp);
        yyjson_mut_obj_add_str(od, oi, "status",
                               errored    ? "error"
                               : all_pass ? "pass"
                                          : "fail");
        size_t ln;
        char *line = yyjson_mut_val_write(oi, 0, &ln);
        int fd = open(op, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd >= 0 && line) {
            ssize_t w1 = write(fd, line, ln), w2 = write(fd, "\n", 1);
            (void)w1;
            (void)w2;
        }
        if (fd >= 0)
            close(fd);
        free(line);
        /* the run's counts */
        yyjson_mut_val *rc = yyjson_mut_obj_get(o, "result_counts");
        static const char *const K[] = {"total", "errored", "failed", "passed"};
        int64_t c[4];
        for (int k = 0; k < 4; k++)
            c[k] = yyjson_mut_get_sint(yyjson_mut_obj_get(rc, K[k]));
        c[0]++;
        c[errored ? 1 : all_pass ? 3 : 2]++;
        yyjson_mut_obj_remove_key(o, "result_counts");
        rc = yyjson_mut_obj_add_obj(d, o, "result_counts");
        for (int k = 0; k < 4; k++)
            yyjson_mut_obj_add_int(d, rc, K[k], c[k]);
        /* per grader: its passes and fails */
        yyjson_mut_val *pt =
            yyjson_mut_obj_get(o, "per_testing_criteria_results");
        size_t ri;
        yyjson_mut_val *rv;
        size_t rmax;
        yyjson_mut_arr_foreach(results, ri, rmax, rv)
        {
            yyjson_mut_val *slot = yyjson_mut_arr_get(pt, ri);
            if (!slot || ri >= nc)
                continue;
            const char *key =
                yyjson_mut_get_bool(yyjson_mut_obj_get(rv, "passed"))
                    ? "passed"
                    : "failed";
            int64_t v = yyjson_mut_get_sint(yyjson_mut_obj_get(slot, key));
            yyjson_mut_obj_remove_key(slot, key);
            yyjson_mut_obj_add_int(d, slot, key, v + 1);
        }
        yyjson_mut_doc_free(od);
        char nx[32];
        snprintf(nx, sizeof nx, "%zu", i + 1);
        pthread_mutex_lock(&mu);
        char *cur = store_get(RUN, run_id, NULL);
        int was_cancelled = cur && strstr(cur, "\"status\":\"canceled\"");
        free(cur);
        if (was_cancelled) {
            yyjson_mut_obj_remove_key(o, "status");
            yyjson_mut_obj_add_str(d, o, "status", "canceled");
        }
        save(RUN, run_id, o, nx, 0);
        pthread_mutex_unlock(&mu);
        if (was_cancelled)
            break;
    }
    pthread_mutex_lock(&mu);
    char *cur = store_get(RUN, run_id, NULL);
    int was_cancelled = cur && strstr(cur, "\"status\":\"canceled\"");
    free(cur);
    if (!stopping && !was_cancelled) {
        yyjson_mut_obj_remove_key(o, "status");
        yyjson_mut_obj_add_str(d, o, "status", "completed");
        char nx[32];
        snprintf(nx, sizeof nx, "%zu", total);
        save(RUN, run_id, o, nx, 0);
    }
    pthread_mutex_unlock(&mu);
    yyjson_doc_free(rd);
    yyjson_doc_free(ed);
    yyjson_doc_free(rowsd);
    yyjson_mut_doc_free(d);
}

/* The oldest run with work left: 1 with its id. */
static int next_run(char *id, size_t cap)
{
    char **ids;
    size_t n = srv_store_list(server->store, RUN, &ids);
    int found = 0;
    for (size_t i = 0; i < n && !found; i++) {
        char *t = store_get(RUN, ids[i], NULL);
        if (t && (strstr(t, "\"status\":\"queued\"") ||
                  strstr(t, "\"status\":\"in_progress\""))) {
            snprintf(id, cap, "%s", ids[i]);
            found = 1;
        }
        free(t);
    }
    srv_store_list_free(ids, n);
    return found;
}

static void *worker_main(void *arg)
{
    (void)arg;
    srv_engine_low = 1;
    char id[64];
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!stopping && !next_run(id, sizeof id))
            pthread_cond_wait(&cv, &mu);
        int stop = stopping;
        pthread_mutex_unlock(&mu);
        if (stop)
            break;
        run_one(id);
    }
    return NULL;
}

int srv_evals_start(struct srv *s)
{
    server = s;
    snprintf(work, sizeof work, "%s/.evals", srv_files_dir(s->files));
    if (mkdir(work, 0700) != 0 && errno != EEXIST)
        return -1;
    stopping = 0;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0)
        return -1;
    running = 1;
    return 0;
}

void srv_evals_stop(void)
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

/* ---- the API: evals ---- */

static const char *grader_error(yyjson_val *g)
{
    const char *type = yyjson_get_str(yyjson_obj_get(g, "type"));
    if (!type || !yyjson_get_str(yyjson_obj_get(g, "name")))
        return "Every grader has a type and a name.";
    if (!strcmp(type, "string_check")) {
        const char *op = yyjson_get_str(yyjson_obj_get(g, "operation"));
        if (!yyjson_get_str(yyjson_obj_get(g, "input")) ||
            !yyjson_get_str(yyjson_obj_get(g, "reference")) || !op ||
            (strcmp(op, "eq") && strcmp(op, "ne") && strcmp(op, "like") &&
             strcmp(op, "ilike")))
            return "A string_check grader has input, reference and an "
                   "operation of eq, ne, like or ilike.";
        return NULL;
    }
    if (!strcmp(type, "text_similarity"))
        return yyjson_get_str(yyjson_obj_get(g, "input")) &&
                       yyjson_get_str(yyjson_obj_get(g, "reference")) &&
                       srv_eval_metric_ok(yyjson_get_str(
                           yyjson_obj_get(g, "evaluation_metric")))
                   ? NULL
                   : "A text_similarity grader has input, reference and an "
                     "evaluation_metric.";
    if (!strcmp(type, "label_model"))
        return yyjson_is_arr(yyjson_obj_get(g, "input")) &&
                       yyjson_arr_size(yyjson_obj_get(g, "labels")) &&
                       yyjson_is_arr(yyjson_obj_get(g, "passing_labels"))
                   ? NULL
                   : "A label_model grader has input messages, labels and "
                     "passing_labels.";
    if (!strcmp(type, "score_model"))
        return yyjson_is_arr(yyjson_obj_get(g, "input"))
                   ? NULL
                   : "A score_model grader has input messages.";
    if (!strcmp(type, "python"))
        return "python graders are not run here.";
    return "A grader is string_check, text_similarity, label_model or "
           "score_model.";
}

int srv_evals_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *dsc = yyjson_obj_get(root, "data_source_config");
    yyjson_val *tc = yyjson_obj_get(root, "testing_criteria");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    const char *dtype = yyjson_get_str(yyjson_obj_get(dsc, "type"));
    const char *why = NULL;
    if (!yyjson_is_obj(root))
        why = "The body is a JSON object.";
    else if (!dtype || (strcmp(dtype, "custom") && strcmp(dtype, "logs") &&
                        strcmp(dtype, "stored_completions")))
        why = "'data_source_config' is custom, logs or stored_completions.";
    else if (!strcmp(dtype, "custom") &&
             !yyjson_is_obj(yyjson_obj_get(dsc, "item_schema")))
        why = "A custom data source config has an item_schema.";
    else if (!yyjson_is_arr(tc) || !yyjson_arr_size(tc))
        why = "'testing_criteria' is a list of graders.";
    else if (!metadata_ok(md))
        why = "'metadata' holds at most 16 strings.";
    size_t i, n;
    yyjson_val *g;
    yyjson_arr_foreach(tc, i, n, g) if (!why) why = grader_error(g);
    if (why) {
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    char id[48];
    srv_new_id(id, sizeof id, "eval_");
    yyjson_mut_obj_add_str(d, o, "object", "eval");
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    const char *name = yyjson_get_str(yyjson_obj_get(root, "name"));
    yyjson_mut_obj_add_strcpy(d, o, "name", name ? name : "");
    /* the config as OpenAI gives it back: its schema of item and sample */
    yyjson_mut_val *c = yyjson_mut_obj_add_obj(d, o, "data_source_config");
    yyjson_mut_obj_add_strcpy(d, c, "type", dtype);
    yyjson_mut_val *sch = yyjson_mut_obj_add_obj(d, c, "schema");
    yyjson_mut_obj_add_str(d, sch, "type", "object");
    yyjson_mut_val *props = yyjson_mut_obj_add_obj(d, sch, "properties");
    yyjson_val *is = yyjson_obj_get(dsc, "item_schema");
    yyjson_mut_obj_add_val(d, props, "item",
                           is ? yyjson_val_mut_copy(d, is) : yyjson_mut_obj(d));
    yyjson_mut_val *req = yyjson_mut_obj_add_arr(d, sch, "required");
    yyjson_mut_arr_add_str(d, req, "item");
    if (yyjson_get_bool(yyjson_obj_get(dsc, "include_sample_schema"))) {
        yyjson_mut_val *s = yyjson_mut_obj_add_obj(d, props, "sample");
        yyjson_mut_obj_add_str(d, s, "type", "object");
        yyjson_mut_arr_add_str(d, req, "sample");
    }
    yyjson_mut_obj_add_val(d, o, "testing_criteria",
                           yyjson_val_mut_copy(d, tc));
    yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
    yyjson_mut_obj_add_val(d, o, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    yyjson_doc_free(doc);
    int rc = save(EVAL, id, o, NULL, 1);
    char *out = rc == 0 ? yyjson_mut_val_write(o, 0, NULL) : NULL;
    yyjson_mut_doc_free(d);
    if (!out)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the eval");
    return reply_text(r, out);
}

int srv_evals_get(struct srv_req *r)
{
    char *t = store_get(EVAL, r->param[0], NULL);
    return t ? reply_text(r, t) : not_found(r, "eval", r->param[0]);
}

int srv_evals_update(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = load(d, EVAL, r->param[0]);
    yyjson_val *name = yyjson_obj_get(root, "name");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    int ret;
    if (!o) {
        ret = not_found(r, "eval", r->param[0]);
    } else if (!yyjson_is_obj(root) || !metadata_ok(md)) {
        ret = bad(r, "The body is {name, metadata}, metadata at most 16 "
                     "strings.");
    } else {
        if (yyjson_is_str(name)) {
            yyjson_mut_obj_remove_key(o, "name");
            yyjson_mut_obj_add_strcpy(d, o, "name", yyjson_get_str(name));
        }
        if (md) {
            yyjson_mut_obj_remove_key(o, "metadata");
            yyjson_mut_obj_add_val(d, o, "metadata",
                                   yyjson_is_obj(md)
                                       ? yyjson_val_mut_copy(d, md)
                                       : yyjson_mut_obj(d));
        }
        ret = save(EVAL, r->param[0], o, NULL, 0) == 0
                  ? reply_text(r, yyjson_mut_val_write(o, 0, NULL))
                  : srv_reply_error(r, 500, "server_error", NULL,
                                    "cannot keep the eval");
    }
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    return ret;
}

int srv_evals_delete(struct srv_req *r)
{
    if (srv_store_del(r->srv->store, EVAL, r->param[0]) != 0)
        return not_found(r, "eval", r->param[0]);
    char out[512];
    snprintf(
        out, sizeof out,
        "{\"object\":\"eval.deleted\",\"deleted\":true,\"eval_id\":\"%s\"}",
        r->param[0]);
    return reply_text(r, strdup(out));
}

/* A page of documents of kind (those of eval_id for runs), filtered by a
   key's value; asc by default as OpenAI lists evals. */
static int list_page(struct srv_req *r, const char *kind, const char *eval_id,
                     const char *status)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after");
    long limit = 20;
    if (l) {
        char *end;
        limit = strtol(l, &end, 10);
        if (*end || limit < 1 || limit > 100)
            return bad(r, "'limit' is a number from 1 to 100.");
    }
    int desc = o && !strcmp(o, "desc");
    char **ids;
    size_t n = srv_store_list(r->srv->store, kind, &ids);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    long shown = 0;
    int more = 0, past = after == NULL;
    char first[64] = "", last[64] = "";
    for (size_t k = 0; k < n; k++) {
        const char *id = ids[desc ? n - 1 - k : k];
        char *t = store_get(kind, id, NULL);
        yyjson_doc *x = t ? yyjson_read(t, strlen(t), 0) : NULL;
        free(t);
        yyjson_val *v = yyjson_doc_get_root(x);
        int keep =
            v &&
            (!eval_id ||
             yyjson_equals_str(yyjson_obj_get(v, "eval_id"), eval_id)) &&
            (!status || yyjson_equals_str(yyjson_obj_get(v, "status"), status));
        if (keep && !past) {
            past = !strcmp(id, after);
            keep = 0;
        }
        if (keep && shown == limit) {
            more = 1;
            yyjson_doc_free(x);
            break;
        }
        if (keep) {
            yyjson_mut_arr_append(data, yyjson_val_mut_copy(d, v));
            if (!shown)
                snprintf(first, sizeof first, "%s", id);
            snprintf(last, sizeof last, "%s", id);
            shown++;
        }
        yyjson_doc_free(x);
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
    char *out = yyjson_mut_val_write(root, 0, NULL);
    yyjson_mut_doc_free(d);
    return reply_text(r, out);
}

int srv_evals_list(struct srv_req *r)
{
    return list_page(r, EVAL, NULL, NULL);
}

/* ---- the API: runs ---- */

int srv_eval_runs_create(struct srv_req *r)
{
    char *et = store_get(EVAL, r->param[0], NULL);
    yyjson_doc *ed = et ? yyjson_read(et, strlen(et), 0) : NULL;
    free(et);
    if (!ed)
        return not_found(r, "eval", r->param[0]);
    yyjson_val *ev = yyjson_doc_get_root(ed);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *ds = yyjson_obj_get(root, "data_source");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    const char *dtype = yyjson_get_str(yyjson_obj_get(ds, "type"));
    const char *why = NULL;
    char tmp[200];
    if (!yyjson_is_obj(root) || !dtype)
        why = "'data_source' is jsonl, completions or responses.";
    else if (strcmp(dtype, "jsonl") && strcmp(dtype, "completions") &&
             strcmp(dtype, "responses"))
        why = "'data_source' is jsonl, completions or responses.";
    else if (strcmp(dtype, "jsonl") && !yyjson_obj_get(ds, "input_messages"))
        why = "A completions data source has input_messages (a template or "
              "an item_reference).";
    else if (!metadata_ok(md))
        why = "'metadata' holds at most 16 strings.";
    else {
        char *rows = rows_of(ds, tmp, sizeof tmp);
        if (!rows)
            why = tmp;
        free(rows);
    }
    if (why) {
        yyjson_doc_free(doc);
        yyjson_doc_free(ed);
        return bad(r, why);
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    char id[48];
    srv_new_id(id, sizeof id, "evalrun_");
    yyjson_mut_obj_add_str(d, o, "object", "eval.run");
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_strcpy(d, o, "eval_id", r->param[0]);
    yyjson_mut_obj_add_str(d, o, "status", "queued");
    yyjson_mut_obj_add_strcpy(d, o, "model", r->srv->cfg.model_id);
    const char *name = yyjson_get_str(yyjson_obj_get(root, "name"));
    yyjson_mut_obj_add_strcpy(d, o, "name", name ? name : id);
    yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
    yyjson_mut_obj_add_str(d, o, "report_url", "");
    yyjson_mut_val *rc = yyjson_mut_obj_add_obj(d, o, "result_counts");
    yyjson_mut_obj_add_int(d, rc, "total", 0);
    yyjson_mut_obj_add_int(d, rc, "errored", 0);
    yyjson_mut_obj_add_int(d, rc, "failed", 0);
    yyjson_mut_obj_add_int(d, rc, "passed", 0);
    yyjson_mut_obj_add_arr(d, o, "per_model_usage");
    yyjson_mut_val *pt =
        yyjson_mut_obj_add_arr(d, o, "per_testing_criteria_results");
    size_t i, n;
    yyjson_val *g;
    yyjson_arr_foreach(yyjson_obj_get(ev, "testing_criteria"), i, n, g)
    {
        yyjson_mut_val *s = yyjson_mut_arr_add_obj(d, pt);
        const char *gn = yyjson_get_str(yyjson_obj_get(g, "name"));
        yyjson_mut_obj_add_strcpy(d, s, "testing_criteria", gn ? gn : "");
        yyjson_mut_obj_add_int(d, s, "passed", 0);
        yyjson_mut_obj_add_int(d, s, "failed", 0);
    }
    yyjson_mut_obj_add_val(d, o, "data_source", yyjson_val_mut_copy(d, ds));
    yyjson_mut_obj_add_val(d, o, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    yyjson_mut_obj_add_null(d, o, "error");
    yyjson_doc_free(doc);
    yyjson_doc_free(ed);
    int ok = save(RUN, id, o, "0", 1) == 0;
    char *out = ok ? yyjson_mut_val_write(o, 0, NULL) : NULL;
    yyjson_mut_doc_free(d);
    if (!out)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot keep the run");
    pthread_mutex_lock(&mu);
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
    return reply_text(r, out);
}

static yyjson_doc *run_of(struct srv_req *r)
{
    char *t = store_get(RUN, r->param[1], NULL);
    yyjson_doc *x = t ? yyjson_read(t, strlen(t), 0) : NULL;
    free(t);
    if (x &&
        !yyjson_equals_str(yyjson_obj_get(yyjson_doc_get_root(x), "eval_id"),
                           r->param[0])) {
        yyjson_doc_free(x);
        x = NULL;
    }
    return x;
}

int srv_eval_runs_get(struct srv_req *r)
{
    yyjson_doc *x = run_of(r);
    if (!x)
        return not_found(r, "run", r->param[1]);
    char *out = yyjson_val_write(yyjson_doc_get_root(x), 0, NULL);
    yyjson_doc_free(x);
    return reply_text(r, out);
}

int srv_eval_runs_list(struct srv_req *r)
{
    if (srv_store_get(r->srv->store, EVAL, r->param[0], NULL, NULL) != 0)
        return not_found(r, "eval", r->param[0]);
    return list_page(r, RUN, r->param[0], srv_query(r, "status"));
}

int srv_eval_runs_cancel(struct srv_req *r)
{
    yyjson_doc *x = run_of(r);
    if (!x)
        return not_found(r, "run", r->param[1]);
    yyjson_mut_doc *d = yyjson_doc_mut_copy(x, NULL);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    const char *st = yyjson_mut_get_str(yyjson_mut_obj_get(o, "status"));
    if (st && (!strcmp(st, "queued") || !strcmp(st, "in_progress"))) {
        yyjson_mut_obj_remove_key(o, "status");
        yyjson_mut_obj_add_str(d, o, "status", "canceled");
        pthread_mutex_lock(&mu);
        save(RUN, r->param[1], o, NULL, 0);
        pthread_mutex_unlock(&mu);
    }
    char *out = yyjson_mut_val_write(o, 0, NULL);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(x);
    return reply_text(r, out);
}

int srv_eval_runs_delete(struct srv_req *r)
{
    yyjson_doc *x = run_of(r);
    if (!x)
        return not_found(r, "run", r->param[1]);
    yyjson_doc_free(x);
    pthread_mutex_lock(&mu);
    srv_store_del(r->srv->store, RUN, r->param[1]);
    pthread_mutex_unlock(&mu);
    char p[4600];
    if (srv_disk_name_ok(r->param[1])) {
        out_path(p, sizeof p, r->param[1]);
        unlink(p);
    }
    char out[512];
    snprintf(out, sizeof out,
             "{\"object\":\"eval.run.deleted\",\"deleted\":true,"
             "\"run_id\":\"%s\"}",
             r->param[1]);
    return reply_text(r, strdup(out));
}

/* The output items of a run, from its file: those with id (NULL: all,
   paged), filtered by status. */
static int items(struct srv_req *r, const char *id)
{
    yyjson_doc *x = run_of(r);
    if (!x)
        return not_found(r, "run", r->param[1]);
    yyjson_doc_free(x);
    char p[4600];
    if (!srv_disk_name_ok(r->param[1]))
        return not_found(r, "run", r->param[1]);
    out_path(p, sizeof p, r->param[1]);
    FILE *f = fopen(p, "rb");
    const char *status = srv_query(r, "status"), *after = srv_query(r, "after");
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    long limit = l ? strtol(l, NULL, 10) : 20;
    if (limit < 1 || limit > 100)
        limit = 20;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *data = yyjson_mut_arr(d);
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    yyjson_mut_val *found = NULL;
    while (f && (n = getline(&line, &cap, f)) > 0) {
        yyjson_doc *ld = yyjson_read(line, (size_t)n, 0);
        yyjson_val *v = yyjson_doc_get_root(ld);
        if (v && (!id || yyjson_equals_str(yyjson_obj_get(v, "id"), id)) &&
            (!status || yyjson_equals_str(yyjson_obj_get(v, "status"), status)))
            yyjson_mut_arr_append(data, yyjson_val_mut_copy(d, v));
        if (v && id && yyjson_equals_str(yyjson_obj_get(v, "id"), id))
            found = yyjson_mut_arr_get_last(data);
        yyjson_doc_free(ld);
    }
    free(line);
    if (f)
        fclose(f);
    if (id) {
        int ret = found ? reply_text(r, yyjson_mut_val_write(found, 0, NULL))
                        : not_found(r, "output item", id);
        yyjson_mut_doc_free(d);
        return ret;
    }
    /* a page of them */
    size_t total = yyjson_mut_arr_size(data);
    int desc = o && !strcmp(o, "desc");
    yyjson_mut_val *root = yyjson_mut_obj(d), *page = yyjson_mut_arr(d);
    long shown = 0;
    int more = 0, past = after == NULL;
    char first[64] = "", last[64] = "";
    for (size_t k = 0; k < total; k++) {
        yyjson_mut_val *v = yyjson_mut_arr_get(data, desc ? total - 1 - k : k);
        const char *vid = yyjson_mut_get_str(yyjson_mut_obj_get(v, "id"));
        if (!past) {
            past = vid && !strcmp(vid, after);
            continue;
        }
        if (shown == limit) {
            more = 1;
            break;
        }
        yyjson_mut_arr_append(page, yyjson_mut_val_mut_copy(d, v));
        if (!shown)
            snprintf(first, sizeof first, "%s", vid ? vid : "");
        snprintf(last, sizeof last, "%s", vid ? vid : "");
        shown++;
    }
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_obj_add_val(d, root, "data", page);
    if (shown) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    char *out = yyjson_mut_val_write(root, 0, NULL);
    yyjson_mut_doc_free(d);
    return reply_text(r, out);
}

int srv_eval_items_list(struct srv_req *r)
{
    return items(r, NULL);
}

int srv_eval_items_get(struct srv_req *r)
{
    return items(r, r->param[2]);
}
