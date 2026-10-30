/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * runs.c - the runs of the Assistants API and their steps. A run goes on a
 * thread of its own: the thread's messages, the assistant's instructions
 * and tools (the run's where it gives them) made into a request to
 * /v1/responses, answered inside the server; its output becomes the run's
 * steps - a file_search call, the message written, added to the thread
 * with its citations as file_citation annotations - or, when the model
 * calls functions, the run waits (requires_action) for their outputs
 * (submit_tool_outputs) and goes on with them. The run is kept after each
 * step, its steps and the items it added to the input (calls and their
 * outputs) as its aux text: a server stopped mid-run takes it up again at
 * the next start.
 */
#include "assistants.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "containers.h"
#include "files_store.h"
#include "llm/json.h"

#define RUN_EXPIRES 600 /* seconds a run may wait for tool outputs */

static pthread_mutex_t live_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t live_cv = PTHREAD_COND_INITIALIZER;
static int live, stopping;
static struct srv *server;

/* the passes running, each with the flag its model's job watches */
struct pass {
    char id[64];
    _Atomic int cancel;
    struct pass *next;
    /* the pass reading /responses streamed: the run's thread and
       assistant, the events not yet whole, the response at the end */
    char thread[64], asst[64];
    struct janas_buf sse;
    yyjson_doc *final;
    char err[256];
    /* the message being written, made at its first piece of text */
    char msg_id[48], step_id[48];
    int64_t msg_at;
    int msg_open;
    /* the search steps told already, by the id of their item */
    char fs_item[8][48], fs_step[8][48];
    int n_fs;
    /* the thread's container, and its files made Files (the images and
       the files the answer names), each once */
    char cid[48];
    char conv_from[16][48], conv_to[16][48];
    int n_conv;
};
static struct pass *passes;

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

static int reply_val(struct srv_req *r, yyjson_mut_val *o)
{
    size_t n = 0;
    char *t = yyjson_mut_val_write(o, 0, &n);
    if (!t)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    return srv_reply_json(r, 200, t, n);
}

static void set_str(yyjson_mut_doc *d, yyjson_mut_val *o, const char *k,
                    const char *v)
{
    yyjson_mut_obj_remove_key(o, k);
    if (v)
        yyjson_mut_obj_add_strcpy(d, o, k, v);
    else
        yyjson_mut_obj_add_null(d, o, k);
}

static void set_int(yyjson_mut_doc *d, yyjson_mut_val *o, const char *k,
                    int64_t v)
{
    yyjson_mut_obj_remove_key(o, k);
    yyjson_mut_obj_add_int(d, o, k, v);
}

static void set_val(yyjson_mut_doc *d, yyjson_mut_val *o, const char *k,
                    yyjson_mut_val *v)
{
    yyjson_mut_obj_remove_key(o, k);
    yyjson_mut_obj_add_val(d, o, k, v ? v : yyjson_mut_null(d));
}

/* ---- a run as kept: its object and its aux {steps, items} ---- */

struct run {
    yyjson_mut_doc *d;
    yyjson_mut_val *o;     /* the run */
    yyjson_mut_val *steps; /* its steps */
    yyjson_mut_val *items; /* what it added to the input */
};

static void run_free(struct run *x)
{
    yyjson_mut_doc_free(x->d);
    memset(x, 0, sizeof(*x));
}

/* (as_lock held) 0, or -1 when there is no such run */
static int run_load(struct srv *s, const char *id, struct run *x)
{
    memset(x, 0, sizeof(*x));
    char *aux = NULL;
    char *t = as_get(s, AS_RUN, id, &aux);
    yyjson_doc *rd = t ? yyjson_read(t, strlen(t), 0) : NULL;
    yyjson_doc *ad = aux ? yyjson_read(aux, strlen(aux), 0) : NULL;
    free(t);
    free(aux);
    if (!rd) {
        yyjson_doc_free(ad);
        return -1;
    }
    x->d = yyjson_mut_doc_new(NULL);
    x->o = yyjson_val_mut_copy(x->d, yyjson_doc_get_root(rd));
    yyjson_val *steps = yyjson_obj_get(yyjson_doc_get_root(ad), "steps");
    yyjson_val *items = yyjson_obj_get(yyjson_doc_get_root(ad), "items");
    x->steps = steps ? yyjson_val_mut_copy(x->d, steps) : yyjson_mut_arr(x->d);
    x->items = items ? yyjson_val_mut_copy(x->d, items) : yyjson_mut_arr(x->d);
    yyjson_doc_free(rd);
    yyjson_doc_free(ad);
    return 0;
}

/* (as_lock held) */
static int run_save(struct srv *s, struct run *x, int fresh)
{
    yyjson_mut_doc *ad = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *a = yyjson_mut_obj(ad);
    yyjson_mut_obj_add_val(ad, a, "steps",
                           yyjson_mut_val_mut_copy(ad, x->steps));
    yyjson_mut_obj_add_val(ad, a, "items",
                           yyjson_mut_val_mut_copy(ad, x->items));
    char *aux = yyjson_mut_val_write(a, 0, NULL);
    yyjson_mut_doc_free(ad);
    const char *id = yyjson_mut_get_str(yyjson_mut_obj_get(x->o, "id"));
    int rc = aux ? as_save(s, AS_RUN, id, x->o, aux, fresh) : -1;
    free(aux);
    return rc;
}

static const char *run_status(const struct run *x)
{
    return yyjson_mut_get_str(yyjson_mut_obj_get(x->o, "status"));
}

static const char *run_str(const struct run *x, const char *k)
{
    return yyjson_mut_get_str(yyjson_mut_obj_get(x->o, k));
}

/* A run waiting for tool outputs past its time: expired. */
static void run_expire(struct run *x)
{
    const char *st = run_status(x);
    int64_t exp = yyjson_mut_get_sint(yyjson_mut_obj_get(x->o, "expires_at"));
    if (st && !strcmp(st, "requires_action") && exp &&
        (int64_t)time(NULL) > exp) {
        set_str(x->d, x->o, "status", "expired");
        set_val(x->d, x->o, "required_action", NULL);
    }
}

static void add_usage(yyjson_mut_doc *d, yyjson_mut_val *o, int64_t in,
                      int64_t out)
{
    yyjson_mut_val *u = yyjson_mut_obj_get(o, "usage");
    if (u && yyjson_mut_is_obj(u)) {
        in += yyjson_mut_get_sint(yyjson_mut_obj_get(u, "prompt_tokens"));
        out += yyjson_mut_get_sint(yyjson_mut_obj_get(u, "completion_tokens"));
    }
    yyjson_mut_val *nu = yyjson_mut_obj(d);
    yyjson_mut_obj_add_int(d, nu, "prompt_tokens", in);
    yyjson_mut_obj_add_int(d, nu, "completion_tokens", out);
    yyjson_mut_obj_add_int(d, nu, "total_tokens", in + out);
    set_val(d, o, "usage", nu);
}

/* A step's object (id NULL: a new one; at 0: now). */
static yyjson_mut_val *step_obj(yyjson_mut_doc *d, const char *id, int64_t at,
                                const char *run, const char *asst,
                                const char *thread, const char *type,
                                const char *status)
{
    yyjson_mut_val *s = yyjson_mut_obj(d);
    char nid[48];
    if (!id) {
        srv_new_id(nid, sizeof nid, "step_");
        id = nid;
    }
    int64_t now = (int64_t)time(NULL);
    yyjson_mut_obj_add_strcpy(d, s, "id", id);
    yyjson_mut_obj_add_str(d, s, "object", "thread.run.step");
    yyjson_mut_obj_add_int(d, s, "created_at", at ? at : now);
    yyjson_mut_obj_add_strcpy(d, s, "run_id", run);
    yyjson_mut_obj_add_strcpy(d, s, "assistant_id", asst);
    yyjson_mut_obj_add_strcpy(d, s, "thread_id", thread);
    yyjson_mut_obj_add_strcpy(d, s, "type", type);
    yyjson_mut_obj_add_strcpy(d, s, "status", status);
    yyjson_mut_obj_add_null(d, s, "cancelled_at");
    if (!strcmp(status, "completed"))
        yyjson_mut_obj_add_int(d, s, "completed_at", now);
    else
        yyjson_mut_obj_add_null(d, s, "completed_at");
    yyjson_mut_obj_add_null(d, s, "expired_at");
    yyjson_mut_obj_add_null(d, s, "failed_at");
    yyjson_mut_obj_add_null(d, s, "last_error");
    yyjson_mut_obj_add_obj(d, s, "metadata");
    yyjson_mut_obj_add_null(d, s, "usage");
    return s;
}

/* A new step of the run (id NULL: a new one). */
static yyjson_mut_val *step_new(struct run *x, const char *id, int64_t at,
                                const char *type, const char *status)
{
    yyjson_mut_val *s =
        step_obj(x->d, id, at, run_str(x, "id"), run_str(x, "assistant_id"),
                 run_str(x, "thread_id"), type, status);
    yyjson_mut_arr_append(x->steps, s);
    return s;
}

/* A file_search_call item of /responses as the details of a step. */
static yyjson_mut_val *search_details(yyjson_mut_doc *d, yyjson_val *it)
{
    yyjson_mut_val *det = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, det, "type", "tool_calls");
    yyjson_mut_val *tc = yyjson_mut_obj_add_arr(d, det, "tool_calls");
    yyjson_mut_val *c1 = yyjson_mut_arr_add_obj(d, tc);
    const char *cid = yyjson_get_str(yyjson_obj_get(it, "id"));
    yyjson_mut_obj_add_strcpy(d, c1, "id", cid ? cid : "");
    yyjson_mut_obj_add_str(d, c1, "type", "file_search");
    yyjson_mut_val *fs = yyjson_mut_obj_add_obj(d, c1, "file_search");
    yyjson_mut_val *ro = yyjson_mut_obj_add_obj(d, fs, "ranking_options");
    yyjson_mut_obj_add_str(d, ro, "ranker", "default_2024_08_21");
    yyjson_mut_obj_add_real(d, ro, "score_threshold", 0);
    yyjson_mut_val *res = yyjson_mut_obj_add_arr(d, fs, "results");
    size_t k, m;
    yyjson_val *h;
    yyjson_arr_foreach(yyjson_obj_get(it, "results"), k, m, h)
    {
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, res);
        const char *fid = yyjson_get_str(yyjson_obj_get(h, "file_id"));
        const char *fn = yyjson_get_str(yyjson_obj_get(h, "filename"));
        const char *tx = yyjson_get_str(yyjson_obj_get(h, "text"));
        yyjson_mut_obj_add_strcpy(d, o, "file_id", fid ? fid : "");
        yyjson_mut_obj_add_strcpy(d, o, "file_name", fn ? fn : "");
        yyjson_mut_obj_add_real(d, o, "score",
                                yyjson_get_num(yyjson_obj_get(h, "score")));
        yyjson_mut_val *ct = yyjson_mut_obj_add_arr(d, o, "content");
        yyjson_mut_val *c2 = yyjson_mut_arr_add_obj(d, ct);
        yyjson_mut_obj_add_str(d, c2, "type", "text");
        yyjson_mut_obj_add_strcpy(d, c2, "text", tx ? tx : "");
    }
    return det;
}

/* A message_creation step's details. */
static yyjson_mut_val *message_details(yyjson_mut_doc *d, const char *msg)
{
    yyjson_mut_val *det = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, det, "type", "message_creation");
    yyjson_mut_val *mc = yyjson_mut_obj_add_obj(d, det, "message_creation");
    yyjson_mut_obj_add_strcpy(d, mc, "message_id", msg);
    return det;
}

/* ---- the request to /v1/responses ---- */

/* The run's tools as the Responses API takes them; file_search with the
   vector stores of the assistant and of the thread. */
static yyjson_mut_val *tools_for(yyjson_mut_doc *d, yyjson_val *tools,
                                 yyjson_val *asst, yyjson_val *thread,
                                 const char *cid)
{
    yyjson_mut_val *out = yyjson_mut_arr(d);
    size_t i, n;
    yyjson_val *t;
    yyjson_arr_foreach(tools, i, n, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (type && !strcmp(type, "function")) {
            yyjson_val *f = yyjson_obj_get(t, "function");
            yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, out);
            yyjson_mut_obj_add_str(d, o, "type", "function");
            static const char *const K[] = {"name", "description", "parameters",
                                            "strict"};
            for (int k = 0; k < 4; k++) {
                yyjson_val *v = yyjson_obj_get(f, K[k]);
                if (v && !yyjson_is_null(v))
                    yyjson_mut_obj_add_val(d, o, K[k],
                                           yyjson_val_mut_copy(d, v));
            }
        } else if (type && !strcmp(type, "code_interpreter")) {
            if (!cid || !*cid)
                continue;
            yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, out);
            yyjson_mut_obj_add_str(d, o, "type", "code_interpreter");
            yyjson_mut_obj_add_strcpy(d, o, "container", cid);
        } else if (type && !strcmp(type, "file_search")) {
            yyjson_mut_val *ids = yyjson_mut_arr(d);
            yyjson_val *src[2] = {asst, thread};
            for (int k = 0; k < 2; k++) {
                size_t j, m;
                yyjson_val *v;
                yyjson_arr_foreach(
                    yyjson_obj_get(
                        yyjson_obj_get(yyjson_obj_get(src[k], "tool_resources"),
                                       "file_search"),
                        "vector_store_ids"),
                    j, m, v)
                    yyjson_mut_arr_append(ids, yyjson_val_mut_copy(d, v));
            }
            if (!yyjson_mut_arr_size(ids))
                continue; /* nothing to search */
            yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, out);
            yyjson_mut_obj_add_str(d, o, "type", "file_search");
            yyjson_mut_obj_add_val(d, o, "vector_store_ids", ids);
            yyjson_val *fs = yyjson_obj_get(t, "file_search");
            yyjson_val *mx = yyjson_obj_get(fs, "max_num_results");
            if (yyjson_is_int(mx))
                yyjson_mut_obj_add_val(d, o, "max_num_results",
                                       yyjson_val_mut_copy(d, mx));
            yyjson_val *ro = yyjson_obj_get(fs, "ranking_options");
            if (yyjson_is_obj(ro))
                yyjson_mut_obj_add_val(d, o, "ranking_options",
                                       yyjson_val_mut_copy(d, ro));
        }
    }
    return out;
}

#define AS_THREAD_CONTAINER "thread.container"

/* The thread's container for code_interpreter (as_lock held): kept from
   run to run (its variables too), made again once it expired; the files of
   the assistant's and the thread's code_interpreter resources copied in
   once. Its id into cid, or "" when it cannot be had. */
static void thread_container(struct srv *s, const char *thread,
                             yyjson_val *asst, yyjson_val *th, char *cid,
                             size_t cap)
{
    cid[0] = 0;
    char *t = as_get(s, AS_THREAD_CONTAINER, thread, NULL);
    yyjson_doc *md = t ? yyjson_read(t, strlen(t), 0) : NULL;
    free(t);
    yyjson_mut_doc *d =
        md ? yyjson_doc_mut_copy(md, NULL) : yyjson_mut_doc_new(NULL);
    yyjson_doc_free(md);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    if (!o) {
        o = yyjson_mut_obj(d);
        yyjson_mut_doc_set_root(d, o);
    }
    const char *have = yyjson_mut_get_str(yyjson_mut_obj_get(o, "container"));
    const char *why = NULL;
    if (have) {
        yyjson_mut_val *hv = yyjson_mut_strcpy(d, have);
        yyjson_doc *hd = yyjson_mut_val_imut_copy(hv, NULL);
        if (srv_container_for(s, yyjson_doc_get_root(hd), cid, cap, &why) != 0)
            have = NULL; /* expired: a new one, the files again */
        yyjson_doc_free(hd);
    }
    if (!have) {
        static const char AUTO[] = "{\"type\":\"auto\"}";
        yyjson_doc *spec = yyjson_read(AUTO, sizeof AUTO - 1, 0);
        if (srv_container_for(s, yyjson_doc_get_root(spec), cid, cap, &why) !=
            0)
            cid[0] = 0;
        yyjson_doc_free(spec);
        yyjson_mut_obj_remove_key(o, "container");
        yyjson_mut_obj_add_strcpy(d, o, "container", cid);
        yyjson_mut_obj_remove_key(o, "files");
        yyjson_mut_obj_add_arr(d, o, "files");
    }
    if (cid[0]) {
        yyjson_mut_val *done = yyjson_mut_obj_get(o, "files");
        yyjson_val *src[2] = {asst, th};
        for (int k = 0; k < 2; k++) {
            size_t i, n;
            yyjson_val *v;
            yyjson_arr_foreach(
                yyjson_obj_get(
                    yyjson_obj_get(yyjson_obj_get(src[k], "tool_resources"),
                                   "code_interpreter"),
                    "file_ids"),
                i, n, v)
            {
                int copied = 0;
                size_t q, qn;
                yyjson_mut_val *c;
                yyjson_mut_arr_foreach(done, q, qn, c)
                {
                    copied |= yyjson_mut_equals_str(c, yyjson_get_str(v));
                }
                if (!copied && yyjson_is_str(v) &&
                    srv_container_add_file(s, cid, yyjson_get_str(v), &why) ==
                        0)
                    yyjson_mut_arr_add_strcpy(d, done, yyjson_get_str(v));
            }
        }
    }
    char *out = yyjson_mut_write(d, 0, NULL);
    if (out)
        srv_store_put(s->store, AS_THREAD_CONTAINER, thread,
                      (int64_t)time(NULL), out, NULL);
    free(out);
    yyjson_mut_doc_free(d);
}

/* Whether the run's tools have one of type t. */
static int has_tool(yyjson_val *tools, const char *t)
{
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(tools, i, n, v)
    {
        if (yyjson_equals_str(yyjson_obj_get(v, "type"), t))
            return 1;
    }
    return 0;
}

/* The body of the request for the run as it stands (as_lock held); the
   thread's container, when the run has code_interpreter, into cid. */
static char *request_for(struct srv *s, struct run *x, char *cid, size_t cap)
{
    char *at = as_get(s, AS_ASSISTANT, run_str(x, "assistant_id"), NULL);
    char *maux = NULL;
    char *tt = as_get(s, AS_THREAD, run_str(x, "thread_id"), &maux);
    yyjson_doc *ad = at ? yyjson_read(at, strlen(at), 0) : NULL;
    yyjson_doc *td = tt ? yyjson_read(tt, strlen(tt), 0) : NULL;
    yyjson_doc *md = maux ? yyjson_read(maux, strlen(maux), 0) : NULL;
    free(at);
    free(tt);
    free(maux);
    char *rt = yyjson_mut_val_write(x->o, 0, NULL);
    yyjson_doc *rd = rt ? yyjson_read(rt, strlen(rt), 0) : NULL;
    free(rt);
    yyjson_val *run = yyjson_doc_get_root(rd);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "model", s->cfg.model_id);
    yyjson_mut_obj_add_bool(d, o, "store", false);
    yyjson_mut_obj_add_bool(d, o, "stream", true); /* the text told as it
                                                      comes (on_data) */
    const char *ins = yyjson_get_str(yyjson_obj_get(run, "instructions"));
    if (ins && *ins)
        yyjson_mut_obj_add_strcpy(d, o, "instructions", ins);
    /* the messages, the last ones only when the run says so */
    yyjson_mut_val *in = yyjson_mut_obj_add_arr(d, o, "input");
    yyjson_val *msgs = yyjson_doc_get_root(md);
    size_t total = yyjson_arr_size(msgs), from = 0;
    yyjson_val *ts = yyjson_obj_get(run, "truncation_strategy");
    if (yyjson_equals_str(yyjson_obj_get(ts, "type"), "last_messages")) {
        yyjson_val *lm = yyjson_obj_get(ts, "last_messages");
        if (yyjson_is_int(lm) && (size_t)yyjson_get_sint(lm) < total)
            from = total - (size_t)yyjson_get_sint(lm);
    }
    for (size_t i = from; i < total; i++) {
        yyjson_val *m = yyjson_arr_get(msgs, i);
        yyjson_mut_val *mi = yyjson_mut_arr_add_obj(d, in);
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        yyjson_mut_obj_add_strcpy(d, mi, "role", role ? role : "user");
        struct janas_buf b = {0};
        size_t k, n;
        yyjson_val *p;
        yyjson_arr_foreach(yyjson_obj_get(m, "content"), k, n, p)
        {
            const char *v = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(p, "text"), "value"));
            if (v) {
                if (b.n)
                    janas_buf_put(&b, "\n", 1);
                janas_buf_puts(&b, v);
            }
        }
        janas_buf_put(&b, "", 1);
        yyjson_mut_obj_add_strcpy(d, mi, "content", b.oom ? "" : b.p);
        janas_buf_free(&b);
    }
    size_t k, n;
    yyjson_mut_val *it;
    yyjson_mut_arr_foreach(x->items, k, n, it)
        yyjson_mut_arr_append(in, yyjson_mut_val_mut_copy(d, it));
    yyjson_val *tools = yyjson_obj_get(run, "tools");
    cid[0] = 0;
    if (has_tool(tools, "code_interpreter"))
        thread_container(s, run_str(x, "thread_id"), yyjson_doc_get_root(ad),
                         yyjson_doc_get_root(td), cid, cap);
    yyjson_mut_val *rtools = tools_for(d, tools, yyjson_doc_get_root(ad),
                                       yyjson_doc_get_root(td), cid);
    if (yyjson_mut_arr_size(rtools)) {
        yyjson_mut_obj_add_val(d, o, "tools", rtools);
        yyjson_val *tc = yyjson_obj_get(run, "tool_choice");
        if (yyjson_is_str(tc))
            yyjson_mut_obj_add_strcpy(d, o, "tool_choice", yyjson_get_str(tc));
        else if (yyjson_is_obj(tc)) {
            yyjson_mut_val *c = yyjson_mut_obj_add_obj(d, o, "tool_choice");
            const char *ty = yyjson_get_str(yyjson_obj_get(tc, "type"));
            yyjson_mut_obj_add_strcpy(d, c, "type", ty ? ty : "auto");
            const char *fn = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(tc, "function"), "name"));
            if (fn)
                yyjson_mut_obj_add_strcpy(d, c, "name", fn);
        }
        yyjson_val *ptc = yyjson_obj_get(run, "parallel_tool_calls");
        if (yyjson_is_bool(ptc))
            yyjson_mut_obj_add_bool(d, o, "parallel_tool_calls",
                                    yyjson_get_bool(ptc));
    }
    static const char *const NUM[] = {"temperature", "top_p"};
    for (int i = 0; i < 2; i++) {
        yyjson_val *v = yyjson_obj_get(run, NUM[i]);
        if (yyjson_is_num(v))
            yyjson_mut_obj_add_real(d, o, NUM[i], yyjson_get_num(v));
    }
    yyjson_val *mx = yyjson_obj_get(run, "max_completion_tokens");
    if (yyjson_is_int(mx))
        yyjson_mut_obj_add_int(d, o, "max_output_tokens", yyjson_get_sint(mx));
    yyjson_val *re = yyjson_obj_get(run, "reasoning_effort");
    if (yyjson_is_str(re)) {
        yyjson_mut_val *r = yyjson_mut_obj_add_obj(d, o, "reasoning");
        yyjson_mut_obj_add_strcpy(d, r, "effort", yyjson_get_str(re));
    }
    /* response_format into text.format */
    yyjson_val *rf = yyjson_obj_get(run, "response_format");
    const char *rft = yyjson_get_str(yyjson_obj_get(rf, "type"));
    if (rft && (!strcmp(rft, "json_object") || !strcmp(rft, "json_schema"))) {
        yyjson_mut_val *tx = yyjson_mut_obj_add_obj(d, o, "text");
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(d, tx, "format");
        yyjson_mut_obj_add_strcpy(d, f, "type", rft);
        yyjson_val *js = yyjson_obj_get(rf, "json_schema");
        static const char *const K[] = {"name", "description", "schema",
                                        "strict"};
        for (int i = 0; i < 4; i++) {
            yyjson_val *v = yyjson_obj_get(js, K[i]);
            if (v && !yyjson_is_null(v))
                yyjson_mut_obj_add_val(d, f, K[i], yyjson_val_mut_copy(d, v));
        }
    }
    char *body = yyjson_mut_write(d, 0, NULL);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(ad);
    yyjson_doc_free(td);
    yyjson_doc_free(md);
    yyjson_doc_free(rd);
    return body;
}

/* A container's file made a File of the Files API (purpose
   assistants_output), once per pass: its id into out, or "". */
static void file_of(struct srv *s, struct pass *me, const char *cfid, char *out,
                    size_t cap)
{
    out[0] = 0;
    for (int i = 0; i < me->n_conv; i++)
        if (!strcmp(me->conv_from[i], cfid)) {
            snprintf(out, cap, "%s", me->conv_to[i]);
            return;
        }
    char name[256];
    int fd = srv_container_file_open(me->cid, cfid, name, sizeof name);
    if (fd < 0)
        return;
    struct janas_buf b = {0};
    char buf[65536];
    ssize_t k;
    while ((k = read(fd, buf, sizeof buf)) > 0 && b.n < ((size_t)64 << 20))
        janas_buf_put(&b, buf, (size_t)k);
    close(fd);
    struct srv_file f;
    char err[200];
    if (!b.oom &&
        srv_files_add(s->files, name, "assistants_output", b.p ? b.p : "", b.n,
                      0, &f, err, sizeof err) == 0) {
        snprintf(out, cap, "%s", f.id);
        if (me->n_conv < 16) {
            snprintf(me->conv_from[me->n_conv], sizeof me->conv_from[0], "%s",
                     cfid);
            snprintf(me->conv_to[me->n_conv], sizeof me->conv_to[0], "%s",
                     f.id);
            me->n_conv++;
        }
    }
    janas_buf_free(&b);
}

/* A code_interpreter_call item of /responses as the details of a step:
   the code, its logs, its images as Files. */
static yyjson_mut_val *code_details(struct srv *s, struct pass *me,
                                    yyjson_mut_doc *d, yyjson_val *it)
{
    yyjson_mut_val *det = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, det, "type", "tool_calls");
    yyjson_mut_val *tc = yyjson_mut_obj_add_arr(d, det, "tool_calls");
    yyjson_mut_val *c1 = yyjson_mut_arr_add_obj(d, tc);
    const char *cid = yyjson_get_str(yyjson_obj_get(it, "id"));
    const char *code = yyjson_get_str(yyjson_obj_get(it, "code"));
    yyjson_mut_obj_add_strcpy(d, c1, "id", cid ? cid : "");
    yyjson_mut_obj_add_str(d, c1, "type", "code_interpreter");
    yyjson_mut_val *ci = yyjson_mut_obj_add_obj(d, c1, "code_interpreter");
    yyjson_mut_obj_add_strcpy(d, ci, "input", code ? code : "");
    yyjson_mut_val *outs = yyjson_mut_obj_add_arr(d, ci, "outputs");
    size_t i, n;
    yyjson_val *o;
    yyjson_arr_foreach(yyjson_obj_get(it, "outputs"), i, n, o)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(o, "type"));
        if (type && !strcmp(type, "logs")) {
            yyjson_mut_val *l = yyjson_mut_arr_add_obj(d, outs);
            const char *logs = yyjson_get_str(yyjson_obj_get(o, "logs"));
            yyjson_mut_obj_add_str(d, l, "type", "logs");
            yyjson_mut_obj_add_strcpy(d, l, "logs", logs ? logs : "");
        } else if (type && !strcmp(type, "image")) {
            /* /v1/containers/<c>/files/<f>/content: the f */
            const char *url = yyjson_get_str(yyjson_obj_get(o, "url"));
            const char *f = url ? strstr(url, "/files/") : NULL;
            char cf[48] = "", fid[48];
            if (f)
                sscanf(f + 7, "%47[^/]", cf);
            file_of(s, me, cf, fid, sizeof fid);
            if (!fid[0])
                continue;
            yyjson_mut_val *im = yyjson_mut_arr_add_obj(d, outs);
            yyjson_mut_obj_add_str(d, im, "type", "image");
            yyjson_mut_val *io = yyjson_mut_obj_add_obj(d, im, "image");
            yyjson_mut_obj_add_strcpy(d, io, "file_id", fid);
        }
    }
    return det;
}

/* Responses' annotations of text as the Assistants API writes them:
   file_citation (index: the character of its '['): the marker [n], its
   start and end in characters, the file; container_file_citation: a
   file_path to the container's file made a File. */
static yyjson_mut_val *annotations(struct srv *s, struct pass *me,
                                   yyjson_mut_doc *d, yyjson_val *anns,
                                   const char *text)
{
    yyjson_mut_val *out = yyjson_mut_arr(d);
    size_t i, n;
    yyjson_val *a;
    yyjson_arr_foreach(anns, i, n, a)
    {
        if (yyjson_equals_str(yyjson_obj_get(a, "type"),
                              "container_file_citation")) {
            char fid[48];
            const char *cf = yyjson_get_str(yyjson_obj_get(a, "file_id"));
            file_of(s, me, cf ? cf : "", fid, sizeof fid);
            const char *fn = yyjson_get_str(yyjson_obj_get(a, "filename"));
            if (!fid[0])
                continue;
            yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, out);
            yyjson_mut_obj_add_str(d, o, "type", "file_path");
            yyjson_mut_obj_add_strcpy(d, o, "text", fn ? fn : "");
            yyjson_mut_obj_add_uint(
                d, o, "start_index",
                yyjson_get_uint(yyjson_obj_get(a, "start_index")));
            yyjson_mut_obj_add_uint(
                d, o, "end_index",
                yyjson_get_uint(yyjson_obj_get(a, "end_index")));
            yyjson_mut_val *fp = yyjson_mut_obj_add_obj(d, o, "file_path");
            yyjson_mut_obj_add_strcpy(d, fp, "file_id", fid);
            continue;
        }
        if (!yyjson_equals_str(yyjson_obj_get(a, "type"), "file_citation"))
            continue;
        size_t at = (size_t)yyjson_get_uint(yyjson_obj_get(a, "index"));
        /* the byte of that character */
        const char *p = text;
        for (size_t c = 0; *p && c < at; p++)
            if (((unsigned char)p[1] & 0xc0) != 0x80)
                c++;
        size_t len = 0;
        if (*p == '[')
            while (p[len] && p[len] != ']' && len < 12)
                len++;
        if (p[len] != ']')
            continue;
        len++;
        char marker[16];
        snprintf(marker, sizeof marker, "%.*s", (int)len, p);
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, out);
        yyjson_mut_obj_add_str(d, o, "type", "file_citation");
        yyjson_mut_obj_add_strcpy(d, o, "text", marker);
        yyjson_mut_obj_add_uint(d, o, "start_index", at);
        yyjson_mut_obj_add_uint(d, o, "end_index", at + len);
        yyjson_mut_val *fc = yyjson_mut_obj_add_obj(d, o, "file_citation");
        yyjson_mut_obj_add_strcpy(d, fc, "file_id",
                                  yyjson_get_str(yyjson_obj_get(a, "file_id")));
    }
    return out;
}

static void fail(struct run *x, const char *code, const char *msg)
{
    set_str(x->d, x->o, "status", "failed");
    set_int(x->d, x->o, "failed_at", (int64_t)time(NULL));
    yyjson_mut_val *e = yyjson_mut_obj(x->d);
    yyjson_mut_obj_add_strcpy(x->d, e, "code", code);
    yyjson_mut_obj_add_strcpy(x->d, e, "message", msg);
    set_val(x->d, x->o, "last_error", e);
}

static void cancelled(struct run *x)
{
    set_str(x->d, x->o, "status", "cancelled");
    set_int(x->d, x->o, "cancelled_at", (int64_t)time(NULL));
    set_val(x->d, x->o, "required_action", NULL);
}

/* The message the pass writes, told at its first piece of text: its step
   and the message itself, in progress. */
static void message_open(struct pass *me)
{
    me->msg_open = 1;
    me->msg_at = (int64_t)time(NULL);
    srv_new_id(me->msg_id, sizeof me->msg_id, "msg_");
    srv_new_id(me->step_id, sizeof me->step_id, "step_");
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *sp =
        step_obj(d, me->step_id, me->msg_at, me->id, me->asst, me->thread,
                 "message_creation", "in_progress");
    set_val(d, sp, "step_details", message_details(d, me->msg_id));
    as_emit(me->id, "thread.run.step.created", sp);
    as_emit(me->id, "thread.run.step.in_progress", sp);
    yyjson_mut_val *m = as_message(d, me->thread, "assistant", "", NULL, NULL,
                                   me->asst, me->id);
    set_str(d, m, "id", me->msg_id);
    set_str(d, m, "status", "in_progress");
    set_val(d, m, "completed_at", NULL);
    set_val(d, m, "content", yyjson_mut_arr(d));
    as_emit(me->id, "thread.message.created", m);
    as_emit(me->id, "thread.message.in_progress", m);
    yyjson_mut_doc_free(d);
}

/* One event of /responses streamed. */
static void sse_event(struct pass *me, const char *data, size_t n)
{
    yyjson_doc *ed = yyjson_read(data, n, 0);
    yyjson_val *e = yyjson_doc_get_root(ed);
    const char *type = yyjson_get_str(yyjson_obj_get(e, "type"));
    if (!type) {
        yyjson_doc_free(ed);
        return;
    }
    if (!strcmp(type, "response.output_text.delta")) {
        const char *dt = yyjson_get_str(yyjson_obj_get(e, "delta"));
        if (!me->msg_open)
            message_open(me);
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *o = yyjson_mut_obj(d);
        yyjson_mut_obj_add_strcpy(d, o, "id", me->msg_id);
        yyjson_mut_obj_add_str(d, o, "object", "thread.message.delta");
        yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, o, "delta");
        yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, dl, "content");
        yyjson_mut_val *p = yyjson_mut_arr_add_obj(d, c);
        yyjson_mut_obj_add_int(d, p, "index", 0);
        yyjson_mut_obj_add_str(d, p, "type", "text");
        yyjson_mut_val *tx = yyjson_mut_obj_add_obj(d, p, "text");
        yyjson_mut_obj_add_strcpy(d, tx, "value", dt ? dt : "");
        yyjson_mut_obj_add_arr(d, tx, "annotations");
        as_emit(me->id, "thread.message.delta", o);
        yyjson_mut_doc_free(d);
    } else if (!strcmp(type, "response.output_item.done")) {
        /* a search, told when done, before the text that uses it */
        yyjson_val *it = yyjson_obj_get(e, "item");
        const char *iid = yyjson_get_str(yyjson_obj_get(it, "id"));
        int search =
            yyjson_equals_str(yyjson_obj_get(it, "type"), "file_search_call");
        int code = yyjson_equals_str(yyjson_obj_get(it, "type"),
                                     "code_interpreter_call");
        if ((search || code) && iid && me->n_fs < 8) {
            snprintf(me->fs_item[me->n_fs], sizeof me->fs_item[0], "%s", iid);
            srv_new_id(me->fs_step[me->n_fs], sizeof me->fs_step[0], "step_");
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *sp =
                step_obj(d, me->fs_step[me->n_fs], 0, me->id, me->asst,
                         me->thread, "tool_calls", "completed");
            set_val(d, sp, "step_details",
                    search ? search_details(d, it)
                           : code_details(server, me, d, it));
            as_emit(me->id, "thread.run.step.created", sp);
            as_emit(me->id, "thread.run.step.completed", sp);
            yyjson_mut_doc_free(d);
            me->n_fs++;
        }
    } else if (!strcmp(type, "response.completed") ||
               !strcmp(type, "response.incomplete") ||
               !strcmp(type, "response.failed")) {
        /* the whole response, as without a stream */
        yyjson_val *resp = yyjson_obj_get(e, "response");
        char *t = resp ? yyjson_val_write(resp, 0, NULL) : NULL;
        yyjson_doc_free(me->final);
        me->final = t ? yyjson_read(t, strlen(t), 0) : NULL;
        free(t);
    } else if (!strcmp(type, "error")) {
        const char *msg = yyjson_get_str(yyjson_obj_get(e, "message"));
        if (!msg)
            msg = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(e, "error"), "message"));
        snprintf(me->err, sizeof me->err, "%s", msg ? msg : "error");
    }
    yyjson_doc_free(ed);
}

/* The bytes of /responses streamed: whole events handed to sse_event. */
static void on_data(void *arg, const char *p, size_t n)
{
    struct pass *me = arg;
    janas_buf_put(&me->sse, p, n);
    if (me->sse.oom)
        return;
    size_t at = 0;
    for (;;) {
        char *end = NULL;
        for (size_t i = at; i + 1 < me->sse.n; i++)
            if (me->sse.p[i] == '\n' && me->sse.p[i + 1] == '\n') {
                end = me->sse.p + i;
                break;
            }
        if (!end)
            break;
        /* its data: line */
        const char *ev = me->sse.p + at;
        for (const char *l = ev; l < end;) {
            const char *le = memchr(l, '\n', (size_t)(end - l));
            if (!le)
                le = end;
            if (le - l > 6 && !strncmp(l, "data: ", 6))
                sse_event(me, l + 6, (size_t)(le - l - 6));
            l = le + 1;
        }
        at = (size_t)(end - me->sse.p) + 2;
    }
    memmove(me->sse.p, me->sse.p + at, me->sse.n - at);
    me->sse.n -= at;
}

/* The run's event for its status, and done when it ends or waits. */
static void tell_status(struct run *x)
{
    const char *id = run_str(x, "id"), *st = run_status(x);
    if (!st)
        return;
    char ev[64];
    snprintf(ev, sizeof ev, "thread.run.%s", st);
    as_emit(id, ev, x->o);
    if (strcmp(st, "queued") && strcmp(st, "in_progress") &&
        strcmp(st, "cancelling"))
        as_emit_done(id);
}

/* One pass of the run: the model asked, its output made into steps; the
   run ends, or waits for tool outputs. */
static void run_pass(struct srv *s, struct pass *me)
{
    const char *id = me->id;
    struct run x;
    as_lock();
    if (run_load(s, id, &x) != 0) {
        as_unlock();
        return;
    }
    const char *st = run_status(&x);
    if (st && !strcmp(st, "cancelling")) {
        cancelled(&x);
        run_save(s, &x, 0);
        tell_status(&x);
    }
    if (!st || (strcmp(st, "queued") && strcmp(st, "in_progress"))) {
        as_unlock();
        run_free(&x);
        return;
    }
    snprintf(me->thread, sizeof me->thread, "%s", run_str(&x, "thread_id"));
    snprintf(me->asst, sizeof me->asst, "%s", run_str(&x, "assistant_id"));
    set_str(x.d, x.o, "status", "in_progress");
    if (!yyjson_mut_is_int(yyjson_mut_obj_get(x.o, "started_at")))
        set_int(x.d, x.o, "started_at", (int64_t)time(NULL));
    run_save(s, &x, 0);
    tell_status(&x);
    char *body = request_for(s, &x, me->cid, sizeof me->cid);
    as_unlock();
    run_free(&x);
    /* /responses streamed, the text told as it comes */
    struct srv_capture c = {.on_data = on_data, .arg = me};
    srv_handler h = srv_route_find("POST", "/responses");
    if (body && h) {
        struct srv_req r = {.srv = s,
                            .cap = &c,
                            .method = "POST",
                            .path = "/responses",
                            .body = body,
                            .n_body = strlen(body)};
        h(&r);
    }
    free(body);
    yyjson_doc *errd = c.body ? yyjson_read(c.body, c.n, 0) : NULL;
    /* an answer not streamed is the response itself */
    if (!me->final && c.status == 200 &&
        yyjson_equals_str(yyjson_obj_get(yyjson_doc_get_root(errd), "object"),
                          "response")) {
        me->final = errd;
        errd = NULL;
    }
    yyjson_val *resp = yyjson_doc_get_root(me->final);
    as_lock();
    pthread_mutex_lock(&live_mu);
    int stop = stopping;
    pthread_mutex_unlock(&live_mu);
    if (run_load(s, id, &x) != 0 || stop) {
        /* gone, or the server stops: taken up again at the next start */
        as_unlock();
        run_free(&x);
        yyjson_doc_free(errd);
        free(c.body);
        return;
    }
    st = run_status(&x);
    const char *rs = yyjson_get_str(yyjson_obj_get(resp, "status"));
    if (st && !strcmp(st, "cancelling")) {
        cancelled(&x);
    } else if (!resp || (rs && !strcmp(rs, "failed"))) {
        /* refused before the stream (an error body), or failed in it */
        const char *msg = yyjson_get_str(yyjson_obj_get(
            yyjson_obj_get(yyjson_doc_get_root(errd), "error"), "message"));
        if (!msg)
            msg = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(resp, "error"), "message"));
        fail(&x, "server_error",
             msg          ? msg
             : me->err[0] ? me->err
                          : "the model did not answer");
    } else {
        yyjson_val *u = yyjson_obj_get(resp, "usage");
        int64_t in = yyjson_get_sint(yyjson_obj_get(u, "input_tokens"));
        int64_t out = yyjson_get_sint(yyjson_obj_get(u, "output_tokens"));
        add_usage(x.d, x.o, in, out);
        yyjson_mut_val *calls = yyjson_mut_arr(x.d);
        size_t i, n;
        yyjson_val *it;
        yyjson_arr_foreach(yyjson_obj_get(resp, "output"), i, n, it)
        {
            const char *type = yyjson_get_str(yyjson_obj_get(it, "type"));
            if (!type)
                continue;
            if (!strcmp(type, "file_search_call") ||
                !strcmp(type, "code_interpreter_call")) {
                /* the step told while streaming keeps its id */
                const char *iid = yyjson_get_str(yyjson_obj_get(it, "id"));
                const char *sid = NULL;
                for (int k = 0; k < me->n_fs && iid; k++)
                    if (!strcmp(me->fs_item[k], iid))
                        sid = me->fs_step[k];
                yyjson_mut_val *sp =
                    step_new(&x, sid, 0, "tool_calls", "completed");
                set_val(x.d, sp, "step_details",
                        type[0] == 'f' ? search_details(x.d, it)
                                       : code_details(s, me, x.d, it));
                if (!sid) {
                    as_emit(id, "thread.run.step.created", sp);
                    as_emit(id, "thread.run.step.completed", sp);
                }
            } else if (!strcmp(type, "message")) {
                yyjson_val *part =
                    yyjson_arr_get_first(yyjson_obj_get(it, "content"));
                const char *text = yyjson_get_str(yyjson_obj_get(part, "text"));
                if (!me->msg_open)
                    message_open(me);
                yyjson_mut_val *m =
                    as_message(x.d, me->thread, "assistant", text ? text : "",
                               NULL, NULL, me->asst, id);
                set_str(x.d, m, "id", me->msg_id);
                set_int(x.d, m, "created_at", me->msg_at);
                yyjson_mut_val *tv = yyjson_mut_obj_get(
                    yyjson_mut_arr_get_first(yyjson_mut_obj_get(m, "content")),
                    "text");
                set_val(x.d, tv, "annotations",
                        annotations(s, me, x.d,
                                    yyjson_obj_get(part, "annotations"),
                                    text ? text : ""));
                as_message_append(s, me->thread, m);
                yyjson_mut_val *sp = step_new(&x, me->step_id, me->msg_at,
                                              "message_creation", "completed");
                set_val(x.d, sp, "step_details",
                        message_details(x.d, me->msg_id));
                as_emit(id, "thread.message.completed", m);
                as_emit(id, "thread.run.step.completed", sp);
                me->msg_open = 0; /* a later message gets its own */
            } else if (!strcmp(type, "function_call")) {
                yyjson_mut_arr_append(calls, yyjson_val_mut_copy(x.d, it));
            }
        }
        if (yyjson_mut_arr_size(calls)) {
            /* the run waits for the outputs of the calls */
            yyjson_mut_val *sp =
                step_new(&x, NULL, 0, "tool_calls", "in_progress");
            yyjson_mut_val *det = yyjson_mut_obj(x.d);
            yyjson_mut_obj_add_str(x.d, det, "type", "tool_calls");
            yyjson_mut_val *tc = yyjson_mut_obj_add_arr(x.d, det, "tool_calls");
            yyjson_mut_val *ra = yyjson_mut_obj(x.d);
            yyjson_mut_obj_add_str(x.d, ra, "type", "submit_tool_outputs");
            yyjson_mut_val *sto =
                yyjson_mut_obj_add_obj(x.d, ra, "submit_tool_outputs");
            yyjson_mut_val *rtc =
                yyjson_mut_obj_add_arr(x.d, sto, "tool_calls");
            yyjson_mut_val *fc;
            yyjson_mut_arr_foreach(calls, i, n, fc)
            {
                const char *cid =
                    yyjson_mut_get_str(yyjson_mut_obj_get(fc, "call_id"));
                const char *name =
                    yyjson_mut_get_str(yyjson_mut_obj_get(fc, "name"));
                const char *args =
                    yyjson_mut_get_str(yyjson_mut_obj_get(fc, "arguments"));
                for (int q = 0; q < 2; q++) {
                    yyjson_mut_val *c1 =
                        yyjson_mut_arr_add_obj(x.d, q ? rtc : tc);
                    yyjson_mut_obj_add_strcpy(x.d, c1, "id", cid ? cid : "");
                    yyjson_mut_obj_add_str(x.d, c1, "type", "function");
                    yyjson_mut_val *f =
                        yyjson_mut_obj_add_obj(x.d, c1, "function");
                    yyjson_mut_obj_add_strcpy(x.d, f, "name", name ? name : "");
                    yyjson_mut_obj_add_strcpy(x.d, f, "arguments",
                                              args ? args : "{}");
                    if (!q)
                        yyjson_mut_obj_add_null(x.d, f, "output");
                }
                yyjson_mut_val *item = yyjson_mut_arr_add_obj(x.d, x.items);
                yyjson_mut_obj_add_str(x.d, item, "type", "function_call");
                yyjson_mut_obj_add_strcpy(x.d, item, "call_id", cid ? cid : "");
                yyjson_mut_obj_add_strcpy(x.d, item, "name", name ? name : "");
                yyjson_mut_obj_add_strcpy(x.d, item, "arguments",
                                          args ? args : "{}");
            }
            set_val(x.d, sp, "step_details", det);
            as_emit(id, "thread.run.step.created", sp);
            set_str(x.d, x.o, "status", "requires_action");
            set_val(x.d, x.o, "required_action", ra);
            set_int(x.d, x.o, "expires_at", (int64_t)time(NULL) + RUN_EXPIRES);
        } else {
            if (rs && !strcmp(rs, "incomplete")) {
                set_str(x.d, x.o, "status", "incomplete");
                yyjson_mut_val *det = yyjson_mut_obj(x.d);
                yyjson_mut_obj_add_str(x.d, det, "reason",
                                       "max_completion_tokens");
                set_val(x.d, x.o, "incomplete_details", det);
            } else {
                set_str(x.d, x.o, "status", "completed");
                set_int(x.d, x.o, "completed_at", (int64_t)time(NULL));
            }
            set_val(x.d, x.o, "required_action", NULL);
        }
    }
    run_save(s, &x, 0);
    tell_status(&x);
    as_unlock();
    run_free(&x);
    yyjson_doc_free(errd);
    free(c.body);
}

static void *run_main(void *arg)
{
    char *id = arg;
    struct pass me = {.next = NULL};
    snprintf(me.id, sizeof me.id, "%s", id);
    atomic_init(&me.cancel, 0);
    pthread_mutex_lock(&live_mu);
    me.next = passes;
    passes = &me;
    pthread_mutex_unlock(&live_mu);
    srv_engine_watch = &me.cancel;
    run_pass(server, &me);
    srv_engine_watch = NULL;
    janas_buf_free(&me.sse);
    yyjson_doc_free(me.final);
    free(id);
    pthread_mutex_lock(&live_mu);
    for (struct pass **p = &passes; *p; p = &(*p)->next)
        if (*p == &me) {
            *p = me.next;
            break;
        }
    live--;
    pthread_cond_broadcast(&live_cv);
    pthread_mutex_unlock(&live_mu);
    return NULL;
}

/* A pass of run id on a thread of its own: 0, or -1. */
static int run_go(const char *id)
{
    pthread_mutex_lock(&live_mu);
    if (stopping) {
        pthread_mutex_unlock(&live_mu);
        return -1;
    }
    live++;
    pthread_mutex_unlock(&live_mu);
    char *arg = strdup(id);
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    int rc = arg ? pthread_create(&th, &at, run_main, arg) : -1;
    pthread_attr_destroy(&at);
    if (rc != 0) {
        free(arg);
        pthread_mutex_lock(&live_mu);
        live--;
        pthread_cond_broadcast(&live_cv);
        pthread_mutex_unlock(&live_mu);
        return -1;
    }
    return 0;
}

void srv_runs_start(struct srv *s)
{
    server = s;
    stopping = 0;
    yyjson_doc *d = as_all(s, AS_RUN, NULL, NULL);
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_doc_get_root(d), i, n, v)
    {
        const char *st = yyjson_get_str(yyjson_obj_get(v, "status"));
        const char *id = yyjson_get_str(yyjson_obj_get(v, "id"));
        if (st && id &&
            (!strcmp(st, "queued") || !strcmp(st, "in_progress") ||
             !strcmp(st, "cancelling")))
            run_go(id);
    }
    yyjson_doc_free(d);
}

void srv_runs_stop(void)
{
    as_feeds_end();
    pthread_mutex_lock(&live_mu);
    stopping = 1;
    while (live > 0)
        pthread_cond_wait(&live_cv, &live_mu);
    pthread_mutex_unlock(&live_mu);
}

/* ---- the API ---- */

/* The run made of a body for a thread (as_lock not held): 0 with the
   reply sent, or the reason. */
static const char *run_make(struct srv_req *r, const char *thread,
                            yyjson_val *root, int with_thread, int *ret)
{
    const char *aid = yyjson_get_str(yyjson_obj_get(root, "assistant_id"));
    if (!aid)
        return "'assistant_id' names the assistant to run.";
    int stream = yyjson_get_bool(yyjson_obj_get(root, "stream"));
    char *at = as_get(r->srv, AS_ASSISTANT, aid, NULL);
    yyjson_doc *ad = at ? yyjson_read(at, strlen(at), 0) : NULL;
    free(at);
    if (!ad)
        return "'assistant_id' names no assistant here.";
    yyjson_val *asst = yyjson_doc_get_root(ad);
    const char *why = NULL;
    yyjson_val *tools = yyjson_obj_get(root, "tools");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    yyjson_val *addm = yyjson_obj_get(root, "additional_messages");
    if ((why = as_tools_error(tools)) || (why = as_metadata_error(md))) {
        yyjson_doc_free(ad);
        return why;
    }
    if (addm && !yyjson_is_null(addm) &&
        (!yyjson_is_arr(addm) || yyjson_arr_size(addm) > 32)) {
        yyjson_doc_free(ad);
        return "'additional_messages' is a list of at most 32 messages.";
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    struct run x = {.d = d};
    x.o = yyjson_mut_obj(d);
    x.steps = yyjson_mut_arr(d);
    x.items = yyjson_mut_arr(d);
    char id[48];
    srv_new_id(id, sizeof id, "run_");
    int64_t now = (int64_t)time(NULL);
    yyjson_mut_obj_add_strcpy(d, x.o, "id", id);
    yyjson_mut_obj_add_str(d, x.o, "object", "thread.run");
    yyjson_mut_obj_add_int(d, x.o, "created_at", now);
    yyjson_mut_obj_add_strcpy(d, x.o, "thread_id", thread);
    yyjson_mut_obj_add_strcpy(d, x.o, "assistant_id", aid);
    yyjson_mut_obj_add_str(d, x.o, "status", "queued");
    yyjson_mut_obj_add_null(d, x.o, "required_action");
    yyjson_mut_obj_add_null(d, x.o, "last_error");
    yyjson_mut_obj_add_null(d, x.o, "expires_at");
    yyjson_mut_obj_add_null(d, x.o, "started_at");
    yyjson_mut_obj_add_null(d, x.o, "cancelled_at");
    yyjson_mut_obj_add_null(d, x.o, "failed_at");
    yyjson_mut_obj_add_null(d, x.o, "completed_at");
    yyjson_mut_obj_add_null(d, x.o, "incomplete_details");
    yyjson_mut_obj_add_strcpy(d, x.o, "model", r->srv->cfg.model_id);
    /* the instructions: the run's, else the assistant's, and more */
    const char *ins = yyjson_get_str(yyjson_obj_get(root, "instructions"));
    if (!ins)
        ins = yyjson_get_str(yyjson_obj_get(asst, "instructions"));
    const char *more =
        yyjson_get_str(yyjson_obj_get(root, "additional_instructions"));
    struct janas_buf b = {0};
    if (ins)
        janas_buf_puts(&b, ins);
    if (more) {
        if (b.n)
            janas_buf_put(&b, "\n\n", 2);
        janas_buf_puts(&b, more);
    }
    janas_buf_put(&b, "", 1);
    yyjson_mut_obj_add_strcpy(d, x.o, "instructions", b.oom ? "" : b.p);
    janas_buf_free(&b);
    yyjson_mut_obj_add_val(
        d, x.o, "tools",
        yyjson_val_mut_copy(
            d, yyjson_is_arr(tools) ? tools : yyjson_obj_get(asst, "tools")));
    yyjson_mut_obj_add_val(d, x.o, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    yyjson_mut_obj_add_null(d, x.o, "usage");
    /* the sampling: the run's, else the assistant's */
    static const char *const OPT[] = {"temperature", "top_p", "response_format",
                                      "reasoning_effort"};
    for (int i = 0; i < 4; i++) {
        yyjson_val *v = yyjson_obj_get(root, OPT[i]);
        if (!v || yyjson_is_null(v))
            v = yyjson_obj_get(asst, OPT[i]);
        yyjson_mut_obj_add_val(
            d, x.o, OPT[i], v ? yyjson_val_mut_copy(d, v) : yyjson_mut_null(d));
    }
    static const char *const ASIS[] = {
        "max_prompt_tokens", "max_completion_tokens", "truncation_strategy",
        "tool_choice", "parallel_tool_calls"};
    for (int i = 0; i < 5; i++) {
        yyjson_val *v = yyjson_obj_get(root, ASIS[i]);
        if (v)
            yyjson_mut_obj_add_val(d, x.o, ASIS[i], yyjson_val_mut_copy(d, v));
        else if (i == 2) {
            yyjson_mut_val *t = yyjson_mut_obj_add_obj(d, x.o, ASIS[i]);
            yyjson_mut_obj_add_str(d, t, "type", "auto");
            yyjson_mut_obj_add_null(d, t, "last_messages");
        } else if (i == 3)
            yyjson_mut_obj_add_str(d, x.o, ASIS[i], "auto");
        else if (i == 4)
            yyjson_mut_obj_add_bool(d, x.o, ASIS[i], true);
        else
            yyjson_mut_obj_add_null(d, x.o, ASIS[i]);
    }
    yyjson_doc_free(ad);
    char other[64];
    as_lock();
    char *tt = as_get(r->srv, AS_THREAD, thread, NULL);
    free(tt);
    if (!tt)
        why = "The thread is not here.";
    else if (as_active_run(r->srv, thread, other, sizeof other))
        why = "The thread already has a run going.";
    /* the additional messages, into the thread first */
    size_t i, n;
    yyjson_val *m;
    yyjson_arr_foreach(addm, i, n, m)
    {
        if (why)
            break;
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        if (!role || (strcmp(role, "user") && strcmp(role, "assistant"))) {
            why = "A message's role is user or assistant.";
            break;
        }
        char *t = as_message_text(yyjson_obj_get(m, "content"), &why);
        yyjson_val *att = yyjson_obj_get(m, "attachments");
        if (t && !(why = as_attach(r->srv, thread, att)))
            as_message_append(r->srv, thread,
                              as_message(d, thread, role, t, att,
                                         yyjson_obj_get(m, "metadata"), NULL,
                                         NULL));
        free(t);
    }
    /* a stream's feed open before the run can go */
    struct as_feed *feed = !why && stream ? as_feed_open(id) : NULL;
    if (!why && stream && !feed)
        why = "out of memory";
    if (!why && run_save(r->srv, &x, 1) != 0)
        why = "cannot keep the run";
    as_unlock();
    if (!why && feed) {
        if (with_thread) {
            char *tt = as_get(r->srv, AS_THREAD, thread, NULL);
            yyjson_doc *td = tt ? yyjson_read(tt, strlen(tt), 0) : NULL;
            free(tt);
            yyjson_mut_doc *d2 = yyjson_mut_doc_new(NULL);
            if (td)
                as_emit(id, "thread.created",
                        yyjson_val_mut_copy(d2, yyjson_doc_get_root(td)));
            yyjson_mut_doc_free(d2);
            yyjson_doc_free(td);
        }
        as_emit(id, "thread.run.created", x.o);
        as_emit(id, "thread.run.queued", x.o);
        run_go(id);
        *ret = as_feed_reply(r, feed);
    } else if (!why) {
        *ret = reply_val(r, x.o);
        run_go(id);
    } else {
        as_feed_close(feed);
    }
    run_free(&x);
    return why;
}

int srv_runs_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    int ret = 0;
    const char *why = !yyjson_is_obj(root)
                          ? "The body is a JSON object."
                          : run_make(r, r->param[0], root, 0, &ret);
    yyjson_doc_free(doc);
    if (why && !strcmp(why, "The thread is not here."))
        return not_found(r, "thread", r->param[0]);
    return why ? bad(r, why) : ret;
}

int srv_runs_create_thread(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    int ret = 0;
    const char *why = NULL;
    char thread[48];
    if (!yyjson_is_obj(root))
        why = "The body is a JSON object.";
    else if (!yyjson_get_str(yyjson_obj_get(root, "assistant_id")))
        why = "'assistant_id' names the assistant to run.";
    else if (!(why = as_thread_make(r->srv, yyjson_obj_get(root, "thread"),
                                    thread, sizeof thread)))
        why = run_make(r, thread, root, 1, &ret);
    yyjson_doc_free(doc);
    return why ? bad(r, why) : ret;
}

/* The run of the request's thread, loaded (as_lock held). */
static int run_of(struct srv_req *r, struct run *x)
{
    if (run_load(r->srv, r->param[1], x) != 0)
        return -1;
    if (!yyjson_mut_equals_str(yyjson_mut_obj_get(x->o, "thread_id"),
                               r->param[0])) {
        run_free(x);
        return -1;
    }
    return 0;
}

int srv_runs_get(struct srv_req *r)
{
    struct run x;
    as_lock();
    int rc = run_of(r, &x);
    if (rc == 0) {
        const char *before = run_status(&x);
        int was = before && !strcmp(before, "requires_action");
        run_expire(&x);
        if (was && strcmp(run_status(&x), "requires_action"))
            run_save(r->srv, &x, 0);
    }
    as_unlock();
    if (rc != 0)
        return not_found(r, "run", r->param[1]);
    int ret = reply_val(r, x.o);
    run_free(&x);
    return ret;
}

int srv_runs_list(struct srv_req *r)
{
    char *tt = as_get(r->srv, AS_THREAD, r->param[0], NULL);
    free(tt);
    if (!tt)
        return not_found(r, "thread", r->param[0]);
    yyjson_doc *d = as_all(r->srv, AS_RUN, "thread_id", r->param[0]);
    int ret = as_list_reply(r, yyjson_doc_get_root(d));
    yyjson_doc_free(d);
    return ret;
}

int srv_runs_modify(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    const char *why = !yyjson_is_obj(root) ? "The body is a JSON object."
                                           : as_metadata_error(md);
    struct run x;
    as_lock();
    int rc = run_of(r, &x);
    if (rc == 0 && !why && md) {
        set_val(x.d, x.o, "metadata",
                yyjson_is_obj(md) ? yyjson_val_mut_copy(x.d, md)
                                  : yyjson_mut_obj(x.d));
        run_save(r->srv, &x, 0);
    }
    as_unlock();
    yyjson_doc_free(doc);
    if (rc != 0)
        return not_found(r, "run", r->param[1]);
    int ret = why ? bad(r, why) : reply_val(r, x.o);
    run_free(&x);
    return ret;
}

int srv_runs_cancel(struct srv_req *r)
{
    struct run x;
    as_lock();
    int rc = run_of(r, &x);
    const char *why = NULL;
    if (rc == 0) {
        const char *st = run_status(&x);
        if (st && !strcmp(st, "in_progress")) {
            set_str(x.d, x.o, "status", "cancelling"); /* the pass sees it */
            pthread_mutex_lock(&live_mu);              /* and its model stops */
            for (struct pass *p = passes; p; p = p->next)
                if (!strcmp(p->id, r->param[1]))
                    atomic_store(&p->cancel, 1);
            pthread_mutex_unlock(&live_mu);
        } else if (st &&
                   (!strcmp(st, "queued") || !strcmp(st, "requires_action")))
            cancelled(&x);
        else
            why = "The run has ended: there is nothing to cancel.";
        if (!why) {
            run_save(r->srv, &x, 0);
            tell_status(&x);
        }
    }
    as_unlock();
    if (rc != 0)
        return not_found(r, "run", r->param[1]);
    int ret = why ? bad(r, why) : reply_val(r, x.o);
    run_free(&x);
    return ret;
}

int srv_runs_submit(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *outs = yyjson_obj_get(root, "tool_outputs");
    /* a stream's feed open before the run goes on */
    struct as_feed *feed = yyjson_get_bool(yyjson_obj_get(root, "stream"))
                               ? as_feed_open(r->param[1])
                               : NULL;
    struct run x;
    as_lock();
    int rc = run_of(r, &x);
    const char *why = NULL;
    if (rc == 0) {
        run_expire(&x);
        const char *st = run_status(&x);
        yyjson_mut_val *want = yyjson_mut_obj_get(
            yyjson_mut_obj_get(yyjson_mut_obj_get(x.o, "required_action"),
                               "submit_tool_outputs"),
            "tool_calls");
        if (!st || strcmp(st, "requires_action"))
            why = "The run is not waiting for tool outputs.";
        else if (!yyjson_is_arr(outs) ||
                 yyjson_arr_size(outs) != yyjson_mut_arr_size(want))
            why = "'tool_outputs' gives an output for every call the run "
                  "waits for.";
        /* each call answered, once */
        size_t i, n;
        yyjson_mut_val *w;
        yyjson_mut_arr_foreach(want, i, n, w)
        {
            if (why)
                break;
            const char *cid = yyjson_mut_get_str(yyjson_mut_obj_get(w, "id"));
            int hits = 0;
            size_t k, m;
            yyjson_val *o;
            yyjson_arr_foreach(outs, k, m, o)
            {
                hits += yyjson_equals_str(yyjson_obj_get(o, "tool_call_id"),
                                          cid ? cid : "");
            }
            if (hits != 1)
                why = "Every call the run waits for gets one output, by its "
                      "tool_call_id.";
        }
        if (!why) {
            /* the outputs into the waiting step and into the input */
            yyjson_mut_val *sp = yyjson_mut_arr_get_last(x.steps);
            yyjson_mut_val *tc = yyjson_mut_obj_get(
                yyjson_mut_obj_get(sp, "step_details"), "tool_calls");
            size_t k, m;
            yyjson_val *o;
            yyjson_arr_foreach(outs, k, m, o)
            {
                const char *cid =
                    yyjson_get_str(yyjson_obj_get(o, "tool_call_id"));
                const char *out = yyjson_get_str(yyjson_obj_get(o, "output"));
                yyjson_mut_val *c1;
                size_t q, qn;
                yyjson_mut_arr_foreach(tc, q, qn, c1)
                {
                    if (yyjson_mut_equals_str(yyjson_mut_obj_get(c1, "id"),
                                              cid))
                        set_str(x.d, yyjson_mut_obj_get(c1, "function"),
                                "output", out ? out : "");
                }
                yyjson_mut_val *item = yyjson_mut_arr_add_obj(x.d, x.items);
                yyjson_mut_obj_add_str(x.d, item, "type",
                                       "function_call_output");
                yyjson_mut_obj_add_strcpy(x.d, item, "call_id", cid);
                yyjson_mut_obj_add_strcpy(x.d, item, "output", out ? out : "");
            }
            set_str(x.d, sp, "status", "completed");
            set_int(x.d, sp, "completed_at", (int64_t)time(NULL));
            set_str(x.d, x.o, "status", "queued");
            set_val(x.d, x.o, "required_action", NULL);
            run_save(r->srv, &x, 0);
        }
    }
    as_unlock();
    yyjson_doc_free(doc);
    if (rc != 0) {
        as_feed_close(feed);
        return not_found(r, "run", r->param[1]);
    }
    int ret;
    if (why) {
        as_feed_close(feed);
        ret = bad(r, why);
    } else if (feed) {
        tell_status(&x); /* queued again */
        run_go(r->param[1]);
        ret = as_feed_reply(r, feed);
    } else {
        ret = reply_val(r, x.o);
        run_go(r->param[1]);
    }
    run_free(&x);
    return ret;
}

/* The run's steps (as_lock not held): its document, or NULL. */
static yyjson_doc *steps_of(struct srv_req *r)
{
    struct run x;
    as_lock();
    int rc = run_of(r, &x);
    as_unlock();
    if (rc != 0)
        return NULL;
    char *t = yyjson_mut_val_write(x.steps, 0, NULL);
    run_free(&x);
    yyjson_doc *d = t ? yyjson_read(t, strlen(t), 0) : NULL;
    free(t);
    return d;
}

int srv_run_steps_list(struct srv_req *r)
{
    yyjson_doc *d = steps_of(r);
    if (!d)
        return not_found(r, "run", r->param[1]);
    int ret = as_list_reply(r, yyjson_doc_get_root(d));
    yyjson_doc_free(d);
    return ret;
}

int srv_run_steps_get(struct srv_req *r)
{
    yyjson_doc *d = steps_of(r);
    if (!d)
        return not_found(r, "run", r->param[1]);
    size_t i, n;
    yyjson_val *s;
    int ret = -1;
    yyjson_arr_foreach(yyjson_doc_get_root(d), i, n, s)
    {
        if (yyjson_equals_str(yyjson_obj_get(s, "id"), r->param[2])) {
            char *t = yyjson_val_write(s, 0, NULL);
            ret = srv_reply_json(r, 200, t, t ? strlen(t) : 0);
        }
    }
    yyjson_doc_free(d);
    return ret < 0 ? not_found(r, "run step", r->param[2]) : ret;
}
