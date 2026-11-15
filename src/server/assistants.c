/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * assistants.c - OpenAI's Assistants API, the assistants, the threads and
 * their messages: kept in the store (assistants.h), the messages of a
 * thread as its aux text. An assistant's or a thread's file_search
 * resources are vector stores of this server (vectors_api.c); the files
 * attached to a message for file_search go into the thread's own store.
 * Runs: runs.c.
 */
#include "assistants.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "containers.h"
#include "llm/json.h"

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

void as_lock(void)
{
    pthread_mutex_lock(&mu);
}

void as_unlock(void)
{
    pthread_mutex_unlock(&mu);
}

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

char *as_get(struct srv *s, const char *kind, const char *id, char **aux)
{
    char *j = NULL;
    if (aux)
        *aux = NULL;
    if (srv_store_get(s->store, kind, id, &j, aux) != 0)
        return NULL;
    return j;
}

int as_save(struct srv *s, const char *kind, const char *id, yyjson_mut_val *o,
            const char *aux, int fresh)
{
    char *t = yyjson_mut_val_write(o, 0, NULL);
    int rc = -1;
    if (t)
        rc = fresh ? srv_store_put(s->store, kind, id, (int64_t)time(NULL), t,
                                   aux)
                   : srv_store_update(s->store, kind, id, t, aux);
    free(t);
    return rc ? -1 : 0;
}

const char *as_metadata_error(yyjson_val *md)
{
    if (!md || yyjson_is_null(md))
        return NULL;
    if (!yyjson_is_obj(md) || yyjson_obj_size(md) > 16)
        return "'metadata' holds at most 16 keys.";
    size_t i, n;
    yyjson_val *k, *v;
    yyjson_obj_foreach(md, i, n, k, v)
    {
        if (!yyjson_is_str(v) || yyjson_get_len(v) > 512 ||
            yyjson_get_len(k) > 64)
            return "'metadata' values are strings of at most 512 "
                   "characters, keys of at most 64.";
    }
    return NULL;
}

const char *as_tools_error(yyjson_val *tools)
{
    if (!tools || yyjson_is_null(tools))
        return NULL;
    if (!yyjson_is_arr(tools) || yyjson_arr_size(tools) > 128)
        return "'tools' is a list of at most 128 tools.";
    size_t i, n;
    yyjson_val *t;
    yyjson_arr_foreach(tools, i, n, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (type && !strcmp(type, "function")) {
            yyjson_val *f = yyjson_obj_get(t, "function");
            if (!yyjson_get_str(yyjson_obj_get(f, "name")))
                return "A function tool has a function with a name.";
        } else if (type && !strcmp(type, "file_search")) {
            yyjson_val *m = yyjson_obj_get(yyjson_obj_get(t, "file_search"),
                                           "max_num_results");
            if (m && (!yyjson_is_int(m) || yyjson_get_sint(m) < 1 ||
                      yyjson_get_sint(m) > 50))
                return "file_search's max_num_results is from 1 to 50.";
        } else if (type && !strcmp(type, "code_interpreter")) {
            if (srv_containers_off())
                return srv_containers_off();
        } else {
            return "A tool is of type function, file_search or "
                   "code_interpreter.";
        }
    }
    return NULL;
}

/* A vector store made of file_ids through the Vector stores API: its id
   into id, or the reason. */
static const char *make_store(struct srv *s, yyjson_val *spec, char *id,
                              size_t cap)
{
    srv_handler h = srv_route_find("POST", "/vector_stores");
    if (!h)
        return "vector stores are not available";
    char *body = spec ? yyjson_val_write(spec, 0, NULL) : strdup("{}");
    struct srv_capture c = {0};
    struct srv_req r = {.srv = s,
                        .cap = &c,
                        .method = "POST",
                        .path = "/vector_stores",
                        .body = body ? body : "{}",
                        .n_body = body ? strlen(body) : 2};
    h(&r);
    free(body);
    yyjson_doc *d = c.body ? yyjson_read(c.body, c.n, 0) : NULL;
    const char *got =
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(d), "id"));
    const char *why = NULL;
    if (c.status != 200 || !got)
        why = "a vector store of the files could not be made (are they "
              "uploaded here?)";
    else
        snprintf(id, cap, "%s", got);
    yyjson_doc_free(d);
    free(c.body);
    return why;
}

const char *as_resources(struct srv *s, yyjson_mut_doc *d, yyjson_val *res,
                         yyjson_mut_val **out)
{
    *out = yyjson_mut_obj(d);
    if (!res || yyjson_is_null(res))
        return NULL;
    if (!yyjson_is_obj(res))
        return "'tool_resources' is an object.";
    yyjson_val *ci = yyjson_obj_get(res, "code_interpreter");
    yyjson_val *cids = yyjson_obj_get(ci, "file_ids");
    if (cids) { /* kept: copied into the thread's container at its runs */
        if (!yyjson_is_arr(cids) || yyjson_arr_size(cids) > 20)
            return "code_interpreter takes at most 20 file_ids.";
        yyjson_mut_val *c = yyjson_mut_obj_add_obj(d, *out, "code_interpreter");
        yyjson_mut_obj_add_val(d, c, "file_ids", yyjson_val_mut_copy(d, cids));
    }
    yyjson_val *fs = yyjson_obj_get(res, "file_search");
    if (!fs)
        return NULL;
    yyjson_val *ids = yyjson_obj_get(fs, "vector_store_ids");
    yyjson_val *make = yyjson_obj_get(fs, "vector_stores");
    if (yyjson_arr_size(ids) + yyjson_arr_size(make) > 1)
        return "file_search takes one vector store.";
    yyjson_mut_val *o = yyjson_mut_obj_add_obj(d, *out, "file_search");
    yyjson_mut_val *arr = yyjson_mut_obj_add_arr(d, o, "vector_store_ids");
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(ids, i, n, v)
    {
        if (!yyjson_is_str(v))
            return "vector_store_ids are ids.";
        yyjson_mut_arr_add_strcpy(d, arr, yyjson_get_str(v));
    }
    yyjson_arr_foreach(make, i, n, v)
    {
        char id[64];
        const char *why = make_store(s, v, id, sizeof id);
        if (why)
            return why;
        yyjson_mut_arr_add_strcpy(d, arr, id);
    }
    return NULL;
}

int as_list_reply(struct srv_req *r, yyjson_val *items)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after"), *before = srv_query(r, "before");
    long limit = 20;
    if (l) {
        char *end;
        limit = strtol(l, &end, 10);
        if (*end || limit < 1 || limit > 100)
            return bad(r, "'limit' is a number from 1 to 100.");
    }
    if (o && strcmp(o, "asc") && strcmp(o, "desc"))
        return bad(r, "'order' is asc or desc.");
    int desc = !o || !strcmp(o, "desc");
    size_t n = yyjson_arr_size(items);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    /* after and before: the items strictly between them, in this order */
    long from = 0, to = (long)n;
    for (size_t k = 0; k < n; k++) {
        yyjson_val *v = yyjson_arr_get(items, desc ? n - 1 - k : k);
        yyjson_val *id = yyjson_obj_get(v, "id");
        if (after && yyjson_equals_str(id, after))
            from = (long)k + 1;
        if (before && yyjson_equals_str(id, before))
            to = (long)k;
    }
    long shown = 0;
    int more = 0;
    const char *first = NULL, *last = NULL;
    for (long k = from; k < to; k++) {
        if (shown == limit) {
            more = 1;
            break;
        }
        yyjson_val *v =
            yyjson_arr_get(items, desc ? n - 1 - (size_t)k : (size_t)k);
        yyjson_mut_arr_append(data, yyjson_val_mut_copy(d, v));
        const char *id = yyjson_get_str(yyjson_obj_get(v, "id"));
        if (!shown)
            first = id;
        last = id;
        shown++;
    }
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_obj_add_val(d, root, "data", data);
    if (first) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    int ret = reply_val(r, root);
    yyjson_mut_doc_free(d);
    return ret;
}

yyjson_doc *as_all(struct srv *s, const char *kind, const char *key,
                   const char *value)
{
    char **ids;
    size_t n = srv_store_list(s->store, kind, &ids);
    struct janas_buf b = {0};
    janas_buf_put(&b, "[", 1);
    int first = 1;
    for (size_t i = 0; i < n; i++) {
        char *t = as_get(s, kind, ids[i], NULL);
        if (!t)
            continue;
        int keep = 1;
        if (key) {
            yyjson_doc *x = yyjson_read(t, strlen(t), 0);
            keep = yyjson_equals_str(
                yyjson_obj_get(yyjson_doc_get_root(x), key), value);
            yyjson_doc_free(x);
        }
        if (keep) {
            if (!first)
                janas_buf_put(&b, ",", 1);
            janas_buf_puts(&b, t);
            first = 0;
        }
        free(t);
    }
    srv_store_list_free(ids, n);
    janas_buf_put(&b, "]", 1);
    yyjson_doc *d = b.oom ? NULL : yyjson_read(b.p, b.n, 0);
    janas_buf_free(&b);
    return d;
}

int as_active_run(struct srv *s, const char *thread, char *id, size_t cap)
{
    yyjson_doc *d = as_all(s, AS_RUN, "thread_id", thread);
    size_t i, n;
    yyjson_val *v;
    int found = 0;
    yyjson_arr_foreach(yyjson_doc_get_root(d), i, n, v)
    {
        const char *st = yyjson_get_str(yyjson_obj_get(v, "status"));
        if (st &&
            (!strcmp(st, "queued") || !strcmp(st, "in_progress") ||
             !strcmp(st, "requires_action") || !strcmp(st, "cancelling"))) {
            snprintf(id, cap, "%s", yyjson_get_str(yyjson_obj_get(v, "id")));
            found = 1;
        }
    }
    yyjson_doc_free(d);
    return found;
}

/* ---- assistants ---- */

static const char *const SAMPLING[] = {"temperature", "top_p",
                                       "response_format", "reasoning_effort"};

/* The fields an assistant's body may set into o; NULL or the reason. */
static const char *assistant_fields(struct srv *s, yyjson_mut_doc *d,
                                    yyjson_mut_val *o, yyjson_val *root,
                                    int fresh)
{
    const char *why;
    if ((why = as_metadata_error(yyjson_obj_get(root, "metadata"))) ||
        (why = as_tools_error(yyjson_obj_get(root, "tools"))))
        return why;
    yyjson_val *t = yyjson_obj_get(root, "temperature");
    if (t && !yyjson_is_null(t) &&
        (!yyjson_is_num(t) || yyjson_get_num(t) < 0 || yyjson_get_num(t) > 2))
        return "'temperature' is from 0 to 2.";
    yyjson_val *p = yyjson_obj_get(root, "top_p");
    if (p && !yyjson_is_null(p) &&
        (!yyjson_is_num(p) || yyjson_get_num(p) < 0 || yyjson_get_num(p) > 1))
        return "'top_p' is from 0 to 1.";
    yyjson_mut_val *res = NULL;
    yyjson_val *rv = yyjson_obj_get(root, "tool_resources");
    if (rv && (why = as_resources(s, d, rv, &res)))
        return why;
    static const char *const TEXT[] = {"name", "description", "instructions",
                                       "model"};
    for (int i = 0; i < 4; i++) {
        yyjson_val *v = yyjson_obj_get(root, TEXT[i]);
        if (v || fresh) {
            yyjson_mut_obj_remove_key(o, TEXT[i]);
            if (yyjson_is_str(v))
                yyjson_mut_obj_add_strcpy(d, o, TEXT[i], yyjson_get_str(v));
            else if (i == 3)
                yyjson_mut_obj_add_strcpy(d, o, TEXT[i], s->cfg.model_id);
            else
                yyjson_mut_obj_add_null(d, o, TEXT[i]);
        }
    }
    yyjson_val *tools = yyjson_obj_get(root, "tools");
    if (tools || fresh) {
        yyjson_mut_obj_remove_key(o, "tools");
        yyjson_mut_obj_add_val(d, o, "tools",
                               yyjson_is_arr(tools)
                                   ? yyjson_val_mut_copy(d, tools)
                                   : yyjson_mut_arr(d));
    }
    if (rv || fresh) {
        yyjson_mut_obj_remove_key(o, "tool_resources");
        yyjson_mut_obj_add_val(d, o, "tool_resources",
                               res ? res : yyjson_mut_obj(d));
    }
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    if (md || fresh) {
        yyjson_mut_obj_remove_key(o, "metadata");
        yyjson_mut_obj_add_val(d, o, "metadata",
                               yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                                 : yyjson_mut_obj(d));
    }
    for (int i = 0; i < 4; i++) {
        yyjson_val *v = yyjson_obj_get(root, SAMPLING[i]);
        if (v || fresh) {
            yyjson_mut_obj_remove_key(o, SAMPLING[i]);
            if (v && !yyjson_is_null(v))
                yyjson_mut_obj_add_val(d, o, SAMPLING[i],
                                       yyjson_val_mut_copy(d, v));
            else if (i < 2)
                yyjson_mut_obj_add_real(d, o, SAMPLING[i], 1.0);
            else if (i == 2)
                yyjson_mut_obj_add_str(d, o, SAMPLING[i], "auto");
            else
                yyjson_mut_obj_add_null(d, o, SAMPLING[i]);
        }
    }
    return NULL;
}

int srv_assistants_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return bad(r, "The body is a JSON object.");
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    char id[48];
    srv_new_id(id, sizeof id, "asst_");
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "assistant");
    yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
    const char *why = assistant_fields(r->srv, d, o, root, 1);
    int ret;
    if (why)
        ret = bad(r, why);
    else if (as_save(r->srv, AS_ASSISTANT, id, o, NULL, 1) != 0)
        ret = srv_reply_error(r, 500, "server_error", NULL,
                              "cannot keep the assistant");
    else
        ret = reply_val(r, o);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    return ret;
}

int srv_assistants_get(struct srv_req *r)
{
    char *t = as_get(r->srv, AS_ASSISTANT, r->param[0], NULL);
    if (!t)
        return not_found(r, "assistant", r->param[0]);
    return srv_reply_json(r, 200, t, strlen(t));
}

int srv_assistants_modify(struct srv_req *r)
{
    char *t = as_get(r->srv, AS_ASSISTANT, r->param[0], NULL);
    if (!t)
        return not_found(r, "assistant", r->param[0]);
    yyjson_doc *cur = yyjson_read(t, strlen(t), 0);
    free(t);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_mut_doc *d = yyjson_doc_mut_copy(cur, NULL);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    const char *why = !yyjson_is_obj(root)
                          ? "The body is a JSON object."
                          : assistant_fields(r->srv, d, o, root, 0);
    int ret;
    if (why)
        ret = bad(r, why);
    else if (as_save(r->srv, AS_ASSISTANT, r->param[0], o, NULL, 0) != 0)
        ret = srv_reply_error(r, 500, "server_error", NULL,
                              "cannot keep the assistant");
    else
        ret = reply_val(r, o);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    yyjson_doc_free(cur);
    return ret;
}

static int deleted(struct srv_req *r, const char *object)
{
    char out[400];
    snprintf(out, sizeof out,
             "{\"id\":\"%s\",\"object\":\"%s\",\"deleted\":true}",
             r->param[r->n_param - 1], object);
    char *t = strdup(out);
    return srv_reply_json(r, 200, t, t ? strlen(t) : 0);
}

int srv_assistants_delete(struct srv_req *r)
{
    if (srv_store_del(r->srv->store, AS_ASSISTANT, r->param[0]) != 0)
        return not_found(r, "assistant", r->param[0]);
    return deleted(r, "assistant.deleted");
}

int srv_assistants_list(struct srv_req *r)
{
    yyjson_doc *d = as_all(r->srv, AS_ASSISTANT, NULL, NULL);
    int ret = as_list_reply(r, yyjson_doc_get_root(d));
    yyjson_doc_free(d);
    return ret;
}

/* ---- messages ---- */

char *as_message_text(yyjson_val *content, const char **why)
{
    if (yyjson_is_str(content))
        return strdup(yyjson_get_str(content));
    if (!yyjson_is_arr(content) || !yyjson_arr_size(content)) {
        *why = "A message's content is a string or a list of text parts.";
        return NULL;
    }
    struct janas_buf b = {0};
    size_t i, n;
    yyjson_val *p;
    yyjson_arr_foreach(content, i, n, p)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(p, "type"));
        if (type &&
            (!strcmp(type, "image_file") || !strcmp(type, "image_url"))) {
            *why = "Images in messages are not read on this server yet.";
            janas_buf_free(&b);
            return NULL;
        }
        const char *text = yyjson_get_str(yyjson_obj_get(p, "text"));
        if (!type || strcmp(type, "text") || !text) {
            *why = "A message's parts are {\"type\": \"text\", \"text\"}.";
            janas_buf_free(&b);
            return NULL;
        }
        if (b.n)
            janas_buf_put(&b, "\n", 1);
        janas_buf_puts(&b, text);
    }
    janas_buf_put(&b, "", 1);
    if (b.oom) {
        janas_buf_free(&b);
        *why = "out of memory";
        return NULL;
    }
    return b.p;
}

yyjson_mut_val *as_message(yyjson_mut_doc *d, const char *thread,
                           const char *role, const char *text,
                           yyjson_val *attachments, yyjson_val *metadata,
                           const char *assistant, const char *run)
{
    yyjson_mut_val *m = yyjson_mut_obj(d);
    char id[48];
    srv_new_id(id, sizeof id, "msg_");
    int64_t now = (int64_t)time(NULL);
    yyjson_mut_obj_add_strcpy(d, m, "id", id);
    yyjson_mut_obj_add_str(d, m, "object", "thread.message");
    yyjson_mut_obj_add_int(d, m, "created_at", now);
    yyjson_mut_obj_add_strcpy(d, m, "thread_id", thread);
    yyjson_mut_obj_add_str(d, m, "status", "completed");
    yyjson_mut_obj_add_null(d, m, "incomplete_details");
    yyjson_mut_obj_add_int(d, m, "completed_at", now);
    yyjson_mut_obj_add_null(d, m, "incomplete_at");
    yyjson_mut_obj_add_strcpy(d, m, "role", role);
    yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, m, "content");
    yyjson_mut_val *part = yyjson_mut_arr_add_obj(d, c);
    yyjson_mut_obj_add_str(d, part, "type", "text");
    yyjson_mut_val *tx = yyjson_mut_obj_add_obj(d, part, "text");
    yyjson_mut_obj_add_strcpy(d, tx, "value", text ? text : "");
    yyjson_mut_obj_add_arr(d, tx, "annotations");
    if (assistant)
        yyjson_mut_obj_add_strcpy(d, m, "assistant_id", assistant);
    else
        yyjson_mut_obj_add_null(d, m, "assistant_id");
    if (run)
        yyjson_mut_obj_add_strcpy(d, m, "run_id", run);
    else
        yyjson_mut_obj_add_null(d, m, "run_id");
    yyjson_mut_obj_add_val(d, m, "attachments",
                           yyjson_is_arr(attachments)
                               ? yyjson_val_mut_copy(d, attachments)
                               : yyjson_mut_arr(d));
    yyjson_mut_obj_add_val(d, m, "metadata",
                           yyjson_is_obj(metadata)
                               ? yyjson_val_mut_copy(d, metadata)
                               : yyjson_mut_obj(d));
    return m;
}

int as_message_append(struct srv *s, const char *thread, yyjson_mut_val *m)
{
    char *aux = NULL;
    char *t = as_get(s, AS_THREAD, thread, &aux);
    if (!t)
        return -1;
    free(t);
    yyjson_doc *x = aux ? yyjson_read(aux, strlen(aux), 0) : NULL;
    free(aux);
    yyjson_mut_doc *d =
        x ? yyjson_doc_mut_copy(x, NULL) : yyjson_mut_doc_new(NULL);
    yyjson_doc_free(x);
    yyjson_mut_val *arr = yyjson_mut_doc_get_root(d);
    if (!arr) {
        arr = yyjson_mut_arr(d);
        yyjson_mut_doc_set_root(d, arr);
    }
    yyjson_mut_arr_append(arr, yyjson_mut_val_mut_copy(d, m));
    char *out = yyjson_mut_write(d, 0, NULL);
    yyjson_mut_doc_free(d);
    int rc =
        out ? srv_store_update(s->store, AS_THREAD, thread, NULL, out) : -1;
    free(out);
    return rc ? -1 : 0;
}

/* Whether the attachment a names the tool. */
static int attach_for(yyjson_val *a, const char *tool)
{
    size_t k, m;
    yyjson_val *t;
    yyjson_arr_foreach(yyjson_obj_get(a, "tools"), k, m, t)
    {
        if (yyjson_equals_str(yyjson_obj_get(t, "type"), tool))
            return 1;
    }
    return 0;
}

/* The object o[key], made when missing. */
static yyjson_mut_val *obj_at(yyjson_mut_doc *d, yyjson_mut_val *o,
                              const char *key)
{
    yyjson_mut_val *v = yyjson_mut_obj_get(o, key);
    if (!yyjson_mut_is_obj(v)) {
        yyjson_mut_obj_remove_key(o, key);
        v = yyjson_mut_obj_add_obj(d, o, key);
    }
    return v;
}

static yyjson_mut_val *arr_at(yyjson_mut_doc *d, yyjson_mut_val *o,
                              const char *key)
{
    yyjson_mut_val *v = yyjson_mut_obj_get(o, key);
    if (!yyjson_mut_is_arr(v)) {
        yyjson_mut_obj_remove_key(o, key);
        v = yyjson_mut_obj_add_arr(d, o, key);
    }
    return v;
}

const char *as_attach(struct srv *s, const char *thread, yyjson_val *att)
{
    if (!att || yyjson_is_null(att))
        return NULL;
    if (!yyjson_is_arr(att) || yyjson_arr_size(att) > 10)
        return "'attachments' is a list of at most 10 files.";
    size_t i, n;
    yyjson_val *a;
    int fs = 0, ci = 0;
    yyjson_arr_foreach(att, i, n, a)
    {
        const char *fid = yyjson_get_str(yyjson_obj_get(a, "file_id"));
        if (!fid)
            return "An attachment has a file_id.";
        size_t k, m;
        yyjson_val *t;
        yyjson_arr_foreach(yyjson_obj_get(a, "tools"), k, m, t)
        {
            const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
            if (type && !strcmp(type, "code_interpreter") &&
                srv_containers_off())
                return srv_containers_off();
            if (!type || (strcmp(type, "file_search") &&
                          strcmp(type, "code_interpreter")))
                return "An attachment's tools are file_search or "
                       "code_interpreter.";
        }
        fs += attach_for(a, "file_search");
        ci += attach_for(a, "code_interpreter");
    }
    char *tt = as_get(s, AS_THREAD, thread, NULL);
    yyjson_doc *td = tt ? yyjson_read(tt, strlen(tt), 0) : NULL;
    free(tt);
    if (!td)
        return "The thread is gone.";
    yyjson_mut_doc *d = yyjson_doc_mut_copy(td, NULL);
    yyjson_doc_free(td);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    yyjson_mut_val *res = obj_at(d, o, "tool_resources");
    const char *why = NULL;
    char vsid[64] = "";
    if (fs) { /* the thread's own store: made at the first file */
        yyjson_mut_val *ids =
            arr_at(d, obj_at(d, res, "file_search"), "vector_store_ids");
        const char *vs = yyjson_mut_get_str(yyjson_mut_arr_get_first(ids));
        if (vs) {
            snprintf(vsid, sizeof vsid, "%s", vs);
        } else {
            static const char spec[] = "{\"name\":\"thread files\"}";
            yyjson_doc *spd = yyjson_read(spec, sizeof spec - 1, 0);
            why = make_store(s, yyjson_doc_get_root(spd), vsid, sizeof vsid);
            yyjson_doc_free(spd);
            if (!why)
                yyjson_mut_arr_add_strcpy(d, ids, vsid);
        }
    }
    if (ci && !why) { /* copied into the thread's container at its runs */
        yyjson_mut_val *ids =
            arr_at(d, obj_at(d, res, "code_interpreter"), "file_ids");
        yyjson_arr_foreach(att, i, n, a)
        {
            const char *fid = yyjson_get_str(yyjson_obj_get(a, "file_id"));
            int have = 0;
            size_t k, m;
            yyjson_mut_val *v;
            yyjson_mut_arr_foreach(ids, k, m, v)
            {
                have |= yyjson_mut_equals_str(v, fid);
            }
            if (attach_for(a, "code_interpreter") && !have)
                yyjson_mut_arr_add_strcpy(d, ids, fid);
        }
        if (yyjson_mut_arr_size(ids) > 20)
            why = "A thread's code_interpreter takes at most 20 files.";
    }
    if (!why)
        as_save(s, AS_THREAD, thread, o, NULL, 0);
    yyjson_mut_doc_free(d);
    yyjson_arr_foreach(att, i, n, a)
    {
        if (why)
            break;
        const char *fid = yyjson_get_str(yyjson_obj_get(a, "file_id"));
        if (attach_for(a, "file_search") &&
            srv_vs_attach(s, vsid, fid, NULL, NULL, &why) != 0 && !why)
            why = "A file could not be attached.";
    }
    return why;
}

int srv_messages_create(struct srv_req *r)
{
    const char *thread = r->param[0];
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *role = yyjson_get_str(yyjson_obj_get(root, "role"));
    const char *why = NULL;
    char run[64];
    char *text = NULL;
    as_lock();
    char *tt = as_get(r->srv, AS_THREAD, thread, NULL);
    free(tt);
    if (!tt) {
        as_unlock();
        yyjson_doc_free(doc);
        return not_found(r, "thread", thread);
    }
    if (!yyjson_is_obj(root))
        why = "The body is a JSON object.";
    else if (!role || (strcmp(role, "user") && strcmp(role, "assistant")))
        why = "'role' is user or assistant.";
    else if (as_active_run(r->srv, thread, run, sizeof run))
        why = "The thread has a run going: messages are added once it "
              "ends.";
    else if (!(why = as_metadata_error(yyjson_obj_get(root, "metadata"))))
        text = as_message_text(yyjson_obj_get(root, "content"), &why);
    yyjson_val *att = yyjson_obj_get(root, "attachments");
    if (!why)
        why = as_attach(r->srv, thread, att);
    int ret;
    if (why) {
        ret = bad(r, why);
    } else {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *m =
            as_message(d, thread, role, text, att,
                       yyjson_obj_get(root, "metadata"), NULL, NULL);
        ret = as_message_append(r->srv, thread, m) == 0
                  ? reply_val(r, m)
                  : srv_reply_error(r, 500, "server_error", NULL,
                                    "cannot keep the message");
        yyjson_mut_doc_free(d);
    }
    as_unlock();
    free(text);
    yyjson_doc_free(doc);
    return ret;
}

/* The thread's messages (as_lock held): NULL when there is no thread. */
static yyjson_doc *messages_of(struct srv *s, const char *thread)
{
    char *aux = NULL;
    char *t = as_get(s, AS_THREAD, thread, &aux);
    if (!t)
        return NULL;
    free(t);
    yyjson_doc *x = yyjson_read(aux ? aux : "[]", aux ? strlen(aux) : 2, 0);
    free(aux);
    return x;
}

int srv_messages_list(struct srv_req *r)
{
    as_lock();
    yyjson_doc *x = messages_of(r->srv, r->param[0]);
    as_unlock();
    if (!x)
        return not_found(r, "thread", r->param[0]);
    const char *run = srv_query(r, "run_id");
    int ret;
    if (run) { /* those the run wrote */
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *arr = yyjson_mut_arr(d);
        yyjson_mut_doc_set_root(d, arr);
        size_t i, n;
        yyjson_val *m;
        yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, m)
        {
            if (yyjson_equals_str(yyjson_obj_get(m, "run_id"), run))
                yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, m));
        }
        yyjson_doc *f = yyjson_mut_doc_imut_copy(d, NULL);
        yyjson_mut_doc_free(d);
        ret = as_list_reply(r, yyjson_doc_get_root(f));
        yyjson_doc_free(f);
    } else {
        ret = as_list_reply(r, yyjson_doc_get_root(x));
    }
    yyjson_doc_free(x);
    return ret;
}

/* The message's index in the thread's list, or -1. */
static long message_at(yyjson_val *arr, const char *id)
{
    size_t i, n;
    yyjson_val *m;
    yyjson_arr_foreach(arr, i, n, m)
    {
        if (yyjson_equals_str(yyjson_obj_get(m, "id"), id))
            return (long)i;
    }
    return -1;
}

int srv_messages_get(struct srv_req *r)
{
    as_lock();
    yyjson_doc *x = messages_of(r->srv, r->param[0]);
    as_unlock();
    if (!x)
        return not_found(r, "thread", r->param[0]);
    long k = message_at(yyjson_doc_get_root(x), r->param[1]);
    int ret;
    if (k < 0) {
        ret = not_found(r, "message", r->param[1]);
    } else {
        char *t = yyjson_val_write(
            yyjson_arr_get(yyjson_doc_get_root(x), (size_t)k), 0, NULL);
        ret = srv_reply_json(r, 200, t, t ? strlen(t) : 0);
    }
    yyjson_doc_free(x);
    return ret;
}

/* The thread's messages with message k changed (metadata) or removed. */
static int message_change(struct srv_req *r, int remove)
{
    yyjson_doc *doc = remove ? NULL : yyjson_read(r->body, r->n_body, 0);
    yyjson_val *md = yyjson_obj_get(yyjson_doc_get_root(doc), "metadata");
    const char *why = remove ? NULL : as_metadata_error(md);
    if (!remove && !yyjson_is_obj(yyjson_doc_get_root(doc)))
        why = "The body is a JSON object.";
    if (why) {
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    as_lock();
    yyjson_doc *x = messages_of(r->srv, r->param[0]);
    long k = x ? message_at(yyjson_doc_get_root(x), r->param[1]) : -1;
    int ret;
    if (!x || k < 0) {
        as_unlock();
        ret = !x ? not_found(r, "thread", r->param[0])
                 : not_found(r, "message", r->param[1]);
    } else {
        yyjson_mut_doc *d = yyjson_doc_mut_copy(x, NULL);
        yyjson_mut_val *arr = yyjson_mut_doc_get_root(d);
        yyjson_mut_val *m = yyjson_mut_arr_get(arr, (size_t)k);
        if (remove) {
            yyjson_mut_arr_remove(arr, (size_t)k);
        } else if (md) {
            yyjson_mut_obj_remove_key(m, "metadata");
            yyjson_mut_obj_add_val(d, m, "metadata",
                                   yyjson_is_obj(md)
                                       ? yyjson_val_mut_copy(d, md)
                                       : yyjson_mut_obj(d));
        }
        char *out = yyjson_mut_write(d, 0, NULL);
        int rc = out ? srv_store_update(r->srv->store, AS_THREAD, r->param[0],
                                        NULL, out)
                     : -1;
        free(out);
        as_unlock();
        if (rc)
            ret = srv_reply_error(r, 500, "server_error", NULL,
                                  "cannot keep the thread");
        else if (remove)
            ret = deleted(r, "thread.message.deleted");
        else
            ret = reply_val(r, m);
        yyjson_mut_doc_free(d);
    }
    yyjson_doc_free(x);
    yyjson_doc_free(doc);
    return ret;
}

int srv_messages_modify(struct srv_req *r)
{
    return message_change(r, 0);
}

int srv_messages_delete(struct srv_req *r)
{
    return message_change(r, 1);
}

/* ---- threads ---- */

/* A thread made of a body {messages, tool_resources, metadata}: 0 with its
   id, or the reason. */
const char *as_thread_make(struct srv *s, yyjson_val *root, char *id,
                           size_t cap)
{
    yyjson_val *msgs = yyjson_obj_get(root, "messages");
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    const char *why;
    if (root && !yyjson_is_obj(root))
        return "The thread is a JSON object.";
    if ((why = as_metadata_error(md)))
        return why;
    if (msgs && (!yyjson_is_arr(msgs) || yyjson_arr_size(msgs) > 32))
        return "'messages' is a list of at most 32 messages.";
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *res = NULL;
    if ((why = as_resources(s, d, yyjson_obj_get(root, "tool_resources"),
                            &res))) {
        yyjson_mut_doc_free(d);
        return why;
    }
    /* the messages checked before the thread is kept */
    size_t i, n;
    yyjson_val *m;
    yyjson_arr_foreach(msgs, i, n, m)
    {
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        if (!role || (strcmp(role, "user") && strcmp(role, "assistant")))
            why = "A message's role is user or assistant.";
        char *t =
            why ? NULL : as_message_text(yyjson_obj_get(m, "content"), &why);
        free(t);
        if (!why)
            why = as_metadata_error(yyjson_obj_get(m, "metadata"));
        if (why)
            break;
    }
    if (why) {
        yyjson_mut_doc_free(d);
        return why;
    }
    yyjson_mut_val *o = yyjson_mut_obj(d);
    srv_new_id(id, cap, "thread_");
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "thread");
    yyjson_mut_obj_add_int(d, o, "created_at", (int64_t)time(NULL));
    yyjson_mut_obj_add_val(d, o, "tool_resources", res);
    yyjson_mut_obj_add_val(d, o, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    as_lock();
    if (as_save(s, AS_THREAD, id, o, "[]", 1) != 0)
        why = "cannot keep the thread";
    yyjson_arr_foreach(msgs, i, n, m)
    {
        if (why)
            break;
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        char *t = as_message_text(yyjson_obj_get(m, "content"), &why);
        yyjson_val *att = yyjson_obj_get(m, "attachments");
        if (t && !(why = as_attach(s, id, att))) {
            yyjson_mut_val *mm = as_message(
                d, id, role, t, att, yyjson_obj_get(m, "metadata"), NULL, NULL);
            if (as_message_append(s, id, mm) != 0)
                why = "cannot keep the message";
        }
        free(t);
    }
    if (why)
        srv_store_del(s->store, AS_THREAD, id);
    as_unlock();
    yyjson_mut_doc_free(d);
    return why;
}

int srv_threads_create(struct srv_req *r)
{
    yyjson_doc *doc = r->n_body ? yyjson_read(r->body, r->n_body, 0) : NULL;
    if (r->n_body && !doc)
        return bad(r, "The body is a JSON object.");
    char id[48];
    const char *why =
        as_thread_make(r->srv, yyjson_doc_get_root(doc), id, sizeof id);
    yyjson_doc_free(doc);
    if (why)
        return bad(r, why);
    char *t = as_get(r->srv, AS_THREAD, id, NULL);
    return srv_reply_json(r, 200, t, t ? strlen(t) : 0);
}

int srv_threads_get(struct srv_req *r)
{
    char *t = as_get(r->srv, AS_THREAD, r->param[0], NULL);
    if (!t)
        return not_found(r, "thread", r->param[0]);
    return srv_reply_json(r, 200, t, strlen(t));
}

int srv_threads_modify(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    const char *why = !yyjson_is_obj(root) ? "The body is a JSON object."
                                           : as_metadata_error(md);
    as_lock();
    char *t = as_get(r->srv, AS_THREAD, r->param[0], NULL);
    yyjson_doc *cur = t ? yyjson_read(t, strlen(t), 0) : NULL;
    free(t);
    int ret;
    if (!cur) {
        ret = not_found(r, "thread", r->param[0]);
    } else {
        yyjson_mut_doc *d = yyjson_doc_mut_copy(cur, NULL);
        yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
        yyjson_mut_val *res = NULL;
        yyjson_val *rv = yyjson_obj_get(root, "tool_resources");
        if (!why && rv)
            why = as_resources(r->srv, d, rv, &res);
        if (!why && res) {
            yyjson_mut_obj_remove_key(o, "tool_resources");
            yyjson_mut_obj_add_val(d, o, "tool_resources", res);
        }
        if (!why && md) {
            yyjson_mut_obj_remove_key(o, "metadata");
            yyjson_mut_obj_add_val(d, o, "metadata",
                                   yyjson_is_obj(md)
                                       ? yyjson_val_mut_copy(d, md)
                                       : yyjson_mut_obj(d));
        }
        if (why)
            ret = bad(r, why);
        else if (as_save(r->srv, AS_THREAD, r->param[0], o, NULL, 0) != 0)
            ret = srv_reply_error(r, 500, "server_error", NULL,
                                  "cannot keep the thread");
        else
            ret = reply_val(r, o);
        yyjson_mut_doc_free(d);
    }
    as_unlock();
    yyjson_doc_free(cur);
    yyjson_doc_free(doc);
    return ret;
}

int srv_threads_delete(struct srv_req *r)
{
    char run[64];
    as_lock();
    int active = as_active_run(r->srv, r->param[0], run, sizeof run);
    int rc = active ? 0 : srv_store_del(r->srv->store, AS_THREAD, r->param[0]);
    as_unlock();
    if (active)
        return bad(r, "The thread has a run going: cancel it first.");
    if (rc != 0)
        return not_found(r, "thread", r->param[0]);
    return deleted(r, "thread.deleted");
}
