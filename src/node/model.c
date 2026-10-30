/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model.c - open() and the Model class: a model file opened on an engine
 * thread of its own, its chat, how its preparations are going, close().
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

#define GIB 1073741824.0

struct open_job {
    char *path;
    struct janas_llm_params p;
    char *mtp, *draft;
    int find_mtp; /* mtp: true, the block beside the model if there is one */
    int no_gpu;
    char found[4096]; /* the block found */
    /* what the model says of itself, for the Model's properties */
    char *name, *description, *system;
    int reasons;
};

static void open_drop(void *data)
{
    struct open_job *oj = data;
    if (!oj)
        return;
    free(oj->path);
    free(oj->mtp);
    free(oj->draft);
    free(oj->name);
    free(oj->description);
    free(oj->system);
    free(oj);
}

static const char *const open_known[] = {
    "context",   "cacheGiB", "reserveGiB", "mtp",        "draft",
    "mode",      "gpu",      "preload",    "expertBits", "headBits",
    "attention", "quality",  NULL};

/* one of the strings, as its index; -1 after throwing */
static int choice(napi_env env, napi_value o, const char *name,
                  const char *const *names, int *out)
{
    char *s = NULL;
    int r = janas_node_opt_string(env, o, "open", name, &s);
    if (r <= 0)
        return r;
    for (int i = 0; names[i]; i++)
        if (!strcmp(names[i], s)) {
            free(s);
            *out = i;
            return 1;
        }
    janas_node_type_error(env, "janas: open: %s cannot be '%s'", name, s);
    free(s);
    return -1;
}

static int open_options(napi_env env, napi_value o, struct open_job *oj)
{
    const char *w = "open";
    static const char *const modes[] = {"auto", "eco", "max", NULL};
    static const char *const attn[] = {"auto", "exact", "fast", NULL};
    static const char *const quality[] = {"low", "medium", "high", NULL};
    int64_t i;
    double d;
    int b, r, k;
    if (janas_node_options(env, o, w, open_known) < 0)
        return -1;
    if ((r = janas_node_opt_int(env, o, w, "context", 1, UINT32_MAX, &i)) < 0)
        return -1;
    if (r)
        oj->p.n_ctx = (uint32_t)i;
    if ((r = janas_node_opt_number(env, o, w, "cacheGiB", 0, 1 << 20, &d)) < 0)
        return -1;
    if (r)
        oj->p.cache_bytes = (uint64_t)(d * GIB);
    if ((r = janas_node_opt_number(env, o, w, "reserveGiB", 0, 1 << 20, &d)) <
        0)
        return -1;
    if (r)
        oj->p.reserve_bytes = (uint64_t)(d * GIB);
    napi_value v;
    napi_valuetype t;
    if (o && napi_typeof(env, o, &t) == napi_ok && t == napi_object &&
        napi_get_named_property(env, o, "mtp", &v) == napi_ok &&
        napi_typeof(env, v, &t) == napi_ok && t != napi_undefined) {
        if (t == napi_boolean) {
            bool on = false;
            napi_get_value_bool(env, v, &on);
            oj->find_mtp = on;
        } else if (janas_node_opt_string(env, o, w, "mtp", &oj->mtp) < 0)
            return -1;
    }
    if (janas_node_opt_string(env, o, w, "draft", &oj->draft) < 0)
        return -1;
    if ((r = choice(env, o, "mode", modes, &k)) < 0)
        return -1;
    if (r)
        oj->p.mode = k == 1   ? JANAS_LLM_MODE_ECO
                     : k == 2 ? JANAS_LLM_MODE_MAX
                              : JANAS_LLM_MODE_AUTO;
    if ((r = janas_node_opt_bool(env, o, w, "gpu", &b)) < 0)
        return -1;
    if (r)
        oj->no_gpu = !b;
    if ((r = janas_node_opt_bool(env, o, w, "preload", &b)) < 0)
        return -1;
    if (r)
        oj->p.no_preload = !b;
    if ((r = janas_node_opt_int(env, o, w, "expertBits", 2, 6, &i)) < 0)
        return -1;
    if (r && i != 2 && i != 4 && i != 6) {
        janas_node_type_error(env, "janas: open: expertBits is 2, 4 or 6");
        return -1;
    }
    if (r)
        oj->p.expert_bits = (int32_t)i;
    if ((r = janas_node_opt_int(env, o, w, "headBits", 4, 6, &i)) < 0)
        return -1;
    if (r && i == 5) {
        janas_node_type_error(env, "janas: open: headBits is 4 or 6");
        return -1;
    }
    if (r)
        oj->p.head_bits = (int32_t)i;
    if ((r = choice(env, o, "attention", attn, &k)) < 0)
        return -1;
    if (r)
        oj->p.attn_scores = k == 1   ? JANAS_LLM_ATTN_EXACT
                            : k == 2 ? JANAS_LLM_ATTN_FAST
                                     : JANAS_LLM_ATTN_AUTO;
    if ((r = choice(env, o, "quality", quality, &k)) < 0)
        return -1;
    if (r)
        oj->p.min_quality = k; /* JANAS_LLM_QUALITY_*: low 0, high 2 */
    return 0;
}

/* a text the library writes (janas_llm_name and the like), malloc'd */
static char *take(int32_t (*get)(const janas_llm *, char *, int32_t, int32_t *),
                  const janas_llm *llm)
{
    char small[256];
    int32_t len = 0;
    if (get(llm, small, sizeof(small), &len) != JANAS_LLM_OK || len < 0)
        return strdup("");
    if (len < (int32_t)sizeof(small))
        return strdup(small);
    char *s = malloc((size_t)len + 1);
    if (s && get(llm, s, len + 1, &len) != JANAS_LLM_OK)
        s[0] = 0;
    return s;
}

static void open_run(struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    struct open_job *oj = j->data;
    if (oj->find_mtp && janas_llm_find_mtp(oj->path, oj->found,
                                           sizeof(oj->found)) == JANAS_LLM_OK)
        oj->p.mtp_path = oj->found;
    else if (oj->mtp)
        oj->p.mtp_path = oj->mtp;
    oj->p.draft_path = oj->draft;
    j->rc = janas_llm_open(oj->path, &oj->p, &m->llm);
    if (j->rc != JANAS_LLM_OK) {
        snprintf(j->error, sizeof(j->error), "%s", janas_llm_last_error());
        m->llm = NULL;
        j->last = 1; /* no model: the thread ends */
        return;
    }
    /* where there is no GPU there is nothing to keep it off */
    if (oj->no_gpu)
        janas_llm_set_gpu(m->llm, 0);
    oj->name = take(janas_llm_name, m->llm);
    oj->description = take(janas_llm_describe, m->llm);
    oj->system = take(janas_llm_default_system, m->llm);
    oj->reasons = janas_llm_reasons(m->llm);
}

static void model_finalize(napi_env env, void *data, void *hint)
{
    (void)hint;
    janas_node_release(env, data);
}

static napi_value text_or_empty(napi_env env, const char *s)
{
    return janas_node_string(env, s ? s : "", JANAS_NAPI_AUTO_LENGTH);
}

static napi_value open_value(napi_env env, struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    struct open_job *oj = j->data;
    napi_value obj = janas_node_new(env, m->st->model_class,
                                    &janas_node_model_tag, m, model_finalize);
    if (!obj)
        return NULL;
    janas_node_hold(m);
    napi_property_descriptor d[] = {
        {.utf8name = "path",
         .value = text_or_empty(env, oj->path),
         .attributes = napi_enumerable},
        {.utf8name = "name",
         .value = text_or_empty(env, oj->name),
         .attributes = napi_enumerable},
        {.utf8name = "description",
         .value = text_or_empty(env, oj->description),
         .attributes = napi_enumerable},
        {.utf8name = "defaultSystem",
         .value = text_or_empty(env, oj->system),
         .attributes = napi_enumerable},
        {.utf8name = "reasons",
         .value = janas_node_bool(env, oj->reasons),
         .attributes = napi_enumerable},
        {.utf8name = "mtp",
         .value = oj->p.mtp_path ? text_or_empty(env, oj->p.mtp_path)
                                 : janas_node_bool(env, 0),
         .attributes = napi_enumerable},
    };
    napi_define_properties(env, obj, sizeof(d) / sizeof(d[0]), d);
    return obj;
}

static void open_after(napi_env env, struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    if (j->rc < 0)
        m->closing = 1; /* the thread has ended: back as the last job */
    else if (!m->holders)
        janas_node_close(env, m, NULL); /* its object could not be made */
}

napi_value janas_node_open(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {NULL, NULL};
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    struct janas_node_env *st = janas_node_env(env);
    size_t len = 0;
    char *path = argc ? janas_node_text(env, argv[0], &len) : NULL;
    if (!st || !path)
        return len == SIZE_MAX
                   ? janas_node_throw(env, JANAS_LLM_ENOMEM,
                                      "janas: out of memory")
                   : janas_node_type_error(
                         env, "janas: open(path, options): the path of a "
                              ".jns file, as a string");
    struct open_job *oj = calloc(1, sizeof(*oj));
    if (!oj) {
        free(path);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    oj->path = path;
    janas_llm_params_default(&oj->p);
    if (open_options(env, argc > 1 ? argv[1] : NULL, oj) < 0) {
        open_drop(oj);
        return NULL;
    }
    struct janas_node_model *m = janas_node_engine(env, st);
    struct janas_node_job *j = m ? janas_node_job(m, open_run) : NULL;
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        open_drop(oj);
        free(j);
        if (m) /* the thread ends with the close it is given */
            janas_node_close(env, m, NULL);
        return janas_node_throw(env, JANAS_LLM_ENOMEM,
                                "janas: open: no memory or no thread for "
                                "the engine");
    }
    j->data = oj;
    j->drop = open_drop;
    j->value = open_value;
    j->after = open_after;
    janas_node_post(env, j);
    return p;
}

static napi_value model_chat(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {NULL};
    struct janas_node_model *m = janas_node_self(
        env, info, &janas_node_model_tag, &argc, argv, "Model.chat");
    if (!m)
        return NULL;
    return janas_node_chat(env, m, argc ? argv[0] : NULL);
}

static napi_value model_close(napi_env env, napi_callback_info info)
{
    struct janas_node_model *m = janas_node_self(
        env, info, &janas_node_model_tag, NULL, NULL, "Model.close");
    if (!m)
        return NULL;
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p)
        janas_node_close(env, m, d);
    return p;
}

/* How the model's preparations are going: the expert cache filling from
   what this machine used most, the NPU made ready. The library allows both
   questions while the engine's thread works. */
static napi_value model_status(napi_env env, napi_callback_info info)
{
    struct janas_node_model *m = janas_node_self(
        env, info, &janas_node_model_tag, NULL, NULL, "Model.status");
    if (!m)
        return NULL;
    int open = m->engine && !m->closing && m->llm;
    int64_t done = 0, total = 0;
    int32_t ndone = 0, ntotal = 0;
    double seconds = 0;
    int preparing = 0;
    if (open) {
        janas_llm_preload(m->llm, &done, &total);
        preparing = janas_llm_npu_preparing(m->llm, &ndone, &ntotal, &seconds);
    }
    napi_value o, pre, npu;
    if (napi_create_object(env, &o) != napi_ok ||
        napi_create_object(env, &pre) != napi_ok ||
        napi_create_object(env, &npu) != napi_ok)
        return NULL;
    janas_node_put(env, pre, "done", janas_node_number(env, (double)done));
    janas_node_put(env, pre, "total", janas_node_number(env, (double)total));
    janas_node_put(env, npu, "preparing", janas_node_bool(env, preparing > 0));
    janas_node_put(env, npu, "done", janas_node_number(env, ndone));
    janas_node_put(env, npu, "total", janas_node_number(env, ntotal));
    janas_node_put(env, npu, "seconds", janas_node_number(env, seconds));
    janas_node_put(env, o, "open", janas_node_bool(env, open));
    janas_node_put(env, o, "busy", janas_node_bool(env, m->pending > 0));
    janas_node_put(env, o, "preload", pre);
    janas_node_put(env, o, "npu", npu);
    return o;
}

napi_value janas_node_model_class(napi_env env)
{
    const napi_property_attributes method = napi_writable | napi_configurable;
    napi_property_descriptor d[] = {
        {.utf8name = "chat", .method = model_chat, .attributes = method},
        {.utf8name = "status", .method = model_status, .attributes = method},
        {.utf8name = "close", .method = model_close, .attributes = method},
    };
    napi_value cls = NULL;
    if (napi_define_class(env, "Model", JANAS_NAPI_AUTO_LENGTH,
                          janas_node_constructor,
                          (void *)"a Model comes from open()",
                          sizeof(d) / sizeof(d[0]), d, &cls) != napi_ok)
        return NULL;
    return cls;
}
