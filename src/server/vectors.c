/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vectors.c - the vector stores' data, indexing and search (see vectors.h).
 *
 * On disk, beside the uploaded files (.vectors in their directory), a
 * store is a directory: meta.json the store as OpenAI writes it, files/
 * one JSON object for each file attached (its status, attributes, chunking
 * strategy), and for each file indexed <file>.chunks, its chunks as JSON
 * lines, and <file>.vec, their embeddings (a 4-byte dimension, then the
 * floats). A file is cut at whitespace into chunks of about the tokens the
 * strategy says (four bytes a token), each overlapping the one before.
 *
 * A thread indexes the files waiting, one at a time and behind every
 * client's request (srv_engine_low): their chunks' embeddings come from
 * the same handler as /v1/embeddings, called from inside. Without an
 * embedding model the chunks are kept alone, and the search reads the
 * words (BM25) instead of the meaning.
 */
#include "vectors.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "files_store.h"
#include "llm/json.h"
#include "server.h"
#include "store_disk.h"

#define EMBED_BATCH 16
#define TEXT_MAX ((size_t)32 << 20) /* a file indexed at most */

static struct srv *server;
static char root[4200];
static pthread_t worker;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int running, stopping, kicked;

void srv_vectors_dir(char *out, size_t cap, const char *vs)
{
    if (vs)
        snprintf(out, cap, "%s/%s", root, vs);
    else
        snprintf(out, cap, "%s", root);
}

/* ---- small file helpers ---- */

static char *read_text(const char *path, size_t *n)
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
        if (n)
            *n = (size_t)len;
    }
    return b;
}

int srv_vectors_write(const char *path, const char *text, size_t n)
{
    char tmp[4400];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return -1;
    int ok = fwrite(text, 1, n, f) == n;
    ok = !fclose(f) && ok;
    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* ---- chunks ---- */

struct chunks {
    char **t;
    size_t n, cap;
};

static int chunk_add(struct chunks *c, const char *s, size_t n)
{
    if (c->n == c->cap) {
        size_t cap = c->cap ? 2 * c->cap : 64;
        char **t = realloc(c->t, cap * sizeof(*t));
        if (!t)
            return -1;
        c->t = t;
        c->cap = cap;
    }
    char *x = malloc(n + 1);
    if (!x)
        return -1;
    memcpy(x, s, n);
    x[n] = 0;
    c->t[c->n++] = x;
    return 0;
}

static void chunks_free(struct chunks *c)
{
    for (size_t i = 0; i < c->n; i++)
        free(c->t[i]);
    free(c->t);
    memset(c, 0, sizeof(*c));
}

/* The text cut into chunks of about size bytes, each beginning overlap
   bytes before the last one ended, at whitespace and at the start of a
   UTF-8 character. */
static int cut(const char *t, size_t n, size_t size, size_t overlap,
               struct chunks *out)
{
    size_t at = 0;
    while (at < n) {
        size_t end = at + size < n ? at + size : n;
        if (end < n) { /* back to a space, not further than half */
            size_t e = end;
            while (e > at + size / 2 && !isspace((unsigned char)t[e]))
                e--;
            if (e > at + size / 2)
                end = e;
            while (end < n && ((unsigned char)t[end] & 0xc0) == 0x80)
                end++;
        }
        if (chunk_add(out, t + at, end - at) != 0)
            return -1;
        if (end >= n)
            break;
        size_t next = end > overlap ? end - overlap : end;
        if (next <= at)
            next = end;
        while (next < end && !isspace((unsigned char)t[next]))
            next++; /* begin at a word */
        while (next < end && ((unsigned char)t[next] & 0xc0) == 0x80)
            next++;
        at = next;
    }
    return 0;
}

/* Text a file holds: no NUL bytes, mostly valid UTF-8. */
static int is_text(const char *t, size_t n)
{
    if (memchr(t, 0, n))
        return 0;
    size_t bad = 0;
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)t[i];
        size_t k = c < 0x80         ? 1
                   : (c >> 5) == 6  ? 2
                   : (c >> 4) == 14 ? 3
                   : (c >> 3) == 30 ? 4
                                    : 0;
        if (!k || i + k > n) {
            bad++;
            i++;
            continue;
        }
        for (size_t j = 1; j < k; j++)
            if (((unsigned char)t[i + j] & 0xc0) != 0x80) {
                bad++;
                k = 1;
                break;
            }
        i += k;
    }
    return bad * 100 <= n; /* a byte in a hundred at most */
}

/* ---- embeddings, through the server's own handler ---- */

/* The embeddings of n texts (prefix before each, may be ""), into a
   malloc'd [n][*dim]: 0, 1 when there is no embedding model, -1 on an
   error. */
static int embed(const char *const *texts, size_t n, const char *prefix,
                 float **out, int *dim)
{
    *out = NULL;
    srv_handler h = srv_route_find("POST", "/embeddings");
    if (!h)
        return -1;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d), *in = yyjson_mut_arr(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "model", "embedding");
    for (size_t i = 0; i < n; i++) {
        size_t pn = strlen(prefix), tn = strlen(texts[i]);
        char *s = malloc(pn + tn + 1);
        if (s) {
            memcpy(s, prefix, pn);
            memcpy(s + pn, texts[i], tn + 1);
            yyjson_mut_arr_add_strcpy(d, in, s);
            free(s);
        }
    }
    yyjson_mut_obj_add_val(d, o, "input", in);
    size_t bn = 0;
    char *body = yyjson_mut_write(d, 0, &bn);
    yyjson_mut_doc_free(d);
    struct srv_capture cap = {0};
    struct srv_req r = {.srv = server,
                        .cap = &cap,
                        .method = "POST",
                        .path = "/embeddings",
                        .body = body ? body : "{}",
                        .n_body = body ? bn : 2};
    h(&r);
    free(body);
    int rc = -1;
    if (cap.status == 200 && cap.body) {
        yyjson_doc *ad = yyjson_read(cap.body, cap.n, 0);
        yyjson_val *data = yyjson_obj_get(yyjson_doc_get_root(ad), "data");
        size_t k = 0, idx, mx;
        yyjson_val *it;
        int dm = 0;
        float *v = NULL;
        yyjson_arr_foreach(data, idx, mx, it)
        {
            yyjson_val *e = yyjson_obj_get(it, "embedding");
            if (!dm) {
                dm = (int)yyjson_arr_size(e);
                v = dm ? malloc(n * (size_t)dm * sizeof(float)) : NULL;
                if (!v)
                    break;
            }
            if ((int)yyjson_arr_size(e) != dm || k >= n)
                break;
            size_t j, m;
            yyjson_val *x;
            yyjson_arr_foreach(e, j, m, x) v[k * (size_t)dm + j] =
                (float)yyjson_get_num(x);
            k++;
        }
        yyjson_doc_free(ad);
        if (v && k == n) {
            *out = v;
            *dim = dm;
            rc = 0;
        } else {
            free(v);
        }
    } else if (cap.body && strstr(cap.body, "embedding model")) {
        rc = 1; /* none: the search goes by the words */
    }
    free(cap.body);
    return rc;
}

/* What makes a query read as one by an instructed embedding model
   (Qwen3-Embedding); the chunks go as they are. */
#define QUERY_PREFIX                                                           \
    "Instruct: Given a search query, retrieve relevant passages that "         \
    "answer the query\nQuery: "

/* ---- indexing ---- */

static void file_path(char *out, size_t cap, const char *vs, const char *id,
                      const char *suf)
{
    snprintf(out, cap, "%s/%s/%s%s", root, vs, id, suf);
}

static void meta_path(char *out, size_t cap, const char *vs, const char *id)
{
    snprintf(out, cap, "%s/%s/files/%s.json", root, vs, id);
}

/* A file attached's JSON with status and the rest changed: 0, or -1. */
int srv_vectors_set_status(const char *vs, const char *id, const char *status,
                           const char *err_code, const char *err_msg,
                           long long usage)
{
    char p[4500];
    meta_path(p, sizeof p, vs, id);
    pthread_mutex_lock(&mu);
    size_t n = 0;
    char *t = read_text(p, &n);
    yyjson_doc *d = t ? yyjson_read(t, n, 0) : NULL;
    free(t);
    int rc = -1;
    if (d) {
        yyjson_mut_doc *m = yyjson_doc_mut_copy(d, NULL);
        yyjson_mut_val *o = yyjson_mut_doc_get_root(m);
        yyjson_mut_obj_remove_key(o, "status");
        yyjson_mut_obj_add_strcpy(m, o, "status", status);
        yyjson_mut_obj_remove_key(o, "last_error");
        if (err_code) {
            yyjson_mut_val *e = yyjson_mut_obj_add_obj(m, o, "last_error");
            yyjson_mut_obj_add_strcpy(m, e, "code", err_code);
            yyjson_mut_obj_add_strcpy(m, e, "message", err_msg);
        } else {
            yyjson_mut_obj_add_null(m, o, "last_error");
        }
        if (usage >= 0) {
            yyjson_mut_obj_remove_key(o, "usage_bytes");
            yyjson_mut_obj_add_int(m, o, "usage_bytes", usage);
        }
        size_t wn;
        char *w = yyjson_mut_write(m, 0, &wn);
        rc = w ? srv_vectors_write(p, w, wn) : -1;
        free(w);
        yyjson_mut_doc_free(m);
        yyjson_doc_free(d);
    }
    pthread_mutex_unlock(&mu);
    return rc;
}

static const char *status_of(const char *vs, const char *id, char *buf,
                             size_t cap)
{
    char p[4500];
    meta_path(p, sizeof p, vs, id);
    pthread_mutex_lock(&mu);
    size_t n = 0;
    char *t = read_text(p, &n);
    pthread_mutex_unlock(&mu);
    yyjson_doc *d = t ? yyjson_read(t, n, 0) : NULL;
    free(t);
    const char *s =
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(d), "status"));
    snprintf(buf, cap, "%s", s ? s : "");
    yyjson_doc_free(d);
    return buf;
}

/* The chunking of a file attached: its static strategy's sizes. */
static void sizes(const char *vs, const char *id, size_t *size, size_t *over)
{
    *size = 800, *over = 400;
    char p[4500];
    meta_path(p, sizeof p, vs, id);
    pthread_mutex_lock(&mu);
    size_t n = 0;
    char *t = read_text(p, &n);
    pthread_mutex_unlock(&mu);
    yyjson_doc *d = t ? yyjson_read(t, n, 0) : NULL;
    free(t);
    yyjson_val *st = yyjson_obj_get(
        yyjson_obj_get(yyjson_doc_get_root(d), "chunking_strategy"), "static");
    if (st) {
        *size = (size_t)yyjson_get_uint(
            yyjson_obj_get(st, "max_chunk_size_tokens"));
        *over =
            (size_t)yyjson_get_uint(yyjson_obj_get(st, "chunk_overlap_tokens"));
    }
    yyjson_doc_free(d);
    if (*size < 100)
        *size = 800;
    if (*over > *size / 2)
        *over = *size / 2;
    *size *= 4; /* tokens to bytes, about */
    *over *= 4;
}

static void index_file(const char *vs, const char *id)
{
    char buf[32];
    struct srv_file f;
    if (srv_files_get(server->files, id, &f) != 0) {
        srv_vectors_set_status(vs, id, "failed", "invalid_file",
                               "The file is not there any more.", 0);
        return;
    }
    if (f.bytes > TEXT_MAX) {
        srv_vectors_set_status(vs, id, "failed", "invalid_file",
                               "The file is larger than 32 MiB.", 0);
        return;
    }
    int fd = srv_files_open_content(server->files, id);
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
        srv_vectors_set_status(vs, id, "failed", "server_error",
                               "The file could not be read.", 0);
        return;
    }
    t[got] = 0;
    if (!is_text(t, got)) {
        free(t);
        srv_vectors_set_status(vs, id, "failed", "unsupported_file",
                               "Only text files are read (text, Markdown, "
                               "code, JSON, CSV).",
                               0);
        return;
    }
    size_t size, over;
    sizes(vs, id, &size, &over);
    struct chunks c = {0};
    int ok = cut(t, got, size, over, &c) == 0;
    free(t);
    /* the chunks as JSON lines */
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    struct janas_buf lines = {0};
    for (size_t i = 0; ok && i < c.n; i++) {
        yyjson_mut_val *s = yyjson_mut_str(d, c.t[i]);
        size_t ln;
        char *l = yyjson_mut_val_write(s, 0, &ln);
        if (l) {
            janas_buf_put(&lines, l, ln);
            janas_buf_put(&lines, "\n", 1);
        }
        free(l);
    }
    yyjson_mut_doc_free(d);
    char cp[4500], vp[4500];
    file_path(cp, sizeof cp, vs, id, ".chunks");
    file_path(vp, sizeof vp, vs, id, ".vec");
    ok = ok && !lines.oom &&
         srv_vectors_write(cp, lines.p ? lines.p : "", lines.n) == 0;
    janas_buf_free(&lines);
    /* their embeddings, a few at a time, stopping if the file is taken
       away meanwhile */
    float *all = NULL;
    int dim = 0, none = 0;
    for (size_t i = 0; ok && !none && i < c.n; i += EMBED_BATCH) {
        if (strcmp(status_of(vs, id, buf, sizeof buf), "in_progress") != 0 ||
            stopping) {
            ok = 0;
            break;
        }
        size_t k = c.n - i < EMBED_BATCH ? c.n - i : EMBED_BATCH;
        float *v;
        int dm;
        int rc = embed((const char *const *)c.t + i, k, "", &v, &dm);
        if (rc == 1) {
            none = 1; /* no embedding model: the words alone */
            break;
        }
        if (rc != 0 || (dim && dm != dim)) {
            ok = 0;
            break;
        }
        float *a = realloc(all, (i + k) * (size_t)dm * sizeof(float));
        if (!a) {
            free(v);
            ok = 0;
            break;
        }
        all = a;
        memcpy(all + i * (size_t)dm, v, k * (size_t)dm * sizeof(float));
        dim = dm;
        free(v);
    }
    if (ok && !none && c.n) {
        size_t bytes = 4 + c.n * (size_t)dim * sizeof(float);
        char *vb = malloc(bytes);
        if (vb) {
            int32_t d32 = dim;
            memcpy(vb, &d32, 4);
            memcpy(vb + 4, all, bytes - 4);
            ok = srv_vectors_write(vp, vb, bytes) == 0;
            free(vb);
        } else {
            ok = 0;
        }
    } else if (none) {
        unlink(vp);
    }
    free(all);
    size_t usage = 0;
    for (size_t i = 0; i < c.n; i++)
        usage += strlen(c.t[i]);
    chunks_free(&c);
    if (stopping && !ok)
        return; /* left in progress: indexed again at the next start */
    if (strcmp(status_of(vs, id, buf, sizeof buf), "in_progress") != 0)
        return; /* cancelled or detached meanwhile */
    if (ok)
        srv_vectors_set_status(vs, id, "completed", NULL, NULL,
                               (long long)usage);
    else
        srv_vectors_set_status(vs, id, "failed", "server_error",
                               "The chunks could not be embedded.", 0);
}

/* The oldest file waiting, of any store: 1 with vs and id, or 0. */
static int next_file(char *vs, size_t vcap, char *id, size_t icap)
{
    DIR *d = opendir(root);
    struct dirent *e;
    long long best = -1;
    while (d && (e = readdir(d))) {
        if (!srv_disk_name_ok(e->d_name))
            continue;
        char fd[4500];
        snprintf(fd, sizeof fd, "%s/%s/files", root, e->d_name);
        DIR *f = opendir(fd);
        struct dirent *g;
        while (f && (g = readdir(f))) {
            size_t n = strlen(g->d_name);
            if (n < 6 || strcmp(g->d_name + n - 5, ".json") != 0)
                continue;
            char p[4800];
            snprintf(p, sizeof p, "%s/%s", fd, g->d_name);
            size_t len;
            char *t = read_text(p, &len);
            yyjson_doc *jd = t ? yyjson_read(t, len, 0) : NULL;
            free(t);
            yyjson_val *o = yyjson_doc_get_root(jd);
            const char *st = yyjson_get_str(yyjson_obj_get(o, "status"));
            long long at = yyjson_get_sint(yyjson_obj_get(o, "created_at"));
            size_t vn = strlen(e->d_name);
            if (st && !strcmp(st, "in_progress") && (best < 0 || at < best) &&
                n - 5 < icap && vn < vcap) {
                best = at;
                memcpy(vs, e->d_name, vn + 1);
                memcpy(id, g->d_name, n - 5);
                id[n - 5] = 0;
            }
            yyjson_doc_free(jd);
        }
        if (f)
            closedir(f);
    }
    if (d)
        closedir(d);
    return best >= 0;
}

static void *worker_main(void *arg)
{
    (void)arg;
    srv_engine_low = 1; /* behind every client's request */
    char vs[64], id[64];
    for (;;) {
        pthread_mutex_lock(&mu);
        int stop = stopping;
        kicked = 0;
        pthread_mutex_unlock(&mu);
        if (stop)
            break;
        if (next_file(vs, sizeof vs, id, sizeof id)) {
            index_file(vs, id);
            continue;
        }
        pthread_mutex_lock(&mu); /* nothing waiting: until woken */
        while (!stopping && !kicked)
            pthread_cond_wait(&cv, &mu);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

void srv_vectors_lock(void)
{
    pthread_mutex_lock(&mu);
}

void srv_vectors_unlock(void)
{
    pthread_mutex_unlock(&mu);
}

void srv_vectors_wake(void)
{
    pthread_mutex_lock(&mu);
    kicked = 1;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
}

int srv_vectors_start(struct srv *s)
{
    server = s;
    snprintf(root, sizeof root, "%s/.vectors", srv_files_dir(s->files));
    if (mkdir(root, 0700) != 0 && errno != EEXIST)
        return -1;
    stopping = 0;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0)
        return -1;
    running = 1;
    return 0;
}

void srv_vectors_stop(void)
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

/* ---- search ---- */

/* Whether a file's attributes pass a filter (comparison or compound). */
static int passes(yyjson_val *attrs, yyjson_val *f)
{
    if (!f || yyjson_is_null(f))
        return 1;
    const char *type = yyjson_get_str(yyjson_obj_get(f, "type"));
    if (!type)
        return 1;
    if (!strcmp(type, "and") || !strcmp(type, "or")) {
        int and_ = !strcmp(type, "and");
        size_t i, m;
        yyjson_val *x;
        yyjson_arr_foreach(yyjson_obj_get(f, "filters"), i, m, x)
        {
            int p = passes(attrs, x);
            if (and_ && !p)
                return 0;
            if (!and_ && p)
                return 1;
        }
        return and_;
    }
    const char *key = yyjson_get_str(yyjson_obj_get(f, "key"));
    yyjson_val *want = yyjson_obj_get(f, "value");
    yyjson_val *have = key ? yyjson_obj_get(attrs, key) : NULL;
    if (!strcmp(type, "in") || !strcmp(type, "nin")) {
        int in = 0;
        size_t i, m;
        yyjson_val *x;
        yyjson_arr_foreach(want, i, m, x) in |= have && yyjson_equals(x, have);
        return !strcmp(type, "in") ? in : !in;
    }
    if (!have)
        return !strcmp(type, "ne");
    if (!strcmp(type, "eq"))
        return yyjson_equals(have, want);
    if (!strcmp(type, "ne"))
        return !yyjson_equals(have, want);
    if (yyjson_is_num(have) && yyjson_is_num(want)) {
        double a = yyjson_get_num(have), b = yyjson_get_num(want);
        return !strcmp(type, "gt")    ? a > b
               : !strcmp(type, "gte") ? a >= b
               : !strcmp(type, "lt")  ? a < b
               : !strcmp(type, "lte") ? a <= b
                                      : 0;
    }
    if (yyjson_is_str(have) && yyjson_is_str(want)) {
        int c = strcmp(yyjson_get_str(have), yyjson_get_str(want));
        return !strcmp(type, "gt")    ? c > 0
               : !strcmp(type, "gte") ? c >= 0
               : !strcmp(type, "lt")  ? c < 0
               : !strcmp(type, "lte") ? c <= 0
                                      : 0;
    }
    return 0;
}

/* Words of a text, lower case ASCII letters and digits, and every byte of
   a UTF-8 character: for BM25. */
static int next_word(const char **p, char *w, size_t cap)
{
    const unsigned char *s = (const unsigned char *)*p;
    while (*s && !(isalnum(*s) || *s >= 0x80))
        s++;
    if (!*s)
        return 0;
    size_t n = 0;
    while (*s && (isalnum(*s) || *s >= 0x80)) {
        if (n + 1 < cap)
            w[n++] = (char)tolower(*s);
        s++;
    }
    w[n] = 0;
    *p = (const char *)s;
    return 1;
}

/* Words too common to tell chunks apart (no idf: a store is read once a
   query), English and Italian */
static int stopword(const char *w)
{
    static const char *const S[] = {
        "the",   "and",    "for",   "with",  "how",  "what",  "who",   "which",
        "that",  "this",   "are",   "was",   "were", "from",  "into",  "about",
        "does",  "can",    "you",   "your",  "its",  "his",   "her",   "their",
        "not",   "but",    "have",  "has",   "had",  "will",  "would", "should",
        "could", "there",  "where", "when",  "why",  "all",   "any",   "some",
        "one",   "two",    "also",  "per",   "che",  "chi",   "con",   "come",
        "dove",  "quando", "del",   "della", "dei",  "delle", "degli", "nel",
        "nella", "una",    "uno",   "sono",  "non",  "gli",   "le",    "il",
        "la",    "di",     "da",    "in",    "su",   "tra",   "fra",   "anche"};
    for (size_t i = 0; i < sizeof(S) / sizeof(S[0]); i++)
        if (!strcmp(w, S[i]))
            return 1;
    return 0;
}

/* BM25's term frequencies against the chunk's length */
static double bm25(const char *query, const char *chunk, double avg_len)
{
    double len = 0, score = 0;
    const char *p = chunk;
    char w[64];
    while (next_word(&p, w, sizeof w))
        len++;
    const char *q = query;
    char qw[64];
    while (next_word(&q, qw, sizeof qw)) {
        if (strlen(qw) < 3 || stopword(qw))
            continue;
        double tf = 0;
        p = chunk;
        while (next_word(&p, w, sizeof w))
            tf += !strcmp(w, qw);
        double k1 = 1.2, b = 0.75;
        score += tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len));
    }
    return score;
}

static int by_score(const void *a, const void *b)
{
    const struct srv_vhit *x = a, *y = b;
    return x->score < y->score ? 1 : x->score > y->score ? -1 : 0;
}

void srv_vhits_free(struct srv_vhit *h, int n)
{
    for (int i = 0; h && i < n; i++) {
        free(h[i].text);
        free(h[i].attributes);
    }
    free(h);
}

int srv_vectors_search(const char *vs, const char *const *queries, int nq,
                       yyjson_val *filters, int max, double threshold,
                       struct srv_vhit **hits, char *err, size_t err_len)
{
    *hits = NULL;
    char dir[4500], meta[4600];
    if (!srv_disk_name_ok(vs)) {
        snprintf(err, err_len, "no vector store %s", vs);
        return -1;
    }
    srv_vectors_dir(dir, sizeof dir, vs);
    snprintf(meta, sizeof meta, "%s/meta.json", dir);
    if (access(meta, F_OK) != 0) {
        snprintf(err, err_len, "no vector store %s", vs);
        return -1;
    }
    /* the queries' embeddings, when the model is there */
    float *qv = NULL;
    int qdim = 0;
    if (embed(queries, (size_t)nq, QUERY_PREFIX, &qv, &qdim) != 0)
        qv = NULL;
    struct srv_vhit *all = NULL;
    size_t n = 0, cap = 0;
    char fdir[4600];
    snprintf(fdir, sizeof fdir, "%s/files", dir);
    DIR *d = opendir(fdir);
    struct dirent *e;
    while (d && (e = readdir(d))) {
        size_t nl = strlen(e->d_name);
        if (nl < 6 || strcmp(e->d_name + nl - 5, ".json") != 0 || nl - 5 >= 48)
            continue;
        char id[48];
        memcpy(id, e->d_name, nl - 5);
        id[nl - 5] = 0;
        char p[5000];
        snprintf(p, sizeof p, "%s/%.255s", fdir, e->d_name);
        size_t len;
        pthread_mutex_lock(&mu);
        char *t = read_text(p, &len);
        pthread_mutex_unlock(&mu);
        yyjson_doc *jd = t ? yyjson_read(t, len, 0) : NULL;
        free(t);
        yyjson_val *o = yyjson_doc_get_root(jd);
        const char *st = yyjson_get_str(yyjson_obj_get(o, "status"));
        yyjson_val *attrs = yyjson_obj_get(o, "attributes");
        if (!st || strcmp(st, "completed") != 0 || !passes(attrs, filters)) {
            yyjson_doc_free(jd);
            continue;
        }
        struct srv_file f;
        const char *fname =
            srv_files_get(server->files, id, &f) == 0 ? f.filename : id;
        char *attr_text = attrs ? yyjson_val_write(attrs, 0, NULL) : NULL;
        yyjson_doc_free(jd);
        /* the chunks, and their vectors if any */
        char cp[4500], vp[4500];
        file_path(cp, sizeof cp, vs, id, ".chunks");
        file_path(vp, sizeof vp, vs, id, ".vec");
        size_t cn = 0, vn = 0;
        char *ct = read_text(cp, &cn), *vt = qv ? read_text(vp, &vn) : NULL;
        int32_t dim = 0;
        if (vt && vn >= 4)
            memcpy(&dim, vt, 4);
        int use_vec = vt && dim == qdim && dim > 0;
        /* chunk lengths for BM25's average */
        double avg = 0;
        long count = 0;
        for (char *l = ct; l && *l;) {
            char *nlp = strchr(l, '\n');
            avg += nlp ? (double)(nlp - l) / 6 : 0;
            count++;
            l = nlp ? nlp + 1 : l + strlen(l);
        }
        avg = count ? avg / (double)count : 1;
        long k = 0;
        for (char *l = ct; l && *l; k++) {
            char *nlp = strchr(l, '\n');
            if (nlp)
                *nlp = 0;
            yyjson_doc *cd = yyjson_read(l, strlen(l), 0);
            const char *chunk = yyjson_get_str(yyjson_doc_get_root(cd));
            double s = 0;
            if (chunk && use_vec &&
                (size_t)(k + 1) * (size_t)dim * 4 + 4 <= vn) {
                const float *v = (const float *)(void *)(vt + 4) + k * dim;
                for (int q = 0; q < nq; q++) { /* the best of the queries */
                    double dot = 0, a = 0, b = 0;
                    for (int j = 0; j < dim; j++) {
                        dot += (double)v[j] * qv[(size_t)q * qdim + j];
                        a += (double)v[j] * v[j];
                        b += (double)qv[(size_t)q * qdim + j] *
                             qv[(size_t)q * qdim + j];
                    }
                    double c = a > 0 && b > 0 ? dot / sqrt(a * b) : 0;
                    if (c > s)
                        s = c;
                }
            } else if (chunk) {
                for (int q = 0; q < nq; q++) {
                    double b = bm25(queries[q], chunk, avg > 1 ? avg : 1);
                    double c = b / (b + 4.0); /* 0 to 1 */
                    if (c > s)
                        s = c;
                }
            }
            if (chunk && s > threshold && s > 0) {
                if (n == cap) {
                    size_t nc = cap ? 2 * cap : 64;
                    struct srv_vhit *x = realloc(all, nc * sizeof(*x));
                    if (!x) {
                        yyjson_doc_free(cd);
                        break;
                    }
                    all = x;
                    cap = nc;
                }
                struct srv_vhit *h = &all[n++];
                memset(h, 0, sizeof(*h));
                snprintf(h->file_id, sizeof h->file_id, "%s", id);
                snprintf(h->filename, sizeof h->filename, "%s", fname);
                h->score = s > 1 ? 1 : s;
                h->text = strdup(chunk);
                h->attributes = attr_text ? strdup(attr_text) : NULL;
            }
            yyjson_doc_free(cd);
            l = nlp ? nlp + 1 : l + strlen(l);
        }
        free(ct);
        free(vt);
        free(attr_text);
    }
    if (d)
        closedir(d);
    free(qv);
    if (n)
        qsort(all, n, sizeof(*all), by_score);
    for (size_t i = (size_t)max; i < n; i++) {
        free(all[i].text);
        free(all[i].attributes);
    }
    *hits = all;
    return (int)(n < (size_t)max ? n : (size_t)max);
}
