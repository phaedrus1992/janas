/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * runs_stream.c - the events of a run of the Assistants API, for the
 * clients that asked for them (stream: true): a feed per client, opened
 * before the run is started so nothing is missed, filled by runs.c as the
 * run goes (thread.run.*, thread.run.step.*, thread.message.*, the text as
 * the model writes it) and ended with done when the run ends or waits for
 * tool outputs.
 */
#include "assistants.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/json.h"

struct as_feed {
    char run[64];
    struct janas_buf b; /* the events not yet sent */
    size_t off;
    int ended;
    struct as_feed *next;
};

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static struct as_feed *feeds;

struct as_feed *as_feed_open(const char *run)
{
    struct as_feed *f = calloc(1, sizeof(*f));
    if (!f)
        return NULL;
    snprintf(f->run, sizeof f->run, "%s", run);
    pthread_mutex_lock(&mu);
    f->next = feeds;
    feeds = f;
    pthread_mutex_unlock(&mu);
    return f;
}

static void put(const char *run, const char *text, size_t n, int end)
{
    pthread_mutex_lock(&mu);
    for (struct as_feed *f = feeds; f; f = f->next)
        if (!f->ended && (!run || !strcmp(f->run, run))) {
            janas_buf_put(&f->b, text, n);
            f->ended = end;
        }
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
}

void as_emit(const char *run, const char *event, yyjson_mut_val *v)
{
    size_t n = 0;
    char *json = v ? yyjson_mut_val_write(v, 0, &n) : NULL;
    if (!json)
        return;
    size_t cap = strlen(event) + n + 32;
    char *t = malloc(cap);
    if (t) {
        int k = snprintf(t, cap, "event: %s\ndata: %s\n\n", event, json);
        put(run, t, (size_t)k, 0);
    }
    free(t);
    free(json);
}

void as_emit_done(const char *run)
{
    static const char done[] = "event: done\ndata: [DONE]\n\n";
    put(run, done, sizeof done - 1, 1);
}

void as_feeds_end(void)
{
    as_emit_done(NULL);
}

static long next(void *ctx, char *buf, size_t cap)
{
    struct as_feed *f = ctx;
    pthread_mutex_lock(&mu);
    if (f->off == f->b.n && !f->ended) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 200 * 1000000L;
        if (ts.tv_nsec >= 1000000000L)
            ts.tv_sec++, ts.tv_nsec -= 1000000000L;
        pthread_cond_timedwait(&cv, &mu, &ts);
    }
    long k = 0;
    if (f->off < f->b.n) {
        size_t n = f->b.n - f->off < cap ? f->b.n - f->off : cap;
        memcpy(buf, f->b.p + f->off, n);
        f->off += n;
        k = (long)n;
    } else if (f->ended) {
        k = -1;
    }
    pthread_mutex_unlock(&mu);
    return k;
}

void as_feed_close(void *ctx)
{
    struct as_feed *f = ctx;
    if (!f)
        return;
    pthread_mutex_lock(&mu);
    for (struct as_feed **p = &feeds; *p; p = &(*p)->next)
        if (*p == f) {
            *p = f->next;
            break;
        }
    pthread_mutex_unlock(&mu);
    janas_buf_free(&f->b);
    free(f);
}

int as_feed_reply(struct srv_req *r, struct as_feed *f)
{
    return srv_reply_stream(r, next, as_feed_close, f);
}
