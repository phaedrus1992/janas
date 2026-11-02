/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * assistants.h - what assistants.c (assistants, threads, messages) and
 * runs.c (runs and their steps) share: OpenAI's Assistants API, kept in the
 * store; a thread's messages are the aux text of the thread, a run's steps
 * the aux text of the run.
 */
#ifndef JANAS_SERVER_ASSISTANTS_H
#define JANAS_SERVER_ASSISTANTS_H

#include "server.h"

#include <yyjson.h>

#define AS_ASSISTANT "assistant"
#define AS_THREAD "thread"
#define AS_RUN "thread.run"

/* Every read-change-write of a thread's messages or of a run. */
void as_lock(void);
void as_unlock(void);

/* A document and its aux (either NULL), malloc'd; NULL when not there. */
char *as_get(struct srv *s, const char *kind, const char *id, char **aux);
/* o written as the document of kind and id (aux NULL: unchanged, unless
   fresh): 0 or -1. */
int as_save(struct srv *s, const char *kind, const char *id, yyjson_mut_val *o,
            const char *aux, int fresh);

/* NULL when metadata is absent or at most 16 strings, else the reason. */
const char *as_metadata_error(yyjson_val *md);
/* NULL when tools are ones a run here can use, else the reason. */
const char *as_tools_error(yyjson_val *tools);
/* NULL when tool_resources is one these can keep (vector_stores made into
   vector_store_ids in out, a copy made in d), else the reason. */
const char *as_resources(struct srv *s, yyjson_mut_doc *d, yyjson_val *res,
                         yyjson_mut_val **out);

/* A page of items (an array, oldest first, each with an id) as OpenAI's
   lists answer: limit, order (desc by default), after, before. */
int as_list_reply(struct srv_req *r, yyjson_val *items);

/* A new message of the thread (role, text, attachments as given; the
   assistant and the run that wrote it, or NULL). */
yyjson_mut_val *as_message(yyjson_mut_doc *d, const char *thread,
                           const char *role, const char *text,
                           yyjson_val *attachments, yyjson_val *metadata,
                           const char *assistant, const char *run);
/* m appended to the thread's messages (as_lock held): 0, -1 none. */
int as_message_append(struct srv *s, const char *thread, yyjson_mut_val *m);
/* The text of a message given by a client (a string, or text parts), or
   NULL with the reason; malloc'd. */
char *as_message_text(yyjson_val *content, const char **why);
/* The files of attachments with file_search added to the thread's vector
   store (made when the thread has none): NULL, or the reason. */
const char *as_attach(struct srv *s, const char *thread, yyjson_val *att);

/* The documents of a kind as an array, oldest first (those whose key has
   value when key is given). */
yyjson_doc *as_all(struct srv *s, const char *kind, const char *key,
                   const char *value);
/* A thread made of a body {messages, tool_resources, metadata} (root may
   be NULL): NULL with its id, or the reason. */
const char *as_thread_make(struct srv *s, yyjson_val *root, char *id,
                           size_t cap);

/* The run of the thread still going (queued, in_progress, requires_action,
   cancelling): 1 with its id, 0 none. */
int as_active_run(struct srv *s, const char *thread, char *id, size_t cap);

/* runs_stream.c: the events of runs for the clients streaming them. A
   feed opened for a run gets every event emitted for it until done. */
struct as_feed;
struct as_feed *as_feed_open(const char *run);
void as_feed_close(void *f); /* a feed not handed to as_feed_reply */
int as_feed_reply(struct srv_req *r, struct as_feed *f);
void as_emit(const char *run, const char *event, yyjson_mut_val *v);
void as_emit_done(const char *run);
void as_feeds_end(void); /* at stop: every feed ended */

/* runs.c: resumes the runs a stopped server left going, and waits at stop
   for those running. */
void srv_runs_start(struct srv *s);
void srv_runs_stop(void);

#endif
