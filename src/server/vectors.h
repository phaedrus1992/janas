/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vectors.h - vector stores (vectors.c): files cut into overlapping chunks,
 * each with its embedding when the server has an embedding model, searched
 * by the cosine of a query's (or by BM25 over the words, without one). The
 * API is vectors_api.c's; file_search in the Responses API asks the same
 * search.
 */
#ifndef JANAS_SERVER_VECTORS_H
#define JANAS_SERVER_VECTORS_H

#include <stddef.h>
#include <yyjson.h>

struct srv;

/* The directory and the indexing thread (after the files and the engine),
   and their end. */
int srv_vectors_start(struct srv *s);
void srv_vectors_stop(void);
/* The indexing thread has work: a file was attached. */
void srv_vectors_wake(void);
/* A file written aside and renamed: 0, or -1. */
int srv_vectors_write(const char *path, const char *text, size_t n);
/* A file attached's status set (with its last error, code and message, or
   NULL; usage < 0: unchanged): 0, or -1. */
int srv_vectors_set_status(const char *vs, const char *id, const char *status,
                           const char *err_code, const char *err_msg,
                           long long usage);
/* The lock the files' JSON are read and written under. */
void srv_vectors_lock(void);
void srv_vectors_unlock(void);
/* The directory of a store (vs NULL: of them all). */
void srv_vectors_dir(char *out, size_t cap, const char *vs);

struct srv_vhit {
    char file_id[48];
    char filename[256];
    double score;     /* 0 to 1 */
    char *text;       /* the chunk; malloc'd */
    char *attributes; /* the file's, JSON text; malloc'd, or NULL */
};

/* The chunks of store vs that answer the queries best (max at most,
   above threshold, of the files filters lets through - a JSON filter, or
   NULL): how many into *hits (free with srv_vhits_free), or -1 with the
   reason in err (no such store). */
int srv_vectors_search(const char *vs, const char *const *queries, int nq,
                       yyjson_val *filters, int max, double threshold,
                       struct srv_vhit **hits, char *err, size_t err_len);
void srv_vhits_free(struct srv_vhit *h, int n);

#endif
