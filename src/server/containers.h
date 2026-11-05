/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * containers.h - code_interpreter's containers (containers.c): a directory
 * of files, /mnt/data to the code, and a Python session that keeps its
 * variables from one piece of code to the next, in the cage of
 * src/common/sandbox.c. Only with --code-interpreter.
 */
#ifndef JANAS_SERVER_CONTAINERS_H
#define JANAS_SERVER_CONTAINERS_H

#include "server.h"

#include <yyjson.h>

/* What a piece of code did. */
struct srv_cell {
    char *logs;  /* what it printed, and the value of its last expression */
    char *error; /* its traceback, or why it did not run; NULL when fine */
    /* the files it wrote or changed (container file ids and paths under
       /mnt/data), images (the figures it drew) first */
    int n_files, n_images;
    char (*file_id)[48];
    char (*path)[300];
};

void srv_cell_free(struct srv_cell *c);

/* NULL when the server runs code (--code-interpreter, and the cage can be
   built here), else why not. */
const char *srv_containers_off(void);

/* A container for a request's tool: an id given (it must be there and
   running), or {type: auto, file_ids, memory_limit} made now. 0 with its id
   in id, -1 with the reason. */
int srv_container_for(struct srv *s, yyjson_val *spec, char *id, size_t cap,
                      const char **why);

/* code run in the container's session: 0 with out filled (the code's own
   errors in out->error), -1 when the container is not there or running. */
int srv_container_run(struct srv *s, const char *id, const char *code,
                      struct srv_cell *out);

/* A file of the Files API copied into a running container: 0, or -1 with
   the reason. */
int srv_container_add_file(struct srv *s, const char *cid, const char *fid,
                           const char **why);
/* A container file opened for reading (a regular file, links never
   followed), its name (last path part) into name: the fd, or -1. */
int srv_container_file_open(const char *cid, const char *cfid, char *name,
                            size_t cap);

/* A container file's name (its last path part) into name: 0, or -1. */
int srv_container_file_name(struct srv *s, const char *cid, const char *fid,
                            char *name, size_t cap);

#endif
