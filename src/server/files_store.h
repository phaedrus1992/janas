/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * files_store.h - the files clients upload (files_store.c): their bytes in
 * a directory of the server's, one file each under its id, and what OpenAI
 * says of each (name, purpose, size, times) beside it. The client's name
 * for a file is only shown, never part of a path.
 */
#ifndef JANAS_SERVER_FILES_STORE_H
#define JANAS_SERVER_FILES_STORE_H

#include <stddef.h>
#include <stdint.h>

struct srv_files;

/* All the files' bytes at most, and the disk they leave free at least. */
#define SRV_FILES_MAX ((uint64_t)16 << 30)
#define SRV_FILES_DISK_FREE ((uint64_t)1 << 30)

struct srv_file {
    char id[48];
    char filename[256];
    char purpose[32];
    uint64_t bytes;
    int64_t created_at, expires_at; /* expires_at 0: never */
};

/* In store_dir/.files (made), with what an earlier run left there read
   back; store_dir NULL: a directory of its own under TMPDIR, removed by
   srv_files_close. NULL with the reason in err. */
struct srv_files *srv_files_open(const char *store_dir, char *err,
                                 size_t err_len);
void srv_files_close(struct srv_files *f);

/* A new file of n bytes: 0 with *out filled, -1 memory or disk (err), -2
   past the limits (err says which). */
int srv_files_add(struct srv_files *f, const char *filename,
                  const char *purpose, const void *data, size_t n,
                  int64_t expires_after, struct srv_file *out, char *err,
                  size_t err_len);
/* A new file of the n files at paths joined in order (an upload's parts),
   which must make `bytes` bytes and, when md5 is not NULL, have that MD5:
   0 with *out filled; -1 disk or memory, -2 past the limits, -3 the size or
   the MD5 not what was said (err says which). */
int srv_files_add_parts(struct srv_files *f, const char *filename,
                        const char *purpose, const char *const *paths, size_t n,
                        uint64_t bytes, const char *md5, int64_t expires_after,
                        struct srv_file *out, char *err, size_t err_len);
/* The directory the files are in (an upload keeps its parts beside). */
const char *srv_files_dir(const struct srv_files *f);
/* 0 with *out, -1 no such file (or expired). */
int srv_files_get(struct srv_files *f, const char *id, struct srv_file *out);
/* 0, -1 no such file. */
int srv_files_delete(struct srv_files *f, const char *id);
/* The file's bytes opened for reading: a descriptor, or -1. */
int srv_files_open_content(struct srv_files *f, const char *id);
/* The files, newest first (asc: oldest first), of purpose (NULL: any),
   after the id after (NULL: from the first): how many, into a malloc'd
   *list (free it). */
size_t srv_files_list(struct srv_files *f, const char *purpose, int asc,
                      const char *after, struct srv_file **list);

#endif
