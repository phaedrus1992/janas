/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * files_store.c - the files clients upload (see files_store.h). On disk,
 * <id> holds the bytes and <id>.meta one line of what is said of them:
 * created_at expires_at bytes purpose, then the name to the end of the
 * line. Both are written aside and renamed, the bytes first, so a file a
 * crash interrupted is either whole or not there. The index is in memory
 * and read back from the .meta files when the server starts.
 */
#define _GNU_SOURCE
#include "files_store.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#ifdef _WIN32
#include <io.h> /* _commit */
#endif

#include "common/md5.h"
#include "server.h"     /* srv_new_id */
#include "store_disk.h" /* srv_disk_name_ok */

struct srv_files {
    pthread_mutex_t mu;
    char dir[4096];
    int temp; /* made for this run: removed at the end */
    struct srv_file *e;
    size_t n, cap;
    uint64_t total;
};

static int path_of(const struct srv_files *f, const char *id, const char *suf,
                   char *out, size_t cap)
{
    if (!srv_disk_name_ok(id))
        return -1;
    int n = snprintf(out, cap, "%s/%s%s", f->dir, id, suf);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static int keep(struct srv_files *f, const struct srv_file *x)
{
    if (f->n == f->cap) {
        size_t cap = f->cap ? 2 * f->cap : 64;
        struct srv_file *t = realloc(f->e, cap * sizeof(*t));
        if (!t)
            return -1;
        f->e = t;
        f->cap = cap;
    }
    f->e[f->n++] = *x;
    f->total += x->bytes;
    return 0;
}

static void forget(struct srv_files *f, size_t i)
{
    char p[4200];
    if (path_of(f, f->e[i].id, ".meta", p, sizeof p) == 0)
        unlink(p);
    if (path_of(f, f->e[i].id, "", p, sizeof p) == 0)
        unlink(p);
    f->total -= f->e[i].bytes;
    memmove(&f->e[i], &f->e[i + 1], (f->n - i - 1) * sizeof(f->e[0]));
    f->n--;
}

/* The files past their time go, under f->mu. */
static void expire(struct srv_files *f)
{
    int64_t now = (int64_t)time(NULL);
    for (size_t i = f->n; i-- > 0;)
        if (f->e[i].expires_at && f->e[i].expires_at <= now)
            forget(f, i);
}

static int by_created(const void *a, const void *b)
{
    const struct srv_file *x = a, *y = b;
    if (x->created_at != y->created_at)
        return x->created_at < y->created_at ? -1 : 1;
    return strcmp(x->id, y->id);
}

static void load(struct srv_files *f)
{
    DIR *d = opendir(f->dir);
    struct dirent *de;
    while (d && (de = readdir(d))) {
        size_t n = strlen(de->d_name);
        char p[4200];
        if (de->d_name[0] == '.') { /* a write that did not finish */
            if (n > 4 && !strcmp(de->d_name + n - 4, ".tmp") &&
                snprintf(p, sizeof p, "%s/%s", f->dir, de->d_name) <
                    (int)sizeof p)
                unlink(p);
            continue;
        }
        if (n <= 5 || strcmp(de->d_name + n - 5, ".meta") != 0 || n - 5 >= 48)
            continue;
        struct srv_file x = {0};
        memcpy(x.id, de->d_name, n - 5);
        struct stat st;
        char line[512];
        FILE *m =
            path_of(f, x.id, ".meta", p, sizeof p) == 0 ? fopen(p, "r") : NULL;
        int ok = m && fgets(line, sizeof line, m) &&
                 sscanf(line, "%" SCNd64 " %" SCNd64 " %" SCNu64 " %31s",
                        &x.created_at, &x.expires_at, &x.bytes, x.purpose) == 4;
        if (m)
            fclose(m);
        char *name = ok ? strchr(line, '\t') : NULL;
        ok = ok && name && path_of(f, x.id, "", p, sizeof p) == 0 &&
             stat(p, &st) == 0 && (uint64_t)st.st_size == x.bytes;
        if (!ok)
            continue; /* left as it is */
        name++;
        name[strcspn(name, "\n")] = 0;
        snprintf(x.filename, sizeof x.filename, "%s", name);
        keep(f, &x);
    }
    if (d)
        closedir(d);
    if (f->n)
        qsort(f->e, f->n, sizeof(f->e[0]), by_created);
    expire(f);
}

struct srv_files *srv_files_open(const char *store_dir, char *err,
                                 size_t err_len)
{
    struct srv_files *f = calloc(1, sizeof(*f));
    if (!f) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    pthread_mutex_init(&f->mu, NULL);
    if (store_dir) {
        snprintf(f->dir, sizeof f->dir, "%s/.files", store_dir);
        if (mkdir(f->dir, 0700) != 0 && errno != EEXIST) {
            snprintf(err, err_len, "cannot make %s: %s", f->dir,
                     strerror(errno));
            srv_files_close(f);
            return NULL;
        }
    } else {
        const char *t = getenv("TMPDIR");
        snprintf(f->dir, sizeof f->dir, "%s/janas-server-files-XXXXXX",
                 t && *t ? t : "/tmp");
        if (!mkdtemp(f->dir)) {
            snprintf(err, err_len, "cannot make a directory for files: %s",
                     strerror(errno));
            f->dir[0] = 0;
            srv_files_close(f);
            return NULL;
        }
        f->temp = 1;
    }
    load(f);
    return f;
}

void srv_files_close(struct srv_files *f)
{
    if (!f)
        return;
    if (f->temp) { /* this run's alone: what is in it goes with it */
        while (f->n)
            forget(f, f->n - 1);
        rmdir(f->dir);
    }
    free(f->e);
    pthread_mutex_destroy(&f->mu);
    free(f);
}

/* buf's n bytes into path through a file aside, renamed: 0, or -1. */
static int write_aside(const struct srv_files *f, const char *id,
                       const char *suf, const void *buf, size_t n)
{
    char tmp[4200], dst[4200];
    if (path_of(f, id, suf, dst, sizeof dst) != 0 ||
        snprintf(tmp, sizeof tmp, "%s/.%s%s.tmp", f->dir, id, suf) >=
            (int)sizeof tmp)
        return -1;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0)
        return -1;
    const char *p = buf;
    size_t done = 0;
    while (done < n) {
        ssize_t w = write(fd, p + done, n - done);
        if (w <= 0) {
            close(fd);
            unlink(tmp);
            return -1;
        }
        done += (size_t)w;
    }
#ifdef _WIN32
    int synced = _commit(fd);
#else
    int synced = fsync(fd);
#endif
    if (synced != 0 || close(fd) != 0 || rename(tmp, dst) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* The name shown: the last part of what the client said, no control
   characters. */
static void shown_name(char *out, size_t cap, const char *filename)
{
    const char *base = filename ? filename : "";
    for (const char *s = base; *s; s++)
        if (*s == '/' || *s == '\\')
            base = s + 1;
    size_t w = 0;
    for (const char *s = base; *s && w + 1 < cap; s++)
        if ((unsigned char)*s >= 0x20 && *s != 0x7f)
            out[w++] = *s;
    out[w] = 0;
}

int srv_files_add(struct srv_files *f, const char *filename,
                  const char *purpose, const void *data, size_t n,
                  int64_t expires_after, struct srv_file *out, char *err,
                  size_t err_len)
{
    struct srv_file x = {0};
    srv_new_id(x.id, sizeof x.id, "file-");
    shown_name(x.filename, sizeof x.filename, filename);
    snprintf(x.purpose, sizeof x.purpose, "%s", purpose);
    x.bytes = n;
    x.created_at = (int64_t)time(NULL);
    x.expires_at = expires_after > 0 ? x.created_at + expires_after : 0;
    pthread_mutex_lock(&f->mu);
    expire(f);
    struct statvfs vs;
    uint64_t avail = statvfs(f->dir, &vs) == 0
                         ? (uint64_t)vs.f_bavail * (uint64_t)vs.f_frsize
                         : UINT64_MAX;
    int rc = 0;
    if (f->total + n > SRV_FILES_MAX) {
        snprintf(err, err_len,
                 "the files kept would pass %" PRIu64 " GiB: delete some",
                 SRV_FILES_MAX >> 30);
        rc = -2;
    } else if (avail < n + SRV_FILES_DISK_FREE) {
        snprintf(err, err_len, "not enough disk for it");
        rc = -2;
    } else {
        char meta[512];
        int mn = snprintf(
            meta, sizeof meta, "%" PRId64 " %" PRId64 " %" PRIu64 " %s\t%s\n",
            x.created_at, x.expires_at, x.bytes, x.purpose, x.filename);
        if (mn <= 0 || (size_t)mn >= sizeof meta ||
            write_aside(f, x.id, "", data, n) != 0 ||
            write_aside(f, x.id, ".meta", meta, (size_t)mn) != 0 ||
            keep(f, &x) != 0) {
            char p[4200];
            if (path_of(f, x.id, "", p, sizeof p) == 0)
                unlink(p);
            if (path_of(f, x.id, ".meta", p, sizeof p) == 0)
                unlink(p);
            snprintf(err, err_len, "cannot write the file");
            rc = -1;
        }
    }
    pthread_mutex_unlock(&f->mu);
    if (rc == 0)
        *out = x;
    return rc;
}

const char *srv_files_dir(const struct srv_files *f)
{
    return f->dir;
}

int srv_files_add_parts(struct srv_files *f, const char *filename,
                        const char *purpose, const char *const *paths, size_t n,
                        uint64_t bytes, const char *md5, int64_t expires_after,
                        struct srv_file *out, char *err, size_t err_len)
{
    struct srv_file x = {0};
    srv_new_id(x.id, sizeof x.id, "file-");
    shown_name(x.filename, sizeof x.filename, filename);
    snprintf(x.purpose, sizeof x.purpose, "%s", purpose);
    x.bytes = bytes;
    x.created_at = (int64_t)time(NULL);
    x.expires_at = expires_after > 0 ? x.created_at + expires_after : 0;
    char tmp[4200], dst[4200], meta[512];
    if (path_of(f, x.id, "", dst, sizeof dst) != 0 ||
        snprintf(tmp, sizeof tmp, "%s/.%s.tmp", f->dir, x.id) >=
            (int)sizeof tmp) {
        snprintf(err, err_len, "cannot write the file");
        return -1;
    }
    pthread_mutex_lock(&f->mu);
    expire(f);
    struct statvfs vs;
    uint64_t avail = statvfs(f->dir, &vs) == 0
                         ? (uint64_t)vs.f_bavail * (uint64_t)vs.f_frsize
                         : UINT64_MAX;
    int rc = 0;
    if (f->total + bytes > SRV_FILES_MAX) {
        snprintf(err, err_len,
                 "the files kept would pass %" PRIu64 " GiB: delete some",
                 SRV_FILES_MAX >> 30);
        rc = -2;
    } else if (avail < bytes + SRV_FILES_DISK_FREE) {
        snprintf(err, err_len, "not enough disk for it");
        rc = -2;
    }
    pthread_mutex_unlock(&f->mu);
    if (rc)
        return rc;
    /* the parts joined outside the lock: gigabytes take a while */
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    struct janas_md5 h;
    janas_md5_init(&h);
    uint64_t done = 0;
    char *buf = malloc((size_t)1 << 20);
    if (!buf)
        rc = -1;
    for (size_t i = 0; fd >= 0 && rc == 0 && i < n; i++) {
        int in = open(paths[i], O_RDONLY | O_CLOEXEC);
        if (in < 0) {
            rc = -1;
            break;
        }
        ssize_t r;
        while (rc == 0 && (r = read(in, buf, (size_t)1 << 20)) > 0) {
            janas_md5_update(&h, buf, (size_t)r);
            done += (uint64_t)r;
            for (ssize_t w = 0; w < r;) {
                ssize_t k = write(fd, buf + w, (size_t)(r - w));
                if (k <= 0) {
                    rc = -1;
                    break;
                }
                w += k;
            }
        }
        close(in);
    }
    free(buf);
    char got[33];
    janas_md5_hex(&h, got);
    if (fd < 0 || rc != 0) {
        snprintf(err, err_len, "cannot write the file");
        rc = -1;
    } else if (done != bytes) {
        snprintf(err, err_len,
                 "the parts make %" PRIu64 " bytes, not the %" PRIu64
                 " the upload was created with",
                 done, bytes);
        rc = -3;
    } else if (md5 && strcasecmp(md5, got) != 0) {
        snprintf(err, err_len, "the parts' MD5 is %s, not %s", got, md5);
        rc = -3;
    }
#ifdef _WIN32
    if (rc == 0 && _commit(fd) != 0)
#else
    if (rc == 0 && fsync(fd) != 0)
#endif
        rc = -1;
    if (fd >= 0 && close(fd) != 0 && rc == 0)
        rc = -1;
    int mn = snprintf(
        meta, sizeof meta, "%" PRId64 " %" PRId64 " %" PRIu64 " %s\t%s\n",
        x.created_at, x.expires_at, x.bytes, x.purpose, x.filename);
    pthread_mutex_lock(&f->mu);
    if (rc == 0 &&
        (mn <= 0 || (size_t)mn >= sizeof meta || rename(tmp, dst) != 0 ||
         write_aside(f, x.id, ".meta", meta, (size_t)mn) != 0 ||
         keep(f, &x) != 0)) {
        unlink(dst);
        snprintf(err, err_len, "cannot write the file");
        rc = -1;
    }
    pthread_mutex_unlock(&f->mu);
    if (rc != 0)
        unlink(tmp);
    else
        *out = x;
    return rc;
}

static long find(struct srv_files *f, const char *id)
{
    for (size_t i = 0; i < f->n; i++)
        if (strcmp(f->e[i].id, id) == 0)
            return (long)i;
    return -1;
}

int srv_files_get(struct srv_files *f, const char *id, struct srv_file *out)
{
    pthread_mutex_lock(&f->mu);
    expire(f);
    long i = find(f, id);
    if (i >= 0)
        *out = f->e[i];
    pthread_mutex_unlock(&f->mu);
    return i >= 0 ? 0 : -1;
}

int srv_files_delete(struct srv_files *f, const char *id)
{
    pthread_mutex_lock(&f->mu);
    expire(f);
    long i = find(f, id);
    if (i >= 0)
        forget(f, (size_t)i);
    pthread_mutex_unlock(&f->mu);
    return i >= 0 ? 0 : -1;
}

int srv_files_open_content(struct srv_files *f, const char *id)
{
    pthread_mutex_lock(&f->mu);
    expire(f);
    char p[4200];
    int fd = find(f, id) >= 0 && path_of(f, id, "", p, sizeof p) == 0
                 ? open(p, O_RDONLY | O_CLOEXEC)
                 : -1;
    pthread_mutex_unlock(&f->mu);
    return fd;
}

size_t srv_files_list(struct srv_files *f, const char *purpose, int asc,
                      const char *after, struct srv_file **list)
{
    *list = NULL;
    pthread_mutex_lock(&f->mu);
    expire(f);
    struct srv_file *out = f->n ? malloc(f->n * sizeof(*out)) : NULL;
    size_t w = 0;
    int past = after == NULL;
    for (size_t k = 0; out && k < f->n; k++) {
        const struct srv_file *x = &f->e[asc ? k : f->n - 1 - k];
        if (!past) {
            past = strcmp(x->id, after) == 0;
            continue;
        }
        if (!purpose || strcmp(x->purpose, purpose) == 0)
            out[w++] = *x;
    }
    pthread_mutex_unlock(&f->mu);
    *list = out;
    return w;
}
