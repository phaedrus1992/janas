/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * steps.c - janas-get's steps (see steps.h). The converters are the
 * programs of the same build, run as they are: what they write is what
 * MODELS.md fingerprints, whoever runs them.
 */
#define _GNU_SOURCE
#include "steps.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/hf_fetch.h"
#include "common/sha256.h"
#include "common/proc.h"
#include "common/sysinfo.h"

void get_local(char *out, size_t len, const char *dir, const char *path)
{
    const char *base = strrchr(path, '/');
    snprintf(out, len, "%s/%s", dir, base ? base + 1 : path);
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

struct bar {
    double t0;
    uint64_t from;
};

static void progress(uint64_t done, uint64_t size, void *arg)
{
    struct bar *b = arg;
    double dt = now() - b->t0;
    double mbs = dt > 0 ? (double)(done - b->from) / dt / 1e6 : 0;
    fprintf(stderr, "\r   %.2f of %.2f GB (%.0f%%), %.0f MB/s    ",
            (double)done / 1e9, (double)size / 1e9,
            size ? 100.0 * (double)done / (double)size : 100.0, mbs);
    if (done >= size)
        fputc('\n', stderr);
}

int get_same(const char *file, const char *want)
{
    char got[65];
    if (janas_sha256_file(file, got) != 0)
        return -1;
    return strcmp(got, want) == 0;
}

int get_fetch(const char *repo, const char *path, const char *dir,
              const char *want, int quiet)
{
    char local[4096], err[512], listed[65], got[65];
    get_local(local, sizeof(local), dir, path);
    uint64_t size = 0;
    if (janas_hf_info(repo, NULL, path, &size, listed, err, sizeof(err)) != 0) {
        if (!quiet)
            fprintf(stderr, "janas-get: %s\n", err);
        return -2;
    }
    /* the repository may have been changed since MODELS.md was written:
       then its file is not the one the fingerprints are of */
    if (want && listed[0] && strcmp(listed, want) != 0) {
        fprintf(stderr,
                "janas-get: %s has changed on Hugging Face since MODELS.md "
                "was written (SHA-256 %s, not %s); not downloaded\n",
                path, listed, want);
        return -1;
    }
    printf("   %s from %s, %.2f GB\n", path, repo, (double)size / 1e9);
    fflush(stdout);
    /* written aside and named only once checked: a download stopped
       halfway is never taken for the file. A partial one left under the
       final name by an older janas-get goes on aside. */
    char part[4200];
    snprintf(part, sizeof(part), "%s.part", local);
    if (access(part, F_OK) != 0 && access(local, F_OK) == 0)
        rename(local, part);
    struct bar b = {.t0 = now()};
    FILE *f = fopen(part, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long at = ftell(f);
        b.from = at > 0 ? (uint64_t)at : 0;
        fclose(f);
    }
    if (janas_hf_download(repo, NULL, path, part, size, got, progress, &b, err,
                          sizeof(err)) != 0) {
        fprintf(stderr, "\njanas-get: %s (run again to resume)\n", err);
        return -1;
    }
    const char *expect = want ? want : listed;
    if (expect[0] && strcmp(got, expect) != 0) {
        fprintf(stderr,
                "janas-get: %s is not the file it should be (SHA-256 %s, "
                "not %s); deleted\n",
                local, got, expect);
        remove(part);
        return -1;
    }
    if (rename(part, local) != 0) {
        fprintf(stderr, "janas-get: cannot name %s\n", local);
        return -1;
    }
    if (expect[0])
        printf("   SHA-256 checked%s\n", want ? " against MODELS.md" : "");
    else /* a small file kept in git, not LFS: Hugging Face lists no hash */
        printf("   not checked: Hugging Face publishes no SHA-256 for it "
               "(SHA-256 %s)\n",
               got);
    return 0;
}

int get_run(const char *tool, char *const argv[])
{
    char self[4096], path[4200];
    if (janas_exe_path(self, sizeof self) < 0)
        return -1;
    char *slash = strrchr(self, '/');
    if (slash)
        *slash = 0;
    snprintf(path, sizeof(path), "%s/%s", self, tool);
    /* the program by its path, the rest of the line as the caller has it */
    size_t n = 0;
    while (argv[n])
        n++;
    const char **args = calloc(n + 1, sizeof *args);
    if (!args)
        return -1;
    args[0] = path;
    for (size_t i = 1; i < n; i++)
        args[i] = argv[i];
    struct janas_proc pr;
    char err[256];
    int rc =
        janas_proc_start(&pr, args, NULL, NULL, -1, -1, -1, 0, err, sizeof err);
    free(args);
    if (rc != 0) {
        fprintf(stderr, "janas-get: cannot run %s\n", path);
        return -1;
    }
    int code = -1;
    if (janas_proc_wait(&pr, -1, &code) != 1)
        return -1;
    return code;
}
