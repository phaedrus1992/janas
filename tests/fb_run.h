/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fb_run.h - FreeBASIC programs built and run, for the checks of
 * janas_oop.bi (test_fb.c, fb_check.c): the compiler from FBC or the PATH;
 * the directory of this program, where the shared libraries are; a
 * directory of its own for what is built ($TMPDIR, else /var/tmp); and a
 * command run with its output kept and a limit on its time. Linux only.
 */
#ifndef JANAS_TESTS_FB_RUN_H
#define JANAS_TESTS_FB_RUN_H

#include <dirent.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* the FreeBASIC compiler: $FBC, else fbc from the PATH; 0 when none */
static int fb_find(char *out, size_t cap)
{
    const char *env = getenv("FBC");
    if (env && *env) {
        if ((size_t)snprintf(out, cap, "%s", env) < cap &&
            access(out, X_OK) == 0)
            return 1;
        return 0;
    }
    const char *path = getenv("PATH");
    while (path && *path) {
        const char *end = strchr(path, ':');
        size_t n = end ? (size_t)(end - path) : strlen(path);
        if (n && (size_t)snprintf(out, cap, "%.*s/fbc", (int)n, path) < cap &&
            access(out, X_OK) == 0)
            return 1;
        path = end ? end + 1 : NULL;
    }
    return 0;
}

/* this program's directory (bin/<cpu>-<os>), and whether it is a variant's
   (test_fb-asan): the shared libraries are only the release build's */
static int fb_bin(char *out, size_t cap, int *variant)
{
    char self[4096];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0)
        return 0;
    self[n] = 0;
    char *base = strrchr(self, '/');
    if (!base)
        return 0;
    *base++ = 0;
    *variant = strchr(base, '-') != NULL;
    return (size_t)snprintf(out, cap, "%s", self) < cap;
}

/* a new directory for what is built */
static int fb_tmpdir(char *out, size_t cap)
{
    const char *t = getenv("TMPDIR");
    if ((size_t)snprintf(out, cap, "%s/janas-fb-XXXXXX",
                         t && *t ? t : "/var/tmp") >= cap)
        return 0;
    return mkdtemp(out) != NULL;
}

/* the directory and the files in it (it holds nothing else) */
static void fb_rmdir(const char *dir)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    char p[4352];
    while (d && (e = readdir(d)))
        if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..") &&
            (size_t)snprintf(p, sizeof(p), "%s/%s", dir, e->d_name) < sizeof(p))
            unlink(p);
    if (d)
        closedir(d);
    rmdir(dir);
}

/* a file with this text (test_fb.c writes small programs) */
__attribute__((unused)) static int fb_write(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return 0;
    int ok = fputs(text, f) >= 0;
    return fclose(f) == 0 && ok;
}

/*
 * Runs argv (argv[0] a path) with LD_LIBRARY_PATH set to lib when it is not
 * NULL, its standard output and error into out (cap bytes, NUL-terminated,
 * the rest dropped). The exit status, or -1 when it could not run, died of a
 * signal, or was killed after timeout_s seconds.
 */
static int fb_exec(char *const argv[], const char *lib, int timeout_s,
                   char *out, size_t cap)
{
    int fd[2];
    if (pipe(fd) != 0)
        return -1;
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        close(fd[0]);
        close(fd[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(fd[1], 1);
        dup2(fd[1], 2);
        close(fd[0]);
        close(fd[1]);
        if (lib)
            setenv("LD_LIBRARY_PATH", lib, 1);
        execv(argv[0], argv);
        _exit(127);
    }
    close(fd[1]);
    size_t len = 0;
    int status = -1, done = 0;
    time_t end = time(NULL) + timeout_s;
    for (;;) {
        struct pollfd p = {.fd = fd[0], .events = POLLIN};
        int r = poll(&p, 1, 100);
        if (r > 0) {
            char buf[4096];
            ssize_t k = read(fd[0], buf, sizeof(buf));
            if (k > 0) {
                size_t take =
                    (size_t)k < cap - 1 - len ? (size_t)k : cap - 1 - len;
                memcpy(out + len, buf, take);
                len += take;
                continue;
            }
            if (k == 0 && done)
                break;
        }
        if (!done) {
            int st;
            pid_t w = waitpid(pid, &st, WNOHANG);
            if (w == pid) {
                done = 1;
                status = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            } else if (time(NULL) > end) {
                kill(pid, SIGKILL);
                waitpid(pid, &st, 0);
                done = 1;
                status = -1;
            }
        } else if (r == 0)
            break;
    }
    close(fd[0]);
    out[len] = 0;
    return status;
}

#endif
