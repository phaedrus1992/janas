/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * node_run.h - node running a script, for the checks of the module for
 * Node.js (test_node.c, node_check.c): node from the PATH; the module of
 * the test's own variant, beside it (janas.node, janas-asan.node...); and
 * the sanitizer's runtime the test was linked with, preloaded into node, so
 * that the module's checks run there too (node itself is not instrumented:
 * its leaks at exit are not looked for). Linux only.
 */
#ifndef JANAS_TESTS_NODE_RUN_H
#define JANAS_TESTS_NODE_RUN_H

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* node's path from the PATH; 0 when there is none */
static int node_find(char *out, size_t cap)
{
    const char *path = getenv("PATH");
    while (path && *path) {
        const char *end = strchr(path, ':');
        size_t n = end ? (size_t)(end - path) : strlen(path);
        if (n && (size_t)snprintf(out, cap, "%.*s/node", (int)n, path) < cap &&
            access(out, X_OK) == 0)
            return 1;
        path = end ? end + 1 : NULL;
    }
    return 0;
}

/* the module beside this program, of its variant: test_node-asan takes
   janas-asan.node */
static int node_addon(char *out, size_t cap)
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
    const char *variant = strchr(base, '-');
    if ((size_t)snprintf(out, cap, "%s/janas%s.node", self,
                         variant ? variant : "") >= cap)
        return 0;
    return access(out, R_OK) == 0;
}

/* the sanitizer's runtime mapped in this process, if any */
static int node_preload(char *out, size_t cap)
{
    FILE *f = fopen("/proc/self/maps", "r");
    char line[4608];
    int found = 0;
    while (f && !found && fgets(line, sizeof(line), f)) {
        char *p = strchr(line, '/');
        if (!p || !(strstr(p, "/libasan.so") || strstr(p, "/libtsan.so") ||
                    strstr(p, "/libubsan.so")))
            continue;
        p[strcspn(p, "\n")] = 0;
        found = (size_t)snprintf(out, cap, "%s", p) < cap;
    }
    if (f)
        fclose(f);
    return found;
}

/*
 * Runs node with flags, the script (-e) and args, in an environment with
 * the sanitizer's runtime preloaded and the variables of env ("NAME=value",
 * NULL-terminated, may be NULL). Its standard output goes into out (cap
 * bytes, NUL-terminated, the rest dropped), its standard error to ours. The
 * exit status, or -1 when it could not run or was killed after timeout_s
 * seconds.
 */
static int node_run(const char *node, const char *const *flags,
                    const char *script, const char *const *args,
                    const char *const *env, int timeout_s, char *out,
                    size_t cap)
{
    const char *argv[64];
    int n = 0;
    argv[n++] = node;
    for (int i = 0; flags && flags[i] && n < 50; i++)
        argv[n++] = flags[i];
    argv[n++] = "-e";
    argv[n++] = script;
    for (int i = 0; args && args[i] && n < 63; i++)
        argv[n++] = args[i];
    argv[n] = NULL;
    char preload[4096];
    int have_preload = node_preload(preload, sizeof(preload));
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
        close(fd[0]);
        close(fd[1]);
        if (have_preload) {
            setenv("LD_PRELOAD", preload, 1);
            /* node's own allocations at exit are not the module's */
            setenv("ASAN_OPTIONS", "detect_leaks=0", 0);
        }
        for (int i = 0; env && env[i]; i++)
            putenv((char *)env[i]);
        execv(node, (char *const *)argv);
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
                fprintf(stderr, "node did not end in %d s: killed\n",
                        timeout_s);
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
