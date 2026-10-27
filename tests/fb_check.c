/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fb_check.c - Janas's classes for FreeBASIC against the library, on a
 * model:
 *
 *     fb_check model.jns [seconds]
 *
 * First the library itself, here: a message and its reply, then the whole
 * conversation loaded with a second message, greedy, on the CPU alone
 * (JANAS_GPU=0, JANAS_NPU=0, for the same numbers in both processes). Then
 * tests/fb/check_oop.bas, built with the FreeBASIC compiler (FBC, or fbc in
 * the PATH), says the same through janas_oop.bi: the replies must be the
 * same text, byte for byte, read piece by piece, whole, through a handler,
 * after a reset and in a new chat; and it tries the rest (one chat at a
 * time, a conversation's start reused, tokens, a request on the side, the
 * model closed under a live chat). Not one of the tests run by build.sh: it
 * needs a model (Qwen3-0.6B takes seconds). Run from the sources' directory.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"

#ifdef _WIN32
int main(void)
{
    printf("fb_check: tried on Linux only\n");
    return 0;
}
#else
#include "fb_run.h"

static const char p1[] = "Write one sentence about the sea.";
static const char p2[] = "And one about the mountains.";

/* the reply's answer, its reasoning left out */
static char *reply_text(janas_llm_chat *c, int32_t rc)
{
    size_t len = 0, cap = 4096;
    char *s = malloc(cap), buf[4096];
    int32_t n;
    while (s && rc == JANAS_LLM_OK &&
           (rc = janas_llm_chat_next(c, buf, sizeof(buf), &n)) ==
               JANAS_LLM_OK) {
        if (n <= 0 || janas_llm_chat_thinking(c))
            continue;
        if (len + (size_t)n + 1 > cap) {
            char *b = realloc(s, cap *= 2);
            if (!b) {
                free(s);
                return NULL;
            }
            s = b;
        }
        memcpy(s + len, buf, (size_t)n);
        len += (size_t)n;
    }
    if (!s || rc < 0) {
        fprintf(stderr, "fb_check: %s\n", janas_llm_last_error());
        free(s);
        return NULL;
    }
    s[len] = 0;
    return s;
}

static char *to_hex(const char *s)
{
    size_t n = strlen(s);
    char *h = malloc(2 * n + 1);
    for (size_t i = 0; h && i < n; i++)
        sprintf(h + 2 * i, "%02x", (unsigned char)s[i]);
    if (h)
        h[2 * n] = 0;
    return h;
}

/* the line "key value" of the program's output: its value is want */
static int same(const char *out, const char *key, const char *want)
{
    size_t kl = strlen(key);
    for (const char *line = out; *line;) {
        size_t n = strcspn(line, "\n");
        if (n > kl && !strncmp(line, key, kl) && line[kl] == ' ')
            return n - kl - 1 == strlen(want) &&
                   !strncmp(line + kl + 1, want, n - kl - 1);
        line += n + (line[n] == '\n');
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: fb_check model.jns [seconds]\n");
        return 2;
    }
    int timeout_s = argc > 2 ? atoi(argv[2]) : 600;
    char fbc[4096], bin[4096], tmp[4096], exe[4352];
    int variant = 0;
    if (!fb_find(fbc, sizeof(fbc)) || !fb_bin(bin, sizeof(bin), &variant) ||
        access("tests/fb/check_oop.bas", R_OK) != 0) {
        fprintf(stderr, "fb_check: no FreeBASIC compiler (FBC, or fbc in the "
                        "PATH), or not run from the sources' directory\n");
        return 2;
    }
    if (variant) {
        fprintf(stderr, "fb_check: the shared libraries are the release "
                        "build's: run bin/<cpu>-<os>/fb_check\n");
        return 2;
    }
    /* the same numbers in both processes: the CPU's */
    setenv("JANAS_GPU", "0", 1);
    setenv("JANAS_NPU", "0", 1);

    struct janas_llm_params p;
    janas_llm_params_default(&p);
    p.n_ctx = 4096;
    janas_llm *llm;
    if (janas_llm_open(argv[1], &p, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "fb_check: %s\n", janas_llm_last_error());
        return 2;
    }
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    cp.temperature = 0;
    cp.thinking = 0;
    cp.max_reply = 48;
    janas_llm_chat *c;
    char *t1 = NULL, *t2 = NULL;
    if (janas_llm_chat_create(llm, &cp, &c) == JANAS_LLM_OK) {
        t1 = reply_text(c, janas_llm_chat_send(c, p1, -1));
        if (t1) {
            const int32_t roles[] = {JANAS_LLM_ROLE_USER,
                                     JANAS_LLM_ROLE_ASSISTANT,
                                     JANAS_LLM_ROLE_USER};
            const char *texts[] = {p1, t1, p2};
            t2 = reply_text(c, janas_llm_chat_load(c, 3, roles, texts, NULL));
        }
        janas_llm_chat_destroy(c);
    }
    janas_llm_close(llm);
    if (!t1 || !t2) {
        fprintf(stderr, "fb_check: the library's replies failed\n");
        return 1;
    }
    printf("library: \"%s\" / \"%s\"\n", t1, t2);

    if (!fb_tmpdir(tmp, sizeof(tmp))) {
        fprintf(stderr, "fb_check: no directory to build in: set TMPDIR\n");
        return 2;
    }
    static char out[1 << 20];
    snprintf(exe, sizeof(exe), "%s/check_oop", tmp);
    char *build[] = {fbc,  "-w",      "all", "tests/fb/check_oop.bas",
                     "-i", "include", "-p",  bin,
                     "-x", exe,       NULL};
    int failures = 0;
    int st = fb_exec(build, NULL, 300, out, sizeof(out));
    if (st != 0) {
        printf("FAIL check_oop.bas does not build:\n%s", out);
        fb_rmdir(tmp);
        return 1;
    }
    char *run[] = {exe, argv[1], (char *)p1, (char *)p2, NULL};
    st = fb_exec(run, bin, timeout_s, out, sizeof(out));
    fb_rmdir(tmp);
    char *h1 = to_hex(t1), *h2 = to_hex(t2);
    struct {
        const char *key, *want, *what;
    } cmp[] = {
        {"t1", h1, "the reply piece by piece"},
        {"t2", h2, "the conversation loaded, read whole"},
        {"t1again", h1, "the same message after a reset, Ask"},
        {"t1handler", h1, "the same message through a handler"},
        {"t1chat2", h1, "the same message in a new chat"},
    };
    printf("FreeBASIC: exit %d\n", st);
    failures += st != 0;
    for (size_t i = 0; i < sizeof(cmp) / sizeof(cmp[0]); i++) {
        int ok = h1 && h2 && same(out, cmp[i].key, cmp[i].want);
        printf("%-5s %s\n", ok ? "ok" : "FAIL", cmp[i].what);
        failures += !ok;
    }
    for (const char *line = out; *line;) {
        size_t n = strcspn(line, "\n");
        if (!strncmp(line, "FAIL", 4) || !strncmp(line, "model ", 6) ||
            !strncmp(line, "cached ", 7))
            printf("FreeBASIC: %.*s\n", (int)n, line);
        line += n + (line[n] == '\n');
    }
    printf("fb_check: %s\n", failures ? "FAIL" : "ok");
    free(h1);
    free(h2);
    free(t1);
    free(t2);
    return failures ? 1 : 0;
}
#endif
