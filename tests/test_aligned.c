/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_aligned.c - janas_aligned_alloc on every boundary the engine asks
 * for, from a pointer's to a huge page's, each block written whole and
 * given back to janas_aligned_free. In the asan variant
 * (JANAS_ALIGNED_CHECK) also the two mismatches, each in a child of its
 * own that must not survive: an aligned block given to free(), an
 * ordinary one to janas_aligned_free.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/aligned.h"

#if defined(JANAS_ALIGNED_CHECK) && !defined(_WIN32)
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static int failures;

#if defined(JANAS_ALIGNED_CHECK) && !defined(_WIN32)
/* the mismatch in a child, its report silenced: 1 if the child survived */
static int survives(int aligned_to_free)
{
    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0)
        return 1;
    if (pid == 0) {
        int null = open("/dev/null", O_WRONLY);
        if (null >= 0)
            dup2(null, 2);
        if (aligned_to_free) {
            void *volatile p = janas_aligned_alloc(4096, 4096);
            free(p);
        } else {
            void *volatile p = malloc(4096);
            janas_aligned_free(p);
        }
        _exit(0);
    }
    int st = 0;
    if (waitpid(pid, &st, 0) != pid)
        return 1;
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
#endif

int main(void)
{
    static const size_t aligns[] = {sizeof(void *), 16, 64, 4096,
                                    (size_t)2 << 20};
    static const size_t sizes[] = {1, 100, 4096, 12345, (size_t)3 << 20};
    for (size_t a = 0; a < sizeof(aligns) / sizeof(*aligns); a++)
        for (size_t s = 0; s < sizeof(sizes) / sizeof(*sizes); s++) {
            unsigned char *p = janas_aligned_alloc(aligns[a], sizes[s]);
            if (!p) {
                printf("FAIL %zu bytes on %zu: no memory\n", sizes[s],
                       aligns[a]);
                failures++;
                continue;
            }
            if ((uintptr_t)p % aligns[a]) {
                printf("FAIL %zu bytes on %zu: at %p\n", sizes[s], aligns[a],
                       (void *)p);
                failures++;
            }
            memset(p, 0xa5, sizes[s]);
            janas_aligned_free(p);
        }
    janas_aligned_free(NULL);
#if defined(JANAS_ALIGNED_CHECK) && !defined(_WIN32)
    if (survives(1)) {
        printf("FAIL an aligned block given to free() went unnoticed\n");
        failures++;
    }
    if (survives(0)) {
        printf("FAIL an ordinary block given to janas_aligned_free went "
               "unnoticed\n");
        failures++;
    }
#endif
    printf("test_aligned: %s%s\n", failures ? "FAILED" : "ok",
#if defined(JANAS_ALIGNED_CHECK) && !defined(_WIN32)
           " (mismatches caught)"
#else
           ""
#endif
    );
    return failures ? 1 : 0;
}
