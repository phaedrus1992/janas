/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_fb.c - Janas's classes for FreeBASIC (include/janas/janas_oop.bi),
 * without a model: tests/fb/test_oop.bas built with every warning on, which
 * must give none, and run (options, a conversation, a model and a chat that
 * do not open, a handler, the fake MCP server's tools); a program of two
 * modules, the second declaring only (JANAS_DECLARE_ONLY), which must link;
 * and a Model copied, which must not compile. Skipped without the FreeBASIC
 * compiler (FBC, or fbc in the PATH), out of the sources' directory, and in
 * the variants but release, which alone makes the shared libraries.
 */
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
int main(void)
{
    printf("test_fb: skipped, tried on Linux only\n");
    return 0;
}
#else
#include <unistd.h>

#include "fb_run.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

static const char two_main[] =
    "#include once \"janas/janas_oop.bi\"\n"
    "declare function other() as long\n"
    "dim as Janas.Model m = Janas.Model(\"/nonexistent/a.jns\")\n"
    "if m.IsOpen = false andalso other() = JANAS_LLM_EOPEN then print \"two "
    "modules ok\"\n";

static const char two_other[] =
    "#define JANAS_DECLARE_ONLY\n"
    "#include once \"janas/janas_oop.bi\"\n"
    "function other() as long\n"
    "    dim as Janas.Model m = Janas.Model(\"/nonexistent/b.jns\")\n"
    "    dim as Janas.Chat c = Janas.Chat(m)\n"
    "    return m.ErrorCode\n"
    "end function\n";

static const char copied[] = "#include once \"janas/janas_oop.bi\"\n"
                             "dim as Janas.Model a\n"
                             "dim as Janas.Model b = a\n";

int main(void)
{
    char fbc[4096], bin[4096], tmp[4096], path[4352], path2[4352], exe[4352],
        lib_flag[4352], fake[4352];
    static char out[1 << 16];
    int variant = 0;
    if (!fb_find(fbc, sizeof(fbc))) {
        printf("test_fb: skipped, no FreeBASIC compiler (FBC, or fbc in the "
               "PATH)\n");
        return 0;
    }
    if (access("include/janas/janas_oop.bi", R_OK) != 0 ||
        access("tests/fb/test_oop.bas", R_OK) != 0) {
        printf("test_fb: skipped, the sources are not here\n");
        return 0;
    }
    if (!fb_bin(bin, sizeof(bin), &variant) || variant) {
        printf("test_fb: skipped, the shared libraries are the release "
               "build's\n");
        return 0;
    }
    if (!fb_tmpdir(tmp, sizeof(tmp))) {
        printf("test_fb: FAIL, no directory to build in: set TMPDIR\n");
        return 1;
    }
    snprintf(lib_flag, sizeof(lib_flag), "%s", bin);
    snprintf(fake, sizeof(fake), "%s/mcp_fake_server", bin);

    /* the test program, with every warning */
    snprintf(exe, sizeof(exe), "%s/test_oop", tmp);
    char *build[] = {fbc,  "-w",      "all", "tests/fb/test_oop.bas",
                     "-i", "include", "-p",  lib_flag,
                     "-x", exe,       NULL};
    int st = fb_exec(build, NULL, 300, out, sizeof(out));
    CHECK(st == 0, "test_oop.bas does not build:\n%s", out);
    CHECK(st != 0 || !strstr(out, "warning"), "warnings:\n%s", out);
    if (st == 0) {
        char *run[] = {exe, fake, NULL};
        st = fb_exec(run, bin, 120, out, sizeof(out));
        CHECK(st == 0 && strstr(out, "checks, ok"), "test_oop (exit %d):\n%s",
              st, out);
        if (st == 0)
            printf("%s", out);
    }

    /* two modules, one of them declaring only */
    snprintf(path, sizeof(path), "%s/two_main.bas", tmp);
    snprintf(path2, sizeof(path2), "%s/two_other.bas", tmp);
    snprintf(exe, sizeof(exe), "%s/two", tmp);
    CHECK(fb_write(path, two_main) && fb_write(path2, two_other),
          "cannot write in %s", tmp);
    char *two[] = {fbc,       "-w", "all",    path, path2, "-i",
                   "include", "-p", lib_flag, "-x", exe,   NULL};
    st = fb_exec(two, NULL, 300, out, sizeof(out));
    CHECK(st == 0, "two modules do not link:\n%s", out);
    if (st == 0) {
        char *run[] = {exe, NULL};
        st = fb_exec(run, bin, 60, out, sizeof(out));
        CHECK(st == 0 && strstr(out, "two modules ok"),
              "two modules (exit %d):\n%s", st, out);
    }

    /* a Model is not copied */
    snprintf(path, sizeof(path), "%s/copied.bas", tmp);
    snprintf(exe, sizeof(exe), "%s/copied", tmp);
    CHECK(fb_write(path, copied), "cannot write in %s", tmp);
    char *copy[] = {fbc,      path, "-i", "include", "-p",
                    lib_flag, "-x", exe,  NULL};
    st = fb_exec(copy, NULL, 300, out, sizeof(out));
    CHECK(st != 0 && strstr(out, "public access"),
          "a Model was copied (it would be closed twice), or the build failed "
          "for another reason:\n%s",
          out);

    fb_rmdir(tmp);
    printf("test_fb: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
#endif
