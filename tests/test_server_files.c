/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_files.c - the files janas-server keeps for its clients
 * (src/server/files_store.c) and the uploads they come in
 * (src/server/multipart.c): a body taken apart right, mangled ones neither
 * crashing nor leaking (under asan), files kept, listed in order, read
 * back after a restart, expired, deleted, and names that never become
 * paths.
 */
#include <fcntl.h>
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/md5.c"
#include "server/files_store.c"
#include "server/multipart.c"
#include "server/store_disk.c"

/* files_store.c names its files with it; it lives in http.c */
void srv_new_id(char *buf, size_t cap, const char *prefix)
{
    static int n;
    snprintf(buf, cap, "%s%04d", prefix, ++n);
}

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static int rm_one(const char *p, const struct stat *s, int f, struct FTW *w)
{
    (void)s;
    (void)f;
    (void)w;
    return remove(p);
}

static const char BODY[] =
    "preamble\r\n"
    "--XyZ\r\n"
    "Content-Disposition: form-data; name=\"purpose\"\r\n\r\n"
    "batch\r\n"
    "--XyZ\r\n"
    "Content-Disposition: form-data; name=\"file\"; "
    "filename=\"../../etc/x \\\"q\\\".jsonl\"\r\n"
    "Content-Type: application/jsonl\r\n\r\n"
    "{\"a\":1}\r\n--X\r\n\0\x01\r\n"
    "--XyZ--\r\n";

static void multipart(void)
{
    struct srv_part p[8];
    char err[128];
    int n = srv_multipart_parse("multipart/form-data; boundary=XyZ", BODY,
                                sizeof(BODY) - 1, p, 8, err, sizeof err);
    CHECK(n == 2, "parts: %d (%s)", n, n < 0 ? err : "");
    if (n != 2)
        return;
    const struct srv_part *f = srv_multipart_get(p, n, "file");
    const char want[] = "{\"a\":1}\r\n--X\r\n\0\x01";
    CHECK(f && f->has_filename &&
              !strcmp(f->filename, "../../etc/x \"q\".jsonl") &&
              f->n == sizeof(want) - 1 && !memcmp(f->data, want, f->n) &&
              !strcmp(f->type, "application/jsonl"),
          "the file part");
    const struct srv_part *pu = srv_multipart_get(p, n, "purpose");
    CHECK(pu && !pu->has_filename && pu->n == 5 &&
              !memcmp(pu->data, "batch", 5),
          "the purpose part");
    CHECK(srv_multipart_parse("multipart/form-data; boundary=\"XyZ\"", BODY,
                              sizeof(BODY) - 1, p, 8, err, sizeof err) == 2,
          "a quoted boundary");
    CHECK(srv_multipart_parse("application/json", BODY, sizeof(BODY) - 1, p, 8,
                              err, sizeof err) < 0,
          "not multipart, taken");
    CHECK(srv_multipart_parse("multipart/form-data; boundary=XyZ", BODY, 60, p,
                              8, err, sizeof err) < 0,
          "a cut body taken");
    /* mangled: bytes flipped, cut anywhere */
    char buf[sizeof(BODY)];
    uint64_t s = 12345;
    for (int it = 0; it < 20000; it++) {
        memcpy(buf, BODY, sizeof(BODY));
        size_t len = sizeof(BODY) - 1;
        s ^= s << 13, s ^= s >> 7, s ^= s << 17;
        int k = 1 + (int)(s % 4);
        for (int j = 0; j < k; j++) {
            s ^= s << 13, s ^= s >> 7, s ^= s << 17;
            buf[s % len] ^= (char)(1 << (s >> 32) % 8);
        }
        len = (size_t)(s >> 20) % (len + 1);
        int m = srv_multipart_parse("multipart/form-data; boundary=XyZ", buf,
                                    len, p, 8, err, sizeof err);
        for (int i = 0; i < m; i++)
            CHECK(p[i].data >= buf && p[i].data + p[i].n <= buf + len,
                  "a part outside the body");
    }
}

static void store(void)
{
    const char *t = getenv("TMPDIR");
    char dir[4096];
    snprintf(dir, sizeof dir, "%s/janas-files-test-%d", t && *t ? t : "/tmp",
             (int)getpid());
    mkdir(dir, 0700);
    char err[200];
    struct srv_files *f = srv_files_open(dir, err, sizeof err);
    CHECK(f, "open: %s", err);
    if (!f)
        return;
    struct srv_file a, b, c, d, x;
    CHECK(srv_files_add(f, "C:\\docs\\a.jsonl", "batch", "abc", 3, 0, &a, err,
                        sizeof err) == 0 &&
              !strcmp(a.filename, "a.jsonl"),
          "add a: %s", err);
    CHECK(srv_files_add(f, "b\n.txt", "user_data", "", 0, 3600, &b, err,
                        sizeof err) == 0 &&
              !strcmp(b.filename, "b.txt") &&
              b.expires_at == b.created_at + 3600,
          "add b");
    CHECK(srv_files_add(f, "c", "assistants", "xyz", 3, 0, &c, err,
                        sizeof err) == 0,
          "add c");
    struct srv_file *l;
    size_t n = srv_files_list(f, NULL, 0, NULL, &l);
    CHECK(n == 3 && !strcmp(l[0].id, c.id) && !strcmp(l[2].id, a.id),
          "newest first: %zu", n);
    free(l);
    n = srv_files_list(f, NULL, 1, a.id, &l);
    CHECK(n == 2 && !strcmp(l[0].id, b.id), "after the first, oldest first");
    free(l);
    n = srv_files_list(f, "batch", 0, NULL, &l);
    CHECK(n == 1 && !strcmp(l[0].id, a.id), "by purpose");
    free(l);
    /* d expires: added after the listings, which a slow disk (Windows'
       runners) could otherwise outlast */
    CHECK(srv_files_add(f, "d", "assistants", "xyz", 3, 1, &d, err,
                        sizeof err) == 0,
          "add d");
    sleep(2);
    CHECK(srv_files_get(f, d.id, &x) != 0, "an expired file still there");
    int fd = srv_files_open_content(f, a.id);
    char got[8] = {0};
    CHECK(fd >= 0 && read(fd, got, 8) == 3 && !memcmp(got, "abc", 3),
          "content");
    if (fd >= 0)
        close(fd);
    CHECK(srv_files_get(f, "../a", &x) != 0 &&
              srv_files_open_content(f, "../../x") < 0,
          "a path as an id");
    srv_files_close(f);

    /* read back */
    f = srv_files_open(dir, err, sizeof err);
    CHECK(f, "reopen: %s", err);
    if (!f)
        return;
    CHECK(srv_files_get(f, a.id, &x) == 0 && x.bytes == 3 &&
              !strcmp(x.filename, "a.jsonl") && !strcmp(x.purpose, "batch"),
          "a after reopening");
    CHECK(srv_files_get(f, b.id, &x) == 0 && x.expires_at == b.expires_at,
          "b after reopening");
    CHECK(srv_files_delete(f, a.id) == 0 && srv_files_get(f, a.id, &x) != 0 &&
              srv_files_delete(f, a.id) != 0,
          "delete");
    srv_files_close(f);
    nftw(dir, rm_one, 16, FTW_DEPTH | FTW_PHYS);

    /* no store: a directory of its own, gone at the end */
    f = srv_files_open(NULL, err, sizeof err);
    CHECK(f, "temporary: %s", err);
    if (f) {
        char tdir[4096];
        snprintf(tdir, sizeof tdir, "%s", f->dir);
        srv_files_add(f, "t", "evals", "1", 1, 0, &x, err, sizeof err);
        srv_files_close(f);
        CHECK(access(tdir, F_OK) != 0, "the temporary directory stayed");
    }
}

static void md5(void)
{
    static const char *const in[] = {"", "abc",
                                     "12345678901234567890123456789012345678"
                                     "901234567890123456789012345678901234567"
                                     "890"};
    static const char *const want[] = {"d41d8cd98f00b204e9800998ecf8427e",
                                       "900150983cd24fb0d6963f7d28e17f72",
                                       "57edf4a22be3c955ac49da2e2107b67a"};
    for (int i = 0; i < 3; i++) {
        struct janas_md5 h;
        char got[33];
        janas_md5_init(&h);
        /* in pieces of 7: the buffering between them */
        for (size_t k = 0; k < strlen(in[i]); k += 7)
            janas_md5_update(&h, in[i] + k,
                             strlen(in[i]) - k < 7 ? strlen(in[i]) - k : 7);
        janas_md5_hex(&h, got);
        CHECK(!strcmp(got, want[i]), "md5 %d: %s", i, got);
    }
    struct janas_md5 h;
    char got[33], a[1000];
    memset(a, 'a', sizeof a);
    janas_md5_init(&h);
    for (int i = 0; i < 1000; i++)
        janas_md5_update(&h, a, sizeof a);
    janas_md5_hex(&h, got);
    CHECK(!strcmp(got, "7707d6ae4e027c70eea2a935c2296f21"), "md5 of a million");
}

/* An upload's parts joined into a file. */
static void parts(void)
{
    char err[200];
    struct srv_files *f = srv_files_open(NULL, err, sizeof err);
    CHECK(f, "parts: %s", err);
    if (!f)
        return;
    char p1[4200], p2[4200];
    snprintf(p1, sizeof p1, "%s/.p1", srv_files_dir(f));
    snprintf(p2, sizeof p2, "%s/.p2", srv_files_dir(f));
    FILE *a = fopen(p1, "wb"), *b = fopen(p2, "wb");
    if (a) {
        fputs("ab", a);
        fclose(a);
    }
    if (b) {
        fputs("c", b);
        fclose(b);
    }
    const char *const order[] = {p1, p2};
    struct srv_file x;
    CHECK(srv_files_add_parts(f, "j.txt", "batch", order, 2, 4, NULL, 0, &x,
                              err, sizeof err) == -3,
          "parts of another size taken");
    CHECK(srv_files_add_parts(f, "j.txt", "batch", order, 2, 3,
                              "00000000000000000000000000000000", 0, &x, err,
                              sizeof err) == -3,
          "parts of another MD5 taken");
    CHECK(srv_files_add_parts(f, "j.txt", "batch", order, 2, 3,
                              "900150983CD24FB0D6963F7D28E17F72", 0, &x, err,
                              sizeof err) == 0,
          "parts refused: %s", err);
    int fd = srv_files_open_content(f, x.id);
    char got[8] = {0};
    CHECK(fd >= 0 && read(fd, got, 8) == 3 && !memcmp(got, "abc", 3),
          "the parts joined in order");
    if (fd >= 0)
        close(fd);
    unlink(p1);
    unlink(p2);
    srv_files_close(f);
}

int main(void)
{
    multipart();
    store();
    md5();
    parts();
    if (!failures)
        printf("test_server_files: ok\n");
    return failures != 0;
}
