/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_sandbox.c - the cage of src/common/sandbox.c, tried as code that
 * wants out would: the user's files, writing outside /mnt/data, the
 * network, other processes, new processes, too much memory, files too
 * large, a /tmp filled, a loop that never ends; and what it must allow
 * (threads, the data directory, numpy where the system has it). Skipped
 * where the cage cannot be built (no user namespaces) or there is no
 * python3.
 */
#include "common/sandbox.h"

#ifdef _WIN32
#include <stdio.h>

int main(void)
{
    printf("test_sandbox: skipped (Linux only)\n");
    return 0;
}
#else

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            fprintf(stderr, "test_sandbox: " __VA_ARGS__);                     \
            fputc('\n', stderr);                                               \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static char data[256];

/* python's code run in the cage: its output (and error) into out, the exit
   status returned; -2 when it ran past secs and was killed. */
static int run(const char *code, char *out, size_t cap, int secs)
{
    char *argv[] = {"/usr/bin/python3", "-I", "-c", (char *)code, NULL};
    char *envp[] = {"PATH=/usr/bin:/bin", "HOME=/tmp", "LANG=C.UTF-8",
                    "OPENBLAS_NUM_THREADS=2", NULL};
    const char *ro[] = {NULL};
    struct janas_sandbox_cfg c = {.data_dir = data,
                                  .ro = ro,
                                  .mem_bytes = (uint64_t)1 << 30,
                                  .file_bytes = (uint64_t)8 << 20,
                                  .tmp_bytes = (uint64_t)16 << 20};
    struct janas_sandbox sb;
    char why[256];
    if (janas_sandbox_spawn(&c, argv, envp, &sb, why, sizeof why) != 0) {
        snprintf(out, cap, "spawn: %s", why);
        return -1;
    }
    close(sb.in);
    sb.in = -1;
    size_t n = 0;
    out[0] = 0;
    time_t end = time(NULL) + secs;
    int open_fds = 2, killed = 0;
    while (open_fds > 0) {
        struct pollfd p[2] = {{sb.out, POLLIN, 0}, {sb.err, POLLIN, 0}};
        int left = (int)(end - time(NULL));
        if (left <= 0) {
            killed = 1;
            break;
        }
        if (poll(p, 2, left * 1000) <= 0)
            continue;
        for (int i = 0; i < 2; i++)
            if (p[i].revents & (POLLIN | POLLHUP)) {
                char buf[512];
                ssize_t k = read(i ? sb.err : sb.out, buf, sizeof buf);
                if (k <= 0) {
                    p[i].fd = -1;
                    if (i)
                        sb.err = (close(sb.err), -1);
                    else
                        sb.out = (close(sb.out), -1);
                    open_fds--;
                    continue;
                }
                size_t m = (size_t)k < cap - 1 - n ? (size_t)k : cap - 1 - n;
                memcpy(out + n, buf, m);
                n += m;
                out[n] = 0;
            }
    }
    if (killed) {
        pid_t pid = sb.pid;
        janas_sandbox_kill(&sb);
        /* nothing left of it */
        CHECK(kill(pid, 0) != 0, "the cage outlived its kill");
        return -2;
    }
    int st = 0;
    while (waitpid(sb.pid, &st, 0) < 0 && errno == EINTR)
        ;
    sb.pid = 0;
    janas_sandbox_kill(&sb);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

/* code must print "ok" and end well */
static void expect_ok(const char *name, const char *code)
{
    char out[4096];
    int rc = run(code, out, sizeof out, 30);
    CHECK(rc == 0 && strstr(out, "ok"), "%s: exit %d, output: %s", name, rc,
          out);
}

int main(int argc, char **argv)
{
    janas_sandbox_main(argc, argv);
    char why[256];
    if (!janas_sandbox_available(why, sizeof why)) {
        printf("test_sandbox: skipped (%s)\n", why);
        return 0;
    }
    if (access("/usr/bin/python3", X_OK) != 0) {
        printf("test_sandbox: skipped (no /usr/bin/python3)\n");
        return 0;
    }
    snprintf(data, sizeof data, "%s/tmp/janas-sandbox-test-XXXXXX",
             getenv("HOME") ? getenv("HOME") : "/tmp");
    char parent[256];
    snprintf(parent, sizeof parent, "%s/tmp",
             getenv("HOME") ? getenv("HOME") : "/tmp");
    mkdir(parent, 0700);
    if (!mkdtemp(data)) {
        perror("mkdtemp");
        return 1;
    }
    char code[2048];

    expect_ok("plain", "print('ok')");
    /* the user's files are not there */
    snprintf(code, sizeof code,
             "import os\n"
             "assert not os.path.exists('/home'), 'home'\n"
             "assert not os.path.exists('%s'), 'the data dir by its host "
             "path'\n"
             "assert os.listdir('/etc') and not os.path.exists('/etc/"
             "passwd'), 'etc'\n"
             "assert os.getcwd() == '/mnt/data'\n"
             "print('ok')",
             data);
    expect_ok("files", code);
    /* writing: only in /mnt/data, and it reaches the host */
    expect_ok("write",
              "import os\n"
              "for p in ('/usr/x', '/etc/x', '/x', '/proc/x'):\n"
              "    try:\n"
              "        open(p, 'w').write('x'); raise SystemExit('wrote '"
              " + p)\n"
              "    except OSError: pass\n"
              "open('/mnt/data/hello.txt', 'w').write('hi')\n"
              "print('ok')");
    char hp[512];
    snprintf(hp, sizeof hp, "%s/hello.txt", data);
    FILE *f = fopen(hp, "r");
    char hb[8] = "";
    if (f) {
        if (!fgets(hb, sizeof hb, f))
            hb[0] = 0;
        fclose(f);
    }
    CHECK(!strcmp(hb, "hi"), "the file written in /mnt/data is not on the "
                             "host");
    unlink(hp);
    /* no network */
    expect_ok("network",
              "import socket\n"
              "for host in (('1.1.1.1', 53), ('127.0.0.1', 22)):\n"
              "    s = socket.socket(); s.settimeout(2)\n"
              "    try:\n"
              "        s.connect(host); raise SystemExit('connected')\n"
              "    except OSError: pass\n"
              "print('ok')");
    /* no other process to see, no new process, no privilege */
    expect_ok("processes",
              "import os, subprocess\n"
              "pids = [p for p in os.listdir('/proc') if p.isdigit()]\n"
              "assert pids == ['1'], pids\n"
              "try:\n"
              "    os.fork(); raise SystemExit('forked')\n"
              "except OSError: pass\n"
              "try:\n"
              "    subprocess.run(['/usr/bin/true']); raise SystemExit('ran')\n"
              "except OSError: pass\n"
              "eff = [l for l in open('/proc/self/status') if "
              "l.startswith('CapEff')][0]\n"
              "assert int(eff.split()[1], 16) == 0, eff\n"
              "print('ok')");
    /* threads are fine */
    expect_ok("threads", "import threading\n"
                         "r = []\n"
                         "t = [threading.Thread(target=r.append, args=(i,)) "
                         "for i in range(8)]\n"
                         "[x.start() for x in t]; [x.join() for x in t]\n"
                         "assert sorted(r) == list(range(8))\n"
                         "print('ok')");
    /* memory, file size, /tmp: bounded */
    expect_ok("memory", "try:\n"
                        "    b = bytearray(3 << 30); raise SystemExit('got "
                        "3 GiB')\n"
                        "except MemoryError: pass\n"
                        "print('ok')");
    expect_ok("file size", "try:\n"
                           "    open('/mnt/data/big', 'wb').write(b'x' * (16 "
                           "<< 20)); raise SystemExit('wrote 16 MiB')\n"
                           "except OSError: pass\n"
                           "print('ok')");
    snprintf(hp, sizeof hp, "%s/big", data);
    unlink(hp);
    expect_ok("tmp", "try:\n"
                     "    f = open('/tmp/fill', 'wb')\n"
                     "    for i in range(32): f.write(b'x' * (1 << 20)); "
                     "f.flush()\n"
                     "    raise SystemExit('filled /tmp')\n"
                     "except OSError: pass\n"
                     "print('ok')");
    /* a loop without end is killed, and the cage is gone */
    char out[256];
    CHECK(run("while True: pass", out, sizeof out, 2) == -2,
          "the endless loop ended by itself");
    /* numpy, where the system has it */
    char probe[64];
    if (run("import numpy; print('ok')", probe, sizeof probe, 30) == 0)
        expect_ok("numpy", "import numpy as np\n"
                           "a = np.arange(1000000.0)\n"
                           "assert abs(a.sum() - 499999500000.0) < 1\n"
                           "print('ok')");
    rmdir(data);
    if (failures) {
        fprintf(stderr, "test_sandbox: %d failures\n", failures);
        return 1;
    }
    printf("test_sandbox: ok\n");
    return 0;
}

#endif
