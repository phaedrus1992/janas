# E1 Portability Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do NOT use `superpowers:subagent-driven-development`.** The owner's global rules forbid it: it hides work inside subagent transcripts. Work inline in the current session, and use `executing-plans` for step tracking only.
>
> **Do NOT run the `executing-plans` tail.** Stop after the last task above the "Closing" section. Its final whole-branch review, `superpowers:finishing-a-development-branch`, and its plan-workspace delete under `.superpowers/sdd/` are replaced by the closing steps written here.

**Goal:** Make Janas compile and pass `release test` and `asan test` on macOS arm64 and Linux arm64, without moving any x86-64 behaviour, delivered as six small upstream pull requests.

**Architecture:** One concern per `port/*` branch, cut from the fork's `clean` branch. Each branch is independently reviewable, touches disjoint files, and is inert on x86-64. The work is ordered so that each branch can be verified by a real compiler run before the next one starts.

**Tech Stack:** C11, POSIX, Apple clang 21 and GCC 13.3, `build.sh` (no make), Docker through Colima for native `aarch64`, GitHub Actions for CI.

**Spec:** `docs/superpowers/specs/2026-10-13-arm64-metal-port/e1-portability.md`

## Global Constraints

Every task's requirements implicitly include this section.

- Upstream rules, `design.md` §4: C11 with no dependency beyond the C library; `clang-format -i` with the root `.clang-format`; `-Wall -Wextra -Werror` clean; `./build.sh release test` and `./build.sh asan test` pass.
- Every commit on a `port/*` branch carries a `Signed-off-by` line. `CONTRIBUTING.md:39` requires it. Use `git commit -s`.
- A `port/*` branch is cut from `clean`, never from `main`, and carries no fork tooling. `.fork/check-clean.sh` must print `is clean` before the push.
- One commit per branch. The branch is the unit the maintainer imports.
- Never commit on `main`. Fork-side changes land through a pull request.
- No attribution trailers of any kind in any commit message.
- x86-64 is verified by the owner on their own x86-64 Linux box. It is a required step, not a hand-off to the maintainer.
- Re-measure every line number before cutting each branch. Upstream releases several times a day and the spec's §2 numbers are a 2026-10-14 snapshot.
- The endianness, the arithmetic order, and the results on x86-64 must not move. A change that could alter a computed value needs an explicit reason in the commit message.
- Every container command needs `tzdata`, not `gcc` alone: four tests assert `Europe/Rome` local times and report failures without it. Install `gcc tzdata` and pass `-e TZ=Europe/Rome`. Verified both ways on the same binary.
- `./build.sh release clean-test` runs 34 test programs on Linux arm64, of which 30 print `: ok`; one more prints `skipped` when the build has no GPU. Report the run count, never the `: ok` count.
- Commit as `Phaedrus <phaedrus1992x@gmail.com>`. The repo's configured identity is the sync bot, which must not carry a DCO sign-off to a third party. Use `git -c user.name=... -c user.email=...` and leave the config alone.
- A branch that is already pushed cannot be amended. Review fixes land as a further commit on the same branch; the maintainer squashes on import, so the branch's commit count does not matter.
- A `port/*` branch carries no `.fork/`, so the guard cannot be run from the working tree. Read it from `main`: `git show main:.fork/check-clean.sh > .tmp/check-clean.sh`.

## Review Focus

The spec is a vision document. These are the inputs and conditions it does not name. Each one is pinned by a test, or by a stated check where no test can reach it, in the task that owns the code.

1. **A path with no `/proc`.** `src/common/reopen.c` builds `/proc/self/fd/<n>` today. macOS has `/dev/fd/<n>`. A wrong path makes `janas_reopen_read` return -1, and the expert cache silently falls back to buffered reads rather than failing loudly.
2. **A signal that is already pending before the write.** `janas_mcp_proc_send` blocks `SIGPIPE`, writes, then drains. If `SIGPIPE` was pending on entry, it must be left pending on exit, as the current code does.
3. **A machine with one performance level.** `hw.perflevel1` is absent on an Intel Mac and on many arm64 servers. `janas_cpu_layout` must still return a usable layout rather than -1.
4. **A battery machine.** `janas_on_mains` returning -1 means "unknown", and `src/llm/api.c` maps that to the maximum setting. On a MacBook this must report battery correctly, or eco mode never engages.
5. **An empty or unexpected file.** `janas_exe_path` and `janas_cpu_name` must leave a NUL-terminated string in the caller's buffer on every failure path, because callers print it directly.

---

## Task 1: `port/21` — names the x86 guards leak

Closes #422. Spec §4 (`port/21`), §9 step 1.

The only slice that fixes a defect in shipped code, and the only one a Linux arm64 machine can verify with no Mac.

**Files:**
- Modify: `src/llm/quant.c` (`use_avx2_fma` at `:113`, and the `e8m0_tab` fill loop in `dot_kernels_init`)
- Modify: `src/llm/attention.c:661` (`attn_item`)
- Modify: `src/llm/deltanet.c:173`
- Modify: `src/llm/model_ple.c:133`, `src/llm/model_ple.c:229`
- Modify: `tests/test_quant.c:481`
- Test: the compiler itself. There is no unit test for a guard leak; the failing build is the test.

There are six sites, not five. `src/llm/quant.c:113 use_avx2_fma` is invisible at first: the hard error on `e8m0_tab` stops the compiler before it reports the unused-variable warning. It surfaces only after that error is gone. Expect the same masking elsewhere as each fix lands.

**Interfaces:**
- Consumes: nothing.
- Produces: nothing that later tasks call. This task only moves declarations; every function signature is unchanged.

- [ ] **Step 1: Sync and cut the branch from `clean`**

Run from the repo root, on `main`, with a clean tree:

```bash
./.fork/sync-upstream.sh
git switch -c port/21-arch-guard-leaks clean
```

Expected: the sync reports no new upstream commits, or merges them and finishes. `git branch --show-current` prints `port/21-arch-guard-leaks`.

- [ ] **Step 2: Re-take the line numbers**

Upstream moves. Confirm each site still matches before editing:

```bash
grep -n "e8m0_tab" src/llm/quant.c
grep -n "const int8_t \*pf = s0" src/llm/attention.c
grep -n "static int use_avx2, use_f16c" src/llm/deltanet.c
grep -n "int v = have_avx2" src/llm/model_ple.c
grep -n "float g\[1024\], u\[1024\], a\[1024\], b\[1024\]" tests/test_quant.c
```

Expected: five matches, near the spec's line numbers. If a site is gone, stop and re-derive it from the spec's cause description, not from the line number.

- [ ] **Step 3: Reproduce the failure on Linux arm64**

This is the failing test. It must fail before any edit.

```bash
docker run --rm -e TZ=Europe/Rome -v "$PWD:/w" -w /w ubuntu:24.04 bash -lc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq gcc tzdata
  uname -m && ./build.sh release build'
```

Expected: `aarch64`, then FAIL at `src/llm/attention.c:661: error: unused variable 'pf' [-Werror=unused-variable]`.

`tzdata` is not optional: without it four tests assert the wrong local time and report failures that have nothing to do with the code.

- [ ] **Step 4: Fix `quant.c` — move the fill loop inside the guard**

`e8m0_tab` is declared at `:1127`, inside the `#if defined(__x86_64__)` block that opens at `:1104` and closes at `:1144`. Its fill loop sits at `:1192-1194`, *between* that block and the next one that opens at `:1195`. On a non-x86 build the table does not exist, so the fill loop is a hard error.

Move the loop inside the guard that opens at `:1195`, so it reads:

```c
__attribute__((constructor)) static void dot_kernels_init(void)
{
#if defined(__x86_64__)
    for (int e = 0; e < 256; e++)
        e8m0_tab[e] = janas_e8m0_half((uint8_t)e);
    __builtin_cpu_init();
```

Do not move the declaration out of its guard. The table is read only by `quant_dotn.h`, which is included from inside a guard, so on arm64 it would be a set-but-never-read static.

- [ ] **Step 5: Fix `attention.c` — cast the parameter that only x86 uses**

At `attn_item`, `pf` is declared and then used only inside `#if defined(__x86_64__)` blocks. Add the same escape this file already uses at `:610-613`, directly after the declaration:

```c
        const int8_t *pf = s0 ? NULL : V;
#if !defined(__x86_64__)
        (void)pf;
#endif
```

- [ ] **Step 6: Fix `deltanet.c` — guard `use_f16c` only**

`use_f16c` is read only inside `#if defined(__x86_64__)`. `use_avx2` is not: `janas_gdn_kernel_name` (`:212-215`) reads it on every architecture and returns "avx2" or "ref". Guarding `use_avx2` breaks the build on arm64, so the two must be split:

```c
static int use_avx2; /* read by janas_gdn_kernel_name on every architecture */
#if defined(__x86_64__)
static int use_f16c;
#endif

__attribute__((constructor)) static void gdn_init(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
```

On arm64 `use_avx2` stays 0 and the name reported is "ref", which is the kernel that actually runs. The constructor and both dispatch sites at `:191` and `:204` already carry their own guards, so nothing else changes.

- [ ] **Step 7: Fix `model_ple.c` — cast `v` in both workers**

`v` is set by `have_avx2()` and read only inside an `#if defined(__x86_64__)` block, in `bf16_worker` and again in `f32_worker`. Add after each of the two `int v = have_avx2();` lines:

```c
    int v = have_avx2();
#if !defined(__x86_64__)
    (void)v;
#endif
```

- [ ] **Step 8: Fix `tests/test_quant.c` — guard the whole comparison**

In `test_det`, the scalar-versus-AVX2 comparison is x86-only in its entirety. Guarding only the inner block is not enough: `g` and `u` are then set but never read, and GCC reports `-Wunused-but-set-variable` (Apple clang does not — the two compilers disagree, so check both). Move `g`, `u`, `bad`, the round loop and the `CHECK` under one guard:

```c
#if defined(__x86_64__)
    float g[1024], u[1024];
    int bad = 0;
    for (int round = 0; round < 1000; round++) {
        ...
        float a[1024], b[1024];
        for (int i = 0; i < 1024; i++)
            a[i] = janas_silu_mul_det(g[i], u[i]);
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            det8(g, u, b, 1024);
            for (int i = 0; i < 1024; i++)
                bad += memcmp(&a[i], &b[i], 4) != 0;
        }
    }
    CHECK(bad == 0, "silu_mul_det: %d scalar/AVX2 differences", bad);
#endif
```

Do not move `a` and `b` back to the top declaration: on arm64 they would be unused there, which is the warning this step exists to remove. The x86-64 result is unchanged — the same comparisons run, in the same order.

- [ ] **Step 9: Verify Linux arm64**

Run the same command as Step 3.

Expected: the build proceeds past the five sites. If another `-Werror` failure appears further along, it is part of this slice only if its cause is a name a guard hides; add it here. If its cause is anything else, stop and report it — it belongs to another slice.

- [ ] **Step 10: Verify macOS arm64**

Libmicrohttpd still blocks a full `build.sh` on macOS until Task 2 lands, so check the five files directly:

```bash
for f in src/llm/quant.c src/llm/attention.c src/llm/deltanet.c \
         src/llm/model_ple.c tests/test_quant.c; do
  cc -fsyntax-only -std=gnu11 -D_GNU_SOURCE= -pthread -Isrc -Iinclude -fPIC \
     -fvisibility=hidden -Wall -Wextra -Werror -DJANAS_VERSION='"plan"' \
     -Isrc/third_party/libmicrohttpd/src/include -Isrc/third_party/yyjson \
     "$f" && echo "OK $f"
done
```

Expected: five `OK` lines. `json.h`'s `gnu_printf` does not reach these five files, so this is a complete check for them.

- [ ] **Step 11: Verify x86-64 on the owner's box**

Ask the owner to run, on the x86-64 Linux machine, from a clone on this branch:

```bash
./build.sh release test
```

Expected: PASS, and no change in any number it prints. If the owner is unavailable, mark this step blocked and do not open the pull request; this slice touches code compiled on x86-64 in the `#else` path and a silent change there is the one failure this plan cannot detect locally.

- [ ] **Step 12: Run the guard, then commit**

```bash
clang-format -i src/llm/quant.c src/llm/attention.c src/llm/deltanet.c \
    src/llm/model_ple.c tests/test_quant.c
git diff            # clang-format 23 also reflows pre-existing lines; revert those
git show main:.fork/check-clean.sh > .tmp/check-clean.sh
bash .tmp/check-clean.sh "$(git branch --show-current)"
git add src/llm/quant.c src/llm/attention.c src/llm/deltanet.c \
    src/llm/model_ple.c tests/test_quant.c
git commit -s -m "fix: names the x86 guards hide break every arm64 build"
```

Expected: `check-clean: ... is clean.` The commit carries a `Signed-off-by` line.

The guard cannot be run from the working tree on a `port/*` branch: `.fork/` is fork tooling and a port branch carries none. Read it from `main`, exactly as the pre-push hook does.

`clang-format` reflows lines it did not author when its version differs from the one upstream used. Revert those hunks by hand and keep the diff to the change under review.

- [ ] **Step 13: Push and open the upstream pull request**

The local pre-push hook is the only guard on this path, because a pull request against upstream never runs the fork's CI (`design.md` §5).

```bash
git push -u origin port/21-arch-guard-leaks
gh pr create --repo prabanta-dev/janas --base main \
  --head phaedrus1992:port/21-arch-guard-leaks \
  --title "make the non-x86 path compile: names the x86 guards hide" \
  --body-file .tmp/pr21.md
```

Write `.tmp/pr21.md` first. It states: what breaks (`e8m0_tab` outside its guard is a hard error on any arm64 target), the five sites, that every change is inside a `#if !defined(__x86_64__)` block or a moved declaration, and that `release test` passes on Linux arm64 and on x86-64. Do not close the fork's issue #422 from this body; refer to it without a closing keyword.

---

## Task 2: `port/20` — build portability

Closes #421. Spec §4 (`port/20`), §9 step 2.

**Files:**
- Modify: `build.sh:306`, `build.sh:312`, `build.sh:341`
- Create: `src/third_party/libmicrohttpd/darwin/MHD_config.h`
- Modify: `src/third_party/libmicrohttpd/origin.md`
- Test: a real `./build.sh release build` on macOS must reach the engine sources.

**Interfaces:**
- Consumes: nothing.
- Produces: a macOS build that gets past third-party, which Task 3 needs in order to see its own failures.

- [ ] **Step 1: Sync and cut the branch**

```bash
./.fork/sync-upstream.sh
git switch -c port/20-build-portability clean
```

- [ ] **Step 2: Reproduce**

```bash
./build.sh release build
```

Expected: FAIL at `src/third_party/libmicrohttpd/src/microhttpd/mhd_sockets.h:110: fatal error: 'sys/epoll.h' file not found`.

- [ ] **Step 3: Replace the GNU `find -printf`**

`build.sh:306` reads:

```bash
headers_newest=$(find src include "$SPV" "$GEN" -name '*.h' -printf '%T@\n' | sort -n | tail -1)
```

BSD `find` has no `-printf`. `date -r` is not portable either: `-r` takes seconds since the epoch on GNU and a file on BSD. Let `find` produce the names and a shell loop keep the newest:

```bash
newest_h=
while IFS= read -r -d '' h; do
    if [ -z "$newest_h" ] || [ "$h" -nt "$newest_h" ]; then
        newest_h=$h
    fi
done < <(find src include "$SPV" "$GEN" -name '*.h' -print0)
headers_newest=0
if [ -n "$newest_h" ]; then
    headers_newest=$(stat -c %Y "$newest_h" 2>/dev/null || stat -f %m "$newest_h")
fi
```

Do not use a pipeline that ends in `head -1`. `head` exits early, the writer dies of SIGPIPE, and under `set -o pipefail` the build stops right after the GPU note with no message.

`stat -c %Y` is GNU and `stat -f %m` is BSD. The GNU spelling goes first. A failing BSD `stat` prints nothing, but a failing GNU `stat -f` prints the file system block to stdout, and the command substitution captures it. GNU `stat` is also what a Mac with Homebrew coreutils or uutils on `PATH` runs.

An empty tree leaves `newest_h` empty, the `stat` never runs, and `headers_newest` stays 0 — which is the existing behaviour for a tree with no objects yet.

- [ ] **Step 4: Replace the GNU `stat -c`**

`build.sh:312` reads:

```bash
[ "$(stat -c %Y "$obj")" -lt "${headers_newest%.*}" ]; then
```

Change it to the same two-spelling form:

```bash
[ "$(stat -c %Y "$obj" 2>/dev/null || stat -f %m "$obj")" -lt "$headers_newest" ]; then
```

`headers_newest` is now plain seconds, so drop the `${headers_newest%.*}` suffix strip.

- [ ] **Step 5: Drop the duplicate `-D_GNU_SOURCE`**

`build.sh:128` puts `-D_GNU_SOURCE=` into `CFLAGS`. `build.sh:341` adds a bare `-D_GNU_SOURCE` for libmicrohttpd. `-D_GNU_SOURCE=` defines the macro as empty; `-D_GNU_SOURCE` defines it as `1`. The two differ, so GCC and clang both report `_GNU_SOURCE redefined` under `-Werror`.

Remove the bare `-D_GNU_SOURCE` from the libmicrohttpd arm of `tp_flags`, because `CFLAGS` already carries it and `tp_flags` is copied into `TP_CFLAGS`:

```bash
    libmicrohttpd)
        # Windows has a configuration of its own (see its origin.md)
        [ "$OS" = windows ] && echo "-I$d/w32"
        echo "-DHAVE_CONFIG_H -I$d -I$d/src/include" \
            "-I$d/src/microhttpd" ;;
```

- [ ] **Step 6: Generate the darwin configuration**

`MHD_config.h` is not part of the upstream tarball; it is what `configure` writes. `origin.md` records the tarball, its hash, and the exact options. Work in a directory of its own, because the tarball is untrusted data:

```bash
mkdir -p .tmp/lmhd && cd .tmp/lmhd
curl -fsSLO https://ftp.gnu.org/gnu/libmicrohttpd/libmicrohttpd-1.0.10.tar.gz
shasum -a 256 -c <<<"04bfe8ef75db7d629a33de767599765cecadc56274a39822d5d081030d577685  libmicrohttpd-1.0.10.tar.gz"
tar xf libmicrohttpd-1.0.10.tar.gz && cd libmicrohttpd-1.0.10
./configure --enable-https=no --disable-bauth --disable-dauth \
  --disable-postprocessor --disable-cookie --disable-httpupgrade \
  --enable-md5=no --enable-sha256=no --disable-sha512-256 --disable-doc \
  --disable-examples --disable-tools --disable-curl --with-threads=posix \
  --disable-shared
```

Expected: `configure` completes. A failed hash check means stop.

- [ ] **Step 7: Install the configuration and diff it**

```bash
mkdir -p src/third_party/libmicrohttpd/darwin
cp .tmp/lmhd/libmicrohttpd-1.0.10/MHD_config.h \
   src/third_party/libmicrohttpd/darwin/MHD_config.h
diff <(grep -c . src/third_party/libmicrohttpd/MHD_config.h) \
     <(grep -c . src/third_party/libmicrohttpd/darwin/MHD_config.h)
grep -n "EPOLL\|EVENTFD\|SENDFILE\|ACCEPT4\|SETNAME_NP\|_MHD_ITC" \
   src/third_party/libmicrohttpd/darwin/MHD_config.h
```

Expected: `EPOLL_SUPPORT`, `HAVE_EPOLL_CREATE1`, `HAVE_LINUX_SENDFILE`, `HAVE_ACCEPT4`, `HAVE_SOCK_NONBLOCK` and `_MHD_ITC_EVENTFD` all commented out or absent; `HAVE_PTHREAD_SETNAME_NP_DARWIN` defined; `HAVE_DARWIN_SENDFILE` defined. If any of those disagree, `configure` did not see the macOS headers and the copy must not be used.

- [ ] **Step 8: Select the configuration in `build.sh`**

Extend the libmicrohttpd arm of `tp_flags` to mirror the Windows precedent:

```bash
    libmicrohttpd)
        # Windows and darwin have configurations of their own (see its origin.md)
        [ "$OS" = windows ] && echo "-I$d/w32"
        [ "$OS" = darwin ] && echo "-I$d/darwin"
        echo "-DHAVE_CONFIG_H -I$d -I$d/src/include" \
            "-I$d/src/microhttpd" ;;
```

- [ ] **Step 9: Record the new configuration in `origin.md`**

Add a paragraph beside the existing `w32/MHD_config.h` paragraph. State the date, that `configure` ran natively on macOS arm64 (Apple clang 21, macOS 27), the same options as the Linux run, and that the same 14 sources compile. `origin.md` is the file that makes this reproducible; an unrecorded configuration does not belong in the tree.

- [ ] **Step 10: Verify macOS arm64**

```bash
./build.sh release build
```

Expected: the build gets past `src/third_party/libmicrohttpd/` and fails later, in the engine sources. A later failure is Task 3's, not this one's. Do not fix engine files here.

- [ ] **Step 11: Verify Linux arm64**

```bash
docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04 bash -lc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq gcc
  ./build.sh release build'
```

Expected: unchanged from Task 1's result. This branch must not alter the Linux build path.

- [ ] **Step 12: Verify x86-64 on the owner's box**

Ask the owner to run `./build.sh clean-build` (not `build`) on the x86-64 machine. `clean-build` matters here and nowhere else in this plan: this branch changes the build script's own staleness logic, and a stale object would hide a mistake in it.

Expected: a full build from nothing, then PASS on `./build.sh release test`.

- [ ] **Step 13: Guard, commit, push, open the pull request**

```bash
./.fork/check-clean.sh
git add build.sh src/third_party/libmicrohttpd/darwin/MHD_config.h \
    src/third_party/libmicrohttpd/origin.md
git commit -s -m "build: portable file times, one GNU_SOURCE, a darwin config"
git push -u origin port/20-build-portability
```

Then open the pull request as in Task 1 Step 13, with `--base main --head phaedrus1992:port/20-build-portability` and a `.tmp/pr20.md` that names the two GNU extensions, the duplicate define, and the new configuration with its `origin.md` record.

---

## Task 3: `port/23` — what Apple clang refuses

Closes #424. Spec §4 (`port/23`), §9 step 3.

This is the branch that takes the macOS failure count from 169 to 15, with Task 2 applied. On `clean` alone it removes 146 failing files; the other 8 are Task 2's libmicrohttpd files.

**Files:**
- Create: `src/common/printf_like.h`
- Modify: the 14 `format(gnu_printf` sites in 13 files: `src/llm/json.h`, `src/llm/chat.h`, `src/llm/metrics.c`, `src/chat/term.h`, `src/common/mcp_client.h`, `src/server/server.h`, `src/server/request.h`, `src/services/git/git.h`, `src/services/prices/prices.h`, `src/services/quakes/tools.c`, `src/services/system/tools.c`, `src/bench/main.c`, and `src/chat/main.c`
- Modify: `src/chat/main.c:17`
- Modify: `src/chat/tune.c:32`
- Modify: `tests/bench_kernels.c:123`
- Test: the compiler. The failing build is the test.

**Interfaces:**
- Consumes: Task 2's `build.sh`, so that `./build.sh` reaches the engine sources on macOS.
- Produces: nothing later tasks call.

- [ ] **Step 1: Sync and cut the branch**

```bash
./.fork/sync-upstream.sh
git switch -c port/23-clang-dialect clean
```

- [ ] **Step 2: Reproduce**

```bash
./build.sh release build
```

Expected: FAIL in an engine source with `error: 'format' attribute argument not supported: gnu_printf`.

- [ ] **Step 3: Guard the format attribute**

`src/llm/json.h:68` reads:

```c
    __attribute__((format(gnu_printf, 2, 3)));
```

Apple clang does not accept the `gnu_printf` argument and reports `-Wignored-attributes`, which `-Werror` makes fatal. `json.h` is not the only site: `git grep 'format(gnu_printf'` finds 14 sites in 13 files. Put the macro in a header of its own, `src/common/printf_like.h`, and include it in each of the 13 files directly, not through another header:

```c
/* gnu_printf checks the GNU extensions in a format string; Apple clang
   does not accept the argument, so it gets the plain one. */
#if defined(__APPLE__)
#define JANAS_PRINTF_LIKE(a, b) __attribute__((format(printf, a, b)))
#else
#define JANAS_PRINTF_LIKE(a, b) __attribute__((format(gnu_printf, a, b)))
#endif
```

Then each declaration becomes `JANAS_PRINTF_LIKE(a, b);` with its own argument numbers. Guard only on `__APPLE__` and leave `gnu_printf` everywhere else: changing it globally would alter `-Wformat` on the x86-64 path, which must not move.

- [ ] **Step 4: Verify the count drops**

```bash
git show main:.fork/sweep-syntax.sh > .tmp/sweep-syntax.sh
bash .tmp/sweep-syntax.sh .tmp/sweep-after-json.txt
```

The sweep is `.fork/sweep-syntax.sh` on `main`. A port branch carries no `.fork/`, so copy it from `main` first. It compiles every source `build.sh` compiles, with `build.sh`'s flags, applying `-w` to third-party and `-Wall -Wextra -Werror` to everything else. It writes stub headers for the generated web pages, so a missing or stale `lib/` does not change the count. The baselines on `clean` are 169 of 375 on macOS arm64 and 4 of 375 on Linux arm64.

Expected: 143 `json.h` files leave the list. A smaller drop means the macro did not reach every use of the attribute; `git grep 'format(gnu_printf'` must print `src/common/printf_like.h` only.

- [ ] **Step 5: Fix `chat/main.c`**

`:17` defines `_XOPEN_SOURCE 700` for `wcwidth`. On macOS that alone hides the Darwin extensions, so `SIGWINCH` (`:1868`) and `_SC_NPROCESSORS_ONLN` (`:518`) lose their declarations. Add the companion macro:

```c
#define _XOPEN_SOURCE 700 /* wcwidth, for the width of what is printed */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE /* _XOPEN_SOURCE alone hides SIGWINCH and _SC_* */
#endif
```

- [ ] **Step 6: Verify the two symbols come back**

```bash
for f in src/llm/json.h src/chat/main.c; do :; done
cc -fsyntax-only -std=gnu11 -D_GNU_SOURCE= -pthread -Isrc -Iinclude -fPIC \
   -fvisibility=hidden -Wall -Wextra -Werror -DJANAS_VERSION='"plan"' \
   -Isrc/third_party/libmicrohttpd/src/include -Isrc/third_party/yyjson \
   src/chat/main.c && echo "OK chat/main.c"
```

Expected: `OK chat/main.c`.

- [ ] **Step 7: Reflow the split literal**

`src/chat/tune.c:32` splits one string across two lines inside an array initializer. Apple clang reports `-Wstring-concatenation`, which is a false positive here but is still fatal under `-Werror`. Put it on one line:

```c
    "Write a C function that reverses a singly linked list, with a short comment.",
```

- [ ] **Step 8: Cast the constant conversion**

`tests/bench_kernels.c:123` computes `(float)rand() / RAND_MAX`. `RAND_MAX` is `2147483647`, and converting it to `float` rounds to `2147483648`; Apple clang reports `-Wimplicit-const-int-float-conversion`. Make the conversion explicit:

```c
        v[i] = (float)rand() / (float)RAND_MAX - 0.5f;
```

The value is unchanged: the implicit conversion already did exactly this. State that in the commit message so the maintainer does not read it as a behaviour change.

- [ ] **Step 9: Verify macOS arm64**

```bash
bash .tmp/sweep-syntax.sh .tmp/sweep-final.txt
```

Expected: `tests/bench_kernels.c` and `src/chat/tune.c` are gone from the list. `src/llm/json.h`'s 143 files are gone. With Task 2 applied, 15 files fail, and all of them belong to Tasks 1, 4 and 5.

- [ ] **Step 10: Verify Linux arm64**

Run Task 2 Step 11's container command.

Expected: unchanged. `__APPLE__` is not defined there, so every change in this branch is inert.

- [ ] **Step 11: Verify x86-64 on the owner's box**

Ask the owner to run `./build.sh release test`.

Expected: PASS with no change in output. Note in particular that `RAND_MAX`'s conversion and the `-Wformat` set are both unchanged.

- [ ] **Step 12: Verify the failure count with `-Werror` on Linux too**

Clippy and clang can hide a warning GCC would raise. Confirm GCC still builds clean:

```bash
docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04 bash -lc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq gcc
  bash .tmp/sweep-syntax.sh .tmp/sweep-gcc.txt'
```

Expected: the same list as before this branch, minus nothing. A new entry means a change altered a GCC warning; investigate it before committing.

- [ ] **Step 13: Guard, commit, push, open the pull request**

```bash
./.fork/check-clean.sh
git add src/common/printf_like.h $(git diff --name-only)
git commit -s -m "build: the four constructs Apple clang refuses"
git push -u origin port/23-clang-dialect
```

Then open the pull request as in Task 1 Step 13, with a `.tmp/pr23.md` that leads with the `json.h` one-liner and its 143-file effect, because that is the change a reviewer most needs to understand.

---

## Task 4: `port/22` — the Linux-only calls

Closes #423. Spec §4 (`port/22`), §9 step 4.

**Files:**
- Modify: `src/common/sock.c:120`
- Modify: `src/common/reopen.c:41` and its `/proc/self/fd` path
- Modify: `src/common/mcp_stdio.c:105`
- Modify: `src/common/proc.c:86`
- Modify: `src/llm/api_keep_disk.c:57`
- Modify: `src/llm/expert_cache.c:520`
- Modify: `src/server/http.c:44`
- Modify: `tests/bench_long.c:33`
- Modify: `tests/jns_check.c:76`
- Test: the compiler, plus the full test suite on both arm64 targets at the end.

**Interfaces:**
- Consumes: Tasks 2 and 3, so that macOS reaches these files.
- Produces: nothing new. Every function in this task keeps its existing signature and its existing return contract; the changes are inside the bodies. `janas_sock_open` still returns a descriptor or -1, and `janas_reopen_read` still returns a descriptor or -1.

- [ ] **Step 1: Sync and cut the branch**

```bash
./.fork/sync-upstream.sh
git switch -c port/22-os-call-sites clean
```

- [ ] **Step 2: Fix `sock.c` — the two socket flags**

macOS has neither flag, verified by compiling them. Keep the Linux call exactly as it is and add a macOS branch, rather than replacing both: passing the two flags to `socket` is atomic, and a descriptor that is briefly without `FD_CLOEXEC` is a real window in a program with a thread pool.

```c
int janas_sock_open(int family, int type, int protocol)
{
#if defined(__linux__)
    return socket(family, type | SOCK_CLOEXEC | SOCK_NONBLOCK, protocol);
#else
    int fd = socket(family, type, protocol);
    if (fd < 0)
        return -1;
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(fd, F_SETFL, O_NONBLOCK) != 0) {
        int e = errno;
        close(fd);
        errno = e;
        return -1;
    }
    return fd;
#endif
}
```

- [ ] **Step 3: Fix `reopen.c` — the path and the direct flag**

Two problems, not one. The spec named only `O_DIRECT`. The function also builds `/proc/self/fd/%d`, which macOS does not have; it has `/dev/fd/%d`. Write the two branches whole, so the Linux line stays identical:

```c
int janas_reopen_read(int fd, int direct)
{
    char path[64];
#if defined(__APPLE__)
    snprintf(path, sizeof(path), "/dev/fd/%d", fd);
    int n = open(path, O_RDONLY | O_CLOEXEC);
    if (n < 0)
        return -1;
    /* F_NOCACHE is what macOS has: it turns the unified buffer cache off
       for the file, rather than requiring aligned transfers as O_DIRECT
       does. Close enough for the expert cache's reads. */
    if (direct && fcntl(n, F_SETFL, F_NOCACHE) != 0) {
        int e = errno;
        close(n);
        errno = e;
        return -1;
    }
    return n;
#else
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    return open(path, O_RDONLY | (direct ? O_DIRECT : 0) | O_CLOEXEC);
#endif
}
```

The `F_NOCACHE` on macOS is not equivalent to `O_DIRECT`: it disables the unified buffer cache rather than demanding aligned I/O. That is the closest macOS has, and it is what the spec chose. Say so in the commit message so the maintainer is not left to infer it.

- [ ] **Step 4: Fix `mcp_stdio.c` — replace `sigtimedwait` portably**

`sigtimedwait` is not on macOS. The drain has an exact portable equivalent: check whether `SIGPIPE` is pending on this thread, and consume it only if it is. Replace the `if (rc && !was_pending) { ... }` block with:

```c
    if (rc && !was_pending) {
        sigset_t now;
        sigpending(&now);
        if (sigismember(&now, SIGPIPE)) {
            int s;
            sigwait(&pipe_set, &s);
        }
    }
```

`sigwait` blocks, so it must only be called when the signal is already pending; the `sigpending` check guards that. Both calls are POSIX and exist on macOS.

Do not simplify the outer condition to `if (rc)`. `was_pending` is read before the mask changes, and it is what keeps a `SIGPIPE` that was already pending on entry from being swallowed here. Verify with the existing pair — `tests/test_mcp.c` drives `tests/mcp_fake_server.c`, whose server closes its pipe, which is exactly the case this drain exists for:

```bash
./build.sh release test 2>&1 | grep -E "test_mcp|FAILED"
```

Expected: `test_mcp: ok`.

- [ ] **Step 5: Fix `proc.c` — the deprecated spawn call**

`posix_spawn_file_actions_addchdir_np` is deprecated as of macOS 26 and the deprecation is an error. Use the new spelling where it exists:

```c
    if (dir) {
#if defined(__APPLE__) && defined(__MAC_26_0)
        posix_spawn_file_actions_addchdir(&fa, dir);
#else
        posix_spawn_file_actions_addchdir_np(&fa, dir);
#endif
    }
```

Check that `__MAC_26_0` is the spelling the SDK's `<Availability.h>` uses before relying on it; if it is not defined on an older SDK the `_np` form is still correct there.

- [ ] **Step 6: Fix `api_keep_disk.c` — the stat field**

macOS spells the modification time `st_mtimespec`:

```c
static int64_t mtime_ns(const struct stat *st)
{
#ifdef _WIN32
    return (int64_t)st->st_mtime * 1000000000; /* seconds only there */
#elif defined(__APPLE__)
    return (int64_t)st->st_mtimespec.tv_sec * 1000000000 +
           st->st_mtimespec.tv_nsec;
#else
    return (int64_t)st->st_mtim.tv_sec * 1000000000 + st->st_mtim.tv_nsec;
#endif
}
```

- [ ] **Step 7: Fix `expert_cache.c` — the madvise that macOS lacks**

The `madvise` is already gated at run time by `JANAS_ARENA_HUGE`. It also needs a compile-time gate, because `MADV_HUGEPAGE` does not exist on macOS:

```c
#ifdef MADV_HUGEPAGE
    if (getenv("JANAS_ARENA_HUGE") && atoi(getenv("JANAS_ARENA_HUGE")) > 0)
        madvise(c->arena, arena_bytes, MADV_HUGEPAGE);
#endif
```

The long comment above it stays: it records why the call is off by default, and that reason does not change.

- [ ] **Step 8: Fix `http.c` — `getrandom`**

macOS has `getentropy`, which takes two arguments and caps at 256 bytes; this call wants 12.

```c
void srv_new_id(char *buf, size_t cap, const char *prefix)
{
    unsigned char r[12];
#if defined(__APPLE__)
    int have = getentropy(r, sizeof(r)) == 0;
#else
    int have = getrandom(r, sizeof(r), 0) == (ssize_t)sizeof(r);
#endif
    if (!have)
        memset(r, 0, sizeof(r));
```

Keep the existing `memset` fallback and the hex loop below unchanged.

- [ ] **Step 9: Fix the two tests**

`tests/bench_long.c:33` includes `<sys/sysmacros.h>` for `major()` and `minor()` at `:74`. That header is Linux-only. macOS has the same macros in `<sys/types.h>`, which the file already includes transitively:

```c
#ifndef _WIN32
#if defined(__linux__)
#include <sys/sysmacros.h>
#endif
#endif
```

`tests/jns_check.c:76` opens with `O_DIRECT` and prints `cannot open with O_DIRECT`. Replace the flag the same way as `reopen.c`, and update the message so it names what it actually did:

```c
#if defined(__APPLE__)
        int fd = open(argv[1], O_RDONLY);
        if (fd >= 0 && fcntl(fd, F_SETFL, F_NOCACHE) != 0) {
            close(fd);
            fd = -1;
        }
#else
        int fd = open(argv[1], O_RDONLY | O_DIRECT);
#endif
```

and change the failure message from `cannot open with O_DIRECT` to `cannot open the file uncached`. The test's alignment requirement (`JANAS_JNS_ALIGN`) still holds on both.

- [ ] **Step 10: Verify macOS arm64**

```bash
./build.sh release build
```

Expected: PASS, or a failure in `src/common/sysinfo.c` — which is Task 5's file, not this one's. If `sysinfo.c` is the only thing left failing, this branch is complete.

- [ ] **Step 11: Verify Linux arm64**

Run Task 2 Step 11's container command.

Expected: PASS, or the same `sysinfo.c`-only gap. `uring.c` needs nothing: its stub already returns NULL, and `io_worker` already reads with `pread`.

- [ ] **Step 12: Verify x86-64 on the owner's box**

Ask the owner to run `./build.sh release test`.

Expected: PASS. The two tests this branch touches must still pass; `jns_check`'s `--verify` path is the one to watch, because it is the only place the direct-I/O flag is exercised.

- [ ] **Step 13: Guard, commit, push, open the pull request**

```bash
clang-format -i src/common/sock.c src/common/reopen.c src/common/mcp_stdio.c \
    src/common/proc.c src/llm/api_keep_disk.c src/llm/expert_cache.c \
    src/server/http.c tests/bench_long.c tests/jns_check.c
./.fork/check-clean.sh
git add src/common/sock.c src/common/reopen.c src/common/mcp_stdio.c \
    src/common/proc.c src/llm/api_keep_disk.c src/llm/expert_cache.c \
    src/server/http.c tests/bench_long.c tests/jns_check.c
git status --short   # confirm the nine files and nothing else
git commit -s -m "port the nine linux-only calls to macos"
git push -u origin port/22-os-call-sites
```

Then open the pull request as in Task 1 Step 13. `.tmp/pr22.md` must state, for each site, that the Linux and Windows forms are unchanged, and must name `F_NOCACHE` as an approximation of `O_DIRECT` rather than an equivalent.

---

## Task 5: `port/24` — sysinfo and the thread pool on macOS

Closes #425. Spec §4 (`port/24`), §5, §9 step 5.

The largest slice. The whole non-Windows half of `src/common/sysinfo.c` is Linux-specific.

**Files:**
- Modify: `src/common/sysinfo.c` (lines 54-291, the `#ifndef _WIN32` region)
- Modify: `src/common/pool.c` only if the pin call needs a guard; the call site at `:130-132` should not change
- Create: `tests/test_sysinfo.c`
- Test: `tests/test_sysinfo.c` and the full suite on both arm64 targets.

**Interfaces:**
- Consumes: nothing new.
- Produces: the same `sysinfo.h` contract for every function. No signature changes. `janas_cpu_split` keeps returning 1, 0 or -1, and `janas_on_mains` keeps returning 1, 0 or -1.

- [ ] **Step 1: Sync and cut the branch**

```bash
./.fork/sync-upstream.sh
git switch -c port/24-sysinfo-macos clean
```

- [ ] **Step 2: Fix the two callers first**

`janas_cpu_layout` and `janas_cpu_split` call `janas_os_allowed_cpus` and `janas_os_cpu_kinds`, which live in the same file. Bring the file to a state where the shared half compiles on both systems before adding the macOS half: move the Linux-only helpers (`read_cpu_list`, `core_leader`, `read_long`, `meminfo`, `janas_cpu_split`, `janas_os_allowed_cpus`, `janas_os_cpu_kinds`, `janas_cpus_save`, `janas_cpus_restore`, `janas_pin_current_thread`, `janas_exe_path`, `janas_cpu_name`, `janas_on_mains`, `janas_mem_available`, `janas_mem_total`) inside `#if defined(__linux__)`, leaving `janas_cpu_layout` unguarded. Confirm the file still fails on macOS, now with missing symbols rather than wrong ones.

- [ ] **Step 3: Add the macOS CPU count and kinds**

macOS has no affinity, so a CPU index is a label and not a placement. That is why the perflevel *counts* are enough: `janas_cpu_layout` only needs an ordering, and the scheduler ignores the tags anyway.

```c
#elif defined(__APPLE__)
#include <sys/sysctl.h>

static int sysctl_int(const char *name, int *out)
{
    size_t n = sizeof(*out);
    return sysctlbyname(name, out, &n, NULL, 0) == 0 && *out > 0;
}

int janas_os_allowed_cpus(int *out, int max)
{
    int n = 0;
    if (!sysctl_int("hw.logicalcpu", &n))
        return -1;
    if (n > max)
        n = max;
    for (int i = 0; i < n; i++)
        out[i] = i;
    return n;
}

void janas_os_cpu_kinds(const int *cand, int nc, int *kind, int *leader)
{
    int p = 0, e = 0;
    /* one performance level: every CPU is a performance core of its own */
    if (!sysctl_int("hw.perflevel0.logicalcpu", &p) || p > nc)
        p = nc;
    (void)sysctl_int("hw.perflevel1.logicalcpu", &e);
    for (int i = 0; i < nc; i++) {
        kind[i] = i < p ? 0 : 1;
        leader[i] = cand[i]; /* no sibling information: each leads itself */
    }
}
```

`hw.perflevel1` is absent on a single-level machine; `e` staying 0 is correct there, and every index at or past `p` would not exist because `p == nc`.

- [ ] **Step 4: Add the macOS split, save, restore and pin**

```c
int janas_cpu_split(int *compute, int *n_compute, int *io, int *n_io, int max)
{
    *n_compute = janas_os_allowed_cpus(compute, max);
    if (*n_compute <= 0)
        return -1;
    /* no affinity: the I/O threads are not pinned to anything */
    *n_io = 0;
    return 0;
}

int janas_cpus_save(unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    (void)save;
    return 0; /* nothing to save: there is no affinity to give back */
}

void janas_cpus_restore(const unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    (void)save;
}
```

For the pin, replace the Linux `pthread_setaffinity_np` with a QoS class and no placement:

```c
static void set_qos(int qos_class)
{
    pthread_set_qos_class_self_np(qos_class, 0);
}

int janas_pin_current_thread(int cpu)
{
    (void)cpu;
    set_qos(QOS_CLASS_USER_INITIATED);
    return 0; /* macOS has no CPU affinity */
}
```

Include `<pthread/qos.h>` and `<pthread.h>` for the macOS arm. The worker threads get `QOS_CLASS_USER_INITIATED`; the caller keeps its own class (`QOS_CLASS_USER_INTERACTIVE`, which a foreground process already has), and I/O threads `QOS_CLASS_UTILITY`. If `pool.c` has a place where the I/O pool is created, set `QOS_CLASS_UTILITY` there; otherwise set only the worker class here and note it in the pull request.

- [ ] **Step 5: Add the macOS paths, name and memory**

```c
int janas_exe_path(char *buf, size_t len)
{
    uint32_t n = (uint32_t)len;
    if (len == 0)
        return -1;
    if (_NSGetExecutablePath(buf, &n) != 0) {
        buf[0] = 0;
        return -1;
    }
    buf[len - 1] = 0;
    return (int)strlen(buf);
}

void janas_cpu_name(char *buf, size_t len)
{
    size_t n = len;
    buf[0] = 0;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, NULL, 0) != 0)
        buf[0] = 0;
    buf[len - 1] = 0;
}
```

`_NSGetExecutablePath` needs `<mach-o/dyld.h>`.

For memory, `hw.memsize` is the total and `host_statistics64` gives the free pages:

```c
uint64_t janas_mem_total(void)
{
    uint64_t v = 0;
    size_t n = sizeof(v);
    if (sysctlbyname("hw.memsize", &v, &n, NULL, 0) != 0)
        return 0;
    return v;
}

uint64_t janas_mem_available(void)
{
    mach_port_t host = mach_host_self();
    vm_size_t page = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(host, &page) != KERN_SUCCESS)
        return 0;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm,
                          &count) != KERN_SUCCESS)
        return 0;
    return ((uint64_t)vm.free_count + vm.speculative_count) * page;
}
```

Needs `<mach/mach.h>` and `<mach/vm_statistics.h>`. `free_count` alone under-reports because macOS keeps a large speculative pool; both are reclaimable on demand, which is what "available to new allocations" means in `sysinfo.h`.

- [ ] **Step 5b: Pin the contract with a test**

The two things this task can get quietly wrong are a single-performance-level machine and an unterminated buffer. `build.sh` builds every `tests/*.c` without a build-script change, so add one:

Create `tests/test_sysinfo.c`:

```c
/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_sysinfo.c - the machine's own answers stay inside the contract in
 * sysinfo.h on every system: a layout or a refusal, a split that agrees
 * with its own return value, and a name that is always terminated.
 */
#include <stdio.h>
#include <string.h>

#include "common/sysinfo.h"

int main(void)
{
    int failures = 0;

    struct janas_cpu_layout l;
    int r = janas_cpu_layout(NULL, 0, &l);
    if (r == 0) {
        /* a layout that claims more threads than cores, or more cores
           than cpus, would make every prefix count wrong */
        if (l.n <= 0 || l.n > JANAS_MAX_CPUS || l.n_p_cores < 0 ||
            l.n_p_cores > l.n_p_threads || l.n_p_threads > l.n) {
            printf("layout: n=%d p_cores=%d p_threads=%d\n", l.n,
                   l.n_p_cores, l.n_p_threads);
            failures++;
        }
    } else if (r != -1) {
        printf("layout: returned %d, not 0 or -1\n", r);
        failures++;
    }

    int compute[JANAS_MAX_CPUS], io[JANAS_MAX_CPUS], nc = 0, ni = 0;
    r = janas_cpu_split(compute, &nc, io, &ni, JANAS_MAX_CPUS);
    if (r != 0 && r != 1 && r != -1) {
        printf("split: returned %d, not 0, 1 or -1\n", r);
        failures++;
    } else if (r == 1 && ni <= 0) {
        printf("split: says io lists cpus but n_io=%d\n", ni);
        failures++;
    } else if (r == 0 && ni != 0) {
        printf("split: says the io threads are unpinned but n_io=%d\n", ni);
        failures++;
    }

    /* every length leaves a terminated string, because callers print it */
    for (size_t len = 1; len <= 4; len++) {
        char b[8];
        memset(b, 'x', sizeof(b));
        janas_cpu_name(b, len);
        if (b[len - 1] != 0) {
            printf("cpu_name: len %zu left the buffer unterminated\n", len);
            failures++;
        }
    }
    for (size_t len = 1; len <= 4; len++) {
        char b[8];
        memset(b, 'x', sizeof(b));
        (void)janas_exe_path(b, len);
        if (b[0] != 0 && b[len - 1] != 0) {
            printf("exe_path: len %zu left the buffer unterminated\n", len);
            failures++;
        }
    }

    printf("test_sysinfo: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
```

Run it on both arm64 targets:

```bash
./build.sh release test 2>&1 | grep -E "test_sysinfo|FAILED"
```

Expected: `test_sysinfo: ok`. On this Mac, `janas_cpu_layout` returns a two-level layout (8 performance, 2 efficiency), so the single-level branch is not exercised here — Task 6's arm64 Linux runner is where that happens, and this test is what will catch it there.

- [ ] **Step 6: Add the macOS power source**

`janas_on_mains` returning -1 is contract-valid but wrong on a MacBook: `src/llm/api.c` treats anything that is not 0 as mains, so eco mode would never engage. Load IOKit the way the engine loads Vulkan — at run time, with `dlopen`, so the build keeps its no-dependency rule.

Compare against IOKit's exported constant objects rather than hard-coded strings. The framework binary lives in the dyld shared cache, so its text is not greppable on disk; `dlsym` reaches it at run time and needs no literal.

```c
#include <dlfcn.h>

int janas_on_mains(void)
{
    void *io = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit",
                      RTLD_LAZY);
    void *cf = dlopen("/System/Library/Frameworks/CoreFoundation.framework/"
                      "CoreFoundation", RTLD_LAZY);
    if (!io || !cf) {
        if (io)
            dlclose(io);
        if (cf)
            dlclose(cf);
        return -1;
    }
    int r = -1;
    const void *(*copy_info)(void) =
        (const void *(*)(void))dlsym(io, "IOPSCopyPowerSourcesInfo");
    const void *(*prov_type)(const void *) =
        (const void *(*)(const void *))dlsym(io,
                                             "IOPSGetProvidingPowerSourceType");
    const void *const *ac =
        (const void *const *)dlsym(io, "kIOPMACPowerKey");
    const void *const *bat =
        (const void *const *)dlsym(io, "kIOPMBatteryPowerKey");
    unsigned char (*equal)(const void *, const void *) =
        (unsigned char (*)(const void *, const void *))dlsym(cf, "CFEqual");
    void (*release)(const void *) =
        (void (*)(const void *))dlsym(cf, "CFRelease");
    if (copy_info && prov_type && ac && bat && equal && release) {
        const void *info = copy_info();
        if (info) {
            const void *t = prov_type(info);
            if (t && equal(t, *ac))
                r = 1;
            else if (t && equal(t, *bat))
                r = 0;
            release(info); /* Copy gave us ownership of info, not of t */
        }
    }
    dlclose(io);
    dlclose(cf);
    return r;
}
```

`IOPSCopyPowerSourcesInfo` follows the Copy rule, so `info` is released; `IOPSGetProvidingPowerSourceType` returns a borrowed reference, so `t` is not. If any symbol is missing, `r` stays -1, which the header already documents as "unknown".

Verify on this MacBook by printing `janas_on_mains()` with the charger plugged in and again unplugged: 1 then 0. If the charger cannot be unplugged, check that the plugged-in case gives 1, and file a follow-up issue for the battery case rather than shipping an unverified branch.

- [ ] **Step 7: Verify macOS arm64**

```bash
./build.sh release test
```

Expected: PASS. This is the first task where the full suite can run on macOS.

- [ ] **Step 8: Verify Linux arm64**

```bash
docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04 bash -lc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq gcc
  ./build.sh release test'
```

Expected: PASS. Every Linux path is inside its own guard and must be byte-identical to before.

- [ ] **Step 9: Verify x86-64 on the owner's box**

Ask the owner to run `./build.sh release test` and, if a model is at hand, `janas-bench` on the same model as the last recorded run. Expected: PASS, and no change in the benchmark.

- [ ] **Step 10: Guard, commit, push, open the pull request**

```bash
clang-format -i src/common/sysinfo.c tests/test_sysinfo.c
./.fork/check-clean.sh
git add src/common/sysinfo.c src/common/pool.c tests/test_sysinfo.c
git status --short   # confirm only those files, and drop pool.c if untouched
git commit -s -m "sysinfo: macos cpus, qos classes and memory"
git push -u origin port/24-sysinfo-macos
```

Then open the pull request as in Task 1 Step 13. `.tmp/pr24.md` must disclose the `dlopen` of IOKit and ask whether the maintainer prefers it to returning -1, because it is the one place this branch reaches outside the C library.

---

## Task 6: `port/25` — arm64 CI, with issue #13

Closes #426 and #13. Spec §4 (`port/25`), §9 step 6.

**Files:**
- Modify: `.github/workflows/release.yml` (runners, plus #13's hardening)
- Test: a real workflow run on the pull request.

**Interfaces:**
- Consumes: Tasks 1 to 5. Do not open this pull request until the other five are merged upstream, because the maintainer gated arm64 CI on the portability work landing.
- Produces: nothing.

- [ ] **Step 1: Check the gate, then cut the branch**

Confirm all five upstream pull requests are merged. If any is open, stop and report; this branch must not open early.

```bash
./.fork/sync-upstream.sh
git switch -c port/25-ci-arm64 clean
```

- [ ] **Step 2: Re-verify the runner labels**

They move. Do not trust the spec's list.

```bash
gh api repos/actions/runner-images --jq '.description'
```

and read GitHub's hosted-runner table for the current arm64 labels. Expected: `macos-14`, `macos-15`, `macos-26` are arm64, and `ubuntu-24.04-arm` exists. If the list has changed, use the new names and say so in the pull request.

- [ ] **Step 3: Add the arm64 jobs**

Extend the existing `linux-x86_64` job's pattern with two arm64 jobs. Take the `actions/checkout` SHA from the pin that Step 4 puts on the file — resolve it once and reuse it, rather than pinning the same action twice in one diff.

```yaml
  linux-arm64:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@<resolved in Step 4> # v7
        with:
          persist-credentials: false
      - run: ./build.sh release test
      - run: ./build.sh asan test

  macos-arm64:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@<resolved in Step 4> # v7
        with:
          persist-credentials: false
      - run: ./build.sh release test
      - run: ./build.sh asan test
```

- [ ] **Step 4: Apply #13 to the same file**

The file carries ten `uses:` lines today, all unpinned: five `actions/checkout@v7`, three `actions/upload-artifact@v7`, two `actions/download-artifact@v7`. Resolve each action's current release tag to a commit SHA — do not take the SHA from memory or from another repository:

```bash
for a in actions/checkout actions/upload-artifact actions/download-artifact; do
  tag=$(gh api "repos/$a/releases/latest" --jq .tag_name)
  sha=$(gh api "repos/$a/git/ref/tags/$tag" --jq .object.sha)
  echo "$a  $tag  $sha"
done
```

Write each as `uses: <action>@<sha> # <tag>`. Then move `contents: write` from the workflow level (`release.yml:21-22`) down to the three jobs that actually write, and give the file a default of `contents: read`:

- `linux-x86_64` — `gh release create` at `:118`.
- `windows-package` — `gh release upload` at `:256`.
- `wiki` — pushes the wiki repository at `:$277`.

`windows-build` and `windows-test` need `contents: read` only; they upload and download workflow artifacts, which the artifact scope covers.

Finally, add `with: persist-credentials: false` to all five checkout steps; the file has none today.

#13 requires this pairing, because both changes touch one file and the fork must not diverge on it.

- [ ] **Step 5: Lint the workflow**

```bash
actionlint .github/workflows/release.yml
zizmor .github/workflows/release.yml
```

Expected: no findings for the lines this branch touches. `zizmor` reported the original issues; the count of its findings must drop.

- [ ] **Step 6: Guard, commit, push, open the pull request**

```bash
./.fork/check-clean.sh
git add .github/workflows/release.yml
git commit -s -m "ci: arm64 runners, and the hardening from #13"
git push -u origin port/25-ci-arm64
gh pr create --repo prabanta-dev/janas --base main \
  --head phaedrus1992:port/25-ci-arm64 \
  --title "CI: macOS arm64 and Linux arm64 runners" --body-file .tmp/pr25.md
```

`.tmp/pr25.md` must state that the arm64 jobs are expected to pass because the five prior pull requests are merged, and must carry #13's three points as its own summary.

- [ ] **Step 7: Watch the run**

```bash
gh pr checks --watch --repo prabanta-dev/janas
```

Expected: the two new arm64 jobs pass. A failure here runs upstream's CI, not the fork's, so debug it on the pull request itself. Use the `nbl-dev:ci-failure-debugger` approach: reproduce locally with the same two commands, fix, and push.

---

## Closing

After the last task, and not before:

- [ ] **Confirm every fork issue is closed or updated.** Tasks 1 to 6 are #421 to #426; the epic is #420; #13 rides in Task 6. Each upstream pull request carries no closing keyword for a fork issue, so close them by hand as each is merged, with a comment naming the upstream pull request.
- [ ] **Update `docs/fork.md`** on a `docs/*` branch off `main`, then open and merge a fork pull request. It currently says "No port code exists yet". After Task 6 that is false. `docs/fork.md` and `docs/superpowers/` are tooling paths, so they never go to upstream.
- [ ] **Update `docs/superpowers/specs/2026-10-13-arm64-metal-port/design.md`** §6 (E1) with the exit-test result, and §7's testing table with the x86-64 machine this plan added. The spec is a snapshot; it must not contradict the code.
- [ ] **File issues for anything deferred.** A TODO comment, a note in a pull request, and a line in a chat log are not deferrals. Anything left undone gets a fork issue in milestone E1 or a later one, with a severity, a size, and an area label.
- [ ] **Do not run the `executing-plans` tail.** Its whole-branch review, `superpowers:finishing-a-development-branch`, and its `.superpowers/sdd/` workspace delete do not apply here. A `port/*` branch has no fork review stage and no fork merge: its review is the upstream maintainer's, and it merges upstream, not into the fork. This plan's closing checklist replaces that tail for the fork-side work only.
