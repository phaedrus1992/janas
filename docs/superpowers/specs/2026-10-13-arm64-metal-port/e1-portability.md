# E1: the portability layer

Status: the design was approved in brainstorming on 2026-10-14.
This spec fills in `design.md` §6 (E1) from that document.
It is a design spec, not an implementation plan.
The implementation plan is a separate step, written later.

## 1. Goal and exit test

Goal: Janas builds and runs on macOS arm64 and Linux arm64 with the scalar kernels.
The x86-64 output does not change.

Exit test: `./build.sh release test` and `./build.sh asan test` pass on macOS arm64.
The same two commands pass on Linux arm64.
`./build.sh release test` passes on x86-64, with the same numbers as before.

## 2. What blocks the build today

The numbers below come from a measurement, not from an estimate.
The sweep compiled every source `build.sh` compiles, with the flags `build.sh` uses.
It ran on 2026-10-14 on the M1 Max: macOS 27.0.1, Apple clang 21.0.0, target `arm64-apple-darwin27.0.0`.

The host has no `gcc` of its own.
`/usr/bin/gcc` is Apple clang, so `build.sh` starts with its default `CC` unchanged.

Of 409 files, 170 fail on macOS arm64.
The causes fall into a small number of groups.

| Cause | Files | Nature |
| --- | --- | --- |
| `src/llm/json.h:68` uses `__attribute__((format(gnu_printf, 2, 3)))` | 143 (only cause) | Apple clang rejects `gnu_printf` |
| A name declared inside `#if defined(__x86_64__)`, used or left unused outside it | 5 sites | Not macOS-specific |
| A Linux-only call with no macOS spelling | 7 sites | OS portability |
| `build.sh` uses GNU `find -printf` and GNU `stat -c` | 2 lines | BSD has neither |
| The committed libmicrohttpd configuration is a Linux x86-64 configure output | 8 files | Needs a second configuration |
| `_XOPEN_SOURCE` without `_DARWIN_C_SOURCE` | 1 site | Hides the Darwin extensions |

The same sweep on Linux arm64 leaves five files.
It ran on Ubuntu 24.04 with GCC 13.3, at native `aarch64` through Colima.
Four of the five are `src/llm/attention.c`, `src/llm/model_ple.c`, `src/llm/quant.c` and `tests/test_quant.c`.
Those four are all in `port/21`, so that branch is the whole of the Linux arm64 work.
The fifth is the mbedtls file below, which is not compiled.

The two compilers disagree on one file: `src/llm/deltanet.c` fails under Apple clang and passes under GCC.
The plan must run each branch under both.

Two entries that look like failures are not:

1. `src/third_party/mbedtls/.../FStar_UInt128_extracted.c` does not compile, but `build.sh` never compiles it.
   `tp_sources mbedtls` names 115 files; this one is outside the list.
2. The third-party sources are compiled with `-w`, because `build.sh` strips `-Wall -Wextra -Werror` and appends it (`build.sh:322-333`).
   A warning in libmicrohttpd or mbedtls therefore does not fail the build.
   Only a hard error does.

## 3. Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Slice unit | One concern per `port/*` branch, six in total | The maintainer asked for small PRs, one change each. Each slice below carries one reason a reviewer can check on its own. |
| Order of the slices | Any; the six are independent | They touch disjoint files, and each is inert on x86-64. They can be reviewed in any order. |
| The x86 guards | Keep them. Repair the names they leak. | The scalar reference already compiles on every platform, because the reference functions are unguarded. Only five names leak out of a guard. |
| macOS thread placement | QoS classes and no pinning. See §5. | macOS has no CPU affinity. `pthread_setaffinity_np` has no equivalent. |
| libmicrohttpd on macOS | A second committed configuration in `darwin/` | `w32/` already sets this precedent. A build-time `configure` would need autoconf on every machine. |
| The `_GNU_SOURCE` duplicate | Drop the second define in `tp_flags` | `-D_GNU_SOURCE` and `-D_GNU_SOURCE=` are different definitions. Both compilers warn. |
| `json.h`'s format attribute | `__APPLE__` selects `printf`; other systems keep `gnu_printf` | `gnu_printf` checks the GNU extensions in a format string. Changing it everywhere would alter `-Wformat` on the x86-64 path, which must not move. |

## 4. The six port branches

Each branch comes from `clean`, carries no fork tooling, and passes `.fork/check-clean.sh` before the push.
Each commit carries `Signed-off-by`.
Every branch leaves the x86-64 paths untouched.

### port/20-build-portability

`build.sh`, `src/third_party/libmicrohttpd/darwin/MHD_config.h` (new), `src/third_party/libmicrohttpd/origin.md`.

1. `build.sh:306`: `find ... -printf '%T@\n'` is a GNU extension.
2. `build.sh:312`: `stat -c %Y` is a GNU extension.
3. `build.sh:341`: `tp_flags libmicrohttpd` adds `-D_GNU_SOURCE` on top of the `-D_GNU_SOURCE=` in `CFLAGS` (line 128).
   The two definitions differ, so both compilers report `_GNU_SOURCE redefined`.
4. The committed `MHD_config.h` sets `EPOLL_SUPPORT` (needs `<sys/epoll.h>`), the eventfd transport (needs `<sys/eventfd.h>`), and `HAVE_PTHREAD_SETNAME_NP_GNU`, whose Linux form takes two arguments where the Darwin form takes one.

No change is needed for the architecture or the OS name.
`build.sh:117-118` reads `uname -m` and `uname -s`, which give `arm64` and `darwin` on macOS and `aarch64` and `linux` on Linux arm64.
`bin/arm64-darwin/` with `lib/arm64-darwin-release/`, and `bin/aarch64-linux/` with `lib/aarch64-linux-release/`, therefore follow with no edit.
Both directories already appear on this Mac as empty or generated output.
`build.sh` holds no architecture compiler flag at all: the x86 tiers are selected with `__attribute__((target(...)))` on the functions.

### port/21-arch-guard-leaks

`src/llm/quant.c`, `src/llm/attention.c`, `src/llm/deltanet.c`, `src/llm/model_ple.c`, `tests/test_quant.c`.

A name that a `#if defined(__x86_64__)` block declares, and the code outside that block then uses or leaves unused:

| File | Name | Note |
| --- | --- | --- |
| `src/llm/quant.c` | `e8m0_tab` | Declared at `:1127`, inside the block that opens at `:1104`. Used at `:1194`. A hard error on every non-x86 architecture. |
| `src/llm/attention.c` | `pf` (`:661`) | Unused on arm64; `-Werror` makes that a failure. |
| `src/llm/model_ple.c` | `v` (`:133`, `:229`) | Set by `have_avx2()`, then used only inside an x86 block. |
| `tests/test_quant.c` | `b` (`:481`) | One of four arrays; used only by the AVX2 path. |
| `src/llm/deltanet.c` | `use_f16c` (`:173`) | Apple clang reports it; GCC does not. |

`attention.c:610-613` already shows the pattern this file uses when a parameter is unused on a non-x86 build.

This is the only slice that a Linux arm64 runner can verify without a Mac.
It is also the only slice that fixes a defect in shipped code rather than a portability gap: `quant.c` cannot compile on any arm64 target today.

### port/22-os-call-sites

| File | Call | macOS equivalent |
| --- | --- | --- |
| `src/common/sock.c:120` | `SOCK_CLOEXEC`, `SOCK_NONBLOCK` in `socket` | `fcntl(FD_CLOEXEC)` and `O_NONBLOCK` after the call |
| `src/common/reopen.c:41` | `O_DIRECT` | `fcntl(F_SETFL, F_NOCACHE)` |
| `src/common/mcp_stdio.c:105` | `sigtimedwait` | a self-pipe, or `kqueue` with `EVFILT_SIGNAL` |
| `src/common/proc.c:86` | `posix_spawn_file_actions_addchdir_np` | `posix_spawn_file_actions_addchdir` on macOS 26 and later; the `_np` form is deprecated there |
| `src/llm/api_keep_disk.c:57` | `st_mtim` | `st_mtimespec` |
| `src/llm/expert_cache.c:520` | `MADV_HUGEPAGE` | none; skip the call |
| `src/server/http.c:44` | `getrandom` | `getentropy`, or `arc4random_buf` |
| `tests/bench_long.c:33` | `<sys/sysmacros.h>` | not needed on macOS |
| `tests/jns_check.c:76` | `O_DIRECT` | as `reopen.c` above |

`src/common/uring.c` needs no work.
Its header already says that `janas_uring_create` returns NULL where io_uring is absent, and that the caller then keeps its own way of reading.
The caller is `io_worker` (`src/llm/expert_cache.c:249`), which already reads with `pread`.

### port/23-clang-dialect

`src/llm/json.h`, `src/chat/main.c`, `src/chat/tune.c`, `tests/bench_kernels.c`.

1. `src/llm/json.h:68` uses `__attribute__((format(gnu_printf, 2, 3)))`.
   Apple clang refuses the `gnu_printf` argument and reports `ignored-attributes`.
   That one line is the only cause of 143 of the 170 failures, so it is the single largest cause in the epic.
2. `src/chat/main.c:17` defines `_XOPEN_SOURCE 700`.
   On macOS that alone hides the Darwin extensions, and `SIGWINCH` (`:1868`) and `_SC_NPROCESSORS_ONLN` (`:518`) then have no declaration.
   `#define _DARWIN_C_SOURCE` beside it restores both.
   `wcwidth`, the reason line 17 exists, still resolves.
3. `src/chat/tune.c:32` splits one string literal over two lines inside an array initializer.
   That is the intent, but Apple clang reports `-Wstring-concatenation` there.
4. `tests/bench_kernels.c:123` converts `2147483647` to `float` implicitly.

### port/24-sysinfo-macos

`src/common/sysinfo.c`, `src/common/pool.c`. See §5.

### port/25-ci-arm64

`.github/workflows/`, and the hardening that issue #13 records.

1. Add `macos-14`, `macos-15` and `macos-26` (all arm64) and `ubuntu-24.04-arm` to the upstream workflows.
2. Apply #13 to `.github/workflows/release.yml` in the same pull request: pin the seven action references to a SHA, move `contents: write` from the workflow to the jobs that need it, and add `persist-credentials: false` to the checkout steps.
   #13 asks for exactly this pairing, because both changes touch one file and the fork must not diverge on it.

The runner labels are the ones GitHub documents today, not from memory.
The macOS arm64 runners carry 3 CPUs and 7 GB on a public and on a private repository.
`ubuntu-24.04-arm` carries 4 CPUs and 16 GB on a public repository and 2 CPUs and 8 GB on a private one.
All of them are available to both kinds of repository.
The maintainer agreed to arm64 CI "once the portability work lands" (`design.md` §5, answer 2).

## 5. macOS thread placement

macOS has no CPU affinity, so `janas_pin_current_thread` (`src/common/sysinfo.c:193`) has nothing to do.
Its Linux form calls `pthread_setaffinity_np`; the Windows form calls `SetThreadGroupAffinity`.

The macOS form returns 0 and does nothing.
The call sites in `src/common/pool.c:130-132` and `src/llm/model_load.c` stay as they are.

The scheduler policy moves to the QoS class instead:

| Thread | QoS class |
| --- | --- |
| The caller | `QOS_CLASS_USER_INTERACTIVE` |
| Pool workers | `QOS_CLASS_USER_INITIATED` |
| I/O threads | `QOS_CLASS_UTILITY` |

The pool size comes from `hw.perflevel0.physicalcpu`, the count of performance cores, so that the compute threads land on the fast cores.
`sysctlbyname` replaces the sysfs and procfs reads for the same reason.

This branch changes no Linux or Windows behaviour, so the x86-64 output cannot move.

## 6. The bundled libraries

libmicrohttpd needs the configuration described in §4 (`port/20`).
Its `origin.md` records how the committed configuration was generated, and the new one must be recorded there too.

mbedtls needs nothing.
The one file of it that fails is not compiled.

## 7. Testing

The proof has four parts.

| Where | What it proves |
| --- | --- |
| macOS arm64, the local M1 Max | `./build.sh release test` and `./build.sh asan test` pass |
| Linux arm64, Docker through Colima at native `aarch64` | The same two commands pass |
| x86-64 Linux, the machine this fork's owner has | `./build.sh release test` passes and the x86-64 numbers do not move |
| GitHub runners in CI | The same checks run off this machine, on the fork and upstream |

The x86-64 run is a required step of this epic, not a hand-off.
It covers `design.md` §4 rule 8, which the design had left to the maintainer's machine alone.
`JANAS_KERNELS=ref` runs the scalar reference on any machine, and complements the x86-64 run: it is the path the port adds.

The test suite carries its own fixtures: `test_expert_cache.c:591` writes the file it reads under `TMPDIR`, and no test needs a model from outside the tree.
The local runs are still the acceptance test, because the macOS CI runner has 3 CPUs.
A green CI run alone does not close a branch.

## 8. Work items

One epic issue and six slice issues go on the fork, in milestone E1.
The dev-sprint `create-issue.sh` script creates them, because it requires the milestone and checks each label against the repository.

| Issue | Labels |
| --- | --- |
| #420 E1: portability layer | `enhancement`, `P1`, `size/XL`, `area/build` |
| #421 `port/20` build portability | `enhancement`, `P1`, `size/M`, `area/build` |
| #422 `port/21` arch guard leaks | `bug`, `P1`, `size/S`, `area/cpu-kernels` |
| #423 `port/22` OS call sites | `enhancement`, `P1`, `size/M`, `area/io`, `area/common` |
| #424 `port/23` clang dialect | `enhancement`, `P1`, `size/S`, `area/build` |
| #425 `port/24` sysinfo on macOS | `enhancement`, `P1`, `size/M`, `area/common` |
| #426 `port/25` CI on arm64 | `enhancement`, `P2`, `size/M`, `area/build` |

Five slices carry `P1` because the exit test in §1 needs all five.
`port/25` alone carries `P2`, because the maintainer gated arm64 CI on the rest.

`area/common` is the one label this epic adds.
Issue #13 rides in `port/25` and needs no issue of its own.

## 9. Implementation order

For the later implementation plan, not now:

1. `port/21-arch-guard-leaks`, because it is the only slice with a defect in shipped code and the only one any arm64 machine can verify.
2. `port/20-build-portability`, because `port/23` needs a build that gets far enough to report its own errors.
3. `port/23-clang-dialect`, which takes the failure count from 170 to 27.
4. `port/22-os-call-sites`.
5. `port/24-sysinfo-macos`.
6. `port/25-ci-arm64`, last, with #13.

Steps 1 and 2 are independent of each other and of the rest.
Steps 3 to 5 are independent of each other too; the order above only follows the failure count.

## 10. Notes

This file sits beside `design.md` in the port spec directory.
E2 to E6 get one spec file each: `e2-neon.md`, and so on.

The measurement in §2 is a snapshot of upstream at the fork's `main` on 2026-10-14.
Upstream releases several times a day, so the line numbers move.
The implementation plan must take them again before each branch is cut.

No implementation starts from this document until the user says so.
The implementation plan is a separate step, written later.
