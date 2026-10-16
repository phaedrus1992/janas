# Rendeverance's arm64 branches and this fork's plan

Rendeverance (`Rendeverance/janas`) ports Janas to arm64 on a Snapdragon X, for Linux and for Windows on ARM64.
That work is further along than this fork's NEON plan, so this fork now helps that work instead of a separate NEON track.
This file compares the two, to start that discussion (#439).

Snapshot: 2026-10-16, upstream `main` at `afb874e`.

## Their branches

All four are open pull requests on `prabanta-dev/janas`.
`main` and `develop` on their fork are the same as upstream `main`.

| Branch | Head | Upstream PR | Content |
| --- | --- | --- | --- |
| `neon-quant-dots` | `04c50cb` | #32 | NEON dot products for 12 quant formats with `sdot`, the dotprod and I8MM checks, `-ffp-contract=off` on aarch64. Carries our port/21 (#25). |
| `win-arm64` | `9a88221` | #33 | Windows on ARM64 with llvm-mingw. Stacks on #32. |
| `janas-free-page-aligned` | `84be178` | #31 | `janas_free` does not read an unmapped page in front of a page-aligned block. Windows only. |
| `mtp-own-cache` | `8b7080b` | #34 | The in-file MTP block gets its own expert cache. |

Still to come from them, from the #32 text: the I8MM prefill tile, then NEON for attention, DeltaNet, the norms, and the router.

## Fork issues they cover

- #6 (Windows on ARM) is done by #31 and #33.
- #419 (E2 NEON kernels) is mostly done by #32. The part that stays open is the macOS side (see below).
- #422 (port/21) is in their stack. It must merge upstream first, because #32 builds on it.

## Overlaps and conflicts

- port/23 (#424, upstream #28) and #33 both fix the string literal in `src/chat/tune.c` that Clang's `-Wstring-concatenation` flags.
  The two fixes conflict.
  One of the two must drop its hunk.
- #425 (port/24, sysinfo on macOS) must use their `janas_cpu_arm_dotprod()` and `janas_cpu_arm_i8mm()` in `src/common/sysinfo.h`, not a second API.
- port/20 (#421), port/21 (#422), and the bash 3.2 branch (#438) merge onto `win-arm64` with no conflict.

## What stays macOS work for this fork

Their branches do not build or test on macOS.
These items keep the macOS path correct on top of their work:

1. `-ffp-contract=off` does not apply on macOS.
   `build.sh` reads `CPU` from `$CC -dumpmachine`, and Apple clang gives `arm64-apple-darwin…`, so `CPU` is `arm64`, not `aarch64`.
   Apple clang fuses `a*b+c` into `fmadd` by default, so the scalar references round differently on a Mac.
   The check must accept both names.
2. `janas_cpu_arm_dotprod()` and `janas_cpu_arm_i8mm()` return 0 on any OS except Linux and Windows.
   A Mac therefore runs the scalar kernels.
   macOS needs `sysctlbyname("hw.optional.arm.FEAT_DotProd")` and `"hw.optional.arm.FEAT_I8MM"`.
   The M1 has dotprod and no I8MM.
3. The macOS build itself: port/20 (BSD `build.sh`, libmicrohttpd config), port/22 (Linux-only calls), port/23 (Apple clang), port/24 (sysinfo and thread pool), and the bash 3.2 fixes (#435, #437).
4. #432: `test_attention` prints `avx2` on arm64 when the reference kernels run.
   Their stack does not change it.
5. CI on arm64 (#426), and the GPU through MoltenVK (#8, E3).

## Open questions for Rendeverance

1. Is there one branch that holds all of your work, so that this fork's macOS branches can stack on it?
2. Do you want the macOS dotprod and `-ffp-contract` fixes as pull requests to your branches, or as separate upstream pull requests on top of #32?
3. Which of the two `tune.c` fixes do we keep?
