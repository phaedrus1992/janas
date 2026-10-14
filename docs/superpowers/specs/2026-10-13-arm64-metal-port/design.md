# Janas on arm64 and Apple Silicon: high-level design

Status: the design sections were approved in brainstorming on 2026-10-13.
Updated on 2026-10-14 with the maintainer's answer to upstream #20.
This is a high-level spec.
Each epic gets its own brainstorming session and its own plan later.

Base: upstream `prabanta-dev/janas` at `98c8300` (Release 2026-10-12 (4)).
The fork is `phaedrus1992/janas`.

## 1. Goal

Janas must run natively on arm64 CPUs and on the Apple Silicon GPU.
All current features for x86-64 CPUs and Vulkan GPUs must stay as they are.

The work goes upstream to `prabanta-dev/janas`.
Every part of the design must therefore follow the upstream rules in `CONTRIBUTING.md`.

## 2. Scope

In scope:

- macOS on Apple Silicon (M1 and later): CPU (NEON) and GPU (the Vulkan backend through MoltenVK).
- Linux on arm64 servers (Graviton, Ampere, Neoverse): CPU (NEON). A discrete GPU through the existing Vulkan backend, where one exists.

Not in scope (issues in the `Future` milestone):

- Asahi Linux on Apple hardware (Vulkan through the Honeykrisp driver).
- Windows on ARM.
- Small boards such as the Raspberry Pi 5.
- Intel Macs.
- SVE and SME CPU tiers.
- Native Metal. The maintainer weighs it only if the vkcheck numbers cannot hold the MoltenVK path.

The design must not block the items that are not in scope.

## 3. Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| GPU API on macOS | The Vulkan backend through MoltenVK | The maintainer's order (upstream #20). The vkcheck gate ran on the M1 Max: MoltenVK imports host memory in place, but the subgroup size is not selectable, so the engine would not take the GPU. Native Metal is weighed on those numbers only if MoltenVK cannot do it. |
| Role of MLX | Reference only | MLX converts Q4_K, Q6_K, Q3_K and IQ3_S to f16 at load time, and its planar layout breaks the CPU/GPU row split. See `ref-mlx.md` §1. |
| Metal host code | Pure C11 through the Objective-C runtime (`objc_msgSend`) | It gives the same calls that an Objective-C file gives, at the same speed. It keeps the upstream "C11 only" rule. |
| Metal loading | At run time with `dlopen`, as Vulkan is loaded now | It keeps the rule that the engine builds and runs without the GPU library. It also lets newer Metal APIs turn on only where the OS has them. |
| Vulkan | Kept on Linux (x86-64 and arm64) and Windows | No change to the current GPU users. |
| Strategy | Seams first, in small increments | Each epic gives upstream a reviewable change that can be tested without a Mac. |
| Delivery | Small PRs on `port/*` branches, one milestone per epic | See §5. |

## 4. Upstream constraints

These come from `CONTRIBUTING.md` and apply to every epic.

1. C11 with no dependencies beyond the C library. A GPU library loads at run time. The engine must build and run without it.
2. Kernels agree bit for bit with the scalar reference. A pass over a block of tokens agrees bit for bit with the same tokens one at a time.
3. `clang-format -i` with the root `.clang-format`. Warnings are errors (`-Wall -Wextra -Werror`).
4. `./build.sh release test` and `./build.sh asan test` pass before each PR. `tsan test` also passes when a change touches concurrent code.
5. Every commit carries `Signed-off-by` (Developer Certificate of Origin).
6. NEON and GPU kernel code is original: written for Janas from the start. No kernel code comes from llama.cpp, MLX, or any other project (upstream #20). The reference repos are for understanding only.
7. Ask in an issue before writing code. Epic E0 does this.

Two more rules come from this design:

8. No speed regression on x86-64. `janas-bench` on x86-64 shows no change. The maintainer's machine decides.
9. AI assistance is disclosed in the first upstream issue.

## 5. Repository workflow (epic E0)

Remotes: `origin` is `phaedrus1992/janas`. `upstream` is `prabanta-dev/janas`.

| Branch | Content | Rule |
| --- | --- | --- |
| `clean` | An exact mirror of `upstream/main` | Fast-forward only. Nobody commits here. |
| `port/<NN>-<topic>` | Engine changes for one upstream PR | Cut from `clean`, or from the previous `port/*` branch when epics stack. Tooling paths are not permitted. |
| `main` (fork default) | `clean` + fork tooling + merged `port/*` work | Day-to-day work and runs happen here. |

Tooling paths (permitted on `main` only):

- `.llmenv.yaml`
- `docs/fork.md` (the fork summary)
- `docs/superpowers/`
- `.claude/`, `CLAUDE.md`, `AGENTS.md`
- `.fork/`
- `.github/workflows/fork-*.yml`
- `.github/dependabot.yml`

The sync loop:

1. A scheduled workflow on `main` (`fork-sync.yml`, daily) fast-forwards `clean` from upstream. When `clean` moves, it opens a PR that merges `clean` into `main`.
2. `.fork/sync-upstream.sh` does the same on demand. Each development session starts with it.
3. When upstream takes a PR, possibly squashed or edited, the `clean` to `main` merge brings back the upstream version. The `port/*` branch is then deleted.

The guard: `.fork/check-clean.sh` fails when the diff of a `port/*` branch against `clean` touches a tooling path.
It also fails when a `port/*` branch contains the fork header at the top of `README.md`.
That header is the one fork change inside an upstream file.
A pre-push hook runs the guard.
A fork CI job runs it on every `port/*` PR.

The upstream issue (E0) proposes the port, discloses AI assistance, and asks for agreement on:

1. Metal host code in pure C through `objc_msgSend`, loaded with `dlopen`.
2. macOS arm64 and Linux arm64 CI runners in the upstream workflows.
3. The bit-exact proposal from E2, if E2 research finds that NEON needs one.

The maintainer answered on 2026-10-13 (upstream #20):

1. Original code only.
   NEON and GPU kernel code is written for Janas from the start.
   No code comes from llama.cpp, MLX, or any other project.
2. Metal not yet.
   The Vulkan backend runs through MoltenVK first.
   `tools/vkcheck.c` is the gate.
   Native Metal is weighed on the gate numbers only if MoltenVK cannot do it.
3. CI on macOS arm64 and Linux arm64: yes, once E1 lands.
4. Bit-exact on NEON: no rule change.
   The float part follows `src/llm/quant_dotn.h` in the same order as the scalar reference and the AVX2 path.
   On NEON's 128-bit registers, keep the AVX2 8-lane order, two registers for one.
5. Delivery: small PRs, one change each, x86-64 paths untouched, tests passing.
   The maintainer reviews and tests each on x86-64, imports it into the development branch with the author's sign-off, and ships it in the next release.
6. Order: portability, NEON, GPU through MoltenVK, then expert streaming on macOS.
   Speed bar: llama.cpp on the same Mac.

The answer is recorded verbatim in `e0-fork-workflow.md` §6.

## 6. Epics

Each epic is a GitHub milestone on the fork.
The exit test is the condition that closes the milestone.

### E0: Fork workflow and upstream agreement

Goal: the branch model in §5, and an upstream issue that the maintainer answers.
Exit test: `clean`, `main`, the sync workflow and the guard work; the maintainer answers the questions in §5.

### E1: Portability layer

Goal: Janas builds and runs on macOS arm64 and Linux arm64 with the scalar kernels.

Scope:

- Replace each Linux-only call with a portable equivalent. The list is in `ref-janas-seams.md` §4.
- Use the existing `pread` thread-pool reader on macOS in place of io_uring.
- Thread placement on macOS: QoS classes and the `hw.perflevel*` sysctls, because macOS has no CPU affinity.
- `build.sh`: `arm64` and `aarch64` CPUs, the `darwin` OS, and output in `bin/arm64-darwin/` and `bin/aarch64-linux/`.
- CI: GitHub `macos` arm64 and `ubuntu-24.04-arm` runners.
- The x86 guards stay. Each one gets a scalar path that compiles on arm64.

Exit test: `release test` and `asan test` pass on macOS arm64 and Linux arm64. x86-64 output is unchanged.

Known gap: Vulkan on Linux arm64 probably works without change, but no GPU is available for a test (Docker has none).

### E2: NEON CPU kernels

Goal: CPU kernels for arm64, selected at run time, in tiers.

Scope:

1. The maintainer answered (upstream #20): no rule change.
   Integer sums are exact.
   The float part (block scales, sums across blocks) follows `src/llm/quant_dotn.h` in the same order as the scalar reference and the AVX2 path.
   On NEON's 128-bit registers, keep the AVX2 8-lane order, two registers for one.
2. Feature detection: `getauxval(AT_HWCAP/AT_HWCAP2)` on Linux and `sysctlbyname("hw.optional.arm.FEAT_*")` on macOS.
3. Tiers: NEON (baseline), NEON + dotprod (M1, Neoverse N1), NEON + I8MM (M2 and later, Neoverse V1 and later).
4. All 12 dot kernels in `quant.c`, plus the x86 blocks in `vmath.h`, `attention.c`, `deltanet.c`, `model_ffn.c` and `model.c`.
5. The `JANAS_KERNELS` override, extended to the new tiers.

Reference for understanding only: llama.cpp `ggml-cpu/arch/arm/quants.c`.
No code is taken from it (upstream #20).

Exit test: each kernel is bit-exact with the scalar reference (integer sums exact; the float order follows `quant_dotn.h`); `janas-bench` numbers for the M1 Max and one Linux arm64 machine.

### E3: GPU on macOS through MoltenVK

Goal: the `gpu.h` contract on macOS through MoltenVK (the existing Vulkan backend).

Scope:

- The vkcheck gate: `tools/vkcheck.c`, run on the M1 Max. The record is `vkcheck-m1-max.md`.
- The gate verdict: MoltenVK passes zero-copy, dot, and the subgroup operations, but the subgroup size is not selectable. The engine therefore would not take the GPU. The maintainer weighs native Metal on these numbers.
- Zero-copy weights through `VK_EXT_external_memory_host`: MoltenVK 1.4.2 imports host memory in place, aligned to 16384 bytes.
- Matvec kernels for each weight type, and the CPU/GPU row split.
- The tuner decides the row split. CPU and GPU share one memory bus on Apple Silicon, so the best split can be different from the Intel Arc split.
- Native Metal returns to this epic only if the maintainer weighs that path in.

Exit test: `gpu_check` and `gpu_ffn_check` are bit-exact on the macOS GPU path.

### E4: Whole token on the macOS GPU path

Goal: the `gpu_token.h` contract on the macOS GPU path: one command buffer per token.

Scope:

- Dense attention models (qwen3), Gated DeltaNet models (qwen3.5), and their mixture-of-experts forms (qwen3moe, qwen35moe).
- Prompt kernels with `simdgroup_matrix` tiles.
- A small-M verify kernel that reads each weight one time for all 1+K draft rows.
- One GPU submission thread.
- Prefill chunks bounded in time, because Metal cannot preempt a running kernel.

Exit test: `llm_eval` agrees with the CPU for the four model families.

### E5: macOS I/O and memory

Goal: models larger than memory and memory limits that suit unified memory.

Scope:

- Expert streaming: `pread` on the I/O thread pool with `F_NOCACHE` and `F_RDADVISE`. The queue depth is tuned.
- Resident experts compute while missed experts load.
- Memory guard: `phys_footprint` and `host_statistics64`. Respect `recommendedMaxWorkingSetSize`. Print the `iogpu.wired_limit_mb` sysctl hint. Keep 5% of RAM free.

Exit test: a model larger than its memory cap runs to completion on macOS, with the same output as without the cap.

### E6: Draft and DeltaNet speedups

Goal: faster speculative decoding on all platforms.

Scope:

- Gated DeltaNet state capture per step, or a "tape" replay, in place of the token-log replay in `model_rec.c`.
- The small-M verify kernel from E4, on the CPU and on Vulkan.
- Block verification, as an option.

Exit test: the same tokens as before; fewer passes per accepted token.

### Future

The `Future` milestone holds one issue for each item that is not in the port:

- The SSD-tiered, hash-chained prompt cache from omlx, for `keep`.
- An SVE tier (Graviton 3 and 4) and SME.
- Asahi Vulkan tuning.
- Windows on ARM.
- Small boards.

## 7. Testing

| Where | What it proves |
| --- | --- |
| macOS arm64, local M1 Max | E1 to E6, the macOS GPU path, bit-exact results and speed |
| Docker `linux/arm64` through Colima (native) | E1 and E2 on Linux arm64 |
| Docker `linux/amd64` through Rosetta | The x86 AVX2 bit-exact tests do not regress |
| GitHub `macos` and `ubuntu-24.04-arm` runners | The same checks in CI, on the fork and upstream |

The local M1 Max has dotprod but no I8MM (`sysctl hw.optional.arm.FEAT_I8MM` gives 0 on 2026-10-13).
The I8MM tier therefore cannot run locally or in Docker on this machine.
It needs an M2 or later Mac, or an arm64 CI runner whose CPU has I8MM.
E2 must confirm which GitHub runner gives I8MM before it relies on one.

Rosetta does not run AVX-512 or AVX-VNNI.
Rosetta also gives no valid speed numbers.
Those need real x86-64 hardware, which the maintainer has.

## 8. Risks

1. **Bit-exact NEON.** Float reductions in a different order give different bits. The maintainer's answer fixes the order: `quant_dotn.h`, with the AVX2 8-lane order on two 128-bit registers.
2. **Row split on the macOS GPU path.** No reference project splits one operation between the CPU and the GPU. The split can give less on unified memory than on Intel Arc. The tuner measures it.
3. **Maintainer agreement.** Resolved on 2026-10-13 (upstream #20): original code, MoltenVK first, CI agreed, no bit-exact rule change. The one open call: native Metal, weighed on the vkcheck numbers.
4. **Upstream drift.** Upstream changes fast (several releases each day). The daily sync and small PRs reduce this risk.
5. **Pure-C Metal readability.** Each call needs a cast. A typed wrapper header with one `static inline` function per method keeps the call sites readable.

## 9. Reference material

| File | Content |
| --- | --- |
| `ref-janas-seams.md` | The Janas extension points and the Linux-only calls |
| `ref-llama-cpp.md` | ARM CPU kernels, Metal backend, Vulkan on MoltenVK, mmap on macOS. Understanding only: no code taken |
| `ref-mlx.md` | MLX quant formats, Metal kernels, allocator, and why MLX is reference only |
| `ref-mtplx.md` | MTP drafts, Gated DeltaNet rollback, small-M verify, tuning |
| `ref-omlx.md` | Prompt cache, expert offload, memory guard, macOS integration |
| `vkcheck-m1-max.md` | The vkcheck gate run on the M1 Max (2026-10-14): MoltenVK numbers and the verdict |
| `techniques.md` | Each technique studied from a reference project, its source, and the epic that uses it. Understanding only |

The reference repos are pinned in the local code-explorer cache:
mtplx `v2.12.2`, omlx `v0.7.0`, mlx `v0.32.3`, llama.cpp `b11588`.
