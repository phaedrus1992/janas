# Janas extension points for the port

All citations are at upstream `98c8300` (Release 2026-10-12 (4)).

## 1. CPU kernel dispatch

The dot kernels are function pointers.
A constructor sets them before `main`, because the pool threads call the kernels first (`src/llm/quant.c:1103`).
The default for each pointer is the scalar reference, named `ref`.

On x86-64 the constructor selects one of these tiers (`src/llm/quant.c:1106-1140`):

1. `avx2`: AVX2 + FMA + F16C.
2. `avx-vnni`: the AVX2 tier plus AVX-VNNI. `JANAS_KERNELS=avx2` turns VNNI off.

There are 12 weight types: Q4_K, Q6_K, Q5_K, Q8_0, IQ4_NL, IQ4_XS, Q4_0, Q5_1, Q4_1, Q3_K, IQ3_S, and Q6_K_P (Q6_K in three planes).
The activations are quantized to Q8_K before the dot product.
This is the llama.cpp design, so the llama.cpp ARM kernels map onto these slots directly.

An arm64 tier goes into the same constructor, under `#if defined(__aarch64__)`.

## 2. x86-only code outside the dot kernels

Each of these files has `#if defined(__x86_64__)` blocks with a scalar `#else`:

| File | First guard |
| --- | --- |
| `src/llm/quant.c` | `:15` |
| `src/llm/quant_dotn.h` | about 300 AVX intrinsics, included only on x86 |
| `src/llm/vmath.h` | `:82` |
| `src/llm/attention.c` | `:14` |
| `src/llm/deltanet.c` | `:12` |
| `src/llm/model.c` | `:612` |
| `src/llm/model_ffn.c` | `:75` |
| `src/llm/model_load.c` | `:942` |
| `src/common/pool.c` | `:18` |
| `src/common/sha256.c` | `:74` |

The code has no NEON today.

## 3. GPU contract

The GPU backend sits behind two headers.
A Metal backend implements the same functions.

`src/llm/gpu.h`:

| Function | Line |
| --- | --- |
| `janas_gpu_create`, `janas_gpu_destroy` | `:49-50` |
| `janas_gpu_import` (host memory, no copy) | `:54` |
| `janas_gpu_alloc`, `janas_gpu_flush` | `:62-63` |
| `janas_gpu_can` | `:67` |
| `janas_gpu_begin`, `janas_gpu_ffn_begin`, `janas_gpu_finish` | `:73`, `:88`, `:94` |
| `janas_gpu_matvec_group` | `:103` |
| `janas_gpu_dense_mm` | `:113` |
| `janas_gpu_keep_awake` | `:135` |
| `janas_gpu_name`, `janas_gpu_subgroup`, `janas_gpu_min_n` | `:138-143` |

`src/llm/gpu_token.h` runs a whole token in one submission (`:98-131`).
The header comment says that the products are the CPU's products bit for bit, and that the norms and the attention sum in the GPU's order.

Vulkan loads at run time: `src/llm/gpu.c:405-409` opens `libvulkan.so.1` or `vulkan-1.dll`.
The device checks are at `src/llm/gpu.c:165-254`.
An integrated GPU without `VK_EXT_external_memory_host` is refused (`:216`).

There are 37 GLSL compute shaders in `src/llm/shaders/`: matvec per type, `*_mmq` (prompt), `*_mmid` (mixture of experts), `gdn_*` (Gated DeltaNet), `moe_*`, and `attn_*`.

## 4. Linux-only calls

These calls stop the build on macOS (found with `cc -fsyntax-only` on macOS 27, Apple clang 21).

| File | Call | macOS equivalent |
| --- | --- | --- |
| `src/third_party/libmicrohttpd/src/microhttpd/mhd_sockets.h:110` | `sys/epoll.h` | `poll` or kqueue (the bundled `MHD_config.h` is set for Linux) |
| `src/common/mcp_stdio.c:105` | `sigtimedwait` | kqueue `EVFILT_SIGNAL`, or a self-pipe |
| `src/common/reopen.c:41` | `O_DIRECT` | `fcntl(F_NOCACHE)` |
| `src/common/sock.c:120` | `SOCK_CLOEXEC` | `fcntl(FD_CLOEXEC)` after `socket` |
| `src/common/sysinfo.c:138` | `cpu_set_t` | `hw.perflevel*` sysctls and QoS classes |
| `src/llm/api_keep_disk.c:57` | `st_mtim` | `st_mtimespec` |
| `src/llm/expert_cache.c:516` | `MADV_HUGEPAGE` | none (skip) |
| `src/common/uring.c` | io_uring | the `pread` pool fallback |

`uring.h` already says that `janas_uring_create` returns NULL where io_uring is not available, and that the caller then keeps its own way of reading.

## 5. Speculative decoding and Gated DeltaNet state

The MTP draft step is `mtp_step` (`src/llm/generate.c:202`).
A chained draft continues while the product of the draft probabilities stays above `min_conf` (`:219-245`).

Gated DeltaNet layers cannot cut their state back the way a KV cache can.
Janas logs the tokens of a call and replays them on the base state (`src/llm/model_rec.c:27-33`, field `replay`).
This is the "snapshot and repair forward" method.
mtplx has two methods that remove the replay (see `ref-mtplx.md` §2).

## 6. Expert cache

`src/llm/expert_cache.h:3-12`: a fixed number of slots in one 4 KiB-aligned arena, LRU replacement, and warm-up from the experts that the prompt uses.
Misses are read with `O_DIRECT` by a pool of I/O threads.

## 7. Tests that guard the port

| Test | What it guards |
| --- | --- |
| `tests/test_quant.c` | Kernels against the scalar reference |
| `tests/test_deltanet.c` | Gated DeltaNet |
| `tests/test_attention.c` | Attention |
| `tests/gpu_check.c`, `tests/gpu_ffn_check.c` | GPU products against the CPU |
| `tests/test_expert_cache.c` | Expert cache |
| `tests/llm_eval.c`, `tests/mtp_eval.c` | Whole-model output and drafts |
| `tests/bench_*.c` | Speed |
