# Reference: MLX

Ref: `v0.32.3` (`64ea011c`), from `ml-explore/mlx`.
Paths are relative to the repository root.
Licence: MIT.

Conclusion: MLX is reference material only.
No code is taken from this project (upstream #20).
The reasons are in §7.

## 1. Quant formats

Modes: `Affine`, `Mxfp4`, `Mxfp8`, `Nvfp4` (`mlx/primitives.h:155`).

| Mode | Default group size | Default bits |
| --- | --- | --- |
| affine | 64 | 4 |
| nvfp4 | 16 | 4 |
| mxfp4 | 32 | 4 |
| mxfp8 | 32 | 8 |

Source: `mlx/ops.cpp:4660-4686`.
Affine permits group sizes 32, 64 and 128, and 2, 3, 4, 5, 6 or 8 bits (`mlx/ops.cpp:5024-5035`).

The layout is planar: a packed `uint32` weight tensor, plus separate `scales` and `biases` arrays.
Dequantization is `w = scale * q + bias` (`mlx/backend/metal/kernels/quantized.h:791-793`).
This is not the interleaved block layout of GGUF.

GGUF loading:

- Only Q4_0, Q4_1 and Q8_0 are repacked into affine (`mlx/io/gguf.cpp:352-354`; `mlx/io/gguf_quants.cpp:34-50`, `:100-160`).
- All other types, including Q4_K, Q6_K, Q3_K and IQ3_S, are dequantized to f16 (`mlx/io/gguf.cpp:97-105`).

## 2. Metal quantized kernels

Matvec (`mlx/backend/metal/kernels/quantized.h`):

- `qmv_fast_impl` (`:756-821`): 2 simdgroups, 4 output rows each. Each thread unpacks its weights, does a dot product against `x` in registers, then `simd_sum` reduces. No simdgroup matrices.
- Variants: `qmv` (`:824`), `qmv_quad` (`:700`), `qmv_wide` (`:988`), `qvm` and `qvm_split_k` (`:1082`, `:1834`).

Matmul: `qmm_t` and `qmm_n` (`:1199`, `:1326`) use the steel `BlockMMA` (`mlx/backend/metal/kernels/steel/gemm/mma.h:46`, `:205`). A `QuantizedBlockLoader` dequantizes tiles into threadgroup memory.

Host dispatch (`mlx/backend/metal/quantized.cpp:1868-1935`): matvec below a row limit, else matmul, with split-K for batch 1.

Mixture of experts (`mlx/backend/metal/quantized.cpp:1964-2050`):

- `gather_qmv` and `gather_qmm` take index buffers. Each threadgroup reads its expert index and offsets into the stacked expert weights (`quantized.h:1494-1527`).
- With sorted indices and enough tokens per expert, `affine_gather_qmm_rhs` batches tokens per expert (`quantized.h:2436`).
- Decode with batch 1 uses `gather_qmv`.

## 3. Allocator and buffers

- All buffers are `StorageModeShared` with untracked hazards (`mlx/backend/metal/allocator.cpp:16`).
- `make_buffer(ptr, size)` wraps existing memory without a copy (`allocator.cpp:207-218`). When Metal refuses the pointer, the array constructor copies instead (`mlx/array.cpp:89-101`).
- The MLX loaders do not use mmap; they `read()` into new buffers (`mlx/io/load.h:34-43`).
- Residency sets with a size cap are attached to each queue; the wired budget comes from `set_wired_limit` (`mlx/backend/metal/resident.h:15-26`).
- `set_wired_limit` is capped at the recommended working set (`allocator.cpp:254-261`). The memory limit defaults to 0.95 times `recommendedMaxWorkingSetSize` (`:87-92`).

## 4. Custom kernels

`fast::metal_kernel` (`mlx/fast.h:87-95`) takes only a kernel body.
MLX generates the signature (`mlx/backend/common/metal_kernel.cpp:52-170`) and compiles the source at run time, cached by hash.
The caller does not control threadgroup memory arguments, argument buffers or the encoder.

## 5. Graph and dispatch

- Each `eval` builds the graph with a depth-first walk (`mlx/transforms.cpp:80-130`, `:358-386`).
- Command buffers commit every 20 to 50 operations or 40 to 50 MB, by chip class (`mlx/backend/metal/device.cpp:605-626`).
- `compile` fuses element-wise operations only, not matmuls (`mlx/compile.cpp:79-80`).

## 6. Build

- C++20 (`CMakeLists.txt:25-26`), macOS SDK 14 or later (`:203`), CMake.
- The kernels ship as a separate `mlx.metallib` (`mlx/backend/metal/device.cpp:149-223`).
- The C API is the separate `mlx-c` repository; its coverage was not checked.

## 7. Why MLX is reference only

MLX gives tuned matvec, matmul and gather kernels, a unified-memory allocator, and residency handling.
It does not fit Janas for these reasons:

1. It cannot run GGUF k-quants or IQ types. They become f16, so they are neither bit-exact nor small.
2. Janas formats need custom MSL in either case.
3. The planar layout cannot share bytes with the CPU for a row split.
4. The MLX scheduler owns command buffers and fences, so Janas cannot interleave CPU and GPU work in detail.
5. C++20, CMake and a metallib break the upstream rule of C11 with no dependencies.

Use `quantized.h` as a tiling reference for E3 and E4.
