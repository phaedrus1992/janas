# Reference: llama.cpp

Ref: `b11588` (`7f51953d7d2f`), from `ggml-org/llama.cpp`.
Paths are relative to the repository root.
Licence: MIT.
Reference for understanding only: no code is taken from this project (upstream #20).

Caution: this tree contains features that may not be in mainline (a Hadamard path, Q1_0 and Q2_0 types).
Check a citation against current upstream before you quote it as mainline behavior.

## 1. ARM CPU kernels

The k-quant dot products are selected at compile time with `__ARM_FEATURE_*` macros.
Selection at run time happens for the whole CPU backend: llama.cpp builds several variants and loads the best one for the CPU.

Kernels in `ggml/src/ggml-cpu/arch/arm/quants.c`:

| Type | Line | Paths |
| --- | --- | --- |
| Q3_K | `:2018` | SVE `:2034`, NEON with `vdotq_s32` `:2209` |
| Q4_K | `:2334` | SVE + I8MM `:2360`, NEON I8MM with `vmmlaq_s32` `:2569`, `:2658`, SVE `:2711`, NEON `:2791` |
| Q6_K | `:2966` | SVE + I8MM `:2986`, I8MM `:3177`, `:3275`, SVE `:3344`, NEON `:3492` |
| IQ3_S | `:3926` | NEON only `:3939`, one row per call |

Notes:

- Without dotprod, `ggml_vdotq_s32` is emulated (`ggml/src/ggml-cpu/ggml-cpu-impl.h:307-321`).
- With I8MM, Q4_K does two rows per call (`ggml/src/ggml-cpu/ggml-cpu.c:312-318`). SMMLA computes a 2x2 tile.
- Weight repacking: `ggml_repack_get_optimal_repack_type` (`ggml/src/ggml-cpu/repack.cpp:4925`). Q4_K becomes 8x8 with I8MM, else 8x4 (`:5012-5014`). Q6_K is the same (`:5061-5069`). Q3_K and IQ3_S have no repack.

Feature detection (`ggml/src/ggml-feats.h`):

- Linux: `getauxval(AT_HWCAP/AT_HWCAP2)` for ASIMDDP, FPHP and ASIMDHP, SVE, SVE2, I8MM, SME, SME2 (`:99-108`). The bit values are hard-coded for old headers (`:9-38`).
- macOS: `sysctlbyname("hw.optional.arm.FEAT_DotProd")` and the same for `FEAT_FP16`, `FEAT_SVE`, `FEAT_SVE2`, `FEAT_I8MM`, `FEAT_SME`, `FEAT_SME2` (`:116-145`).
- Windows: `IsProcessorFeaturePresent` (`:151-157`).

Variant scoring: `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp:6-37` gives 0 when a required feature is absent, else weighted points.

Apple variants (`ggml/src/CMakeLists.txt:526-546`):

| Variant | Features |
| --- | --- |
| `apple_m1` | DOTPROD |
| `apple_m2_m3` | DOTPROD, MATMUL_INT8 |
| `apple_m4` | DOTPROD, MATMUL_INT8, SME, no SVE |

For M1 these flags agree with real hardware: the M1 Max of the fork owner gives `FEAT_DotProd: 1`, `FEAT_I8MM: 0`, `FEAT_SME: 0`, `FEAT_BF16: 0` (checked on 2026-10-13).
The M2, M3 and M4 rows are not checked on hardware.

SME is used only through KleidiAI, for Q4_0 and Q8_0 (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:485-493`).
Accelerate is used for vDSP element-wise operations (`ggml/src/ggml-cpu/binary-ops.cpp:64-101`) and BLAS `sgemm` (`ggml/src/ggml-blas/ggml-blas.cpp:142`, `:420-432`).
The Apple AMX unit is not used directly.

## 2. Metal backend

Matvec for one token (`ggml/src/ggml-metal/kernels/mul_mv.metal`):

| Kernel | Line |
| --- | --- |
| `kernel_mul_mv_q3_K_f32_impl` | `:1346` |
| `kernel_mul_mv_q4_K_f32_impl` | `:1511` |
| `kernel_mul_mv_q6_K_f32_impl` | `:1763` |
| `kernel_mul_mv_iq3_s_f32_impl` | `:2285` |

- Partial sums reduce with `simd_sum` (for example `:1611`).
- Q3_K, Q4_K and Q6_K use 2 rows per simdgroup and 2 simdgroups per threadgroup. IQ3_S uses 4 rows, or 8 in a split variant (`ggml/src/ggml-metal/ggml-metal-impl.h:53-91`).
- The dispatch is 32 threads times the simdgroup count (`ggml/src/ggml-metal/ggml-metal-ops.cpp:2581-2583`).

Small batches:

- Up to 16 rows: MMA kernels with 8x8 simdgroup matrices (`mul_mv_mma.metal`, selected in `ggml-metal-common.cpp:143-153`).
- 4 to 8 rows for k-quants: `mul_mv_ext` (`ggml-metal-ops.cpp:2642-2650`).

Prompt matmul:

- Selected when the batch is larger than 8 and the row is at least 64 (`ggml-metal-common.cpp:35-41`).
- `kernel_mul_mm` (`mul_mm.metal:156-176`) uses 64x32x32 tiles, eight `simdgroup_float8x8` accumulators, and `simdgroup_multiply_accumulate` (`:211`, `:315`).
- A Metal 4 tensor-API variant is at `mul_mm.metal:20-145`.

Hardware gates: simdgroup reduction and simdgroup matrices need `MTLGPUFamilyApple7` (M1 and later) (`ggml-metal-device.m:1134-1137`). The tensor API needs Metal 4 (`:1146`).

Zero-copy weights (`ggml-metal-device.m:2179-2250`):

1. Round the pointer down to `sysconf(_SC_PAGESIZE)` and the length up.
2. Call `newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared deallocator:nil`.
3. When a region is larger than `maxBufferLength`, split it into overlapping views. The overlap is the largest tensor plus about 2 pages, so each tensor is fully inside one view (`:2229-2242`).

Own allocations use `vm_allocate`, with a `posix_memalign` fallback (`:2089-2099`).

Residency: one `MTLResidencySet` per buffer (`:2045-2060`). A background loop calls `requestResidency` to keep it alive for 3 minutes (`:976-1008`).

Shaders: either a prebuilt `default.metallib` loaded with `newLibraryWithURL` (`:517`), or the source embedded in the binary and compiled at run time with `newLibraryWithSource` (`:352`, `:630`; embedding in `ggml/src/ggml-metal/CMakeLists.txt:77-144`). Function constants specialize the kernels (`mul_mv_mma.metal:4-9`).

## 3. Vulkan on macOS (MoltenVK)

- Portability enumeration is turned on under `__APPLE__` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:5202-5238`).
- **Host-memory import is off on MoltenVK** "until MoltenVK PR 2622 is in the SDK" (`:4170-4174`). This blocks the Janas zero-copy path.
- `storageBuffer16BitAccess` is required (`:4635-4637`).
- On AMD GPUs under macOS, subgroup arithmetic and shuffle are off (`:4235-4246`).
- The Asahi Honeykrisp driver does not report the Apple vendor ID; llama.cpp forces Apple tiling for it (`:4833-4840`).

Not checked: whether MoltenVK PR 2622 is in a current Vulkan SDK.

## 4. CPU and GPU together

- No operation is split by rows between the CPU and Metal. Work is placed per layer or per tensor.
- Weights on the CPU are offloaded to Metal per operation when the batch is 32 or more (`ggml-metal-device.m:1284`).
- macOS has no thread affinity (`ggml/src/ggml-cpu/ggml-cpu.c:2609-2613`). Priority uses `SCHED_FIFO` (`:2615-2640`). No QoS classes are used.
- The default thread count is `hw.perflevel0.physicalcpu`, the performance cores only (`common/common.cpp:111`).

## 5. mmap on macOS

- Linux only: `posix_fadvise(SEQUENTIAL)`, `MAP_POPULATE` (`src/llama-mmap.cpp:487-494`), `O_DIRECT` (`:198-200`), `MADV_WILLNEED` prefetch (`:838`, `:877`).
- All POSIX: `posix_madvise` WILLNEED or RANDOM (`:500-523`).
- When `mlock` fails on macOS, the message tells the user to raise `vm.user_wire_limit` and `RLIMIT_MEMLOCK` (`:703-713`).
- Not used: `F_NOCACHE`, `F_RDADVISE`, `iogpu.wired_limit_mb`.
- The GPU budget is `recommendedMaxWorkingSetSize`, which is a soft limit (`ggml-metal-device.m:1502-1505`).
