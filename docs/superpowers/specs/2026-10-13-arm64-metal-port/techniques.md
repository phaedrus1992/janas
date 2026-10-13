# Techniques to borrow

Each row is a technique from a reference project, with the epic that uses it.
Status values: `planned` (in an epic), `future` (in the `Future` milestone), `check` (verify against Janas before you use it).

| Technique | Source | Epic | Status |
| --- | --- | --- | --- |
| CPU feature detection with HWCAP on Linux and `hw.optional.arm.FEAT_*` on macOS | llama.cpp `ggml/src/ggml-feats.h:99-145` | E2 | planned |
| NEON dotprod and I8MM k-quant dot products | llama.cpp `ggml/src/ggml-cpu/arch/arm/quants.c:2018-3939` | E2 | planned |
| Two rows per call with I8MM (2x2 SMMLA tile) | llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:312-318` | E2 | planned |
| Weight repack to 8x8 or 8x4 for I8MM | llama.cpp `ggml/src/ggml-cpu/repack.cpp:4925` | E2 | check (changes the layout in memory) |
| Default thread count from `hw.perflevel0.physicalcpu` | llama.cpp `common/common.cpp:111` | E1 | planned |
| `newBufferWithBytesNoCopy` over mapped weights, with overlapping views | llama.cpp `ggml/src/ggml-metal/ggml-metal-device.m:2179-2250` | E3 | planned |
| `MTLResidencySet` with keepalive | llama.cpp `ggml-metal-device.m:976-1008`, `:2045-2060` | E3 | planned |
| Idle residency keepalive (drop after about 2.5 s idle) | mtplx `mtplx/model_scheduler.py:417-445` | E3 | planned |
| Matvec: 32-wide simdgroups, about 2 rows each, `simd_sum` | llama.cpp `mul_mv.metal:1511`; MLX `quantized.h:756-821` | E3 | planned |
| Embedded MSL source compiled at run time, function constants | llama.cpp `ggml-metal-device.m:352`, `:630` | E3 | planned |
| Do not free a buffer that the GPU can still use | omlx `omlx/server.py:8401-8414` | E3 | planned |
| Prompt matmul with `simdgroup_float8x8` tiles of 64x32x32 | llama.cpp `mul_mm.metal:156-315`; MLX `steel/gemm/mma.h` | E4 | planned |
| Mixture-of-experts gather matvec | MLX `quantized.cpp:1964-2050` | E4 | planned |
| Small-M verify kernel that reads each weight one time | mtplx `mtplx/verify_qmv.py`, `docs/turbo-verify.md:21-33` | E4, E6 | planned |
| Fuse element-wise work, keep the matvec kernels | mtplx `mtplx/kernels/gdn_step_fused.py:1-12` | E4 | planned |
| One GPU submission thread | omlx `omlx/engine_core.py:210-224` | E4 | planned |
| Prefill chunks bounded in time | omlx `omlx/scheduler.py:1518-1530` | E4 | planned |
| Resident experts compute while misses load; GPU keepalive | omlx `omlx/patches/moe_expert_offload.py:578-640` | E5 | planned |
| Threaded `pread` prefetch at a tuned queue depth | mtplx `mtplx/models/qwen4_exp.py:5191-5290` | E5 | planned |
| Memory guard on `phys_footprint` and `host_statistics64` | omlx `omlx/process_memory_enforcer.py:657-735`, `omlx/utils/proc_memory.py` | E5 | planned |
| Wired-limit cap, sysctl hint, 5% RAM reserve | omlx `process_memory_enforcer.py:173-326`; mtplx `mtplx/server/openai.py:2925-2948` | E5 | planned |
| Gated DeltaNet state capture at each step, or tape replay | mtplx `mtplx/gdn_capture.py:1922-1960`, `:3107-3185` | E6 | planned |
| Exact accept rule with residual sampling | mtplx `mtplx/sampling.py:295-369` | E6 | check (compare with the Janas sampler) |
| Block verification | mtplx `mtplx/qwen4_block_verify.py:1-40` | E6 | check |
| Tuner key with chip and versions, quality gate, clear on loss | mtplx `mtplx/commands/public.py:4059-4070`, `:4532-4565`, `:5597-5615` | E2, E3 | check (compare with `janas-bench`) |
| Absolute-path sysctl calls | omlx `omlx/utils/hardware.py:28-41` | E1 | planned |
| Hash-chained prompt cache with an SSD tier | omlx `omlx/cache/paged_cache.py:84-125`, `omlx/cache/paged_ssd_cache.py` | Future | future |
| SME kernels | llama.cpp `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` | Future | future |
| Vulkan on MoltenVK | llama.cpp `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4170-4174` | Future | future (blocked on host import) |
