# Reference: omlx

Ref: `v0.7.0` (`4d4f5a28`), from `jundot/omlx`.
Paths are relative to the repository root.
Licence: Apache-2.0.

omlx is a Python server around mlx-lm, mlx-vlm and mlx-embeddings, with a Swift menu-bar app.
It has no engine of its own.
It pins `mlx==0.32.2` (`pyproject.toml:46-49`).

## 1. Process model

- One FastAPI process. All MLX GPU work for all models runs on one global thread, "to prevent Metal command buffer races that cause segfaults" (`omlx/engine_core.py:210-224`).
- The Swift app starts the server as a child process (`apps/omlx-mac/Sources/Server/ServerProcess.swift:17`, `:342`).

## 2. Prompt cache

- Paged metadata ported from the vLLM block pool: reference-counted blocks, an LRU free queue, a hash-to-block map, and copy-on-write prefix sharing (`omlx/cache/paged_cache.py:5-20`).
- Each block hash is a SHA-256 chain of model name, parent hash, token tuple and extra keys (`paged_cache.py:84-125`).
- SSD tier: one safetensors file per block at `<cache_dir>/<hex[0]>/<hex>.safetensors` (`omlx/cache/paged_ssd_cache.py:2166-2182`), with metadata for the token count, layer count, a cache signature and layer types (`:1026-1059`).
- Blocks with a different signature (quant bits, numerics, layer types) are refused (`:2677-2812`).
- Write path: the inference thread copies the bytes out; a writer thread writes a temporary file, calls `fsync`, renames it into place, and calls `fsync` on the directory. The writer makes no MLX calls (`:3203-3244`, `:904-920`).
- A RAM hot tier sits in front of the SSD, with one byte budget for all models (`:1473`).
- Restore: find the longest shared prefix (`omlx/cache/prefix_cache.py:702-730`), read the blocks in parallel (`paged_ssd_cache.py:4292-4310`, about 2 ms per block by a code comment), and join them into a contiguous cache (`prefix_cache.py:3118-3150`).
- Recurrent state gets boundary snapshots (`omlx/cache/boundary_snapshot_store.py`).

## 3. Scheduling

- vLLM-style waiting queue and running set over the mlx-lm `BatchGenerator` (`omlx/scheduler.py:1-10`).
- Prefill is cut into chunks sized in time while decodes run, with a "decode debt", because "Metal cannot preempt a running kernel, so bounding chunk duration IS the interleave mechanism" (`scheduler.py:1518-1530`).

## 4. Memory on unified memory

- `mx.set_wired_limit` is called only when the user set `iogpu.wired_limit_mb`. Otherwise the default of about 75% of RAM stays, and the log shows the sysctl to run (`omlx/process_memory_enforcer.py:173-198`, `:234-326`).
- Keep 5% of RAM free, because a process that jetsam kills leaves its wired memory stuck until a reboot (`:200-233`).
- The process ceiling is the minimum of a static limit, a dynamic limit from `host_statistics64`, and the Metal cap plus the CPU footprint (`:657-735`). Soft and hard watermarks, LRU model eviction, and an admission pause (`:1588-1600`).
- Memory is measured as `phys_footprint` through `proc_pid_rusage(RUSAGE_INFO_V4)`, because jetsam uses that value and RSS under-reports Metal memory (`omlx/utils/proc_memory.py:1-40`).
- A freed buffer that the GPU still uses can cause a kernel panic on M4; omlx keeps freed buffers in the pool to avoid it (`omlx/server.py:8401-8414`).

## 5. Expert offload

- Each layer keeps a resident fraction of the experts (default 0.25) with LRU slot eviction (`omlx/patches/moe_expert_offload.py:350-416`).
- Misses are read with `os.pread` from the original files on a thread pool (`:41`, `:106`, `:186`, `:254-285`).
- Resident experts compute while the reads are pending. A keepalive kernel runs because "Apple GPUs lower their clock after a couple of milliseconds idle" (`:578-590`, `:607-640`).
- Measured on a 26B model with 128 experts: accuracy stays flat down to 12% residency, and throughput falls about as the square root of memory (`:1-28`).

## 6. macOS integration

- Chip name from `sysctl machdep.cpu.brand_string`, RAM from `hw.memsize`, GPU core count from `system_profiler SPDisplaysDataType` (`omlx/utils/hardware.py:60-110`, `:255-275`).
- Tools are called by absolute path, because launchd gives no `/usr/sbin` in `PATH` (`:28-41`).
- Homebrew formula with a `service do` block for `brew services` (`Formula/omlx.rb:44-50`).

## 7. What Janas takes

| Technique | Epic |
| --- | --- |
| Compute resident experts while misses load; GPU keepalive | E5 |
| Memory guard on `phys_footprint` and `host_statistics64`, with watermarks | E5 |
| Wired-limit policy and a 5% RAM reserve | E5 |
| Prefill chunks bounded in time | E4 |
| One GPU submission thread | E4 |
| Do not free a buffer that the GPU can still use | E3 |
| Absolute-path sysctl calls | E1 |
| Hash-chained, SSD-tiered prompt cache | Future |

Not taken: the Python executor, the mlx-lm patch layer, MLX affine quant, and the cluster subsystem.
