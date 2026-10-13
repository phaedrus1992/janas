# Reference: mtplx

Ref: `v2.12.2` (`9882703f`), from `youssofal/mtplx`.
Paths are relative to the repository root.
Licence: Apache-2.0.

mtplx is a Mac app and command line that runs models on Apple Silicon with the model's own multi-token prediction (MTP) heads.
Python drives the loop, MLX runs the operations, and about 75 custom Metal kernels fill the gaps.
It does not run models larger than memory, it does not stream experts, and it refuses GGUF (`mtplx/commands/forge.py:64-65`, `:577-583`).

## 1. Why the custom kernels exist

1. **Small-M verify matmul (2 to 16 rows).** The stock MLX quantized matmul does not reuse a weight load across the verify rows (`mtplx/nax_verify.py:1-15`, `mtplx/verify_qmv.py:1-8`). Measured on a 27B model: 48 to 65 tok/s (`docs/turbo-verify.md:21-33`).
2. **Ternary 2-bit matvec.** Stock 2-bit affine is limited by ALU, at 250 to 420 of 614 GB/s (`mtplx/kernels/ternary_qmv.py:1-27`).
3. **Fused Gated DeltaNet decode step.** It replaces 4 small dispatches between the two library matvecs. Their rule: fuse the element-wise work and keep the library matvecs, because hand-written matvecs "lost 3-for-3" (`mtplx/kernels/gdn_step_fused.py:1-12`).
4. **Gated DeltaNet tape capture and replay** for rollback (§2).

## 2. Speculative decoding

- The verify batch is one sequence of shape `[1, 1+K]`: the primary token and the drafts (`mtplx/generation.py:14815-14831`).
- Accept rule: accept with `min(1, p/q)`; on a reject, sample from the normalized `max(p - q, 0)` (`mtplx/sampling.py:295-339`, `:358-369`).
- Block verification (Sun et al. 2024) clips the running product instead of each factor. Measured: +1.85% tokens per window (`mtplx/qwen4_block_verify.py:1-40`).

Rollback:

- KV layers: `trim(n)` (`mtplx/cache_state.py:5051-5056`).
- Gated DeltaNet layers, three methods:
  1. Copy only the recurrent states before verify, restore them, then run a repair forward (`mtplx/cache_state.py:4953-4968`). This is what Janas does now.
  2. The verify kernel captures the conv and recurrent state at each step, and the commit takes the state at index `keep - 1` (`mtplx/gdn_capture.py:3107-3185`).
  3. Tape: the verify kernel writes the fp32 delta of each step, and a replay kernel rebuilds the state from `state_in` for `keep` steps (`mtplx/gdn_capture.py:1007`, `:1922-1960`; `mtplx/kernels/gdn_tape_headquarter.py:1-27`).
- Each row of a batch can commit a different prefix length (`mtplx/gdn_capture.py:3202-3216`).

Not checked: which method is the default for each model family, and the cost of a repair forward against a capture.

## 3. Memory

- `mx.set_memory_limit` and `mx.set_wired_limit`, capped by `iogpu.wired_limit_mb` (`mtplx/server/openai.py:2759`, `:2925-2948`). The server prints a `sudo sysctl` hint (`:3962`).
- **Idle keepalive.** macOS drops Metal residency about 2.5 s after the last command buffer, and residency costs about 9 ms per GiB to rebuild. A small kernel keeps the weights resident (`mtplx/model_scheduler.py:417-445`).
- An n-gram table of about 30 GiB streams from SSD through a memmap, with threaded `pread` prefetch at queue depth 16 and `madvise`. A cold 100k prefill went from 36 s to 7.5 s (`mtplx/models/qwen4_exp.py:5191-5290`).

## 4. Tuning

`mtplx-tune` (`mtplx/cli.py:5363`):

- Measures plain decoding and MTP depths 1 to 8, each in its own subprocess (`mtplx/commands/public.py:5149-5175`).
- Picks the highest speedup that passes a quality gate and shows no collapse in acceptance (`:5597-5615`).
- Stores the result in `~/.mtplx/tuning.json`, keyed by a SHA-256 of model, chip, chip family, hardware model, software versions, backend and settings (`:4532-4565`).
- Clears a stored winner when a later run measures a loss (`:4059-4070`).

## 5. Per-chip paths

- M5 (GPU family 17): Metal 4 TensorOps, with macOS 26.2 or later (`mtplx/nax_detect.py:1-60`, `mtplx/hardware.py:185-200`).
- M3: a native sparse-attention kernel, because M3 has no TensorOps.
- M1 and M2: FP16 in place of BF16, because they have no native BF16 (`mtplx/commands/forge.py:300`, `:328`).

## 6. What Janas takes

| Technique | Epic |
| --- | --- |
| Gated DeltaNet state capture at each step, or tape replay | E6 |
| Small-M verify kernel that reads each weight one time | E4, E6 |
| Fuse element-wise work, keep the matvec kernels | E4 |
| Exact accept rule and block verification | E6 (check what Janas does first) |
| Idle residency keepalive | E3 |
| Wired-limit cap and sysctl hint | E5 |
| Tuner key with chip and versions, quality gate, clear on loss | E2, E3 |
| Threaded `pread` prefetch at a tuned queue depth | E5 |

Not taken: the Python and MLX host design, the MLX private-ABI extensions, and the MLX affine formats.
