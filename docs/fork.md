# About this fork

`phaedrus1992/janas` is a fork of [`prabanta-dev/janas`](https://github.com/prabanta-dev/janas).
It exists to port Janas to arm64 CPUs and to the Apple Silicon GPU, and to send that work upstream.

The fork keeps all upstream features for x86-64 CPUs and Vulkan GPUs.
It adds:

- NEON CPU kernels for macOS on Apple Silicon and for Linux arm64 servers.
- A Metal GPU backend for macOS, in pure C11, loaded at run time.
- macOS support for the I/O and memory paths, including models larger than memory.

## Status

E0, the fork workflow, is built: `clean` mirrors `upstream/main`, the daily
sync merges upstream into `main` through a pull request, and a pre-push hook
keeps fork tooling off `port/*` branches. The hook tests are in
`.fork/tests/`; run both by hand. Each clone installs the hook once with
`./.fork/install-hooks.sh`.

No port code exists yet.
Each epic gets its own design session before work starts.

The daily sync runs with the `SYNC_TOKEN` repository secret, a fine-grained
token owned by the fork's account. GitHub refuses pull request creation to the
workflow's own token in this repository, whatever the repository settings and
the workflow `permissions` say.
To rotate it, mint a new fine-grained token on that account with `Contents`,
`Issues`, `Pull requests` and `Workflows` write access to this repository, and
replace the secret's value.
The current token's expiry date lives on its own settings page; this file does
not repeat it.
A sync run that cannot read the token stops at the `Check the sync token` step
and names the fix.
`.fork/sync-upstream.sh` in a local clone uses your own credentials, not the
secret, so a lapsed token does not stop a manual sync.

## The spec

The spec and its reference material are in [`docs/superpowers/specs/2026-10-13-arm64-metal-port/`](superpowers/specs/2026-10-13-arm64-metal-port/):

| File | Content |
| --- | --- |
| [`design.md`](superpowers/specs/2026-10-13-arm64-metal-port/design.md) | Goal, scope, decisions, workflow, epics, tests, risks |
| [`ref-janas-seams.md`](superpowers/specs/2026-10-13-arm64-metal-port/ref-janas-seams.md) | The Janas extension points and the Linux-only calls |
| [`ref-llama-cpp.md`](superpowers/specs/2026-10-13-arm64-metal-port/ref-llama-cpp.md) | llama.cpp: ARM kernels, Metal, Vulkan on macOS |
| [`ref-mlx.md`](superpowers/specs/2026-10-13-arm64-metal-port/ref-mlx.md) | MLX, and why it is reference only |
| [`ref-mtplx.md`](superpowers/specs/2026-10-13-arm64-metal-port/ref-mtplx.md) | mtplx: drafts, Gated DeltaNet rollback, tuning |
| [`ref-omlx.md`](superpowers/specs/2026-10-13-arm64-metal-port/ref-omlx.md) | omlx: prompt cache, expert offload, memory guard |
| [`techniques.md`](superpowers/specs/2026-10-13-arm64-metal-port/techniques.md) | Each borrowed technique and the epic that uses it |

## Epics

Each epic is a milestone on this fork.

| Epic | Goal |
| --- | --- |
| E0 | Fork workflow and upstream agreement |
| E1 | Portability layer: builds and runs on macOS arm64 and Linux arm64 |
| E2 | NEON CPU kernels |
| E3 | Metal row-split matvec |
| E4 | Metal whole token |
| E5 | macOS I/O and memory |
| E6 | Draft and Gated DeltaNet speedups |
| Future | Ideas outside the port |

## Branches

| Branch | Content |
| --- | --- |
| `clean` | An exact mirror of upstream `main` |
| `port/<NN>-<topic>` | Engine changes for one upstream PR, with no fork files |
| `main` | `clean`, plus fork files, plus merged port work |

Fork files (this file, the README header, `docs/superpowers/`, `.llmenv.yaml`, and the other paths in `design.md` §5) stay on `main` and never go upstream.
