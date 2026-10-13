# E0: fork workflow and upstream agreement

Status: the design was approved in brainstorming on 2026-10-13.
This spec fills in `design.md` §5 and the E0 epic from that document.
It is a design spec, not an implementation plan.
The implementation plan is a separate step, written later.

## 1. Goal and exit test

Goal: the branch model in `design.md` §5 exists and works, and an upstream issue that the maintainer answers.

Exit test: `clean`, `main`, the sync workflow and the guard work.
The maintainer answers the three questions in `design.md` §5.

## 2. Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Sync policy | Auto-resolve + auto-merge | Upstream releases several times a day and edits `README.md` with each release. The fork header at the top of `README.md` therefore conflicts on nearly every sync. A human per sync cannot keep up. |
| Sync landing | A `sync/YYYY-MM-DD` branch, merged into `main` through an auto-merged PR | `main` only ever moves through PR merges. `clean` stays a pure mirror forever. Each sync leaves a record. |
| README auto-resolve | Take upstream's `README.md`, then prepend the fork header block from `.fork/readme-header.md` | The header block is fixed text. The resolve is deterministic. |
| Guard hook | `.fork/hooks/pre-push.sh`, copied into `.git/hooks/pre-push` by `.fork/install-hooks.sh` | The hook must run on a `port/*` checkout, and that checkout carries no tooling files. prek refuses to run when its config is absent, and it consumes git's stdin refs. prek therefore cannot run this hook. |
| Upstream issue | Filed first, before the plumbing is built | The maintainer's answer gates E1. The clock starts as soon as the issue is up. |
| Local commits to `main` | Not possible; every `main` change arrives as a PR merge | The llmenv hook blocks commits on the default branch. The branch model in `design.md` §5 agrees. |

## 3. Artifacts

1. `clean`, created once on the fork: an exact fast-forward-only mirror of `upstream/main`.
   `sync-upstream.sh` creates it if a clone lacks it.
2. `.fork/sync-upstream.sh`: one sync implementation, used by the local session and by CI.
3. `.fork/check-clean.sh`: the guard.
4. `.fork/hooks/pre-push.sh`: the hook. It reads the pushed refs from stdin and calls the guard for `port/*` refs only. On a checkout without tooling files it reads the guard from `main`.
5. `.fork/readme-header.md`: the fork header block. The sync applies it. The guard checks for it.
6. `.fork/install-hooks.sh`: copies the hook into `.git/hooks/pre-push`. Each clone runs it once.
7. `.github/workflows/fork-sync.yml`: daily schedule plus manual dispatch. It checks out and runs `sync-upstream.sh`.
8. `.github/workflows/fork-guard.yml`: runs the guard on every PR whose head branch is `port/*`.
9. An amendment to `design.md` §5: add `.github/dependabot.yml` to the tooling paths list.
10. Fork issue E0, milestone E0, labels `area/build`, `P1`, `size/M`.
    Create it with the dev-sprint `create-issue.sh` script.
    That script validates the milestone and the labels.
11. The upstream issue on `prabanta-dev/janas`, filed first in implementation. Its draft text is in §6.

This epic adds no engine code.
It does not create `CLAUDE.md`, `AGENTS.md` or `.claude/`.
Upstream `release.yml` stays untouched.
It fires only on `v*` tags, which the fork never pushes.

## 4. Sync mechanics

`sync-upstream.sh` fails fast on a dirty tree and on a local `main` that differs from `origin/main`.
It runs from `main`.

The script resolves `clean` in this order:

1. The local branch `clean`, when it exists.
2. `origin/clean`, which is what CI sees after a full checkout.
3. Otherwise the script creates `clean` from `upstream/main`.
   This is the first run in a fresh clone.
   Artifact 1 and sync step 2 say the same.

The sync steps:

1. Fetch `upstream/main`.
2. Create `clean` from it when `clean` is missing. Fast-forward `clean`.
   Exit 0 when `main` already contains every commit `clean` has.
   The test is containment, not "clean moved": a run whose pull request never merged must try again on the next run.
3. Push `clean` to origin.
4. Fetch `origin/main`. Cut `sync/YYYY-MM-DD` from it.
   When an open PR with the same head exists, exit 0. The script says so.
   When the branch exists without an open PR, delete it and cut it again.
5. Merge `clean` into the sync branch with `git merge --no-ff --no-commit clean`.
   The `--no-ff` keeps the merge commit that the paragraph after the steps names.
   On a `README.md` conflict: take upstream's `README.md` with `git checkout clean -- README.md`, prepend `.fork/readme-header.md`, stage it.
   On any other conflict, do these:
   - Abort the merge.
   - Delete the branch.
   - Open a fork issue titled `sync blocked: <path> conflicts with upstream`, when no open issue with that title exists.
   - Exit 1.
6. Commit the merge with the message `sync: merge upstream/main into main (YYYY-MM-DD)`.
7. Push the sync branch. Open PR `sync/*` → `main`. Merge it at once with `gh pr merge --merge`, delete the branch.
8. Return to `main` and run `git pull --ff-only`. CI skips this step.

The PR merge adds a second merge commit on top of the sync branch's own merge commit.
This is cosmetic and keeps the record of each sync.

CI runs the same script.
`fork-sync.yml` sets `permissions: contents: write, issues: write, pull-requests: write`, a daily cron, `workflow_dispatch`, and a concurrency group so two runs never overlap.
The `issues` scope is for the conflict path, which opens a fork issue.
The runner checks out with `fetch-depth: 0`.

One repo setting must change once, by hand or with `gh api`:

```
PUT /repos/phaedrus1992/janas/actions/permissions/workflow
{"default_workflow_permissions": "write", "can_approve_pull_request_reviews": true}
```

The current value is `read` and pull request creation is off.
With those values, the sync workflow cannot push `clean`, and `gh pr create` fails with the workflow token.

## 5. Guard mechanics

`check-clean.sh [ref]` checks the ref given, or `HEAD`.
It fails fast when `clean` cannot be resolved, with the fix in the message.
Then two checks run:

1. Tooling paths. The diff `clean...ref` must not touch any of these:
   - `.llmenv.yaml`
   - `docs/fork.md`
   - `docs/superpowers/`
   - `.claude/`
   - `CLAUDE.md`
   - `AGENTS.md`
   - `.fork/`
   - `.github/workflows/fork-*.yml`
   - `.github/dependabot.yml`

   Other files under `.github/workflows/` stay allowed.
   A `port/*` branch may change upstream CI.

2. README marker. `git show ref:README.md` must not contain `> **This is a fork.**`.

The guard prints each offending path and exits 1.
A failed diff is an error, not an empty diff: a ref with no shared history with `clean` fails.
The path match ignores case, includes the bare names (`.fork` where a directory is expected), and covers `fork-*.yaml` as well as `fork-*.yml`.

The hook reads the ref lines from stdin.
For each line whose destination ref starts with `port/`, it runs the guard on the pushed commit.
The destination decides, so a push that creates `port/*` from any local ref is checked.
One failure blocks the push.
The guard comes from `main`, never from the working tree: the tree is the content under review.
A push with no `port/*` destination needs no guard and never blocks.
A `port/*` checkout carries no tooling files, so the hook reads the guard from `main` when the working tree has none.
When `main` is missing, the hook fails and names the fix: it never passes in silence.

`.fork/install-hooks.sh` copies the hook into `.git/hooks/pre-push`.
Each clone runs it once.

prek cannot install this hook.
The installed prek hook refuses to run when its config is absent, and a `port/*` checkout must not carry that config.
prek also consumes git's stdin, so the hook would never see the refs.

`fork-guard.yml` runs on pull requests whose head branch starts with `port/`.
It checks out with `fetch-depth: 0`, reads `main`'s copy of `.fork/check-clean.sh`, and runs it on the PR head SHA.
The guard must not come from the pull request's own tree: an added or changed guard would check itself.

A known limit: a PR opened against upstream runs upstream's CI, not the fork's.
The local pre-push hook is the only guard on that path.

## 6. Upstream issue

The issue goes on `prabanta-dev/janas` as the first implementation step, from the fork account.
The text gets a final review before it is filed.

Title: `arm64 and apple silicon port (neon + metal) — three questions first`

Body:

> Full disclosure up front: this port is designed with AI assistance (Claude Code), and I'll use it during implementation too, with everything reviewed and tested before it lands here.
>
> I'd like to port Janas to arm64 and the Apple Silicon GPU, in small PRs, one epic at a time:
>
> - portability layer: the Linux-only calls, thread placement, `build.sh` arm64/darwin targets, CI on macOS arm64 and Linux arm64 runners. Everything x86-64/Vulkan stays as it is.
> - NEON CPU kernels in the same tiers as the AVX2 ones (NEON, +dotprod, +i8mm), bit-exact with the scalar reference.
> - a native Metal backend behind the existing `gpu.h`/`gpu_token.h` contract: zero-copy weights, the CPU/GPU row split, the same arithmetic as the Vulkan paths.
> - macOS I/O and memory: expert streaming and the memory guard, so models bigger than memory work like they do on Linux.
> - draft and DeltaNet speedups, mostly platform-neutral.
>
> Three things I'd like to check before writing code, per CONTRIBUTING.md:
>
> 1. Metal host code in pure C11 through the Objective-C runtime (`objc_msgSend`), with Metal.framework loaded at run time with `dlopen`, the way Vulkan is loaded today. An .m file compiles to the same calls, so this is only about keeping the "C11, no deps" rule. OK?
> 2. CI runners: macOS arm64 and Linux arm64 in the upstream workflows, once the portability epic lands. OK?
> 3. Bit-exactness on NEON: float reductions in a different vector width can land on different bits. The NEON epic starts with research on how the AVX2 path meets the bit-exact rule today, and whether NEON can meet it the same way. If it can't, I'll propose a rule change before any kernel work.
>
> I test on an M1 Max (macOS) and Docker arm64. I have no real x86-64 machine, so the "no x86-64 speed regression" check needs your Core Ultra machine. All commits carry DCO sign-off, and any code borrowed from elsewhere (llama.cpp, mlx, both MIT) is declared in the PRs.

## 7. Testing

The scripts get fixture tests before the workflows trust them:

1. `check-clean.sh`: three throwaway branches.
   One carries a fake tooling file and must fail.
   One carries an engine-only change and must pass.
   One carries the README header and must fail.
2. `sync-upstream.sh`: a throwaway local clone whose `origin` and `upstream` remotes point at two bare test repos.
   Simulate an upstream commit. Run the script. Watch the full PR flow.
   Run it again with an upstream change that touches `README.md`.
   Check that the header comes back.
3. A real run: flip the token permission, dispatch `fork-sync.yml` by hand, and watch the first sync PR open and merge.
4. The guard in CI: open a throwaway `port/*` PR with a tooling file, confirm it fails, close it.

## 8. Implementation order

For the later implementation plan, not now:

1. File the upstream issue after its final text review.
2. Amend `design.md` §5: add `.github/dependabot.yml` to the tooling paths.
3. Create `clean` and push it to the fork.
4. Write the `.fork/` scripts and run the fixture tests from §7.
5. Add `.fork/install-hooks.sh` and run it.
6. Flip the Actions token permission to `write`.
7. Add `fork-sync.yml` and `fork-guard.yml`.
8. Dispatch one real sync run and watch the PR open and merge.
9. Open a throwaway guard PR, confirm it fails, close it.
10. File the fork E0 issue and update the status in `docs/fork.md`.

## 9. Notes

This file sits beside `design.md` in the port spec directory.
E1 to E6 get one spec file each: `e1-portability.md`, `e2-neon.md`, and so on.

No implementation starts from this document until the user says so.
The implementation plan is a separate step, written later.
