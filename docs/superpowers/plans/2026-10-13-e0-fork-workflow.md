# E0 fork workflow implementation plan

> **For agentic workers:** Work this plan inline in the current session with `superpowers:executing-plans` for step tracking. Do NOT use `superpowers:subagent-driven-development` (project rule: it hides work in subagent transcripts). Stop after the last plan task and return to `nbl-dev:ship-issue` Step 6. Do NOT run the executing-plans tail: no final whole-branch review subagent, no `superpowers:finishing-a-development-branch`, no delete of a plan workspace. The sprint's `nbl-dev:pre-pr-review` stage is the only review stage.
>
> Track the steps with `llmenv task`, not TaskCreate. Use each plan task's own heading as the task title, and put this file's path in the task note.

**Goal:** Build the fork branch model (`clean`, the sync, the guard) and file the upstream issue that asks the maintainer the three questions in `design.md` §5.

**Architecture:** One sync script and one guard script do all the work. The local session and the two GitHub workflows call the same scripts, so there is one implementation each. Fixture tests prove the scripts before a workflow trusts them. The epic adds no engine code.

**Tech Stack:** Bash (`set -euo pipefail`, shellcheck, shfmt), git, the `gh` CLI, GitHub Actions (SHA-pinned), Dependabot.

**Spec:** `docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md` (section numbers below refer to it).

**Issue:** #10.

## Global Constraints

- Branch model (`design.md` §5): `clean` is a fast-forward-only mirror of `upstream/main`; `port/*` branches carry engine work only; `main` receives tooling and merged work, and moves only through PR merges.
- This plan's own work lands on `feat/10-e0-fork-workflow` and reaches `main` through a PR. Never commit on `main`; a local hook blocks it.
- Every commit carries `Signed-off-by` (`git commit -s -m ...`), subject under 60 characters, conventional commit style, lowercase after the colon, no trailing period.
- Shell: `set -euo pipefail`, `shellcheck` and `shfmt` clean, no `grep -P`, macOS and Linux both.
- Workflows: `actions/checkout` pinned to `3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1) with a version comment. `actionlint` and `zizmor` clean.
- Dependabot: `github-actions` ecosystem, 7-day cooldown, grouped update.
- Tooling paths live on `main` only: `.llmenv.yaml`, `docs/fork.md`, `docs/superpowers/`, `.claude/`, `CLAUDE.md`, `AGENTS.md`, `.fork/`, `.github/workflows/fork-*.yml`, `.github/dependabot.yml`.
- The `upstream` remote is `https://github.com/prabanta-dev/janas.git`. The fork remote `origin` is `phaedrus1992/janas`.
- Scratch files stay under `.tmp/` in this clone, and the scripts clean up their own scratch.

## File Structure

| File | Responsibility |
| --- | --- |
| `.fork/readme-header.md` | The fork header block. The sync prepends it to upstream's README. |
| `.fork/check-clean.sh` | The guard. One ref in, exit 0 or 1 out. |
| `.fork/hooks/pre-push.sh` | The pre-push hook. Reads git's ref lines, calls the guard for `port/*` refs; reads the guard from `main` when the tree has none. |
| `.fork/sync-upstream.sh` | The sync. Fast-forwards `clean`, then merges it into `main` through a PR. |
| `.fork/tests/test-check-clean.sh` | Fixture test for the guard and the hook. |
| `.fork/tests/test-sync-upstream.sh` | Fixture test for the sync, against two bare repositories and a stub `gh`. |
| `.fork/install-hooks.sh` | Copies the hook into `.git/hooks/pre-push`. One run per clone. |
| `.github/workflows/fork-sync.yml` | Daily schedule and manual dispatch. Runs the sync. |
| `.github/workflows/fork-guard.yml` | Runs the guard on every `port/*` pull request. |
| `.github/dependabot.yml` | Keeps the pinned actions current, grouped, 7-day cooldown. |
| `docs/fork.md` | The fork summary. Gains the E0 status. |
| `docs/superpowers/specs/2026-10-13-arm64-metal-port/design.md` | §5 tooling paths gain the two new fork files. |
| `docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md` | §4 gains the corrected `clean` resolution and the second permission setting. |

## Review Focus

These are the failure modes the spec implies but its artifact list does not spell out. Each one gets a test in the task named.

1. **A clone with no `clean` branch.** The script must create it from `upstream/main` and push it (§3 artifact 1), not stop with an error. → Task 5, scenario 5.
2. **A second sync on the same day, after a failed run.** A `sync/<date>` branch exists with no open PR. The script must delete it and cut it again (§4 step 4). → Task 5, scenario 2.
3. **A push of several refs where the `port/*` ref is not the first line.** The hook must check every line, not the first (§5). → Task 4.
4. **A `port/*` branch that deletes `README.md`.** The guard must fail with a clear message, not pass in silence. → Task 3.
5. **Two syncs at once.** An open PR for the same head must end the run with exit 0 (§4 step 4); the workflow adds a concurrency group. → Task 5, scenario 3.

---

### Task 1: File the upstream issue

This is an outward-facing action. Do NOT file it until ranger has reviewed the text. It is the only task in this plan that blocks on him.

**Files:** none. The work is a GitHub issue plus a record comment.

**Interfaces:**
- Consumes: the issue text in spec §6.
- Produces: the upstream issue URL, recorded in fork issue #10.

- [ ] **Step 1: Show the text and get the final review**

Print spec §6 to ranger and ask for the final review. Apply his edits. Do not continue until he approves the text.

- [ ] **Step 2: Write the body to a file**

Write the approved text, and nothing else, to `.tmp/upstream-issue-body.md`. Keep the `>` quote markers off; they are Markdown quoting in the spec, and the issue body must not carry them.

- [ ] **Step 3: File the issue**

Run:

```bash
GH_SHIM_ACCOUNT=phaedrus1992 gh issue create --repo prabanta-dev/janas \
  --title "arm64 and apple silicon port (neon + metal) — three questions first" \
  --body-file .tmp/upstream-issue-body.md
```

`GH_SHIM_ACCOUNT` is not decoration.
The gh identity shim picks the account from the repository owner, and it falls back to the personal account (`RangerRick`) when the owner matches no login.
`prabanta-dev` matches none, so a bare call files the issue as `RangerRick`.
`GH_SHIM_ACCOUNT` is the shim's documented override, and it names the account directly.
`gh auth status` reads as `phaedrus1992` either way: the shim replaces `GH_TOKEN` after that check.
Check the author after filing (`gh issue view <N> --repo prabanta-dev/janas --json author`).

Expected: a URL on stdout, `https://github.com/prabanta-dev/janas/issues/<N>`.

- [ ] **Step 4: Record the URL on the fork issue**

```bash
gh issue comment 10 --body "upstream issue opened: <URL>"
```

- [ ] **Step 5: Commit nothing**

There is no file change. Note the URL in the session and continue.

---

### Task 2: The `clean` branch and the header block

**Files:**
- Create: `.fork/readme-header.md`
- Create (branch, not a file): `clean`

**Interfaces:**
- Consumes: `README.md` in `main`, and `upstream/main`.
- Produces: the branch `clean` on the fork, equal to `upstream/main`; and `.fork/readme-header.md`, which Task 5 concatenates onto upstream's README.

- [ ] **Step 1: Extract the header block**

```bash
sed -n '1,6p' README.md > .fork/readme-header.md
```

The block is five quoted lines plus one empty line. The empty line keeps the header separate from upstream's `# Janas` title.

- [ ] **Step 2: Check the block against the README it came from**

The fork's README is the header block, then the README of the fork's upstream base.

```bash
diff <(cat .fork/readme-header.md; git show 98c8300:README.md) <(git show main:README.md)
```

Expected: no output. A difference means the header block is wrong. Do not commit a wrong block.

- [ ] **Step 3: Create `clean` and push it**

```bash
git fetch upstream
git branch clean upstream/main
git push --set-upstream origin clean
```

- [ ] **Step 4: Verify the mirror**

```bash
git rev-parse clean origin/clean upstream/main
```

Expected: three identical SHAs.

- [ ] **Step 5: Commit**

```bash
git add .fork/readme-header.md
git commit -s -m "feat(fork): add the fork README header block"
```

---

### Task 3: The guard, its fixture test, and the tooling-path list

**Files:**
- Create: `.fork/check-clean.sh`
- Create: `.fork/tests/test-check-clean.sh`
- Modify: `docs/superpowers/specs/2026-10-13-arm64-metal-port/design.md` §5

**Interfaces:**
- Consumes: the branch `clean` (Task 2).
- Produces: `.fork/check-clean.sh [ref]` — exit 0 when the ref is legal, 1 when it is not; each offending path on stderr. Task 4's hook and Task 6's workflow call it.

- [ ] **Step 1: Write the failing test**

Create `.fork/tests/test-check-clean.sh`:

```bash
#!/usr/bin/env bash
# Fixture test for .fork/check-clean.sh and .fork/hooks/pre-push.sh.
# The test runs in this clone. It touches no file of the working tree: every
# branch is built with git plumbing on top of clean, and the trap removes them.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
guard="$root/.fork/check-clean.sh"
hook="$root/.fork/hooks/pre-push.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-check-clean.XXXXXX")"
fixtures=()

cleanup() {
  local ref
  for ref in "${fixtures[@]}"; do
    git branch --delete --force "$ref" >/dev/null 2>&1 || true
  done
  rm -rf "$scratch"
}
trap cleanup EXIT

cd "$root"
if ! git rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  echo "FAIL: the branch clean is missing; run .fork/sync-upstream.sh once" >&2
  exit 1
fi

fails=0
bad() { echo "FAIL: $*" >&2; fails=$((fails + 1)); }

# make_ref <name> <path> <content>...  Builds a commit on top of clean.
# A content of "-" removes the path.
make_ref() {
  local name="$1" index="$scratch/index.$1"
  shift
  GIT_INDEX_FILE="$index" git read-tree clean
  while (($#)); do
    local path="$1" content="$2"
    shift 2
    if [[ "$content" == "-" ]]; then
      GIT_INDEX_FILE="$index" git update-index --force-remove "$path"
    else
      local blob
      blob="$(printf '%s' "$content" | git hash-object --stdin -w)"
      GIT_INDEX_FILE="$index" git update-index --add --cacheinfo "100644,$blob,$path"
    fi
  done
  local tree commit
  tree="$(GIT_INDEX_FILE="$index" git write-tree)"
  commit="$(git commit-tree "$tree" -p clean -m "fixture: $name")"
  git branch "$name" "$commit"
  fixtures+=("$name")
}

make_ref fix-engine src/llm/quant.c '/* fixture */'
make_ref fix-tooling .llmenv.yaml 'id: fixture'
make_ref fix-header README.md '> **This is a fork.**'
make_ref fix-deleted README.md '-'

expect_pass() { if "$guard" "$1" >/dev/null 2>&1; then return 0; fi; bad "$1 must pass"; }
expect_fail() { if "$guard" "$1" >/dev/null 2>&1; then bad "$1 must fail"; fi; }

expect_pass fix-engine
expect_fail fix-tooling
expect_fail fix-header
expect_fail fix-deleted

# A clone with no clean anywhere must fail with the fix in the message.
clone="$scratch/noclean"
git clone --quiet --shared "$root" "$clone"
git -C "$clone" update-ref -d refs/remotes/origin/clean
if (cd "$clone" && "$guard" HEAD) 2>"$scratch/noclean.err"; then
  bad "a clone without clean must fail"
elif ! grep -q "clean" "$scratch/noclean.err"; then
  bad "the failure message must name the fix"
fi

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-check-clean: all checks passed"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
chmod +x .fork/tests/test-check-clean.sh
.fork/tests/test-check-clean.sh
```

Expected: exit 1, `No such file or directory` for `.fork/check-clean.sh`.

- [ ] **Step 3: Write the guard**

Create `.fork/check-clean.sh`:

```bash
#!/usr/bin/env bash
# The port-branch guard. A port/* branch must not touch fork tooling, and its
# README.md must not carry the fork header.
# Design: docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md §5
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: check-clean.sh [ref]

Checks the ref (default HEAD) against clean. Exit 0 when the ref is a legal
port branch, 1 when it is not, or when clean cannot be resolved.
EOF
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac
ref="${1:-HEAD}"

if git rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  clean=clean
elif git rev-parse --verify --quiet refs/remotes/origin/clean >/dev/null; then
  clean=origin/clean
else
  echo "check-clean: the branch clean is missing." >&2
  echo "check-clean: create it with .fork/sync-upstream.sh, or by hand:" >&2
  echo "check-clean:   git branch clean upstream/main && git push -u origin clean" >&2
  exit 1
fi

if ! git rev-parse --verify --quiet "$ref" >/dev/null; then
  echo "check-clean: $ref is not a valid ref." >&2
  exit 1
fi

# Tooling paths (design.md §5). Port branches must not change them.
exact_paths=(.llmenv.yaml docs/fork.md CLAUDE.md AGENTS.md .github/dependabot.yml)
prefix_paths=(docs/superpowers/ .claude/ .fork/)

is_tooling() {
  local path="$1" pattern
  for pattern in "${exact_paths[@]}"; do
    [[ "$path" == "$pattern" ]] && return 0
  done
  for pattern in "${prefix_paths[@]}"; do
    [[ "$path" == "$pattern"* ]] && return 0
  done
  for pattern in "${glob_paths[@]}"; do
    # Unquoted on purpose: this is a glob match.
    [[ "$path" == $pattern ]] && return 0
  done
  return 1
}

status=0
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  if is_tooling "$path"; then
    echo "check-clean: $path is a fork tooling path; a port branch must not change it" >&2
    status=1
  fi
done < <(git diff --name-only "$clean...$ref")

if ! readme="$(git show "$ref:README.md" 2>/dev/null)"; then
  echo "check-clean: cannot read README.md at $ref" >&2
  status=1
elif [[ "$readme" == *'> **This is a fork.**'* ]]; then
  echo "check-clean: README.md at $ref carries the fork header" >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "check-clean: $ref is clean."
fi
exit "$status"
```

- [ ] **Step 4: Run the test again and watch it pass**

```bash
chmod +x .fork/check-clean.sh
.fork/tests/test-check-clean.sh
```

Expected: `test-check-clean: all checks passed`.

- [ ] **Step 5: Amend the tooling-path list in the design**

In `design.md` §5, the list under "Tooling paths (permitted on `main` only)" becomes:

```markdown
- `.llmenv.yaml`
- `docs/fork.md` (the fork summary)
- `docs/superpowers/`
- `.claude/`, `CLAUDE.md`, `AGENTS.md`
- `.fork/`
- `.github/workflows/fork-*.yml`
- `.github/dependabot.yml`
```

- [ ] **Step 6: Lint and commit**

```bash
shellcheck .fork/check-clean.sh .fork/tests/test-check-clean.sh
shfmt -d .fork/check-clean.sh .fork/tests/test-check-clean.sh
git add .fork/check-clean.sh .fork/tests/test-check-clean.sh \
  docs/superpowers/specs/2026-10-13-arm64-metal-port/design.md
git commit -s -m "feat(fork): add the port-branch guard"
```

---

### Task 4: The pre-push hook and its installer

**Files:**
- Create: `.fork/hooks/pre-push.sh`
- Create: `.fork/install-hooks.sh`
- Modify: `.fork/tests/test-check-clean.sh`

**Interfaces:**
- Consumes: `.fork/check-clean.sh` from Task 3.
- Produces: `.fork/install-hooks.sh`, which copies `.fork/hooks/pre-push.sh` into `.git/hooks/pre-push`. The hook reads git's pre-push lines on stdin and exits 1 when any `port/*` ref fails the guard.

**Do not use prek for this hook.** A `port/*` checkout carries no tooling files, so its tree has no prek config; the installed prek hook then exits 1 and blocks every push. prek also consumes git's stdin, so the hook would never see the refs. The spec's §5 records this.

- [ ] **Step 1: Add the failing hook cases to the test**

Append to `.fork/tests/test-check-clean.sh`, before the final `if [[ "$fails" -gt 0 ]]` block. The cases cover the plain refs, a multi-ref push, a delete, and the guard-from-`main` path on a checkout with no tooling:

```bash
# The pre-push hook runs the guard on every port/* line of its stdin.
zero=0000000000000000000000000000000000000000
tooling_sha="$(git rev-parse fix-tooling)"
port_line="refs/heads/port/x $tooling_sha refs/heads/port/x $zero"
feat_line="refs/heads/feat/x $tooling_sha refs/heads/feat/x $zero"
delete_line="(delete) $zero refs/heads/port/gone $tooling_sha"

if printf '%s\n' "$port_line" | "$hook" >/dev/null 2>&1; then
  bad "a port/* push of a tooling ref must fail"
fi
if ! printf '%s\n' "$feat_line" | "$hook" >/dev/null 2>&1; then
  bad "a feat/* push must pass"
fi
if printf '%s\n%s\n' "$feat_line" "$port_line" | "$hook" >/dev/null 2>&1; then
  bad "a multi-ref push with a port/* line must fail"
fi
if ! printf '%s\n' "$delete_line" | "$hook" >/dev/null 2>&1; then
  bad "a deleted port/* ref must pass"
fi
```

Then add the scratch-repository block that proves the fallback: push the current tree to a bare repository, fetch it into a scratch work tree, check out `clean` (no tooling), and run the installed hook on a tooling commit.

```bash
# A port/* checkout carries no tooling. The hook must read the guard from
# main there, and it must block the push when main is missing.
bare="$scratch/tiny.git"
work="$scratch/tiny-work"
git init --quiet --bare "$bare"
git push --quiet "$bare" "HEAD:refs/heads/main" clean:refs/heads/clean
git init --quiet "$work"
git -C "$work" remote add origin "$bare"
git -C "$work" fetch --quiet origin
git -C "$work" branch --no-track main origin/main
git -C "$work" switch --quiet --detach origin/clean
tiny_index="$scratch/index.tiny"
GIT_INDEX_FILE="$tiny_index" git -C "$work" read-tree origin/clean
blob="$(printf 'id: tiny' | git -C "$work" hash-object --stdin -w)"
GIT_INDEX_FILE="$tiny_index" git -C "$work" update-index --add --cacheinfo "100644,$blob,.llmenv.yaml"
tree="$(GIT_INDEX_FILE="$tiny_index" git -C "$work" write-tree)"
tiny_commit="$(git -C "$work" commit-tree "$tree" -p origin/clean -m "fixture: tiny tooling ref")"
git -C "$work" branch --force port/x "$tiny_commit"
cp "$hook" "$work/.git/hooks/pre-push"
chmod +x "$work/.git/hooks/pre-push"
if printf 'refs/heads/port/x %s refs/heads/port/x %s\n' "$tiny_commit" "$zero" |
  (cd "$work" && .git/hooks/pre-push) >/dev/null 2>&1; then
  bad "a port/* push from a checkout without tooling must fail"
fi
git -C "$work" branch --move main main-hidden
if printf 'refs/heads/port/x %s refs/heads/port/x %s\n' "$tiny_commit" "$zero" |
  (cd "$work" && .git/hooks/pre-push) 2>"$scratch/nomain.err"; then
  bad "a port/* push with no main must fail"
elif ! grep -q "main" "$scratch/nomain.err"; then
  bad "the failure must name main"
fi
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.fork/tests/test-check-clean.sh
```

Expected: exit 1, with the hook cases reported as failures.

- [ ] **Step 3: Write the hook**

Create `.fork/hooks/pre-push.sh`:

```bash
#!/usr/bin/env bash
# The port-branch guard hook. Git writes one line per pushed ref on stdin:
#   <local ref> <local oid> <remote ref> <remote oid>
# Every port/* line runs the guard. One failure blocks the push.
# Install it once per clone with .fork/install-hooks.sh.
#
# A port/* checkout carries no tooling files (design.md §5), so the guard is
# read from main when the working tree has none. There is one guard.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
guard="$root/.fork/check-clean.sh"
scratch=""

cleanup() {
  if [[ -n "$scratch" ]]; then
    rm -f "$scratch"
  fi
}
trap cleanup EXIT

if [[ ! -f "$guard" ]]; then
  scratch="$(mktemp "${TMPDIR:-/tmp}/janas-guard.XXXXXX")"
  if ! git show main:.fork/check-clean.sh >"$scratch" 2>/dev/null; then
    echo "pre-push: cannot read .fork/check-clean.sh from main." >&2
    echo "pre-push: fetch main, then push again." >&2
    exit 1
  fi
  guard="$scratch"
fi

status=0
while read -r local_ref local_sha _; do
  [[ -z "$local_ref" ]] && continue
  [[ "$local_ref" == refs/heads/port/* ]] || continue
  bash "$guard" "$local_sha" || status=1
done

exit "$status"
```

- [ ] **Step 4: Run the test again and watch it pass**

```bash
chmod +x .fork/hooks/pre-push.sh
.fork/tests/test-check-clean.sh
```

Expected: `test-check-clean: all checks passed`.

- [ ] **Step 5: Write the installer**

Create `.fork/install-hooks.sh`:

```bash
#!/usr/bin/env bash
# Install the fork hooks into this clone. Run it once per clone after the
# first checkout of main. Design: e0-fork-workflow.md §5
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
source_hook="$root/.fork/hooks/pre-push.sh"
target="$root/.git/hooks/pre-push"

[[ -f "$source_hook" ]] || {
  echo "install-hooks: $source_hook is missing; run this from main." >&2
  exit 1
}

cp "$source_hook" "$target"
chmod +x "$target"
echo "install-hooks: installed $target"
```

- [ ] **Step 6: Install the hook in this clone**

```bash
.fork/install-hooks.sh
```

Expected: `install-hooks: installed .../.git/hooks/pre-push`.

- [ ] **Step 7: Prove git runs it**

The test cases call the hook file directly. This step goes through the installed copy.

```bash
sha="$(git rev-parse clean)"
printf 'refs/heads/port/probe %s refs/heads/port/probe %s\n' "$sha" "$sha" | .git/hooks/pre-push
```

Expected: `check-clean: <sha> is clean.` and exit 0.

- [ ] **Step 8: Lint and commit**

```bash
shellcheck .fork/hooks/pre-push.sh .fork/install-hooks.sh .fork/tests/test-check-clean.sh
shfmt -i 2 -d .fork/hooks/pre-push.sh .fork/install-hooks.sh .fork/tests/test-check-clean.sh
git add .fork/hooks/pre-push.sh .fork/install-hooks.sh .fork/tests/test-check-clean.sh
git commit -s -m "feat(fork): guard port pushes with a pre-push hook"
```

---

### Task 5: The sync script and its fixture test

**Files:**
- Create: `.fork/sync-upstream.sh`
- Create: `.fork/tests/test-sync-upstream.sh`
- Modify: `docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md` §4

**Interfaces:**
- Consumes: `.fork/readme-header.md` (Task 2), the branch `clean` (Task 2).
- Produces: `.fork/sync-upstream.sh`, no arguments. Task 6's workflow runs it.

- [ ] **Step 1: Write the failing test**

Create `.fork/tests/test-sync-upstream.sh`:

```bash
#!/usr/bin/env bash
# Fixture test for .fork/sync-upstream.sh. Two bare repositories stand in for
# upstream and origin, and a stub gh stands in for GitHub. The stub writes
# every call to a log, so the test can assert the PR flow.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
script="$root/.fork/sync-upstream.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sync.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# A fixed identity and no signing: the fixtures must not need a key.
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=user.name GIT_CONFIG_VALUE_1=fixture
export GIT_CONFIG_KEY_2=user.email GIT_CONFIG_VALUE_2=fixture@example.invalid

fails=0
bad() { echo "FAIL: $*" >&2; fails=$((fails + 1)); }
ok() { echo "ok: $*"; }

upstream="$scratch/upstream.git"
origin="$scratch/origin.git"
git init --quiet --bare --initial-branch=main "$upstream"
git init --quiet --bare --initial-branch=main "$origin"

seed="$scratch/seed"
git clone --quiet "$upstream" "$seed"
cd "$seed"
printf '# Upstream\n\none\n' >README.md
mkdir -p src
printf 'int one(void) { return 1; }\n' >src/engine.c
git add -A
git commit --quiet -m "upstream: first"

fork="$scratch/fork"
git clone --quiet "$origin" "$fork"
cd "$fork"
git remote add upstream "$upstream"
git fetch --quiet upstream
git switch --quiet --create main upstream/main
{ cat "$root/.fork/readme-header.md"; cat README.md; } >README.new
mv README.new README.md
git add README.md
git commit --quiet -m "fork: header"
git push --quiet --set-upstream origin main

# The stub gh. It records every call. GH_STUB_OPEN_PRS is the count that
# `gh pr list` reports.
mkdir -p "$scratch/bin"
cat >"$scratch/bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$GH_STUB_LOG"
case "$1 ${2:-}" in
  "pr list") echo "${GH_STUB_OPEN_PRS:-0}" ;;
  "pr create") echo "https://example.invalid/pull/1" ;;
  "pr merge") echo "merged" ;;
  "issue list") echo "0" ;;
  "issue create") echo "https://example.invalid/issues/1" ;;
  *)
    echo "gh-stub: unhandled: $*" >&2
    exit 1
    ;;
esac
STUB
chmod +x "$scratch/bin/gh"
export PATH="$scratch/bin:$PATH"
export GH_STUB_LOG="$scratch/gh.log"
: >"$GH_STUB_LOG"

today="$(date +%F)"
sync_branch="sync/$today"

run_sync() { cd "$1" && "$script"; }

# Scenario 1: the first sync. Upstream touches README.md, so the merge
# conflicts and the script must restore the header.
cd "$seed"
printf 'two\n' >>README.md
git commit --quiet -am "upstream: readme two"
git push --quiet origin main

if run_sync "$fork"; then ok "sync 1 exits 0"; else bad "sync 1 must exit 0"; fi

clean_sha="$(git --git-dir="$upstream" rev-parse main)"
fork_clean="$(git -C "$fork" rev-parse origin/clean)"
[[ "$clean_sha" == "$fork_clean" ]] && ok "clean mirrors upstream" || bad "clean must mirror upstream"

readme="$(git -C "$fork" show "origin/$sync_branch:README.md")"
[[ "$readme" == '> **This is a fork.**'* ]] && ok "the header is back" || bad "the header must be back"
[[ "$readme" == *"two"* ]] && ok "upstream's README text is kept" || bad "upstream's text must be kept"

parents="$(git --git-dir="$origin" rev-list --parents --no-walk "$sync_branch" | wc -w)"
[[ "$parents" -eq 3 ]] && ok "the sync branch carries a merge commit" || bad "the sync branch needs a merge commit"

grep -q -- "pr create" "$GH_STUB_LOG" && ok "the PR was opened" || bad "gh pr create must run"
grep -q -- "pr merge" "$GH_STUB_LOG" && ok "the PR was merged" || bad "gh pr merge must run"

# Scenario 2: a second sync on the same day. The branch exists, no PR is
# open, so the script deletes it and cuts it again.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'int two(void) { return 2; }\n' >src/engine.c
git commit --quiet -am "upstream: engine two"
git push --quiet origin main
before="$(git --git-dir="$origin" rev-parse "$sync_branch")"
if run_sync "$fork"; then ok "sync 2 exits 0"; else bad "sync 2 must exit 0"; fi
after="$(git --git-dir="$origin" rev-parse "$sync_branch")"
[[ "$before" != "$after" ]] && ok "the sync branch was cut again" || bad "the branch must be cut again"
git -C "$fork" show "origin/$sync_branch:src/engine.c" | grep -q "two" \
  && ok "upstream's engine change arrived" || bad "the engine change must arrive"

# Scenario 3: an open PR for the same head. The script must do nothing.
rm -f "$GH_STUB_LOG"
cd "$fork"
before="$(git --git-dir="$origin" rev-parse "$sync_branch")"
export GH_STUB_OPEN_PRS=1
if run_sync "$fork"; then ok "sync 3 exits 0"; else bad "sync 3 must exit 0"; fi
unset GH_STUB_OPEN_PRS
after="$(git --git-dir="$origin" rev-parse "$sync_branch")"
[[ "$before" == "$after" ]] && ok "nothing was pushed" || bad "nothing must be pushed"
grep -q "pr create" "$GH_STUB_LOG" && bad "no PR must be opened" || ok "no PR was opened"

# Scenario 4: a conflict on a file other than README.md. The script must
# abort, delete the branch, open one issue, and exit 1.
rm -f "$GH_STUB_LOG"
cd "$fork"
git switch --quiet main
printf 'fork edit\n' >src/engine.c
git commit --quiet -am "fork: engine edit"
git push --quiet origin main
cd "$seed"
printf 'upstream edit\n' >src/engine.c
git commit --quiet -am "upstream: engine edit"
git push --quiet origin main
cd "$fork"
if run_sync "$fork" 2>"$scratch/conflict.err"; then
  bad "sync 4 must fail"
else
  ok "sync 4 fails"
fi
grep -q "src/engine.c" "$scratch/conflict.err" && ok "the failure names the path" || bad "the failure must name the path"
grep -q "issue create" "$GH_STUB_LOG" && ok "an issue was opened" || bad "an issue must be opened"
[[ "$(git -C "$fork" rev-parse --abbrev-ref HEAD)" == "main" ]] \
  && ok "the script returned to main" || bad "the script must return to main"
git -C "$fork" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null \
  && bad "the local sync branch must be deleted" || ok "the local sync branch is gone"

# Scenario 5: a clone with no clean branch. The script must create it and
# push it, not fail.
rm -f "$GH_STUB_LOG"
git --git-dir="$origin" update-ref -d refs/heads/clean
fresh="$scratch/fresh"
git clone --quiet "$origin" "$fresh"
cd "$fresh"
git remote add upstream "$upstream"
git fetch --quiet upstream
git switch --quiet main
if run_sync "$fresh"; then ok "sync 5 exits 0"; else bad "sync 5 must exit 0"; fi
git --git-dir="$origin" rev-parse --verify --quiet refs/heads/clean >/dev/null \
  && ok "clean was created and pushed" || bad "clean must be created and pushed"

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-sync-upstream: all checks passed"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
chmod +x .fork/tests/test-sync-upstream.sh
.fork/tests/test-sync-upstream.sh
```

Expected: exit 1, `No such file or directory` for `.fork/sync-upstream.sh`.

- [ ] **Step 3: Write the sync script**

Create `.fork/sync-upstream.sh`:

```bash
#!/usr/bin/env bash
# The fork sync. Fast-forwards clean from upstream/main, then merges clean
# into main through an auto-merged pull request. One implementation for a
# local session and for .github/workflows/fork-sync.yml.
# Design: docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md §4
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sync-upstream.sh

Runs from main. Fast-forwards clean from upstream/main, merges clean into a
sync/YYYY-MM-DD branch, and merges that branch into main through a pull
request. Exits 0 when main is already in sync.
EOF
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

upstream_url="https://github.com/prabanta-dev/janas.git"

die() { echo "sync-upstream: $*" >&2; exit 1; }
note() { echo "sync-upstream: $*"; }

[[ -z "$(git status --porcelain)" ]] || die "the working tree is dirty; commit or stash first"

head_ref="$(git rev-parse --abbrev-ref HEAD)"
[[ "$head_ref" == "main" ]] || die "run this from main, not from $head_ref"

git remote get-url upstream >/dev/null 2>&1 || git remote add upstream "$upstream_url"

# A CI runner has no identity. The merge commit needs one.
git config user.name >/dev/null 2>&1 || git config user.name janas-sync
git config user.email >/dev/null 2>&1 || git config user.email janas-sync@users.noreply.github.com

git fetch --quiet origin
git fetch --quiet upstream
git rev-parse --verify --quiet origin/main >/dev/null || die "origin/main is missing"
git rev-parse --verify --quiet upstream/main >/dev/null || die "upstream/main is missing"
[[ "$(git rev-parse main)" == "$(git rev-parse origin/main)" ]] \
  || die "local main differs from origin/main; run git pull --ff-only first"

# clean: the mirror of upstream/main. Create it when this clone has none.
if git rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  git switch --quiet clean
  before="$(git rev-parse clean)"
elif git rev-parse --verify --quiet refs/remotes/origin/clean >/dev/null; then
  before="$(git rev-parse origin/clean)"
  git switch --quiet --create clean --track origin/clean
else
  before=""
  git switch --quiet --create clean upstream/main
  note "created clean from upstream/main"
fi

git merge --quiet --ff-only upstream/main
moved=true
[[ "$before" == "$(git rev-parse clean)" ]] && moved=false
git push --quiet origin clean
if [[ "$moved" == "false" ]]; then
  note "clean did not move; main is in sync with upstream"
  git switch --quiet main
  exit 0
fi
git switch --quiet main

today="$(date +%F)"
sync_branch="sync/$today"

if [[ "$(gh pr list --state open --head "$sync_branch" --json number --jq 'length')" != "0" ]]; then
  note "an open pull request for $sync_branch exists; nothing to do"
  exit 0
fi

git rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null \
  && git branch --delete --force "$sync_branch"
if git ls-remote --exit-code --heads origin "$sync_branch" >/dev/null 2>&1; then
  git push --quiet origin --delete "$sync_branch"
fi
git switch --quiet --create "$sync_branch" origin/main

if ! git merge --no-ff --no-commit clean; then
  conflicted="$(git diff --name-only --diff-filter=U)"
  if [[ "$conflicted" == "README.md" ]]; then
    note "README.md conflicts; keeping upstream's copy and the fork header"
    git checkout clean -- README.md
    cat .fork/readme-header.md README.md >README.md.new
    mv README.md.new README.md
    git add README.md
  else
    git merge --abort 2>/dev/null || true
    git switch --quiet main
    git branch --delete --force "$sync_branch"
    [[ -n "$conflicted" ]] || die "the merge failed; see the errors above"
    title="sync blocked: $conflicted conflicts with upstream"
    if [[ "$(gh issue list --state open --search "$title in:title" --json number --jq 'length')" == "0" ]]; then
      gh issue create --title "$title" \
        --body "The sync could not merge upstream/main into main. Conflicting paths: $conflicted. Resolve them by hand, then run .fork/sync-upstream.sh again."
    fi
    die "merge conflict in $conflicted; a fork issue has the details"
  fi
fi

git commit --quiet --message "sync: merge upstream/main into main ($today)"

git push --quiet --set-upstream origin "$sync_branch"
gh pr create --base main --head "$sync_branch" \
  --title "sync: merge upstream/main into main ($today)" \
  --body "Automated sync of upstream/main into main, by .fork/sync-upstream.sh."
gh pr merge "$sync_branch" --merge --delete-branch

git switch --quiet main
if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  git pull --quiet --ff-only
fi
note "sync complete"
```

Two decisions that the spec leaves open, and their reasons:

- `git merge --no-ff --no-commit clean`. A plain merge would fast-forward the sync branch, and §4 says the sync branch carries its own merge commit, under the PR's merge commit.
- The script pushes `clean` on every run, before the "did not move" check. That keeps `origin/clean` fresh when an earlier push failed, and a no-op push costs nothing.

- [ ] **Step 4: Run the test again and watch it pass**

```bash
.fork/tests/test-sync-upstream.sh
```

Expected: `ok:` lines for the five scenarios, then `test-sync-upstream: all checks passed`.

- [ ] **Step 5: Correct the spec's `clean` resolution**

In spec §4, replace resolution item 3 ("Otherwise the script fails. The message names the fix: run `sync-upstream.sh` once to create `clean`.") with:

```markdown
3. Otherwise the script creates `clean` from `upstream/main` and pushes it.
   This is the first run in a fresh clone.
```

The spec's own sync steps already say "Create `clean` from it when `clean` is missing", and §3 artifact 1 says the same. The resolution list contradicted them.

- [ ] **Step 6: Note the second permission setting in the spec**

In §4, after the `PUT /repos/.../actions/permissions/workflow` block, add:

```markdown
The same call sets `can_approve_pull_request_reviews` to `true`, because the
sync opens and merges a pull request with the workflow token:

```
PUT /repos/phaedrus1992/janas/actions/permissions/workflow
{"default_workflow_permissions": "write", "can_approve_pull_request_reviews": true}
```
```

- [ ] **Step 7: Lint and commit**

```bash
shellcheck .fork/sync-upstream.sh .fork/tests/test-sync-upstream.sh
shfmt -d .fork/sync-upstream.sh .fork/tests/test-sync-upstream.sh
git add .fork/sync-upstream.sh .fork/tests/test-sync-upstream.sh \
  docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md
git commit -s -m "feat(fork): add the upstream sync script"
```

---

### Task 6: The workflows, Dependabot, and the repository setting

**Files:**
- Create: `.github/workflows/fork-sync.yml`
- Create: `.github/workflows/fork-guard.yml`
- Create: `.github/dependabot.yml`
- Setting: the fork's workflow permissions

**Interfaces:**
- Consumes: `.fork/sync-upstream.sh` (Task 5), `.fork/check-clean.sh` (Task 3).
- Produces: a daily sync on the fork, and a guard check on every `port/*` pull request.

- [ ] **Step 1: Write the sync workflow**

Create `.github/workflows/fork-sync.yml`:

```yaml
# SPDX-License-Identifier: GPL-3.0-or-later
# The daily sync of upstream/main into the fork's main. The script
# fast-forwards clean, then merges it into main through an auto-merged pull
# request. A local session runs the same script: .fork/sync-upstream.sh
name: fork-sync

on:
  schedule:
    - cron: '17 6 * * *'
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

concurrency:
  group: fork-sync
  cancel-in-progress: false

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0
      - name: Merge upstream/main into main
        env:
          GH_TOKEN: ${{ github.token }}
        run: .fork/sync-upstream.sh
```

- [ ] **Step 2: Write the guard workflow**

Create `.github/workflows/fork-guard.yml`:

```yaml
# SPDX-License-Identifier: GPL-3.0-or-later
# The port-branch guard in CI: a port/* pull request must not carry fork
# tooling, and its README.md must not carry the fork header. The prek hook
# runs the same guard in a local clone: .fork/check-clean.sh
name: fork-guard

on:
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  guard:
    if: startsWith(github.head_ref, 'port/')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}
      - name: Check the port branch against clean
        run: .fork/check-clean.sh ${{ github.event.pull_request.head.sha }}
```

- [ ] **Step 3: Write the Dependabot config**

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: daily
    cooldown:
      default-days: 7
    groups:
      github-actions:
        patterns: ['*']
```

- [ ] **Step 4: Validate the workflow files**

```bash
actionlint
zizmor .github/workflows
```

Expected: no output from either. Fix every warning at once.

- [ ] **Step 5: Flip the repository setting**

```bash
gh api --method PUT repos/phaedrus1992/janas/actions/permissions/workflow \
  --input - <<'EOF'
{"default_workflow_permissions": "write", "can_approve_pull_request_reviews": true}
EOF
gh api repos/phaedrus1992/janas/actions/permissions/workflow
```

Expected: `{"default_workflow_permissions":"write","can_approve_pull_request_reviews":true}`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/fork-sync.yml .github/workflows/fork-guard.yml .github/dependabot.yml
git commit -s -m "feat(fork): add the sync and guard workflows"
```

---

### Task 7: The fork summary and the pull request

**Files:**
- Modify: `docs/fork.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the pull request to `main`, which ranger merges.

- [ ] **Step 1: Update the status section of `docs/fork.md`**

Replace the "## Status" body with the new state, in the file's own style:

```markdown
E0, the fork workflow, is built: `clean` mirrors `upstream/main`, the daily
sync merges upstream into `main` through a pull request, and a guard keeps
fork tooling off `port/*` branches. The tests for the scripts are in
`.fork/tests/`; run both by hand. Each clone installs the pre-push hook once
with `./.fork/install-hooks.sh`.

No port code exists yet. Each epic gets its own design session before work
starts.
```

- [ ] **Step 2: Re-read the diff**

```bash
git diff main... --stat
git diff main...
```

Check each file against the Global Constraints. Remove anything that does not belong.

- [ ] **Step 3: Run both fixture tests once more**

```bash
.fork/tests/test-check-clean.sh
.fork/tests/test-sync-upstream.sh
```

Expected: both print `all checks passed`.

- [ ] **Step 4: Commit and push**

```bash
git add docs/fork.md
git commit -s -m "docs(fork): record the E0 workflow"
git push --set-upstream origin feat/10-e0-fork-workflow
```

- [ ] **Step 5: Open the pull request**

```bash
gh pr create --base main --head feat/10-e0-fork-workflow \
  --title "E0: fork workflow and upstream agreement" \
  --body "$(cat <<'EOF'
Builds the E0 branch model and the upstream issue.

Closes #10
EOF
)"
```

- [ ] **Step 6: Wait for ranger to merge**

Do not merge it. Report the PR number and stop here.

---

### Task 8: Live checks after the merge

Do NOT start this task before the PR is merged. `workflow_dispatch` only sees a workflow that is on the default branch.

**Files:** none, unless a check fails.

**Interfaces:**
- Consumes: the merged workflows.
- Produces: proof that the sync and the guard work on the fork.

- [ ] **Step 1: Run one real sync**

```bash
git switch main
git pull --ff-only
gh workflow run fork-sync.yml --ref main
sleep 15   # the run needs a moment to appear; gh run watch needs its id
gh run watch "$(gh run list --workflow fork-sync.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Expected: the run finishes green.

- [ ] **Step 2: Check the sync landed**

```bash
gh run list --workflow fork-sync.yml --limit 1
git fetch origin
git log --oneline -3 origin/main
git rev-parse origin/clean upstream/main
```

Expected: the run list shows a merged sync pull request, `origin/main` carries the merge commit, and the two SHAs from the last command are equal.

- [ ] **Step 3: Check the guard on a real pull request**

```bash
git switch --quiet --create port/00-guard-test origin/clean
printf '# guard test\n' > AGENTS.md
git add AGENTS.md
git commit -s -m "chore: guard test (do not merge)"
git push --set-upstream origin port/00-guard-test
gh pr create --base main --head port/00-guard-test --title "guard test (do not merge)" --body "Throwaway. The fork-guard check must fail."
sleep 20   # the check needs a moment to register; gh pr checks needs one to watch
gh pr checks --watch || true
```

The tooling path must be one `main` does not have, such as `AGENTS.md`.
A path `main` already has, such as `.llmenv.yaml`, makes both sides of the pull request add it, so the pull request has a conflict, and GitHub does not run `pull_request` workflows for a pull request it cannot merge.
The run is then absent, not red.

The local pre-push hook blocks this push first, which is the point.
Add `--no-verify` to the push to reach the CI half of the test.

Expected: `fork-guard` fails and names `AGENTS.md`.

- [ ] **Step 4: Clean up the throwaway branch**

```bash
gh pr close --delete-branch port/00-guard-test
git switch main
git branch --delete --force port/00-guard-test
```

- [ ] **Step 5: Leave the follow-up issue open**

The fixture tests run by hand today. Issue #11 tracks running them in CI. It stays open after this plan; no action is needed here.

---

## Self-review

- **Spec coverage:** §3 artifacts 1-9 and 11 are Task 2 (1, 5), Task 3 (3, 6), Task 4 (4, 6), Task 5 (2), Task 6 (7, 8), and Task 1 (11); artifact 9 is the design.md amendment in Task 3; artifact 10 is issue #10. §4 sync mechanics are Task 5. §5 guard mechanics are Task 3 and Task 4. §6 is Task 1. §7 tests are Tasks 3, 4, 5, and 8. §8 order is the task order, with one change: the design.md amendment comes in Task 3, not before the sync, because the guard encodes the same list.
- **Placeholders:** none. Every step carries its content.
- **Type consistency:** `check-clean.sh [ref]`, the tooling-path list, `sync-upstream.sh` with no arguments, and `sync/YYYY-MM-DD` are used with the same names in every task.
- **Review Focus:** the five items each have a test in the named task.
