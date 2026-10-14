#!/usr/bin/env bash
# Fixture test for .fork/check-clean.sh and .fork/hooks/pre-push.sh.
# The test runs in this clone. It touches no file of the working tree: every
# branch is built with git plumbing on top of clean, and the trap removes them.
set -euo pipefail

# A fixed identity and no signing: the fixtures must not need a key.
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=user.name GIT_CONFIG_VALUE_1=fixture
export GIT_CONFIG_KEY_2=user.email GIT_CONFIG_VALUE_2=fixture@example.invalid
# A hermetic fixture: the operator's git config must not leak in. A CI runner
# has no identity, and git cannot auto-detect one there.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null

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
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}

# make_ref <name> <base> <path> <content>...  Builds a commit on top of the
# base branch. A content of "-" removes the path.
make_ref() {
  local name="$1" base="$2" index="$scratch/index.$1"
  shift 2
  GIT_INDEX_FILE="$index" git read-tree "$base"
  add_to_index "$index" "$@"
  local tree commit
  tree="$(GIT_INDEX_FILE="$index" git write-tree)"
  commit="$(git commit-tree "$tree" -p "$base" -m "fixture: $name")"
  git branch "$name" "$commit"
  fixtures+=("$name")
}

# make_orphan_ref <name> <path> <content>...  A commit with no parent, so it
# shares no history with clean.
make_orphan_ref() {
  local name="$1" index="$scratch/index.$1"
  shift
  GIT_INDEX_FILE="$index" git read-tree --empty
  add_to_index "$index" "$@"
  local tree commit
  tree="$(GIT_INDEX_FILE="$index" git write-tree)"
  commit="$(git commit-tree "$tree" -m "fixture: $name")"
  git branch "$name" "$commit"
  fixtures+=("$name")
}

add_to_index() {
  local index="$1"
  shift
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
}

make_ref fix-engine clean src/llm/quant.c '/* fixture */'
make_ref fix-tooling clean .llmenv.yaml 'id: fixture'
make_ref fix-header clean README.md '> **This is a fork.**'
make_ref fix-deleted clean README.md '-'
make_ref fix-case clean .LLMENV.yaml 'id: fixture'
make_ref fix-forkfile clean .fork 'not a directory'
make_ref fix-prefix clean docs/superpowers/plans/note.md 'note'
make_ref fix-glob clean .github/workflows/fork-x.yml 'name: x'
make_ref fix-glob-yaml clean .github/workflows/fork-y.yaml 'name: y'
make_ref fix-neighbour clean .github/workflows/release.yml 'name: release'
make_ref fix-nonascii clean .fork/café.sh 'echo hi'
make_orphan_ref fix-orphan README.md 'no history' .llmenv.yaml 'id: fixture'

expect_pass() {
  if "$guard" "$1" >/dev/null 2>&1; then return 0; fi
  bad "$1 must pass"
}
expect_fail() { if "$guard" "$1" >/dev/null 2>&1; then bad "$1 must fail"; fi; }

expect_pass fix-engine
expect_fail fix-tooling
expect_fail fix-header
expect_fail fix-deleted
expect_fail fix-case
expect_fail fix-forkfile
expect_fail fix-prefix
expect_fail fix-glob
expect_fail fix-glob-yaml
expect_pass fix-neighbour
expect_fail fix-nonascii
expect_fail fix-orphan

# The guard's marker and the header block it prepends must stay one text.
header_marker="$(head -1 "$root/.fork/readme-header.md")"
if ! grep -qF "$header_marker" "$guard"; then
  bad "the guard must look for the header's first line"
fi

# A clone with no clean anywhere must fail with the fix in the message.
clone="$scratch/noclean"
git clone --quiet --shared "$root" "$clone"
git -C "$clone" update-ref -d refs/remotes/origin/clean
if (cd "$clone" && "$guard" HEAD) 2>"$scratch/noclean.err"; then
  bad "a clone without clean must fail"
elif ! grep -q "clean" "$scratch/noclean.err"; then
  bad "the failure message must name the fix"
fi

# The hook. It reads its guard from origin/main, so the cases run in a scratch
# checkout that has one. That checkout models a port/* tree: no tooling files.
bare="$scratch/tiny.git"
work="$scratch/tiny-work"
git init --quiet --bare "$bare"
git push --quiet "$bare" "HEAD:refs/heads/main" clean:refs/heads/clean
git init --quiet "$work"
git -C "$work" remote add origin "$bare"
git -C "$work" fetch --quiet origin
git -C "$work" branch --no-track main origin/main
git -C "$work" switch --quiet --detach origin/clean

# A commit that touches .llmenv.yaml, built in the scratch repository.
tiny_index="$scratch/index.tiny"
GIT_INDEX_FILE="$tiny_index" git -C "$work" read-tree origin/clean
blob="$(printf 'id: tiny' | git -C "$work" hash-object --stdin -w)"
GIT_INDEX_FILE="$tiny_index" git -C "$work" update-index --add --cacheinfo "100644,$blob,.llmenv.yaml"
tree="$(GIT_INDEX_FILE="$tiny_index" git -C "$work" write-tree)"
tiny_commit="$(git -C "$work" commit-tree "$tree" -p origin/clean -m "fixture: tiny tooling ref")"
git -C "$work" branch --force port/x "$tiny_commit"
cp "$hook" "$work/.git/hooks/pre-push"
chmod +x "$work/.git/hooks/pre-push"

zero=0000000000000000000000000000000000000000
run_hook() { (cd "$work" && .git/hooks/pre-push); }

if printf 'refs/heads/port/x %s refs/heads/port/x %s\n' "$tiny_commit" "$zero" | run_hook >/dev/null 2>&1; then
  bad "a port/* push of a tooling ref must fail"
fi
if ! printf 'refs/heads/feat/x %s refs/heads/feat/x %s\n' "$tiny_commit" "$zero" | run_hook >/dev/null 2>&1; then
  bad "a feat/* push must pass"
fi
# The destination decides. A push that creates port/x from another local ref
# must be checked.
if printf 'refs/heads/scratch %s refs/heads/port/from-scratch %s\n' "$tiny_commit" "$zero" | run_hook >/dev/null 2>&1; then
  bad "a push that creates a port/* ref from another local ref must fail"
fi
if printf '%s\n%s\n' "refs/heads/feat/x $tiny_commit refs/heads/feat/x $zero" \
  "refs/heads/port/x $tiny_commit refs/heads/port/x $zero" | run_hook >/dev/null 2>&1; then
  bad "a multi-ref push with a port/* line must fail"
fi
if ! printf '(delete) %s refs/heads/port/gone %s\n' "$zero" "$tiny_commit" | run_hook >/dev/null 2>&1; then
  bad "a deleted port/* ref must pass"
fi

# The tree is the content under review: a guard file there must not be used.
mkdir -p "$work/.fork"
printf 'exit 0\n' >"$work/.fork/check-clean.sh"
if printf 'refs/heads/port/x %s refs/heads/port/x %s\n' "$tiny_commit" "$zero" | run_hook >/dev/null 2>&1; then
  bad "a guard file in the working tree must not be used"
fi
rm -rf "$work/.fork"

# With no main and no origin/main, the hook must fail and name the fix.
git -C "$work" branch --move main main-hidden
git -C "$work" update-ref -d refs/remotes/origin/main
if printf 'refs/heads/port/x %s refs/heads/port/x %s\n' "$tiny_commit" "$zero" | run_hook 2>"$scratch/nomain.err"; then
  bad "a port/* push with no main must fail"
elif ! grep -q "main" "$scratch/nomain.err"; then
  bad "the failure must name main"
fi

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-check-clean: all checks passed"
