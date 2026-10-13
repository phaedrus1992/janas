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
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}

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

expect_pass() {
  if "$guard" "$1" >/dev/null 2>&1; then return 0; fi
  bad "$1 must pass"
}
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

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-check-clean: all checks passed"
