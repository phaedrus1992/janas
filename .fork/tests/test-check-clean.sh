#!/usr/bin/env bash
# Fixture test for .fork/check-clean.sh and .fork/hooks/pre-push.sh.
# The test runs in this clone. It touches no file of the working tree: every
# branch is built with git plumbing on top of clean, and the trap removes them.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
guard="$root/.fork/check-clean.sh"
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

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-check-clean: all checks passed"
