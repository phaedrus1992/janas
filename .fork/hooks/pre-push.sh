#!/usr/bin/env bash
# The port-branch guard hook. Git writes one line per pushed ref on stdin:
#   <local ref> <local oid> <remote ref> <remote oid>
# Every line whose destination is port/* runs the guard on the pushed commit.
# Install it once per clone with .fork/install-hooks.sh.
#
# The guard comes from origin/main, never from the working tree: the tree is
# the content under review, and a branch that carries a changed guard must not
# check itself.
set -euo pipefail

zero=0000000000000000000000000000000000000000

# Collect first: a push with no port/* destination needs no guard, and a clone
# whose main lacks one must still be able to push its other branches.
pushed=()
while read -r _local_ref local_sha remote_ref _; do
  [[ "$remote_ref" == refs/heads/port/* ]] || continue
  # A deleted ref has a zero local oid; there is nothing to check.
  [[ "$local_sha" == "$zero" ]] && continue
  pushed+=("$local_sha")
done
((${#pushed[@]} > 0)) || exit 0

scratch=""
trap 'if [[ -n "$scratch" ]]; then rm -f "$scratch"; fi' EXIT
scratch="$(mktemp "${TMPDIR:-/tmp}/janas-guard.XXXXXX")"

read_guard() {
  local source_ref
  for source_ref in origin/main main; do
    if git show "$source_ref:.fork/check-clean.sh" >"$scratch" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

if ! read_guard; then
  echo "pre-push: cannot read .fork/check-clean.sh from origin/main or main." >&2
  echo "pre-push: create main (git branch main origin/main), then push again." >&2
  exit 1
fi

status=0
for local_sha in "${pushed[@]}"; do
  bash "$scratch" "$local_sha" || status=1
done

exit "$status"
