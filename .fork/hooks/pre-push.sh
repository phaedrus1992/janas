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

scratch=""

trap 'if [[ -n "$scratch" ]]; then rm -f "$scratch"; fi' EXIT

# Writes the guard to the scratch file and names the ref it came from.
read_guard() {
  local source_ref
  for source_ref in origin/main main; do
    if git show "$source_ref:.fork/check-clean.sh" >"$scratch" 2>/dev/null; then
      printf '%s' "$source_ref"
      return 0
    fi
  done
  return 1
}

scratch="$(mktemp "${TMPDIR:-/tmp}/janas-guard.XXXXXX")"
if ! read_guard >/dev/null; then
  echo "pre-push: cannot read .fork/check-clean.sh from origin/main or main." >&2
  echo "pre-push: create main (git branch main origin/main), then push again." >&2
  exit 1
fi

zero=0000000000000000000000000000000000000000
status=0
while read -r _local_ref local_sha remote_ref _; do
  [[ "$remote_ref" == refs/heads/port/* ]] || continue
  # A deleted ref has a zero local oid; there is nothing to check.
  [[ "$local_sha" == "$zero" ]] && continue
  bash "$scratch" "$local_sha" || status=1
done

exit "$status"
