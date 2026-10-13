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

trap 'if [[ -n "$scratch" ]]; then rm -f "$scratch"; fi' EXIT

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
