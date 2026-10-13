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
