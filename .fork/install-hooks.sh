#!/usr/bin/env bash
# Install the fork hooks into this clone. Run it once per clone after the
# first checkout of main. Design: e0-fork-workflow.md §5
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
source_hook="$root/.fork/hooks/pre-push.sh"

[[ -f "$source_hook" ]] || {
  echo "install-hooks: $source_hook is missing; run this from main." >&2
  exit 1
}

# Let git name the target: an absolute core.hooksPath and a linked worktree
# both move it away from .git/hooks.
hooks_path="$(git -C "$root" rev-parse --git-path hooks/pre-push)"
case "$hooks_path" in
/*) target="$hooks_path" ;;
*) target="$root/$hooks_path" ;;
esac

if [[ -f "$target" ]] && ! cmp -s "$source_hook" "$target"; then
  cp "$target" "$target.bak"
  echo "install-hooks: the old hook is saved at $target.bak"
fi
cp "$source_hook" "$target"
chmod +x "$target"
echo "install-hooks: installed $target"
