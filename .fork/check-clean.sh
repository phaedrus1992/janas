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
  [[ "$path" == .github/workflows/fork-*.yml ]] && return 0
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
