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

die() {
  echo "sync-upstream: $*" >&2
  exit 1
}
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
[[ "$(git rev-parse main)" == "$(git rev-parse origin/main)" ]] ||
  die "local main differs from origin/main; run git pull --ff-only first"

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
if [[ "$before" == "$(git rev-parse clean)" ]]; then
  moved=false
fi
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

if git rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  git branch --delete --force "$sync_branch"
fi
if git ls-remote --exit-code --heads origin "$sync_branch" >/dev/null 2>&1; then
  git push --quiet origin --delete "$sync_branch"
fi
git switch --quiet --create "$sync_branch" origin/main

# A plain merge would fast-forward. --no-ff keeps the sync branch's own merge
# commit, under the pull request's merge commit (design §4).
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
