#!/usr/bin/env bash
# Fixture test for .fork/sync-upstream.sh. Two bare repositories stand in for
# upstream and origin, and a stub gh stands in for GitHub. The stub writes
# every call to a log, so the test can assert the PR flow.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
script="$root/.fork/sync-upstream.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sync.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# A fixed identity and no signing: the fixtures must not need a key.
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=user.name GIT_CONFIG_VALUE_1=fixture
export GIT_CONFIG_KEY_2=user.email GIT_CONFIG_VALUE_2=fixture@example.invalid
# A hermetic fixture: the operator's git config must not leak in. A global
# fetch.prune, for example, prunes the tracking refs the assertions read
# once the stub merge deletes the remote branch.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null

fails=0
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}
ok() { echo "ok: $*"; }

upstream="$scratch/upstream.git"
origin="$scratch/origin.git"
git init --quiet --bare --initial-branch=main "$upstream"
git init --quiet --bare --initial-branch=main "$origin"

seed="$scratch/seed"
git clone --quiet "$upstream" "$seed"
cd "$seed"
printf '# Upstream\n\none\n' >README.md
mkdir -p src
printf 'int one(void) { return 1; }\n' >src/engine.c
git add -A
git commit --quiet -m "upstream: first"
git push --quiet origin main

# The sync script validates the upstream remote against the real URL, so the
# fixtures keep that URL and rewrite it to the local bare repository.
canonical="https://github.com/prabanta-dev/janas.git"

fork="$scratch/fork"
git clone --quiet "$origin" "$fork"
cd "$fork"
git config "url.$upstream.insteadOf" "$canonical"
git remote add upstream "$canonical"
git fetch --quiet upstream
git switch --quiet --create main upstream/main
{
  cat "$root/.fork/readme-header.md"
  cat README.md
} >README.new
mv README.new README.md
git add README.md
git commit --quiet -m "fork: header"
git push --quiet --set-upstream origin main

# The stub gh. It records every call. `pr list` answers with GH_STUB_PRS_JSON;
# `pr merge` deletes the head branch that `pr create` recorded, the way
# GitHub does, unless GH_STUB_KEEP_BRANCH=1 models the not-yet-settled delete.
mkdir -p "$scratch/bin"
cat >"$scratch/bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$GH_STUB_LOG"
case "$1 ${2:-}" in
  "pr list")
    if [[ "${GH_STUB_FAIL_PR_LIST:-0}" == "1" ]]; then
      echo "gh-stub: pr list failed" >&2
      exit 1
    fi
    printf '%s\n' "${GH_STUB_PRS_JSON:-[]}"
    ;;
  "pr create")
    head=""
    prev=""
    for arg in "$@"; do
      if [[ "$prev" == "--head" ]]; then head="$arg"; fi
      prev="$arg"
    done
    printf '%s' "$head" >"$GH_STUB_HEAD"
    echo "https://example.invalid/pull/1"
    ;;
  "pr merge")
    if [[ "${GH_STUB_FAIL_PR_MERGE:-0}" == "1" ]]; then
      echo "gh-stub: pr merge failed" >&2
      exit 1
    fi
    if [[ "${GH_STUB_KEEP_BRANCH:-0}" != "1" ]]; then
      head="$(cat "$GH_STUB_HEAD")" || head=""
      if [[ -n "$head" ]]; then
        git --git-dir="$GH_STUB_ORIGIN" update-ref -d "refs/heads/$head"
      fi
    fi
    echo "merged"
    ;;
  "issue list")
    if [[ "${GH_STUB_FAIL_ISSUE_LIST:-0}" == "1" ]]; then
      echo "gh-stub: issue list failed" >&2
      exit 1
    fi
    echo "0"
    ;;
  "issue create") echo "https://example.invalid/issues/1" ;;
  *)
    echo "gh-stub: unhandled: $*" >&2
    exit 1
    ;;
esac
STUB
chmod +x "$scratch/bin/gh"

# The stub date: the sync names its branch from the clock, so the test fixes
# the clock and knows each run's branch name before the run. An unset value
# fails loudly under set -u.
cat >"$scratch/bin/date" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
+%F) echo "$STUB_DATE" ;;
+%F-%H%M%S) echo "$STUB_DATE-$STUB_TIME" ;;
*) exec /bin/date "$@" ;;
esac
STUB
chmod +x "$scratch/bin/date"

export PATH="$scratch/bin:$PATH"
export GH_STUB_LOG="$scratch/gh.log"
export GH_STUB_HEAD="$scratch/gh-head"
export GH_STUB_ORIGIN="$origin"
: >"$GH_STUB_LOG"

export STUB_DATE="2026-10-13"
sync_branch=""

# Step the clock and set the branch name the next run will use. Each scenario
# gets its own second, so every run cuts a fresh branch, as in production.
at_time() {
  export STUB_TIME="$1"
  sync_branch="sync/$STUB_DATE-$1"
}

run_sync() { cd "$1" && "$script"; }

# Scenario 1: the first sync. Upstream touches README.md, so the merge
# conflicts and the script must restore the header.
cd "$seed"
printf 'two\n' >>README.md
git commit --quiet -am "upstream: readme two"
git push --quiet origin main

at_time 120001
if run_sync "$fork"; then ok "sync 1 exits 0"; else bad "sync 1 must exit 0"; fi

clean_sha="$(git --git-dir="$upstream" rev-parse main)"
fork_clean="$(git -C "$fork" rev-parse origin/clean)"
if [[ "$clean_sha" == "$fork_clean" ]]; then ok "clean mirrors upstream"; else bad "clean must mirror upstream"; fi

readme="$(git -C "$fork" show "origin/$sync_branch:README.md")"
if [[ "$readme" == '> **This is a fork.**'* ]]; then ok "the header is back"; else bad "the header must be back"; fi
header_count="$(printf '%s' "$readme" | grep -cF '> **This is a fork.**')"
if [[ "$header_count" -eq 1 ]]; then ok "the header appears once"; else bad "the header must appear once, not $header_count times"; fi
if [[ "$readme" == *"two"* ]]; then ok "upstream's README text is kept"; else bad "upstream's text must be kept"; fi

parents="$(git -C "$fork" rev-list --parents --no-walk "origin/$sync_branch" | wc -w)"
if [[ "$parents" -eq 3 ]]; then ok "the sync branch carries a merge commit"; else bad "the sync branch needs a merge commit"; fi

if grep -q -- "pr create" "$GH_STUB_LOG" 2>/dev/null; then ok "the PR was opened"; else bad "gh pr create must run"; fi
if grep -q -- "pr merge" "$GH_STUB_LOG"; then ok "the PR was merged"; else bad "gh pr merge must run"; fi

# Scenario 2: a second sync on the same day. The run must cut a fresh branch
# rather than delete and re-create one: a re-created branch cannot open its
# pull request for the first seconds after the delete (#31). The stub keeps
# the branch, so the name it used is checked on origin.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'int two(void) { return 2; }\n' >src/engine.c
git commit --quiet -am "upstream: engine two"
git push --quiet origin main
at_time 120002
export GH_STUB_KEEP_BRANCH=1
if run_sync "$fork"; then ok "sync 2 exits 0"; else bad "sync 2 must exit 0"; fi
unset GH_STUB_KEEP_BRANCH
if git --git-dir="$origin" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  ok "the fresh branch is on origin"
else
  bad "the fresh branch must be on origin"
fi
if git -C "$fork" show "origin/$sync_branch:src/engine.c" | grep -q "two"; then
  ok "upstream's engine change arrived"
else
  bad "the engine change must arrive"
fi

# Scenario 3: an open sync pull request. The run must stop and name it; a
# silent exit 0 would hide that the sync is stalled. The first of two
# same-repo sync pull requests is the one named.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'three\n' >>README.md
git commit --quiet -am "upstream: readme three"
git push --quiet origin main
cd "$fork"
at_time 120003
export GH_STUB_PRS_JSON='[{"url":"https://example.invalid/pull/9","headRefName":"sync/2026-10-13-000000","isCrossRepository":false},{"url":"https://example.invalid/pull/10","headRefName":"sync/2026-10-13-000001","isCrossRepository":false},{"url":"https://example.invalid/pull/11","headRefName":"feat/other","isCrossRepository":false}]'
if run_sync "$fork" 2>"$scratch/openpr.err"; then
  bad "sync 3 must stop on an open sync pull request"
else
  ok "sync 3 stops"
fi
unset GH_STUB_PRS_JSON
if grep -q "https://example.invalid/pull/9" "$scratch/openpr.err"; then
  ok "the failure names the pull request"
else
  bad "the failure must name the pull request"
fi
if grep -q "Merge it, or close it" "$scratch/openpr.err"; then
  ok "the failure names the recovery"
else
  bad "the failure must name the recovery"
fi
if git --git-dir="$origin" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  bad "nothing must be pushed"
else
  ok "nothing was pushed"
fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then bad "no PR must be opened"; else ok "no PR was opened"; fi

# Scenario 3b: pull requests that must not stop the run: one from another
# fork, and heads that only look like sync/* names. The same table runs again
# in the cleanup's re-check within the same run.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'three-b\n' >>README.md
git commit --quiet -am "upstream: readme three-b"
git push --quiet origin main
cd "$fork"
at_time 120004
export GH_STUB_PRS_JSON='[{"url":"https://example.invalid/pull/12","headRefName":"sync/2026-10-13-000002","isCrossRepository":true},{"url":"https://example.invalid/pull/13","headRefName":"syncx/1","isCrossRepository":false},{"url":"https://example.invalid/pull/14","headRefName":"sync","isCrossRepository":false},{"url":"https://example.invalid/pull/15","headRefName":"feat/other","isCrossRepository":false}]'
if run_sync "$fork"; then ok "sync 3b exits 0"; else bad "sync 3b must exit 0"; fi
unset GH_STUB_PRS_JSON
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then ok "sync 3b opens its pull request"; else bad "sync 3b must open its pull request"; fi

# Scenario 4: a conflict on a file other than README.md. The script must
# abort, delete the branch, open one issue, and exit 1.
rm -f "$GH_STUB_LOG"
cd "$fork"
git switch --quiet main
printf 'fork edit\n' >src/engine.c
git commit --quiet -am "fork: engine edit"
git push --quiet origin main
cd "$seed"
printf 'upstream edit\n' >src/engine.c
git commit --quiet -am "upstream: engine edit"
git push --quiet origin main
cd "$fork"
at_time 120005
if run_sync "$fork" 2>"$scratch/conflict.err"; then
  bad "sync 4 must fail"
else
  ok "sync 4 fails"
fi
if grep -q "src/engine.c" "$scratch/conflict.err"; then ok "the failure names the path"; else bad "the failure must name the path"; fi
if grep -q "issue create" "$GH_STUB_LOG"; then ok "an issue was opened"; else bad "an issue must be opened"; fi
if [[ "$(git -C "$fork" rev-parse --abbrev-ref HEAD)" == "main" ]]; then
  ok "the script returned to main"
else
  bad "the script must return to main"
fi
if git -C "$fork" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  bad "the local sync branch must be deleted"
else
  ok "the local sync branch is gone"
fi

# Scenario 5: a clone with no clean branch. The script must create it and
# push it, not fail. The engine conflict from scenario 4 goes away first.
cd "$fork"
git fetch --quiet upstream
git checkout --quiet upstream/main -- src/engine.c
git commit --quiet -am "fork: take upstream's engine"
git push --quiet origin main

rm -f "$GH_STUB_LOG"
git --git-dir="$origin" update-ref -d refs/heads/clean
fresh="$scratch/fresh"
git clone --quiet "$origin" "$fresh"
cd "$fresh"
git config "url.$upstream.insteadOf" "$canonical"
git remote add upstream "$canonical"
git fetch --quiet upstream
git switch --quiet main
at_time 120006
if run_sync "$fresh"; then ok "sync 5 exits 0"; else bad "sync 5 must exit 0"; fi
if git --git-dir="$origin" rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  ok "clean was created and pushed"
else
  bad "clean must be created and pushed"
fi

# Scenario 6: clean sits at upstream/main but main is behind, because an
# earlier run's pull request never merged. The script must try again, not
# report that main is in sync.
rm -f "$GH_STUB_LOG"
cd "$fresh"
at_time 120007
if run_sync "$fresh"; then ok "sync 6 exits 0"; else bad "sync 6 must exit 0"; fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then ok "sync 6 opens a pull request"; else bad "sync 6 must open a pull request"; fi

# Scenario 7: main already contains everything clean has, as it would after a
# CI sync merged. The script must stop before it cuts a branch.
cd "$fork"
git switch --quiet main
git fetch --quiet origin
git fetch --quiet upstream
git merge --no-ff --no-commit clean >/dev/null 2>&1 || true
git checkout clean -- README.md
cat "$root/.fork/readme-header.md" README.md >README.new
mv README.new README.md
git add README.md
git commit --quiet -m "fixture: main contains clean"
git push --quiet origin main
rm -f "$GH_STUB_LOG"
at_time 120008
if run_sync "$fork"; then ok "sync 7 exits 0"; else bad "sync 7 must exit 0"; fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then bad "sync 7 must not open a pull request"; else ok "sync 7 opens nothing"; fi
if git --git-dir="$origin" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  bad "sync 7 must not cut a branch"
else
  ok "sync 7 cuts no branch"
fi

# Scenario 8: gh cannot list pull requests. That is an error, not "a pull
# request is open".
cd "$seed"
printf 'four\n' >>README.md
git commit --quiet -am "upstream: readme four"
git push --quiet origin main
cd "$fresh"
at_time 120009
export GH_STUB_FAIL_PR_LIST=1
if run_sync "$fresh" >/dev/null 2>&1; then
  bad "sync 8 must fail when gh pr list fails"
else
  ok "sync 8 fails"
fi
unset GH_STUB_FAIL_PR_LIST

# Scenario 9: a conflict while gh cannot list issues. The run must still fail
# and must not claim that an issue was filed; it names the conflict instead.
cd "$fork"
git switch --quiet main
printf 'fork edit two\n' >src/engine.c
git commit --quiet -am "fork: engine edit two"
git push --quiet origin main
cd "$seed"
printf 'upstream edit two\n' >src/engine.c
git commit --quiet -am "upstream: engine edit two"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
at_time 120010
export GH_STUB_FAIL_ISSUE_LIST=1
if run_sync "$fork" >"$scratch/conflict2.out" 2>"$scratch/conflict2.err"; then
  bad "sync 9 must fail"
else
  ok "sync 9 fails"
fi
unset GH_STUB_FAIL_ISSUE_LIST
if grep -q "merge conflict in src/engine.c" "$scratch/conflict2.err"; then
  ok "the failure names the conflict"
else
  bad "the failure must name the conflict"
fi
if grep -q "cannot list the issues" "$scratch/conflict2.out"; then
  ok "the failed issue search is reported"
else
  bad "a failed issue search must be reported"
fi
if grep -q "issue create" "$GH_STUB_LOG"; then
  ok "the issue is still created after a failed search"
else
  bad "the issue must be created after a failed search"
fi
if [[ "$(git -C "$fork" rev-parse --abbrev-ref HEAD)" == "main" ]]; then
  ok "sync 9 returned to main"
else
  bad "sync 9 must return to main"
fi

# Scenario 10: the merge of the pull request fails. The run must fail and name
# the recovery, because the branch and the pull request stay behind.
cd "$fork"
git fetch --quiet upstream
git checkout --quiet upstream/main -- src/engine.c
git commit --quiet -am "fork: take upstream's engine again"
git push --quiet origin main
cd "$seed"
printf 'five\n' >>README.md
git commit --quiet -am "upstream: readme five"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
at_time 120011
export GH_STUB_FAIL_PR_MERGE=1
if run_sync "$fork" 2>"$scratch/merge.err"; then
  bad "sync 10 must fail when the merge fails"
else
  ok "sync 10 fails"
fi
unset GH_STUB_FAIL_PR_MERGE
if grep -q "cannot merge" "$scratch/merge.err"; then
  ok "the failure names the recovery"
else
  bad "the failure must name the recovery"
fi

# Scenario 11: branches left behind by earlier runs that never merged. The
# merge deletes this run's branch (the stub models that), and the sweep
# removes every other sync branch, after the merge, never before the create.
cd "$fork"
git switch --quiet main
git --git-dir="$origin" update-ref refs/heads/sync/2020-01-01-000000 "$(git --git-dir="$origin" rev-parse main)"
cd "$seed"
printf 'six\n' >>README.md
git commit --quiet -am "upstream: readme six"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
at_time 120012
if run_sync "$fork"; then ok "sync 11 exits 0"; else bad "sync 11 must exit 0"; fi
remaining="$(git --git-dir="$origin" for-each-ref --format='%(refname:short)' 'refs/heads/sync/*')"
if [[ -z "$remaining" ]]; then
  ok "no sync branches are left"
else
  bad "the stale sync branches must be deleted, remaining: $remaining"
fi

# Scenario 12: the not-yet-settled delete window. When the merge leaves the
# run's own branch in place, the sweep must keep it and delete only the
# others.
cd "$fork"
git switch --quiet main
git --git-dir="$origin" update-ref refs/heads/sync/2020-01-02-000000 "$(git --git-dir="$origin" rev-parse main)"
cd "$seed"
printf 'seven\n' >>README.md
git commit --quiet -am "upstream: readme seven"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
at_time 120013
export GH_STUB_KEEP_BRANCH=1
if run_sync "$fork"; then ok "sync 12 exits 0"; else bad "sync 12 must exit 0"; fi
unset GH_STUB_KEEP_BRANCH
remaining="$(git --git-dir="$origin" for-each-ref --format='%(refname:short)' 'refs/heads/sync/*')"
if [[ "$remaining" == "$sync_branch" ]]; then
  ok "only this run's sync branch remains"
else
  bad "the sweep must keep this run's branch and delete the rest, remaining: $remaining"
fi

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-sync-upstream: all checks passed"
