#!/usr/bin/env bash
# Fixture test for .fork/sweep-syntax.sh. A stub compiler stands in for cc, so
# the test is fast and needs no toolchain. It writes to a scratch directory and
# to one directory under the clone's .tmp/, and the trap removes both.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
sweep="$root/.fork/sweep-syntax.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sweep-test.XXXXXX")"
mkdir -p "$root/.tmp"
inside="$(mktemp -d "$root/.tmp/janas-sweep-test.XXXXXX")"
trap 'rm -rf "$scratch" "$inside"' EXIT

fails=0
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}

# The stub records one line for each source: the source, a tab, then all the
# arguments. It answers the sweep's probe (source "-") without a record.
#   STUB_EXIT        the status for every source (default 0)
#   STUB_FAIL_ODD=1  fail only the sources whose path has an odd length
#   STUB_PROBE=fail  fail the probe
# A source that is not a file gets status 3, as a real compiler would fail.
# For src/server/http.c it also records whether the generated header it
# includes is on the include path.
stub="$scratch/cc"
cat >"$stub" <<'EOF'
#!/usr/bin/env bash
src="${!#}"
if [[ "$src" == - ]]; then
  [[ "${STUB_PROBE:-}" != fail ]]
  exit
fi
printf '%s\t%s\n' "$src" "$*" >>"$STUB_RECORD"
[[ -f "$src" ]] || exit 3
if [[ "$src" == src/server/http.c ]]; then
  for arg in "$@"; do
    if [[ "$arg" == -I* && -f "${arg#-I}/server_test_html.h" ]]; then
      grep -q janas_server_test_html "${arg#-I}/server_test_html.h" &&
        echo "generated header found" >>"$STUB_RECORD.gen"
    fi
  done
fi
if [[ "${STUB_FAIL_ODD:-}" == 1 ]]; then
  exit $((${#src} % 2))
fi
exit "${STUB_EXIT:-0}"
EOF
chmod +x "$stub"
export STUB_RECORD="$scratch/record"
private_tmp="$scratch/tmp"
mkdir "$private_tmp"

run_sweep() { # exit status for the stub, then the sweep arguments
  local status="$1"
  shift
  : >"$STUB_RECORD"
  TMPDIR="$private_tmp" STUB_EXIT="$status" CC="$stub" bash "$sweep" "$@"
}

# Usage errors.
if ! bash "$sweep" --help 2>"$scratch/help"; then
  bad "--help must exit 0"
fi
grep -q '^usage: sweep-syntax.sh' "$scratch/help" || bad "--help must print the usage"
set +e
bash "$sweep" 2>/dev/null
rc=$?
bash "$sweep" "$scratch/missing/list.txt" 2>"$scratch/err"
rc_dir=$?
set -e
[[ $rc -eq 2 ]] || bad "no argument must exit 2, got $rc"
[[ $rc_dir -eq 2 ]] || bad "a missing output directory must exit 2, got $rc_dir"
grep -q "$scratch/missing" "$scratch/err" || bad "the error must name the missing directory"

# Every file passes.
summary="$(run_sweep 0 "$scratch/pass.txt")"
total="$(wc -l <"$STUB_RECORD" | tr -d ' ')"
[[ "$summary" == "0 of $total files fail" ]] ||
  bad "all pass: expected '0 of $total files fail', got '$summary'"
[[ -f "$scratch/pass.txt" && ! -s "$scratch/pass.txt" ]] ||
  bad "all pass: the failing list must exist and be empty"
[[ -f "$scratch/pass.txt.log" ]] || bad "the sweep must write <failing-list>.log"
cp "$STUB_RECORD" "$scratch/record.pass"

# Each source is compiled once, and no glob reaches the compiler unexpanded.
dupes="$(cut -f1 "$scratch/record.pass" | sort | uniq -d)"
[[ -z "$dupes" ]] || bad "these sources were compiled more than once: $dupes"
sources="$(cut -f1 "$scratch/record.pass")"
[[ "$sources" != *'*'* ]] || bad "a glob that matched nothing reached the compiler"
for group in src/third_party/libmicrohttpd/ src/third_party/mbedtls/library/ \
  src/third_party/yyjson/ src/llm/ src/server/ src/services/ tests/ tools/; do
  grep -q "^$group" "$scratch/record.pass" || bad "no source under $group was compiled"
done
expected_tp="$(cd "$root" && find src/third_party/libmicrohttpd src/third_party/yyjson \
  -name '*.c' | wc -l | tr -d ' ')"
got_tp="$(grep -cE '^src/third_party/(libmicrohttpd|yyjson)/' "$scratch/record.pass" || true)"
[[ $got_tp -eq $expected_tp ]] ||
  bad "libmicrohttpd and yyjson: expected $expected_tp sources, got $got_tp"

# The flag split, for every record: -w for third-party sources, -Werror for
# Janas's own.
split_errors="$(awk -F'\t' '{
  tp = ($1 ~ /^src\/third_party\//)
  w = (" " $2 " " ~ / -w /); werror = (" " $2 " " ~ / -Werror /)
  if (tp && (!w || werror)) print $1 ": third-party needs -w and no -Werror"
  if (!tp && (w || !werror)) print $1 ": a Janas source needs -Werror and no -w"
}' "$scratch/record.pass")"
[[ -z "$split_errors" ]] || bad "flag split: $split_errors"
mcp_flags="$(awk -F'\t' '$1 == "src/common/mcp_tls.c" { print $2 }' "$scratch/record.pass")"
[[ " $mcp_flags " == *" -isystemsrc/third_party/mbedtls/include "* ]] ||
  bad "src/common/mcp_tls.c must see the mbedtls headers as system headers"
grep -q 'generated header found' "$STUB_RECORD.gen" ||
  bad "src/server/http.c must see a server_test_html.h that declares janas_server_test_html"

# Every file fails, and a second run replaces the list instead of adding to it.
summary="$(run_sweep 1 "$scratch/list.txt")"
[[ "$summary" == "$total of $total files fail" ]] ||
  bad "all fail: expected '$total of $total files fail', got '$summary'"
listed="$(wc -l <"$scratch/list.txt" | tr -d ' ')"
[[ $listed -eq $total ]] || bad "all fail: expected $total listed files, got $listed"
run_sweep 0 "$scratch/list.txt" >/dev/null
[[ ! -s "$scratch/list.txt" ]] || bad "a second run must replace the failing list"

# Some files fail: the summary, the list, and the compiler agree on which.
: >"$STUB_RECORD"
summary="$(TMPDIR="$private_tmp" STUB_FAIL_ODD=1 CC="$stub" bash "$sweep" "$scratch/odd.txt")"
cut -f1 "$STUB_RECORD" | awk 'length($0) % 2 == 1' | sort >"$scratch/odd.expected"
odd="$(wc -l <"$scratch/odd.expected" | tr -d ' ')"
[[ $odd -gt 0 && $odd -lt $total ]] || bad "the odd-length split must fail some files, not $odd"
[[ "$summary" == "$odd of $total files fail" ]] ||
  bad "some fail: expected '$odd of $total files fail', got '$summary'"
sort "$scratch/odd.txt" | cmp -s - "$scratch/odd.expected" ||
  bad "some fail: the failing list is not the set of files that failed"
grep -q '^=== src/llm/api.c$' "$scratch/odd.txt.log" ||
  bad "the log must mark each file with a === line"

# A broken compiler stops the sweep instead of counting every file.
broken_case() { # what, expected message, then the environment
  local what="$1" message="$2" rc
  shift 2
  : >"$STUB_RECORD"
  set +e
  env TMPDIR="$private_tmp" "$@" bash "$sweep" "$scratch/broken.txt" >"$scratch/broken.out" \
    2>"$scratch/broken.err"
  rc=$?
  set -e
  [[ $rc -eq 1 ]] || bad "$what: expected exit 1, got $rc"
  grep -q "$message" "$scratch/broken.err" || bad "$what: the error must say '$message'"
  [[ ! -e "$scratch/broken.txt" ]] || bad "$what: a stopped sweep must leave no failing list"
  [[ ! -s "$scratch/broken.out" ]] || bad "$what: a stopped sweep must print no summary"
}
broken_case "a missing compiler" "cannot compile an empty file" CC="$scratch/no-such-cc"
broken_case "a failing probe" "cannot compile an empty file" CC="$stub" STUB_PROBE=fail
broken_case "a compiler that dies" "stopped with status 127" CC="$stub" STUB_EXIT=127

# A tree that is not the one build.sh expects stops the sweep.
fake="$scratch/fake"
mkdir -p "$fake/src/third_party/empty" "$fake/src/tool"
cat >"$fake/build.sh" <<'EOF'
tp_flags()
{
    :
}
tp_sources()
{
    find "src/third_party/$1" -name '*.c'
}
EOF
set +e
(cd "$fake" && TMPDIR="$private_tmp" CC="$stub" bash "$sweep" "$scratch/fake.txt" \
  >/dev/null 2>"$scratch/fake.err")
rc=$?
set -e
[[ $rc -eq 1 ]] || bad "a library with no sources: expected exit 1, got $rc"
grep -q 'no sources for src/third_party/empty' "$scratch/fake.err" ||
  bad "a library with no sources: the error must name it"
printf '#!/bin/sh\n' >"$fake/build.sh"
set +e
(cd "$fake" && TMPDIR="$private_tmp" CC="$stub" bash "$sweep" "$scratch/fake.txt" \
  >/dev/null 2>"$scratch/fake.err")
rc=$?
set -e
[[ $rc -eq 1 ]] || bad "a build.sh with no tp_flags: expected exit 1, got $rc"
grep -q 'build.sh has no function tp_flags' "$scratch/fake.err" ||
  bad "a build.sh with no tp_flags: the error must say so"

# Outside a clone the sweep has no tree to sweep.
mkdir "$scratch/outside"
set +e
(cd "$scratch/outside" && bash "$sweep" list.txt 2>"$scratch/err")
rc=$?
set -e
[[ $rc -eq 2 ]] || bad "outside a clone the sweep must exit 2, got $rc"
grep -q 'is not a Janas clone' "$scratch/err" || bad "outside a clone the error must say so"

# Without git, as in a build container, the current directory is the clone.
nogit="$scratch/nogit"
mkdir "$nogit"
for tool in bash env dirname basename uname tr find sort grep cat mktemp mkdir rm mv awk ls; do
  ln -s "$(command -v "$tool")" "$nogit/$tool"
done
: >"$STUB_RECORD"
(cd "$root" && PATH="$nogit" TMPDIR="$private_tmp" STUB_EXIT=0 CC="$stub" \
  bash "$sweep" "$scratch/nogit.txt" >"$scratch/nogit.out") ||
  bad "without git the sweep must run in the current directory"
[[ "$(cat "$scratch/nogit.out")" == "0 of $total files fail" ]] ||
  bad "without git: expected '0 of $total files fail', got '$(cat "$scratch/nogit.out")'"

# A relative output path is relative to the caller, not to the clone root.
rel="${inside#"$root"/}"
(cd "$root" && STUB_EXIT=1 TMPDIR="$private_tmp" CC="$stub" \
  bash "$sweep" "./$rel/root.txt" >/dev/null) || bad "a ./ path from the clone root must work"
[[ -s "$inside/root.txt" ]] || bad "a ./ path from the clone root must land there"
(cd "$root/src/llm" && STUB_EXIT=1 TMPDIR="$private_tmp" CC="$stub" \
  bash "$sweep" "../../$rel/deep.txt" >/dev/null) || bad "a path from two levels down must work"
[[ -s "$inside/deep.txt" ]] || bad "a path from two levels down must land in the caller's directory"

# The sweep leaves nothing in TMPDIR.
leftover="$(find "$private_tmp" -mindepth 1)"
[[ -z "$leftover" ]] || bad "the sweep must remove its scratch files: $leftover"

if ((fails > 0)); then
  echo "test-sweep-syntax: $fails failures" >&2
  exit 1
fi
echo "test-sweep-syntax: ok"
