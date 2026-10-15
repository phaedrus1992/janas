#!/usr/bin/env bash
# The portability sweep. Compiles every source build.sh compiles with
# -fsyntax-only, with build.sh's flags, and lists the files that fail. It
# measures how far a port/* branch moves the failure count.
# Design: docs/superpowers/specs/2026-10-13-arm64-metal-port/e1-portability.md §2
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sweep-syntax.sh <failing-list>

Sweeps the Janas clone that holds the current directory, or the current
directory itself when git is not available (as in a build container). Writes
the failing files to <failing-list>, one for each line, and the compiler
output to <failing-list>.log, with a "=== <file>" line before each file.
Prints "<failing> of <total> files fail" on stdout.

The sweep measures the build without GPU support: it does not define
JANAS_GPU_SPV, so the Vulkan bodies of src/llm/gpu*.c are not compiled.

Exit 0 when the sweep ran, whatever the count; 1 when the compiler or the
tree is broken, so that no count would be true; 2 for a usage error.

CC selects the compiler (default cc). A port/* branch carries no .fork/, so
copy the script from main first:

  git show main:.fork/sweep-syntax.sh > .tmp/sweep-syntax.sh
  bash .tmp/sweep-syntax.sh .tmp/sweep.txt
EOF
}

die() {
  echo "sweep-syntax: $1" >&2
  exit 1
}

case "${1:-}" in
-h | --help)
  usage
  exit 0
  ;;
"")
  usage
  exit 2
  ;;
esac

out_dir="$(dirname "$1")"
if [[ ! -d "$out_dir" ]]; then
  echo "sweep-syntax: the directory $out_dir does not exist." >&2
  echo "sweep-syntax: create it (mkdir -p $out_dir), then run again." >&2
  exit 2
fi
out="$(cd "$out_dir" && pwd)/$(basename "$1")"
log="$out.log"

if ! command -v git >/dev/null; then
  top="$PWD"
elif ! top="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  # Git refuses a clone that another user owns, as in a container mount.
  echo "sweep-syntax: git cannot read a clone here; sweeping $PWD." >&2
  top="$PWD"
fi
if [[ ! -f "$top/build.sh" ]]; then
  echo "sweep-syntax: $top is not a Janas clone: it has no build.sh." >&2
  echo "sweep-syntax: cd into the Janas clone to sweep, then run again." >&2
  exit 2
fi
cd "$top"
CC="${CC:-cc}"
# shellcheck disable=SC2034 # read by tp_flags, which comes from build.sh
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
VERSION="$(git describe --always --dirty 2>/dev/null || echo unknown)"

# The third-party flags and source lists come from this clone's build.sh, so
# that each branch is swept with the build it carries.
for fn in tp_flags tp_sources; do
  body="$(awk -v fn="$fn" '$0 == fn "()" { on = 1 } on { print } on && $0 == "}" { exit }' build.sh)"
  [[ -n "$body" ]] || die "build.sh has no function $fn; update the sweep to match build.sh."
  eval "$body"
done

scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sweep.XXXXXX")"
trap 'rm -rf "$scratch"; rm -f "$out.part"' EXIT
trap 'exit 130' INT TERM HUP

# build.sh writes each src/<tool>/<name>.html as a C array in a generated
# header. A syntax check needs only the declaration, so the sweep writes
# stubs of its own: a header left in lib/ by an old build can be stale or
# absent, and an absent one adds a failure that is not a portability one.
gen="$scratch/gen"
mkdir "$gen"
for html in src/*/*.html; do
  if [[ -f "$html" ]]; then
    tool="$(basename "$(dirname "$html")")"
    page="$(basename "${html%.html}")"
    echo "static const char janas_${tool}_${page}_html[] = {0};" \
      >"$gen/${tool}_${page}_html.h"
  fi
done

# build.sh gives third-party sources -w and its own sources
# -Wall -Wextra -Werror. A sweep that puts -Werror on third-party sources
# reports about 160 failures that the real build does not have.
BASE=(-std=gnu11 -D_GNU_SOURCE= -pthread -Isrc -Iinclude -fPIC -fvisibility=hidden)
BASE+=(-I"$gen" -DJANAS_VERSION="\"$VERSION\"")
JANAS=("${BASE[@]}" -Wall -Wextra -Werror
  -Isrc/third_party/libmicrohttpd/src/include -Isrc/third_party/yyjson)
TP=("${BASE[@]}" -w)

# A compiler that cannot compile an empty file would count every file as a
# portability failure.
if ! printf 'int janas_sweep_probe;\n' | "$CC" -fsyntax-only "${BASE[@]}" -x c - \
  >"$scratch/probe.log" 2>&1; then
  cat "$scratch/probe.log" >&2
  die "the compiler '$CC' cannot compile an empty file with build.sh's flags; fix CC."
fi

# Each list is read into a variable and checked here: a find that fails, or
# that finds nothing, must stop the sweep and not shorten the total.
need() { # what, list
  [[ -n "$2" ]] || die "no sources for $1; the tree is not the one build.sh expects."
}

total=0
fail=0
: >"$out.part"
: >"$log"
sweep_one() { # flags... source
  local src="${!#}" rc=0
  total=$((total + 1))
  printf '=== %s\n' "$src" >>"$log"
  "$CC" -fsyntax-only "$@" >>"$log" 2>&1 || rc=$?
  if ((rc >= 126)); then
    die "the compiler '$CC' stopped with status $rc on $src; see $log."
  fi
  if ((rc != 0)); then
    fail=$((fail + 1))
    echo "$src" >>"$out.part"
  fi
}

for dir in src/third_party/*/; do
  lib="$(basename "$dir")"
  # shellcheck disable=SC2207 # build.sh splits tp_flags the same way
  flags=($(tp_flags "$lib"))
  sources="$(tp_sources "$lib")" ||
    die "build.sh's tp_sources $lib failed: a file in its list or a tool it runs is missing."
  need "src/third_party/$lib" "$sources"
  while IFS= read -r src; do
    sweep_one "${TP[@]}" ${flags[@]+"${flags[@]}"} "$src"
  done <<<"$sources"
done
# shellcheck disable=SC2207 # build.sh splits tp_flags the same way
MBEDTLS_INC=($(tp_flags mbedtls))

# A directory with a main.c is a program. Its other files go to the program
# pass below, as build.sh splits them.
mains=()
for m in src/*/main.c src/services/*/main.c; do
  if [[ -f "$m" ]]; then
    mains+=("$m")
  fi
done
((${#mains[@]} > 0)) || die "no src/*/main.c; the tree is not the one build.sh expects."
tool_dirs="$(for m in "${mains[@]}"; do dirname "$m"; done)"
sources="$(find src -name '*.c' ! -name main.c ! -name win_start.c \
  ! -path 'src/third_party/*' | sort)" || die "cannot list the sources under src/."
need "the library under src/" "$sources"
while IFS= read -r src; do
  if grep -qxF "$(dirname "$src")" <<<"$tool_dirs"; then
    continue
  fi
  extra=()
  if [[ "$src" == src/common/mcp_tls.c ]]; then
    extra=("${MBEDTLS_INC[@]/#-I/-isystem}")
  fi
  sweep_one "${JANAS[@]}" ${extra[@]+"${extra[@]}"} "$src"
done <<<"$sources"
for src in tests/*.c tools/*.c "${mains[@]}"; do
  if [[ ! -f "$src" ]]; then
    continue
  fi
  sweep_one "${JANAS[@]}" "$src"
  if [[ "$(basename "$src")" == main.c ]]; then
    parts="$(find "$(dirname "$src")" -maxdepth 1 -name '*.c' ! -name main.c | sort)" ||
      die "cannot list the sources beside $src."
    if [[ -n "$parts" ]]; then
      while IFS= read -r part; do
        sweep_one "${JANAS[@]}" "$part"
      done <<<"$parts"
    fi
  fi
done

# The list appears only when the sweep is complete, so a stopped sweep
# cannot leave a short list that reads as a result.
mv "$out.part" "$out"
echo "$fail of $total files fail"
