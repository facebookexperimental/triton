#!/usr/bin/env bash
# Build the meta-triton wheel locally on a CentOS 9 / RHEL 9 devgpu.
#
# Steps:
#   1. Prompt for paths (googletest, Triton cache, GCC toolset).
#   2. Verify prereqs: uv, gcc-toolset-14, and proxy env if a fetch is needed.
#   3. Clone googletest v1.17.0 into GOOGLETEST_DIR (skipped if present).
#   4. Download the prebuilt LLVM tarball matching cmake/llvm-info.json into
#      $TRITON_CACHE/llvm/<name>/ and write a version.txt so setup.py picks
#      it up instead of trying to fetch (skipped if already cached).
#   5. Optionally clean prior build/ and dist/*.whl.
#   6. Source gcc-toolset-14 (provides GLIBCXX_3.4.30 + __glibcxx_assert_fail
#      that the prebuilt LLVM static libs depend on).
#   7. Run `uv build --wheel`, which spins up a fresh PEP 517 build env under
#      ~/.cache/uv/builds-v0/ (NOT a persistent venv), installs the build deps
#      from pyproject.toml (setuptools/cmake/ninja/pybind11), invokes
#      setuptools.build_meta.build_wheel which runs cmake/ninja, and writes
#      the wheel to dist/. The build env is discarded after.
#
# Interactive: prompts for paths / decisions when not provided. Override via
# env vars to skip prompts: GOOGLETEST_DIR, TRITON_CACHE, GCC_TOOLSET, CLEAN.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "$REPO_ROOT/dist"
LOG_FILE="$REPO_ROOT/dist/build-$(date +%Y%m%d-%H%M%S).log"

# --- prompt helpers ----------------------------------------------------------

is_tty() { [[ -t 0 && -t 1 ]]; }

# prompt_path VAR DEFAULT QUESTION
prompt_path() {
  local var="$1" default="$2" question="$3" answer
  if [[ -n "${!var:-}" ]]; then
    return
  fi
  if ! is_tty; then
    printf -v "$var" '%s' "$default"
    return
  fi
  read -rp "${question} [${default}]: " answer
  printf -v "$var" '%s' "${answer:-$default}"
}

# prompt_yn VAR DEFAULT QUESTION   (DEFAULT = y or n)
prompt_yn() {
  local var="$1" default="$2" question="$3" answer prompt
  if [[ -n "${!var:-}" ]]; then
    return
  fi
  if ! is_tty; then
    printf -v "$var" '%s' "$default"
    return
  fi
  if [[ "$default" == "y" ]]; then prompt="[Y/n]"; else prompt="[y/N]"; fi
  read -rp "${question} ${prompt}: " answer
  answer="${answer:-$default}"
  case "$answer" in
    [Yy]*) printf -v "$var" 'y' ;;
    *)     printf -v "$var" 'n' ;;
  esac
}

# --- gather inputs -----------------------------------------------------------

cat <<'EOF'
============================================================
 meta-triton local wheel build
============================================================
This script will:
  1. Stage googletest v1.17.0 (clone if missing).
  2. Stage the prebuilt LLVM matching cmake/llvm-info.json
     into the Triton cache (download if missing).
  3. Activate gcc-toolset-14 (newer libstdc++ symbols that
     the prebuilt LLVM libs reference).
  4. Run `uv build --wheel`, which uses an ephemeral PEP 517
     build env under ~/.cache/uv/builds-v0/ (NOT a persistent
     venv) to invoke setuptools -> cmake -> ninja, and writes
     the resulting wheel to dist/.

Network fetches (steps 1-2) require Meta's `with-proxy` wrapper
on the first run; subsequent builds reuse the local cache.
============================================================

EOF

prompt_path GOOGLETEST_DIR "$REPO_ROOT/.googletest" \
  "Where should googletest live?"
prompt_path TRITON_CACHE   "${TRITON_HOME:-$HOME/.triton}" \
  "Triton cache root (LLVM tarball will be staged here)?"
prompt_path GCC_TOOLSET    "/opt/rh/gcc-toolset-14" \
  "GCC toolset prefix?"

if [[ -d "$REPO_ROOT/build" || -n "$(echo "$REPO_ROOT"/dist/*.whl 2>/dev/null)" ]]; then
  prompt_yn CLEAN "n" "Clean previous build/ and dist/*.whl before building?"
else
  CLEAN="n"
fi

LLVM_SUFFIX="almalinux-x64"
# LLVM pin source: newer branches use cmake/llvm-info.json (llvm_hash +
# build_number; the tarball name includes the build number). Older branches use
# the bare cmake/llvm-hash.txt (no build number). Support both.
if [[ -f "$REPO_ROOT/cmake/llvm-info.json" ]]; then
  LLVM_HASH="$(python3 -c "import json;print(json.load(open('$REPO_ROOT/cmake/llvm-info.json'))['llvm_hash'][:8])")"
  LLVM_BUILD="$(python3 -c "import json;print(json.load(open('$REPO_ROOT/cmake/llvm-info.json'))['build_number'])")"
  LLVM_NAME="llvm-${LLVM_HASH}-${LLVM_SUFFIX}-${LLVM_BUILD}"
else
  LLVM_HASH="$(cut -c1-8 "$REPO_ROOT/cmake/llvm-hash.txt")"
  LLVM_NAME="llvm-${LLVM_HASH}-${LLVM_SUFFIX}"
fi
LLVM_URL="https://oaitriton.blob.core.windows.net/public/llvm-builds/${LLVM_NAME}.tar.gz"
LLVM_DIR="${TRITON_CACHE}/llvm/${LLVM_NAME}"

cat <<EOF

Plan:
  Repo:           $REPO_ROOT
  googletest:     $GOOGLETEST_DIR  $( [[ -d "$GOOGLETEST_DIR/.git" ]] && echo "(present)" || echo "(will clone v1.17.0)" )
  Triton cache:   $TRITON_CACHE
  LLVM:           $LLVM_DIR  $( [[ -f "$LLVM_DIR/version.txt" ]] && echo "(cached)" || echo "(will download)" )
  GCC toolset:    $GCC_TOOLSET
  Clean rebuild:  $CLEAN

EOF

prompt_yn PROCEED "y" "Proceed?"
[[ "$PROCEED" == "y" ]] || { echo "Aborted."; exit 0; }

# Interactive phase done; tee everything else (clone, download, build) to the log.
echo "Logging to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- prereq checks -----------------------------------------------------------

if ! command -v uv >/dev/null; then
  echo "ERROR: 'uv' not found on PATH. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -f "$GCC_TOOLSET/enable" ]]; then
  echo "ERROR: $GCC_TOOLSET/enable not found." >&2
  echo "Install with: sudo dnf install -y gcc-toolset-14" >&2
  exit 1
fi

# Network access: github.com (googletest) and oaitriton.blob.core.windows.net
# (LLVM tarball) are not reachable directly from Meta devgpus. Re-run via
# /usr/bin/with-proxy if a fetch is needed and proxy env isn't set.
NEED_NET=0
[[ ! -d "$GOOGLETEST_DIR/.git" ]] && NEED_NET=1
{ [[ ! -f "$LLVM_DIR/version.txt" ]] || [[ "$(cat "$LLVM_DIR/version.txt")" != "$LLVM_URL" ]]; } && NEED_NET=1
if [[ "$NEED_NET" == 1 && -z "${HTTPS_PROXY:-${https_proxy:-}}" && -x /usr/bin/with-proxy ]]; then
  echo "ERROR: network fetch needed (googletest and/or LLVM) but no proxy is set." >&2
  echo "Re-run via with-proxy:" >&2
  echo "  with-proxy $0 $*" >&2
  exit 1
fi

# --- googletest --------------------------------------------------------------

if [[ ! -d "$GOOGLETEST_DIR/.git" ]]; then
  echo ">>> Cloning googletest v1.17.0 to $GOOGLETEST_DIR"
  git clone --depth 1 --branch v1.17.0 \
    https://github.com/google/googletest.git "$GOOGLETEST_DIR"
else
  echo ">>> googletest already at $GOOGLETEST_DIR"
fi

# --- prebuilt LLVM matching cmake/llvm-info.json -----------------------------

if [[ ! -f "$LLVM_DIR/version.txt" ]] || [[ "$(cat "$LLVM_DIR/version.txt")" != "$LLVM_URL" ]]; then
  echo ">>> Downloading $LLVM_NAME"
  mkdir -p "${TRITON_CACHE}/llvm"
  curl -fL -o "${TRITON_CACHE}/llvm/${LLVM_NAME}.tar.gz" "$LLVM_URL"
  tar -xzf "${TRITON_CACHE}/llvm/${LLVM_NAME}.tar.gz" -C "${TRITON_CACHE}/llvm"
  printf '%s' "$LLVM_URL" > "$LLVM_DIR/version.txt"
else
  echo ">>> LLVM already cached at $LLVM_DIR"
fi

# --- clean (optional) --------------------------------------------------------

if [[ "$CLEAN" == "y" ]]; then
  echo ">>> Cleaning build/ and dist/*.whl"
  rm -rf "$REPO_ROOT/build" "$REPO_ROOT"/dist/*.whl
fi

# --- build -------------------------------------------------------------------

# shellcheck disable=SC1091
source "$GCC_TOOLSET/enable"

echo ">>> Building wheel ($(c++ --version | head -1))"
cd "$REPO_ROOT"
TRITON_LLVM_SYSTEM_SUFFIX="$LLVM_SUFFIX" \
TRITON_APPEND_CMAKE_ARGS="-DGOOGLETEST_DIR=$GOOGLETEST_DIR -DTRITON_BUILD_UT=OFF" \
  uv build --wheel . -o dist/ -p python3

echo
echo ">>> Done:"
ls -lh "$REPO_ROOT/dist/"*.whl
echo
echo "Build log saved to: $LOG_FILE"
echo "To test the wheel:  $REPO_ROOT/.claude/skills/fbtriton-backport/test-wheel.sh"
