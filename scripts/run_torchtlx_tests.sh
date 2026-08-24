#!/usr/bin/env bash
# Runner for the torchTLX Inductor unit tests (templates + fusions, incl. the AMD
# FlexAttention tests).
#
# By default it just RUNS the tests against the existing venv (fast). Pass --build
# to bootstrap first: create the venv, install a PRE (nightly) PyTorch — torchTLX
# tracks nightly, not stable — build the fbtriton fork Triton, and apply the torch
# shim. Any other args (and explicit test paths) are forwarded to pytest.
#
# Usage:
#   scripts/run_torchtlx_tests.sh                  # run tests (env already set up)
#   scripts/run_torchtlx_tests.sh --build          # bootstrap (pre torch + build) then run
#   scripts/run_torchtlx_tests.sh -k flex          # forward args to pytest
#   scripts/run_torchtlx_tests.sh --build -k flex  # combine
#
# Env knobs:
#   TORCHTLX_VENV=<dir>       venv location (default <repo>/.venv)
#   TORCH_INDEX_URL=<url>     torch nightly index (auto: cu128 on NV, rocm on AMD)
#   TORCHTLX_ROCM_LINE=<x.y>  rocm nightly line on AMD (default 7.0)
#   TORCHTLX_STDCXX_DIR=<d>   dir with a modern libstdc++.so.6 (auto-detected)
#   PIP_EXTRA_INDEX_URL / LLVM_SYSPATH / TRITON_OFFLINE_BUILD / JSON_SYSPATH / CC /
#   CXX are inherited by the --build step (needed for offline / AMD fork builds).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
VENV="${TORCHTLX_VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
TESTS=(
  python/test/unit/language/test_torchtlx_templates.py
  python/test/unit/language/test_torchtlx_fusions.py
)
log() { echo "[run_torchtlx_tests] $*"; }

# --- args: extract --build, forward the rest to pytest ------------------------
BUILD=0
PYTEST_ARGS=()
for a in "$@"; do
  if [ "$a" = "--build" ]; then BUILD=1; else PYTEST_ARGS+=("$a"); fi
done

# --- runtime library path (import torch + the fork .so need these) ------------
# .venv/lib carries the libzstd.so.1 symlink torch needs; a modern libstdc++
# (GLIBCXX_3.4.30) is needed where the prebuilt LLVM was linked against it.
detect_stdcxx_dir() {
  if [ -n "${TORCHTLX_STDCXX_DIR:-}" ]; then echo "$TORCHTLX_STDCXX_DIR"; return; fi
  local base_home cand
  base_home="$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg" 2>/dev/null || true)"
  for cand in "${base_home%/bin}/lib" "${CONDA_PREFIX:-}/lib" /opt/conda/lib; do
    [ -n "$cand" ] || continue
    if [ -e "$cand/libstdc++.so.6" ] && grep -qa GLIBCXX_3.4.30 "$cand/libstdc++.so.6" 2>/dev/null; then
      echo "$cand"; return
    fi
  done
  echo ""
}
[ -d "$VENV/lib" ] && export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
STDCXX="$(detect_stdcxx_dir)"
[ -n "$STDCXX" ] && export LD_LIBRARY_PATH="$STDCXX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "$BUILD" = 1 ]; then
  # pip/venv backend (prefer uv; fall back to python -m venv + pip)
  if command -v uv >/dev/null 2>&1; then
    PIP()   { uv pip install --python "$PY" "$@"; }
    PIPUN() { uv pip uninstall --python "$PY" "$@" 2>/dev/null || true; }
    [ -x "$PY" ] || { log "creating venv at $VENV"; uv venv "$VENV"; }
  else
    PIP()   { "$PY" -m pip install "$@"; }
    PIPUN() { "$PY" -m pip uninstall -y "$@" 2>/dev/null || true; }
    [ -x "$PY" ] || { log "creating venv at $VENV"; "$(command -v python3)" -m venv "$VENV"; }
  fi
  [ -d "$VENV/lib" ] && export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # torch nightly index: cu128 on NV; on AMD the rocm<line> nightly (its wheel
  # bundles the ROCm runtime so it runs on gfx950 regardless of system ROCm).
  default_index() {
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "https://download.pytorch.org/whl/nightly/cu128"
    elif command -v rocminfo >/dev/null 2>&1 || ls -d /opt/rocm* >/dev/null 2>&1; then
      echo "https://download.pytorch.org/whl/nightly/rocm${TORCHTLX_ROCM_LINE:-7.0}"
    else
      echo "https://download.pytorch.org/whl/nightly/cu128"
    fi
  }
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-$(default_index)}"

  # libstdc++ for the fork link step
  if [ -n "$STDCXX" ]; then
    log "using libstdc++ from $STDCXX"
    export LIBRARY_PATH="$STDCXX${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LDFLAGS="-L$STDCXX -Wl,-rpath,$STDCXX ${LDFLAGS:-}"
  else
    log "WARNING: no libstdc++ with GLIBCXX_3.4.30 found; the Triton link step may fail."
  fi

  # PRE (nightly) torch — torchTLX tracks nightly (never stable): the TLX Inductor
  # templates plug into recent Inductor APIs that stable releases lag.
  if ! "$PY" -c "import torch" 2>/dev/null; then
    log "installing nightly (pre) torch from $TORCH_INDEX_URL"
    PIP --pre torch --index-url "$TORCH_INDEX_URL" \
      ${PIP_EXTRA_INDEX_URL:+--extra-index-url "$PIP_EXTRA_INDEX_URL"}
    PIPUN pytorch-triton triton pytorch-triton-rocm  # don't let a wheel triton shadow the fork
  fi

  # build the fork Triton (with the TLX dialect) into the venv
  if ! "$PY" -c "import triton, triton.language.extra.tlx" 2>/dev/null; then
    log "installing build + test requirements"
    PIP -r python/requirements.txt -r python/test-requirements.txt
    log "building the fbtriton fork Triton (this can take a few minutes)"
    PIP -e . --no-build-isolation
  fi

  # torch shim: tlx_mode knob + fork-loading template_heuristics/tlx.py
  "$PY" scripts/_torchtlx_torch_shim.py
fi

# --- sanity + run -------------------------------------------------------------
"$PY" -c "import triton; print('[run_torchtlx_tests] triton ->', triton.__file__)"

# COMPILE_THREADS=1: autotune subprocess workers otherwise crash with
# '0 active drivers' (no GPU backend in the worker).
export TORCHINDUCTOR_COMPILE_THREADS=1

# Explicit test paths passed by the caller override the defaults.
set -- "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
have_path=0
for a in "$@"; do case "$a" in -*) ;; *) [ -e "$a" ] && have_path=1;; esac; done
if [ "$have_path" -eq 0 ]; then set -- "${TESTS[@]}" "$@"; fi

exec "$PY" -m pytest -p no:cacheprovider "$@" -v
