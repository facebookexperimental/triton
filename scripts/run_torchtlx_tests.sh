#!/usr/bin/env bash
# One-shot runner for the torchTLX Inductor unit tests (templates + fusions).
#
# From a fresh terminal on an fbtriton dev box this will, idempotently:
#   1. create an isolated venv (.venv) if missing,
#   2. install a nightly PyTorch (torchTLX tracks nightly, not stable),
#   3. build the fbtriton fork Triton (with the TLX dialect) into that venv,
#   4. apply a small stopgap so torch carries the torchTLX PyTorch-side pieces
#      (tlx_mode knob + fork-loading template_heuristics/tlx.py) until they land
#      in OSS PyTorch,
#   5. run the two test files with the env fixes they need.
# Re-running when everything is already set up just runs the tests.
#
# Usage:
#   scripts/run_torchtlx_tests.sh                 # bootstrap (if needed) + run both files
#   scripts/run_torchtlx_tests.sh -k matmul_ws    # forward extra args to pytest
#   TORCHTLX_SKIP_SETUP=1 scripts/run_torchtlx_tests.sh   # just run, assume ready env
#
# Env knobs:
#   TORCHTLX_VENV=<dir>     venv location (default: <repo>/.venv)
#   TORCH_INDEX_URL=<url>   torch nightly index (default cu128; use a rocm index on AMD)
#   TORCHTLX_STDCXX_DIR=<d> dir with a modern libstdc++.so.6 (auto-detected if unset)
#   PIP_EXTRA_INDEX_URL=<url>  extra index for torch's deps (helps on networks that
#                              block pypi.nvidia.com)
#   Offline/constrained builds may also export LLVM_SYSPATH / TRITON_OFFLINE_BUILD /
#   JSON_SYSPATH; they are inherited by the build step.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
VENV="${TORCHTLX_VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu128}"
TESTS=(
  python/test/unit/language/test_torchtlx_templates.py
  python/test/unit/language/test_torchtlx_fusions.py
)

log() { echo "[run_torchtlx_tests] $*"; }

# --- pip/venv backend (prefer uv; fall back to python -m venv + pip) ----------
if command -v uv >/dev/null 2>&1; then
  mkvenv() { uv venv "$VENV"; }
  PIP()    { uv pip install --python "$PY" "$@"; }
  PIPUN()  { uv pip uninstall --python "$PY" "$@" 2>/dev/null || true; }
else
  mkvenv() { "$(command -v python3)" -m venv "$VENV"; }
  PIP()    { "$PY" -m pip install "$@"; }
  PIPUN()  { "$PY" -m pip uninstall -y "$@" 2>/dev/null || true; }
fi

# --- locate a modern libstdc++ (the prebuilt LLVM needs GLIBCXX_3.4.30) -------
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

if [ "${TORCHTLX_SKIP_SETUP:-0}" != "1" ]; then
  # 1) venv --------------------------------------------------------------------
  if [ ! -x "$PY" ]; then log "creating venv at $VENV"; mkvenv; fi
  STDCXX="$(detect_stdcxx_dir)"
  if [ -n "$STDCXX" ]; then
    log "using libstdc++ from $STDCXX"
    export LD_LIBRARY_PATH="$STDCXX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$STDCXX${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LDFLAGS="-L$STDCXX -Wl,-rpath,$STDCXX ${LDFLAGS:-}"
  else
    log "WARNING: no libstdc++ with GLIBCXX_3.4.30 found; the Triton link step may fail."
    log "         set TORCHTLX_STDCXX_DIR to a dir containing a newer libstdc++.so.6"
  fi

  # 2) nightly torch -----------------------------------------------------------
  if ! LD_LIBRARY_PATH="${STDCXX:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" -c "import torch" 2>/dev/null; then
    log "installing nightly torch from $TORCH_INDEX_URL"
    PIP --pre --upgrade torch --index-url "$TORCH_INDEX_URL" \
      ${PIP_EXTRA_INDEX_URL:+--extra-index-url "$PIP_EXTRA_INDEX_URL"}
    # drop the pytorch-triton torch pulls so it can't shadow the fork build
    PIPUN pytorch-triton triton
  fi

  # 3) build the fork Triton ---------------------------------------------------
  if ! LD_LIBRARY_PATH="${STDCXX:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" -c "import triton, triton.language.extra.tlx" 2>/dev/null; then
    log "installing build + test requirements"
    PIP -r python/requirements.txt -r python/test-requirements.txt
    log "building the fbtriton fork Triton (this can take a few minutes)"
    PIP -e . --no-build-isolation
  fi

  # 4) torch stopgap (tlx_mode knob + fork-loading tlx.py) ---------------------
  "$PY" scripts/_torchtlx_torch_shim.py
else
  STDCXX="$(detect_stdcxx_dir)"
  [ -n "$STDCXX" ] && export LD_LIBRARY_PATH="$STDCXX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- sanity + run -------------------------------------------------------------
"$PY" -c "import triton; print('[run_torchtlx_tests] triton ->', triton.__file__)"

# TORCHINDUCTOR_COMPILE_THREADS=1: the autotune subprocess workers otherwise
# crash with '0 active drivers' (no GPU backend in the worker).
export TORCHINDUCTOR_COMPILE_THREADS=1

# Explicit test paths passed by the caller override the defaults.
have_path=0
for a in "$@"; do case "$a" in -*) ;; *) [ -e "$a" ] && have_path=1;; esac; done
if [ "$have_path" -eq 0 ]; then set -- "${TESTS[@]}" "$@"; fi

exec "$PY" -m pytest -p no:cacheprovider "$@" -v
