#!/usr/bin/env bash
# Build a PyTorch source checkout into the torchTLX co-development venv.
#
# Run this yourself, not from an agent: fetching the submodules needs github,
# which the agent network filter blocks.
#
#   scripts/build_torch_src.sh [<pytorch-checkout>]
#
# Defaults to /data/users/daohang/pytorch, installing into <repo>/.venv-torchsrc.
# Afterwards, run against that venv's interpreter directly:
#   .venv-torchsrc/bin/python -m pytest python/test/unit/tlx_ops/test_torchtlx_*.py
#   .venv-torchsrc/bin/python -m pytest python/test/tlx_benchmark/test_ops_perf.py -k torchtlx
#
# Env knobs: TORCHTLX_VENV, TORCH_CUDA_ARCH_LIST (default 10.0 = B200), MAX_JOBS.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
SRC="$(cd "${1:-/data/users/daohang/pytorch}" && pwd)"
VENV="${TORCHTLX_VENV:-$REPO/.venv-torchsrc}"
PY="$VENV/bin/python"
log() { echo "[build_torch_src] $*"; }

# --- submodules (needs github) ------------------------------------------------
if [ -z "$(ls -A "$SRC/third_party/pybind11" 2>/dev/null)" ]; then
  log "fetching submodules into $SRC (this is the step an agent cannot do)"
  git -C "$SRC" submodule update --init --recursive
fi

# --- toolchain ----------------------------------------------------------------
# System gcc is 11.5, too old for this source; gcc-toolset-15 is too new for
# nvcc 12.9. 14 is the one that works on this host.
TOOLSET=/opt/rh/gcc-toolset-14/root/usr
if [ -x "$TOOLSET/bin/gcc" ]; then
  export CC="$TOOLSET/bin/gcc" CXX="$TOOLSET/bin/g++" PATH="$TOOLSET/bin:$PATH"
  log "using $($CC --version | head -1)"
fi
# Derive CUDA_HOME from the nvcc actually on PATH. /usr/local/cuda is a distro
# alternative that can point at a different version than that nvcc, and cmake
# rejects the mismatch ("FindCUDA says X, but the CUDA headers say Y").
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# CMake pins CMAKE_CUDA_COMPILER in its cache, so a cache written against a
# different CUDA than $CUDA_HOME fails the configure no matter what PATH says.
# Both keys matter: nvcc and the headers are resolved separately, and a cache
# where they disagree fails the configure regardless of PATH.
CACHE="$SRC/build/CMakeCache.txt"
if [ -f "$CACHE" ] && grep -qE "^(CMAKE_CUDA_COMPILER:FILEPATH|CUDA_TOOLKIT_ROOT_DIR:PATH)=" "$CACHE" &&
  grep -E "^(CMAKE_CUDA_COMPILER:FILEPATH|CUDA_TOOLKIT_ROOT_DIR:PATH)=" "$CACHE" | grep -qv "=$CUDA_HOME[/$]"; then
  log "dropping CMake cache built against a different CUDA than $CUDA_HOME"
  rm -rf "$SRC/build/CMakeCache.txt" "$SRC/build/CMakeFiles"
fi

# --- venv ---------------------------------------------------------------------
# uv defaults to the fbcode platform010 python, whose loader never searches
# /lib64 and so cannot import torch (libzstd.so.1). Match the repo venv instead.
if [ ! -x "$PY" ]; then
  BASE="$(sed -n 's/^home = //p' "$REPO/.venv/pyvenv.cfg" 2>/dev/null || true)"
  log "creating venv at $VENV"
  uv venv ${BASE:+--python "$BASE/python3"} "$VENV"
fi
PIP() { uv pip install --python "$PY" "$@"; }

log "installing build requirements"
PIP -r "$SRC/requirements.txt"
[ -f "$SRC/requirements-build.txt" ] && PIP -r "$SRC/requirements-build.txt"
PIP cmake ninja

# --- build --------------------------------------------------------------------
log "building PyTorch from $SRC (expect 1-2 hours)"
cd "$SRC"
USE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}" \
USE_FLASH_ATTENTION=0 \
USE_MEM_EFF_ATTENTION=0 \
BUILD_TEST=0 \
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / 2 ))}" \
  uv pip install --python "$PY" -e . --no-build-isolation

# --- fork Triton on top -------------------------------------------------------
uv pip uninstall --python "$PY" pytorch-triton triton 2>/dev/null || true
cd "$REPO"
log "building the fbtriton fork Triton into $VENV"
PIP -r python/requirements.txt -r python/test-requirements.txt
PIP -e . --no-build-isolation

# --- verify -------------------------------------------------------------------
"$PY" - <<'CHECK'
import torch, triton
from torch._inductor import config
print("torch   ->", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("triton  ->", triton.__file__)
print("tlx_mode ->", config.triton.tlx_mode)
import torch._inductor.heuristics.template.tlx as t
print("loader  ->", t.__file__)
CHECK
log "done -- run against it with $PY, e.g."
log "  $PY -m pytest python/test/unit/tlx_ops/test_torchtlx_*.py"
log "  $PY -m pytest python/test/tlx_benchmark/test_ops_perf.py -k torchtlx"
