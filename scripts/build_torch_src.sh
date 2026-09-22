#!/usr/bin/env bash
# Create an isolated TorchTLX environment from a PyTorch nightly (default) or
# an explicitly supplied PyTorch source checkout.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
VENV="${TORCHTLX_VENV:-$REPO/.venv-torchtlx}"
PYTORCH_SRC=""
BASE_PYTHON="${TORCHTLX_BASE_PYTHON:-python3}"
TORCH_SPEC="${TORCHTLX_TORCH_SPEC:-torch>=2.15.0.dev20260920}"
TORCH_INDEX="${TORCHTLX_TORCH_INDEX_URL:-}"

usage() {
  echo "usage: $0 [--pytorch-src PATH] [--venv PATH] [--torch-index-url URL] [--python PYTHON]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pytorch-src)
      PYTORCH_SRC="${2:?--pytorch-src requires a path}"
      shift 2
      ;;
    --venv)
      VENV="${2:?--venv requires a path}"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX="${2:?--torch-index-url requires a URL}"
      shift 2
      ;;
    --python)
      BASE_PYTHON="${2:?--python requires an interpreter}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() { echo "[torchtlx-setup] $*"; }

if [ ! -x "$VENV/bin/python" ]; then
  log "creating virtual environment at $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$BASE_PYTHON" "$VENV"
  else
    "$BASE_PYTHON" -m venv "$VENV"
  fi
fi
PY="$VENV/bin/python"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@"
  else
    "$PY" -m pip install "$@"
  fi
}

pip_uninstall() {
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall --python "$PY" "$@"
  else
    "$PY" -m pip uninstall -y "$@"
  fi
}

if [ -n "$PYTORCH_SRC" ]; then
  PYTORCH_SRC="$(cd "$PYTORCH_SRC" && pwd)"
  if [ ! -f "$PYTORCH_SRC/setup.py" ] && [ ! -f "$PYTORCH_SRC/pyproject.toml" ]; then
    echo "not a PyTorch source checkout: $PYTORCH_SRC" >&2
    exit 1
  fi
  if [ ! -d "$PYTORCH_SRC/third_party/pybind11/include" ]; then
    echo "PyTorch submodules are not initialized in $PYTORCH_SRC" >&2
    echo "initialize them explicitly before rerunning; this script will not modify the checkout" >&2
    exit 1
  fi

  log "installing PyTorch build requirements"
  pip_install -r "$PYTORCH_SRC/requirements.txt"
  if [ -f "$PYTORCH_SRC/requirements-build.txt" ]; then
    pip_install -r "$PYTORCH_SRC/requirements-build.txt"
  fi
  pip_install cmake ninja

  log "building editable PyTorch from $PYTORCH_SRC"
  log "the PyTorch build may create or update untracked build artifacts in that checkout"
  (
    export BUILD_TEST=0
    export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / 2 ))}"
    if command -v rocm-smi >/dev/null 2>&1; then
      export USE_ROCM=1
      export USE_CUDA=0
      # Leave PYTORCH_ROCM_ARCH unset unless the caller chose one; PyTorch then
      # derives the target set instead of assuming this host is MI350X.
    elif command -v nvidia-smi >/dev/null 2>&1; then
      export USE_CUDA=1
      export USE_ROCM=0
      # Leave TORCH_CUDA_ARCH_LIST unset unless the caller chose one; PyTorch
      # then derives it from visible hardware instead of assuming B200.
    else
      export USE_CUDA=0
      export USE_ROCM=0
    fi
    pip_install -e "$PYTORCH_SRC" --no-build-isolation
  )
else
  if [ -z "$TORCH_INDEX" ]; then
    if command -v rocm-smi >/dev/null 2>&1; then
      ROCM_VERSION="${TORCHTLX_ROCM_VERSION:-7.2}"
      TORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      CUDA_VERSION="${TORCHTLX_CUDA_VERSION:-129}"
      TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION}"
    else
      TORCH_INDEX="https://download.pytorch.org/whl/nightly/cpu"
    fi
  fi
  log "installing PyTorch nightly from $TORCH_INDEX"
  pip_install --pre --upgrade "$TORCH_SPEC" --index-url "$TORCH_INDEX"
fi

# The wheel's bundled Triton must not shadow the FBTriton checkout.
pip_uninstall pytorch-triton pytorch-triton-rocm triton triton-rocm 2>/dev/null || true
log "installing FBTriton into $VENV"
pip_install -r "$REPO/python/requirements.txt" -r "$REPO/python/test-requirements.txt"
pip_install -e "$REPO" --no-build-isolation

"$PY" - <<'PY'
import pathlib

import torch
import triton
from torch._inductor import config

if not hasattr(config.triton, "tlx_mode"):
    raise SystemExit("incompatible PyTorch: torch._inductor.config.triton.tlx_mode is missing")

import triton.language.extra.tlx.inductor.registry  # noqa: F401

print("torch  ->", torch.__version__)
print("triton ->", pathlib.Path(triton.__file__).resolve())
print("TorchTLX registry -> compatible")
PY

log "ready"
log "$PY python/test/tlx_benchmark/run_torchtlx_fusions.py --list"
