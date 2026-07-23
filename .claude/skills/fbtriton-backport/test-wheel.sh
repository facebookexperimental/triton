#!/usr/bin/env bash
# Test a locally-built meta-triton wheel (produced by .claude/skills/fbtriton-backport/build-wheel.sh).
#
# Installs the wheel (arg > $WHEEL > newest in dist/) into a throwaway uv venv
# alongside the torch build matching the GPU on this box (auto-detected: CUDA
# torch for NVIDIA, ROCm torch for AMD — the default PyPI torch is CUDA-only and
# reports torch.cuda.is_available()==False on an AMD box), then runs:
#   1. a smoke kernel,
#   2. the launch.h crash-fix tests (#1816),
#   3. the TLX correctness suite (arch-gated: on AMD only amd/ikbo kernels run;
#      on NVIDIA only Hopper/Blackwell — the rest auto-skip either way).
#
# Standalone: it sources gcc-toolset so the wheel's libtriton.so finds the newer
# libstdc++ (GLIBCXX_3.4.30+ / __glibcxx_assert_fail) the prebuilt LLVM static
# libs depend on at runtime — the same toolchain the build used.
#
# Usage:
#   ./.claude/skills/fbtriton-backport/test-wheel.sh [path/to/wheel.whl]
# Env overrides: WHEEL, TEST_VENV, GCC_TOOLSET, CUDA_VISIBLE_DEVICES, TLX_TIMEOUT,
#                GPU_VENDOR (nvidia|amd, else auto-detected),
#                TORCH_INDEX_URL (torch wheel index; else picked from the GPU).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GCC_TOOLSET="${GCC_TOOLSET:-/opt/rh/gcc-toolset-14}"
TLX_TIMEOUT="${TLX_TIMEOUT:-1800}"

# --- detect GPU vendor + pick the matching torch wheel index -----------------
# NVIDIA -> default PyPI (CUDA) torch. AMD -> ROCm torch, on an index matching
# the gfx arch: gfx950 (MI350/CDNA4) and gfx12xx need ROCm 7.0; older cards work
# with ROCm 6.4. Override GPU_VENDOR / TORCH_INDEX_URL to bypass detection.
# Capture rocminfo once (grep'ing its pipe directly would SIGPIPE it and, under
# pipefail, make the detection conditional falsely fail).
ROCMINFO_OUT="$(command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null || true)"

GPU_VENDOR="${GPU_VENDOR:-}"
if [[ -z "$GPU_VENDOR" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
  elif grep -q 'Name:.*gfx' <<<"$ROCMINFO_OUT"; then
    GPU_VENDOR="amd"
  else
    echo "ERROR: no NVIDIA or AMD GPU detected (need nvidia-smi or rocminfo)." >&2
    echo "Set GPU_VENDOR=nvidia|amd to override." >&2
    exit 1
  fi
fi

TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [[ -z "$TORCH_INDEX_URL" ]]; then
  case "$GPU_VENDOR" in
    nvidia) TORCH_INDEX_URL="https://download.pytorch.org/whl" ;;  # CUDA (PyPI default)
    amd)
      GFX_ARCH="$(grep -oE -m1 'gfx[0-9a-f]+' <<<"$ROCMINFO_OUT" || true)"
      case "$GFX_ARCH" in
        gfx950|gfx12*) TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm7.0" ;;
        *)             TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.4" ;;
      esac
      echo ">>> AMD GPU $GFX_ARCH -> torch index $TORCH_INDEX_URL"
      ;;
    *) echo "ERROR: unknown GPU_VENDOR '$GPU_VENDOR' (want nvidia|amd)." >&2; exit 1 ;;
  esac
fi
echo ">>> GPU vendor: $GPU_VENDOR"

# --- locate the wheel: positional arg > $WHEEL > newest in dist/ --------------
WHEEL="${1:-${WHEEL:-}}"
if [[ -z "$WHEEL" ]]; then
  WHEEL="$(ls -t "$REPO_ROOT"/dist/*.whl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
  echo "ERROR: no wheel found. Build one first (.claude/skills/fbtriton-backport/build-wheel.sh) or pass a path." >&2
  exit 1
fi

# --- prereqs -----------------------------------------------------------------
command -v uv >/dev/null || { echo "ERROR: 'uv' not found on PATH." >&2; exit 1; }
if [[ ! -f "$GCC_TOOLSET/enable" ]]; then
  echo "ERROR: $GCC_TOOLSET/enable not found (install: sudo dnf install -y gcc-toolset-14)." >&2
  exit 1
fi

# gcc-toolset supplies the newer libstdc++ the wheel needs at runtime.
# shellcheck disable=SC1091
source "$GCC_TOOLSET/enable"

# --- install the wheel into a throwaway venv ---------------------------------
echo ">>> Installing $(basename "$WHEEL") into $REPO_ROOT/.venv"
uv venv -p python3.13 --clear
source .venv/bin/activate
uv pip install torch --index-url "$TORCH_INDEX_URL"
uv pip install pytest pytest-xdist numpy torchao

# ROCm torch pulls in the upstream 'triton-rocm' package (ships its own triton/),
# and the CUDA torch pulls in 'pytorch-triton'; either would shadow our wheel.
# Drop both (plus any prior 'triton') before installing the wheel under test.
uv pip uninstall triton triton-rocm pytorch-triton >/dev/null 2>&1 || true
uv pip install "$WHEEL"

# --- 1. smoke ----------------------------------------------------------------
echo ">>> Smoke: compile + run a trivial kernel"
# Real .py file: @triton.jit needs inspect.getsourcelines(), which fails on stdin.
cat > /tmp/triton_smoke.py <<'PYEOF'
import triton, triton.language as tl, torch
print("triton", triton.__version__)
@triton.jit
def add1(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)
x = torch.arange(1024, device="cuda", dtype=torch.float32); y = torch.empty_like(x)
add1[(4,)](x, y, x.numel(), BLOCK=256)
assert torch.allclose(y, x + 1.0), "smoke mismatch"
print("SMOKE OK")
PYEOF
python /tmp/triton_smoke.py

# --- 2. launch.h crash-fix (#1816) -------------------------------------------
echo ">>> launch.h crash-fix tests (#1816)"
pytest -q \
  "$REPO_ROOT/python/test/unit/runtime/test_launch_metadata.py" \
  "$REPO_ROOT/python/test/unit/runtime/test_launch_kernel_core.py"

# --- 3. TLX correctness suite ------------------------------------------------
# The suite is arch-gated via @pytest.mark.skipif. On AMD, select the kernels
# that actually run on this arch with -k "amd or ikbo" (IKBO node ids contain no
# "amd", so both selectors are needed); on NVIDIA, run the whole file and let the
# Hopper/Blackwell gating decide.
TLX_KFILTER=()
[[ "$GPU_VENDOR" == "amd" ]] && TLX_KFILTER=(-k "amd or ikbo")
echo ">>> TLX correctness suite (${GPU_VENDOR}, ${TLX_TIMEOUT}s cap)"
if timeout "$TLX_TIMEOUT" pytest -q "${TLX_KFILTER[@]}" \
     "$REPO_ROOT/third_party/tlx/tutorials/testing/test_correctness.py"; then
  echo "TLX SUITE OK"
else
  echo "TLX suite failed/timed out; killing GPU stragglers"
fi

echo ">>> Tests done."
