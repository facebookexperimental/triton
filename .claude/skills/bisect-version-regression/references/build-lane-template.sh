#!/bin/bash
# One self-contained Triton source-build lane for version bisecting.
#
# Usage: build-lane-template.sh <lane_name> <git_ref_or_tree_dir>
#   - Creates a conda env named <lane_name> (python 3.12).
#   - Checks out <git_ref_or_tree_dir> in ./src-<lane_name> (a git ref in
#     this clone) or copies the tree if the argument is a directory
#     (fbcode snapshot trees have no git history).
#   - Builds with a per-lane TRITON_HOME so parallel lanes never wipe
#     each other's LLVM download. Fixed LLVM pin and ptxas come from the
#     tree itself; never override them mid-bisect.
#
# Run two lanes in parallel shells/worktrees at most; measurements still
# serialize on the GPU.
set -euo pipefail

LANE="${1:?lane name}"
REF="${2:?git ref or tree dir}"
ENV_NAME="$LANE"
SRC="src-$LANE"
export TRITON_HOME="$PWD/.triton-home-$LANE"
mkdir -p "$TRITON_HOME"

# --- source tree -----------------------------------------------------------
if [ -d "$REF" ]; then
  rm -rf "$SRC" && cp -r "$REF" "$SRC"
else
  git clone --quiet https://github.com/facebookexperimental/triton.git "$SRC"
  git -C "$SRC" checkout --quiet "$REF"
fi
git -C "$SRC" submodule update --init 2>/dev/null || true  # e.g. sleef

# --- conda env -------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.12
conda install -y -n "$ENV_NAME" -c conda-forge \
  gcc=13.2.0 gxx=13.2.0 cmake ninja pybind11 \
  libstdcxx-ng lld zstd zlib libgcc-ng
conda activate "$ENV_NAME"

# PyTorch nightly first, then remove its bundled Triton.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip uninstall -y triton || true

# --- build -----------------------------------------------------------------
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$SRC"
# fbcode snapshot trees may need: LLVM_ENABLE_WERROR=OFF (record if used)
pip install -e . --no-build-isolation

# --- verify before trusting any benchmark ----------------------------------
python -c "import triton, torch; print(triton.__version__, triton.__file__); print(torch.__version__, torch.cuda.is_available())"
echo "LANE $LANE READY: src=$SRC env=$ENV_NAME TRITON_HOME=$TRITON_HOME"
