
# Basic logic to build Triton from scratch:
# 1. build LLVM with MLRI enabled
# 2. setup python virtual env locally
# 3. build Triton with the customized LLVM


# Step 1: Build LLVM with default cmake settings
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_CCACHE_BUILD=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DCMAKE_INSTALL_PREFIX= `pwd`/destdir \
  -B`pwd`/build `pwd`/llvm
ninja -C build


# to re-build without changing cmake settings, simply run at ./llvm-project
ninja -C build


# set up python venv without conda
cd ./triton
python3.11 -m venv .venv --prompt triton
source .venv/bin/activate

# install dependencies if not there
pip3 install ninja cmake wheel scipy numpy pytest pytest-xdist pytest-forked lit pandas matplotlib pybind11 expecttest hypothesis pre-commit
# for amd gpu
export CUDA_OR_ROCM=rocm
export CUDA_MAJOR=6
export CUDA_MINOR=".3"
export TORCH_URL=https://download.pytorch.org/whl/nightly/$CUDA_OR_ROCM$CUDA_MAJOR$CUDA_MINOR # pip install that get us torch 2.7.1 won't work, since triton is newer
pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url $TORCH_URL



# After running the above, to rebuild Triton, run:
LLVM_BUILD_DIR=`pwd`/../llvm-project-fbinternal/build \
DEBUG=1 \
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=0 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
    pip3 install -e . --no-build-isolation

# or
LLVM_BUILD_DIR=`pwd`/../triton-llvm/build \
DEBUG=1 \
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=0 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
    pip3 install -e . --no-build-isolation

# triton from internal repo has different pyproject.toml, so we need to run:
LLVM_BUILD_DIR=`pwd`/../llvm-project/build \
DEBUG=1 \
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=0 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
    pip3 install -e python --no-build-isolation


# run the tutorial
python3 python/tutorials/01-vector-add.py

# run python in debugging mode
lldb -- python3 python/tutorials/01-vector-add.py

# Triton hash: 2b797c903ccf4f19b17b7bb632260b6b77088b6f
# LLVM hash: e12cbd8339b89563059c2bb2a312579b652560d0
