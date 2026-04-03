#!/bin/bash
# Build script with correct MLIR paths

set -e

echo "Building Triton-Lint with correct MLIR paths..."
echo ""

cd /data/users/pka/triton/triton-lint

# Set correct paths
export MLIR_DIR=/data/users/pka/llvm-build/lib/cmake/mlir
export LLVM_DIR=/data/users/pka/llvm-build/lib/cmake/llvm

echo "Using:"
echo "  MLIR_DIR=$MLIR_DIR"
echo "  LLVM_DIR=$LLVM_DIR"
echo ""

# Build
PATH=$PATH:../../llvm-build/bin ./build.sh
