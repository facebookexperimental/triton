#!/bin/bash
# Test if the build system works

echo "Testing Tritlint Build System"
echo "=============================="
echo ""

cd /data/users/pka/triton/tritlint

# Check if Triton is built
echo "1. Checking if Triton is built..."
if [ -d "../build" ]; then
    echo "   ✓ Triton build directory exists"

    # Check for LLVM
    if find ../build -name "MLIRConfig.cmake" 2>/dev/null | head -1 | grep -q .; then
        echo "   ✓ Triton's LLVM/MLIR found"
    else
        echo "   ✗ Triton's LLVM not found"
        echo "     Build Triton first: cd .. && python setup.py develop"
    fi
else
    echo "   ✗ Triton build directory not found"
    echo "     Build Triton first: cd .. && python setup.py develop"
fi

echo ""
echo "2. Checking llvm-config..."
if command -v llvm-config &> /dev/null; then
    echo "   ✓ llvm-config found: $(which llvm-config)"
    echo "     Version: $(llvm-config --version)"
else
    echo "   ✗ llvm-config not found"
fi

echo ""
echo "3. Checking for llvm-build..."
if [ -d "../../llvm-build/lib/cmake/mlir" ]; then
    echo "   ✓ llvm-build found with MLIR"
else
    echo "   ✗ llvm-build not found or missing MLIR"
fi

echo ""
echo "=============================="
echo "Ready to build? Run: ./build.sh"
