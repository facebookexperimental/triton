#!/bin/bash
# Build script for Triton-Lint TMA Verification Pass

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Building Triton-Lint TMA Verification Pass                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for LLVM/MLIR
if ! command -v llvm-config &> /dev/null; then
    echo "❌ Error: llvm-config not found"
    echo "Please ensure LLVM/MLIR is installed and in your PATH"
    echo ""
    echo "You may need to:"
    echo "  1. Load LLVM module: module load llvm"
    echo "  2. Or set LLVM_DIR: export LLVM_DIR=/path/to/llvm/lib/cmake/llvm"
    exit 1
fi

LLVM_VERSION=$(llvm-config --version)
echo "✓ Found LLVM version: $LLVM_VERSION"

# Create build directory
BUILD_DIR="$SCRIPT_DIR/lib/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Configuring CMake..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try to find MLIR/LLVM
if [ -n "$MLIR_DIR" ]; then
    echo "Using provided MLIR_DIR: $MLIR_DIR"
    CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR"
    if [ -n "$LLVM_DIR" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DLLVM_DIR=$LLVM_DIR"
    fi
elif [ -n "$LLVM_DIR" ]; then
    echo "Using provided LLVM_DIR: $LLVM_DIR"
    CMAKE_ARGS="-DLLVM_DIR=$LLVM_DIR"
else
    # Try to find Triton's build directory first
    echo "Searching for LLVM/MLIR..."
    echo ""

    TRITON_BUILD_DIR="$SCRIPT_DIR/../build"
    if [ -d "$TRITON_BUILD_DIR" ]; then
        echo "✓ Found Triton build directory: $TRITON_BUILD_DIR"

        # Look for LLVM in Triton's build
        TRITON_LLVM=$(find "$TRITON_BUILD_DIR" -name "MLIRConfig.cmake" 2>/dev/null | head -1)

        if [ -n "$TRITON_LLVM" ]; then
            MLIR_CMAKE_DIR=$(dirname "$TRITON_LLVM")
            LLVM_CMAKE_DIR=$(dirname "$MLIR_CMAKE_DIR")/llvm

            echo "✓ Found Triton's LLVM/MLIR:"
            echo "  MLIR: $MLIR_CMAKE_DIR"
            echo "  LLVM: $LLVM_CMAKE_DIR"
            CMAKE_ARGS="-DMLIR_DIR=$MLIR_CMAKE_DIR -DLLVM_DIR=$LLVM_CMAKE_DIR"
        else
            echo "⚠️  Warning: Triton build exists but LLVM not found"
            echo "   Did you build Triton? Run: cd .. && python setup.py develop"
        fi
    else
        echo "⚠️  Triton build directory not found at: $TRITON_BUILD_DIR"
        echo ""
        echo "   To build Triton first:"
        echo "     cd /data/users/pka/triton"
        echo "     python setup.py develop"
        echo ""
    fi

    # If still no MLIR found, try llvm-config
    if [ -z "$CMAKE_ARGS" ]; then
        if command -v llvm-config &> /dev/null; then
            LLVM_PREFIX=$(llvm-config --prefix)
            echo "✓ Found llvm-config at: $(which llvm-config)"

            if [ -d "$LLVM_PREFIX/lib/cmake/mlir" ]; then
                CMAKE_ARGS="-DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm"
                echo "✓ Using system MLIR: $LLVM_PREFIX/lib/cmake/mlir"
            elif [ -d "$LLVM_PREFIX/lib64/cmake/mlir" ]; then
                CMAKE_ARGS="-DMLIR_DIR=$LLVM_PREFIX/lib64/cmake/mlir -DLLVM_DIR=$LLVM_PREFIX/lib64/cmake/llvm"
                echo "✓ Using system MLIR: $LLVM_PREFIX/lib64/cmake/mlir"
            else
                echo "⚠️  llvm-config found but CMake files not in expected location"
            fi
        fi
    fi

    # Last resort: try user's llvm-build
    if [ -z "$CMAKE_ARGS" ] && [ -d "$SCRIPT_DIR/../../llvm-build" ]; then
        LLVM_BUILD="$SCRIPT_DIR/../../llvm-build"
        if [ -d "$LLVM_BUILD/lib/cmake/mlir" ]; then
            echo "✓ Found LLVM build at: $LLVM_BUILD"
            CMAKE_ARGS="-DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm"
        fi
    fi

    # Final fallback
    if [ -z "$CMAKE_ARGS" ]; then
        echo ""
        echo "❌ Could not find MLIR automatically!"
        echo ""
        echo "Please either:"
        echo "  1. Build Triton first:"
        echo "       cd /data/users/pka/triton && python setup.py develop"
        echo ""
        echo "  2. Set paths manually:"
        echo "       export MLIR_DIR=/path/to/llvm/lib/cmake/mlir"
        echo "       export LLVM_DIR=/path/to/llvm/lib/cmake/llvm"
        echo "       ./build.sh"
        echo ""
        exit 1
    fi
fi

# Configure with CMake
if command -v ninja &> /dev/null; then
    echo "Using Ninja build system"
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/install" \
        $CMAKE_ARGS \
        ..
else
    echo "Using Make build system"
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/install" \
        $CMAKE_ARGS \
        ..
fi

echo ""
echo "Building..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build
if command -v ninja &> /dev/null; then
    ninja
else
    make -j$(nproc)
fi

echo ""
echo "Installing..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install
if command -v ninja &> /dev/null; then
    ninja install
else
    make install
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Build Successful!                                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if library was created
LIB_PATH="$SCRIPT_DIR/install/lib/libtriton_lint_passes.so"
if [ -f "$LIB_PATH" ]; then
    echo "✓ Library created: $LIB_PATH"
    ls -lh "$LIB_PATH"
    echo ""
    echo "Usage:"
    echo "  triton-opt --load=$LIB_PATH --triton-lint-verify-tma input.mlir"
    echo ""
else
    echo "⚠️  Warning: Library not found at expected location"
    echo "Looking for library files..."
    find "$SCRIPT_DIR/install" -name "*.so" 2>/dev/null || true
fi

echo "To clean and rebuild: rm -rf $BUILD_DIR && ./build_now.sh"
