#!/usr/bin/env bash
set -euo pipefail
sdk="$RUNNER_TEMP/llvm-sdk-relocated"
consumer="$RUNNER_TEMP/llvm-sdk-consumer"
mv "$LLVM_INSTALL_DIR" "$sdk"
test -f "$sdk/include/llvm/IR/Module.h"
test -f "$sdk/lib/libLLVMAMDGPUCodeGen.a"
if [[ "$LLVM_PACKAGE" == llvm ]]; then
  test -f "$sdk/lib/cmake/mlir/MLIRConfig.cmake"
  "$sdk/bin/mlir-tblgen" --version
fi
mkdir -p "$consumer"
cat > "$consumer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(llvm_sdk_consumer LANGUAGES C CXX)
find_package(LLVM REQUIRED CONFIG)
add_executable(sdk_consumer main.cpp)
target_compile_features(sdk_consumer PRIVATE cxx_std_17)
target_include_directories(sdk_consumer PRIVATE ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs Core Support)
target_link_libraries(sdk_consumer PRIVATE ${llvm_libs})
CMAKE
cat > "$consumer/main.cpp" <<'CPP'
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
int main() {
  llvm::LLVMContext context;
  llvm::Module module("sdk", context);
  return module.getName() == "sdk" ? 0 : 1;
}
CPP
cmake -G Ninja -S "$consumer" -B "$consumer/build" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_DIR="$sdk/lib/cmake/llvm" \
  -DCMAKE_DISABLE_FIND_PACKAGE_LibLZMA=TRUE
cmake --build "$consumer/build"
"$consumer/build/sdk_consumer"
mv "$sdk" "$LLVM_INSTALL_DIR"
