cmake_minimum_required(VERSION 3.20)
project(triton_lint_passes C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "Building Triton-Lint TMA Verification Pass")

# Find Triton's build directory first
set(TRITON_BUILD_DIR "${PROJECT_SOURCE_DIR}/../../build")
file(GLOB TRITON_CMAKE_DIRS "${TRITON_BUILD_DIR}/cmake.*")
if(TRITON_CMAKE_DIRS)
  list(GET TRITON_CMAKE_DIRS 0 TRITON_BUILD_PATH)
  message(STATUS "Found Triton build: ${TRITON_BUILD_PATH}")

  # Try to extract LLVM path from Triton's CMakeCache
  set(TRITON_CMAKE_CACHE "${TRITON_BUILD_PATH}/CMakeCache.txt")
  if(EXISTS "${TRITON_CMAKE_CACHE}")
    file(STRINGS "${TRITON_CMAKE_CACHE}" LLVM_DIR_LINE REGEX "^LLVM_DIR:PATH=")
    if(LLVM_DIR_LINE)
      string(REGEX REPLACE "^LLVM_DIR:PATH=" "" LLVM_DIR "${LLVM_DIR_LINE}")
      string(REGEX REPLACE "/lib/cmake/llvm$" "" LLVM_BUILD_ROOT "${LLVM_DIR}")
      set(MLIR_DIR "${LLVM_BUILD_ROOT}/lib/cmake/mlir")
      message(STATUS "Using Triton's LLVM from CMakeCache: ${LLVM_DIR}")
      message(STATUS "Using Triton's MLIR from CMakeCache: ${MLIR_DIR}")
    endif()
  endif()

  # Fallback: Check if MLIR/LLVM are in Triton's build directory
  if(NOT LLVM_DIR)
    set(MLIR_DIR "${TRITON_BUILD_PATH}/lib/cmake/mlir")
    set(LLVM_DIR "${TRITON_BUILD_PATH}/lib/cmake/llvm")
    message(STATUS "Using Triton's MLIR from: ${MLIR_DIR}")
    message(STATUS "Using Triton's LLVM from: ${LLVM_DIR}")
  endif()
else()
  message(WARNING "Triton build directory not found at ${TRITON_BUILD_DIR}")
  message(WARNING "Will try to use system LLVM/MLIR (may have ABI issues)")
endif()

# Find LLVM and MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

# Set up include paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/../include)

# Find Triton includes - check multiple locations
set(TRITON_BUILD_DIR "${PROJECT_SOURCE_DIR}/../../build")

# 1. Check Triton's build directory (../build/cmake.linux-x86_64-cpython-3.11/include)
message(STATUS "Searching for Triton includes...")
file(GLOB TRITON_BUILD_INCLUDES "${TRITON_BUILD_DIR}/cmake.*/include")
if(TRITON_BUILD_INCLUDES)
  list(GET TRITON_BUILD_INCLUDES 0 TRITON_INCLUDE_DIR)
  message(STATUS "✓ Found Triton build includes: ${TRITON_INCLUDE_DIR}")
  include_directories(${TRITON_INCLUDE_DIR})
endif()

# 2. Check source includes (../../include)
set(TRITON_SOURCE_INCLUDE "${PROJECT_SOURCE_DIR}/../../include")
if(EXISTS "${TRITON_SOURCE_INCLUDE}/triton")
  message(STATUS "✓ Found Triton source includes: ${TRITON_SOURCE_INCLUDE}")
  include_directories(${TRITON_SOURCE_INCLUDE})
endif()

# 3. Check python build directory
set(TRITON_PYTHON_BUILD "${PROJECT_SOURCE_DIR}/../../python/build")
if(EXISTS ${TRITON_PYTHON_BUILD})
  file(GLOB PYTHON_TRITON_INCLUDES "${TRITON_PYTHON_BUILD}/cmake.*/include")
  if(PYTHON_TRITON_INCLUDES)
    list(GET PYTHON_TRITON_INCLUDES 0 PYTHON_INCLUDE)
    message(STATUS "✓ Found Python build includes: ${PYTHON_INCLUDE}")
    include_directories(${PYTHON_INCLUDE})
  endif()
endif()

# If still not found, warn but continue (pass works without Triton headers)
if(NOT TRITON_BUILD_INCLUDES AND NOT EXISTS "${TRITON_SOURCE_INCLUDE}/triton" AND NOT PYTHON_TRITON_INCLUDES)
  message(WARNING "Triton headers not found - this is OK, pass uses string-based matching")
  message(WARNING "For better type checking, build Triton first: cd ../.. && python setup.py develop")
endif()

# Get LLVM/MLIR preprocessor definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Disable RTTI if LLVM was built without it
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# Build as a shared library (plugin)
add_library(triton_lint_passes SHARED
  TMAVerificationPass.cpp
)

# Set output directory
set_target_properties(triton_lint_passes PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../install/lib"
)

# Link against Triton dialect object files directly
# These are built as OBJECT libraries in Triton's build
set(TRITON_OBJS "")
if(TRITON_BUILD_PATH)
  file(GLOB TRITON_IR_OBJS "${TRITON_BUILD_PATH}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/*.o")
  file(GLOB TRITON_NVIDIA_OBJS "${TRITON_BUILD_PATH}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir/*.o")
  file(GLOB TRITON_GPU_OBJS "${TRITON_BUILD_PATH}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/*.o")

  list(APPEND TRITON_OBJS ${TRITON_IR_OBJS})
  list(APPEND TRITON_OBJS ${TRITON_NVIDIA_OBJS})
  list(APPEND TRITON_OBJS ${TRITON_GPU_OBJS})

  if(TRITON_OBJS)
    list(LENGTH TRITON_OBJS TRITON_OBJS_COUNT)
    message(STATUS "✓ Found ${TRITON_OBJS_COUNT} Triton object files")
    message(STATUS "  Linking against Triton dialects for ABI compatibility")
  else()
    message(WARNING "Could not find Triton object files - may have ABI issues")
  endif()
endif()

# Link against MLIR libraries
# Note: For a plugin, we don't link most MLIR libraries - they should be provided by the host executable
target_link_libraries(triton_lint_passes
  ${TRITON_OBJS}
)

# Installation
install(TARGETS triton_lint_passes
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

message(STATUS "Triton-Lint passes will be built as: libtriton_lint_passes.so")
