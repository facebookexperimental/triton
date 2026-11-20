include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  if(TRITON_BUILD_NVIDIA_GSAN_RUNTIME)
    set(GSAN_RUNTIME_DIR "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src")
    set(GSAN_RUNTIME_HDRS
      "${GSAN_RUNTIME_DIR}/GSan.h"
      "${GSAN_RUNTIME_DIR}/Compat.h"
      "${GSAN_RUNTIME_DIR}/TMA.cuh"
      "${GSAN_RUNTIME_DIR}/AllocationProfile.h"
      "${GSAN_RUNTIME_DIR}/Hash.cuh")

  find_program(TRITON_GSAN_CLANGXX NAMES clang++
               PATHS "${LLVM_SYSPATH}/bin"
               REQUIRED NO_DEFAULT_PATH)
  if(NOT TRITON_GSAN_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build gsan.ll")
  endif()

  set(GSAN_RUNTIME_CUDA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/backend")
  foreach(candidate "$ENV{CUDA_PATH}" "$ENV{CUDA_HOME}" "/usr/local/cuda")
    if(candidate AND EXISTS "${candidate}/include/cuda_runtime.h")
      set(GSAN_RUNTIME_CUDA_PATH "${candidate}")
      break()
    endif()
  endforeach()

  set(GSAN_RUNTIME_PLATFORM_FLAGS
      "--cuda-path=${GSAN_RUNTIME_CUDA_PATH}")
  if(APPLE AND CMAKE_OSX_SYSROOT)
    list(APPEND GSAN_RUNTIME_PLATFORM_FLAGS -isysroot "${CMAKE_OSX_SYSROOT}")
  endif()

  set(TRITON_GSAN_INCLUDE_FLAGS "")
  foreach(include_path IN ITEMS "${TRITON_CUDACRT_PATH}" "${TRITON_CUDART_PATH}")
    if(NOT "${include_path}" STREQUAL "")
      list(APPEND TRITON_GSAN_INCLUDE_FLAGS "-I${include_path}")
    endif()
  endforeach()

  set(GSAN_RUNTIME_TOOLCHAIN_FLAGS)
  # Detect the right include directory for C++ stdlib.
  # The way this is currently done is GCC specific.
  # TODO: Support libc++-based systems.
  set(GSAN_HOST_GNU_CXX "${CMAKE_CXX_COMPILER}")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Force g++ to be used for querying -print-file-name on libstdc++.so
    # clang++ does not print an absolute path in this case.
    find_program(GSAN_HOST_GNU_CXX NAMES g++ c++)
  endif()
  if(GSAN_HOST_GNU_CXX)
    execute_process(
      COMMAND "${GSAN_HOST_GNU_CXX}" -print-file-name=libstdc++.so
      OUTPUT_VARIABLE LIBSTDCXX_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(IS_ABSOLUTE "${LIBSTDCXX_PATH}")
      get_filename_component(GCC_INSTALL_DIR "${LIBSTDCXX_PATH}" DIRECTORY)
      list(APPEND GSAN_RUNTIME_TOOLCHAIN_FLAGS "--gcc-install-dir=${GCC_INSTALL_DIR}")
    endif()
  endif()

    # Keep both runtimes on the same compiler flags and dependency set.
    # They share the instrumentation ABI and descriptor decoder.
    set(GSAN_RUNTIME_IRS "")
    foreach(runtime IN ITEMS gsan gsan-profile)
      if(runtime STREQUAL "gsan-profile")
        set(runtime_src "${GSAN_RUNTIME_DIR}/AllocationProfile.cu")
      else()
        set(runtime_src "${GSAN_RUNTIME_DIR}/GSanLibrary.cu")
      endif()
      set(runtime_ir "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/${runtime}.ll")
      add_custom_command(
          OUTPUT "${runtime_ir}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
                  "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib"
          COMMAND "${TRITON_GSAN_CLANGXX}"
                  -x cuda
                  -std=c++17 -O3 -S -emit-llvm
                  --cuda-device-only
                  -nocudalib
                  --no-cuda-version-check
                  -fno-exceptions
                  -fcuda-flush-denormals-to-zero
                  --cuda-gpu-arch=sm_80
                  --cuda-feature=+ptx70
                  -Wno-unknown-cuda-version
                  ${GSAN_RUNTIME_TOOLCHAIN_FLAGS}
                  ${GSAN_RUNTIME_PLATFORM_FLAGS}
                  ${TRITON_GSAN_INCLUDE_FLAGS}
                  -isystem "${CMAKE_CURRENT_SOURCE_DIR}/clang_cuda_shims"
                  -isystem "${CMAKE_CURRENT_SOURCE_DIR}/backend/include"
                  "${runtime_src}" -o "${runtime_ir}"
          DEPENDS "${runtime_src}" ${GSAN_RUNTIME_HDRS}
          COMMENT "Building ${runtime} runtime"
          VERBATIM)
      list(APPEND GSAN_RUNTIME_IRS "${runtime_ir}")
    endforeach()
    add_custom_target(TritonNVIDIAGSanRuntime ALL DEPENDS ${GSAN_RUNTIME_IRS})
  endif()

  add_triton_plugin(TritonNVIDIA ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc LINK_LIBS TritonNVIDIAGPUToLLVM NVGPUToLLVM)
  target_link_libraries(TritonNVIDIA PRIVATE Python3::Module nanobind-static)
  if(TRITON_BUILD_NVIDIA_GSAN_RUNTIME)
    add_dependencies(TritonNVIDIA TritonNVIDIAGSanRuntime)
  endif()
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(hopper)
