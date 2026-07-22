set(TLX_WAVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../wave")
set(TLX_WAVE_LLVM_INSTALL_DIR "${TLX_WAVE_SOURCE_DIR}/build/llvm-install"
  CACHE PATH "Wave-pinned LLVM/MLIR install prefix used by the Wave submodule")
set(TLX_WAVE_STANDALONE_BUILD_DIR "${TLX_WAVE_SOURCE_DIR}/build/wave-build"
  CACHE PATH "Standalone Wave build directory used by the tlx_wave backend")

if(NOT EXISTS "${TLX_WAVE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "tlx_wave requires the third_party/wave submodule. "
    "Run `git submodule update --init --recursive third_party/wave`.")
endif()

function(tlx_wave_mlir_has_python_bindings LLVM_INSTALL_DIR OUT_VAR)
  set(_mlir_config "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
  if(EXISTS "${_mlir_config}")
    file(STRINGS "${_mlir_config}" _mlir_python_bindings
      REGEX "set\\(MLIR_ENABLE_BINDINGS_PYTHON[ \t]+\"?ON\"?\\)")
    if(_mlir_python_bindings)
      set(${OUT_VAR} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${OUT_VAR} FALSE PARENT_SCOPE)
endfunction()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLXWave ${CMAKE_CURRENT_SOURCE_DIR}/triton_tlx_wave.cc)
  target_link_libraries(TritonTLXWave PRIVATE Python3::Module pybind11::headers)
endif()

set(TLX_WAVE_BUILD_JOBS "" CACHE STRING
  "Parallelism for the standalone Wave build; defaults to MAX_JOBS when set")
if("${TLX_WAVE_BUILD_JOBS}" STREQUAL "" AND DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
  set(TLX_WAVE_BUILD_JOBS "$ENV{MAX_JOBS}")
endif()

set(_tlx_wave_build_parallel_args)
if(NOT "${TLX_WAVE_BUILD_JOBS}" STREQUAL "")
  list(APPEND _tlx_wave_build_parallel_args "--parallel" "${TLX_WAVE_BUILD_JOBS}")
endif()

tlx_wave_mlir_has_python_bindings("${TLX_WAVE_LLVM_INSTALL_DIR}" _tlx_wave_local_has_mlir_python)
if(_tlx_wave_local_has_mlir_python)
  add_custom_target(TLXWavePinnedLLVM)
else()
  if(TRITON_OFFLINE_BUILD)
    message(FATAL_ERROR
      "tlx_wave requires the Wave-pinned LLVM/MLIR install with MLIR Python "
      "bindings at '${TLX_WAVE_LLVM_INSTALL_DIR}'. Run "
      "`python third_party/wave/build_tools/build_llvm.py --python-bindings` "
      "before configuring an offline build.")
  endif()

  set(_tlx_wave_llvm_stamp
    "${TLX_WAVE_SOURCE_DIR}/build/.tlx_wave_llvm_with_python.stamp")
  set(_tlx_wave_build_llvm_args
    "${Python3_EXECUTABLE}"
    "${TLX_WAVE_SOURCE_DIR}/build_tools/build_llvm.py"
    "--python-bindings")
  if(NOT "${TLX_WAVE_BUILD_JOBS}" STREQUAL "")
    list(APPEND _tlx_wave_build_llvm_args "--jobs" "${TLX_WAVE_BUILD_JOBS}")
  endif()
  add_custom_command(
    OUTPUT "${_tlx_wave_llvm_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E env
      "LLVM_INSTALL_DIR="
      "LLVM_DIR="
      "MLIR_DIR="
      "Clang_DIR="
      "LLD_DIR="
      ${_tlx_wave_build_llvm_args}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_tlx_wave_llvm_stamp}"
    WORKING_DIRECTORY "${TLX_WAVE_SOURCE_DIR}"
    DEPENDS
      "${TLX_WAVE_SOURCE_DIR}/build_tools/build_llvm.py"
      "${TLX_WAVE_SOURCE_DIR}/llvm-commit.txt"
    VERBATIM
    USES_TERMINAL)
  add_custom_target(TLXWavePinnedLLVM DEPENDS "${_tlx_wave_llvm_stamp}")
endif()

add_custom_target(TLXWaveStandaloneConfigure
  COMMAND "${CMAKE_COMMAND}" -E env
    "LLVM_INSTALL_DIR="
    "LLVM_DIR="
    "MLIR_DIR="
    "Clang_DIR="
    "LLD_DIR="
    "${CMAKE_COMMAND}"
    -S "${TLX_WAVE_SOURCE_DIR}"
    -B "${TLX_WAVE_STANDALONE_BUILD_DIR}"
    -G Ninja
    -U LLVM_DIR
    -U MLIR_DIR
    -U Clang_DIR
    -U LLD_DIR
    "-DLLVM_INSTALL_DIR=${TLX_WAVE_LLVM_INSTALL_DIR}"
    "-DWAVE_ENABLE_PYTHON_BINDINGS=ON"
  DEPENDS TLXWavePinnedLLVM
  VERBATIM
  USES_TERMINAL)

add_custom_target(TLXWaveSubmoduleRuntime ALL
  COMMAND "${CMAKE_COMMAND}" --build "${TLX_WAVE_STANDALONE_BUILD_DIR}"
    ${_tlx_wave_build_parallel_args}
  DEPENDS TLXWaveStandaloneConfigure
  VERBATIM
  USES_TERMINAL)
