#!/bin/bash
# Source this script so LD_LIBRARY_PATH is available to subsequent commands.
set -euo pipefail

TORCH_NVIDIA_LIB_DIR=$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent.parent / "nvidia" / "cu13" / "lib" )')

if [ ! -f "${TORCH_NVIDIA_LIB_DIR}/libcublas.so" ]; then
  for cublas_library in "${TORCH_NVIDIA_LIB_DIR}"/libcublas.so.*; do
    if [ -f "${cublas_library}" ]; then
      ln -sf "${cublas_library}" "${TORCH_NVIDIA_LIB_DIR}/libcublas.so"
      break
    fi
  done
  for cublas_library in "${TORCH_NVIDIA_LIB_DIR}"/libcublasLt.so.*; do
    if [ -f "${cublas_library}" ]; then
      ln -sf "${cublas_library}" "${TORCH_NVIDIA_LIB_DIR}/libcublasLt.so"
      break
    fi
  done
fi

if [ ! -f "${TORCH_NVIDIA_LIB_DIR}/libcublas.so" ]; then
  echo "ERROR: Neither libcublas.so nor libcublas.so.* exists in ${TORCH_NVIDIA_LIB_DIR}" >&2
  exit 1
fi

if [ ! -f "${TORCH_NVIDIA_LIB_DIR}/libcublasLt.so" ]; then
  echo "ERROR: Neither libcublasLt.so nor libcublasLt.so.* exists in ${TORCH_NVIDIA_LIB_DIR}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${TORCH_NVIDIA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
