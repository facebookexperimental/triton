// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Hardware-agnostic LatencyModel utilities. The abstract interface lives in
// LatencyModel.h; the NVIDIA/Blackwell implementation is in NVLatencyModel.cpp.
// This file is part of the backend-neutral modulo-scheduling core library
// (TritonGPUModuloCore), so it must NOT depend on any backend dialect.

#include "LatencyModel.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::gpu {

llvm::StringRef getPipelineName(HWPipeline pipeline) {
  switch (pipeline) {
  case HWPipeline::TMA:
    return "TMA";
  case HWPipeline::TC:
    return "TC";
  case HWPipeline::CUDA:
    return "CUDA";
  case HWPipeline::SFU:
    return "SFU";
  case HWPipeline::MFMA:
    return "MFMA";
  case HWPipeline::LDS:
    return "LDS";
  case HWPipeline::GLOBAL:
    return "GLOBAL";
  case HWPipeline::VALU:
    return "VALU";
  case HWPipeline::NONE:
    return "NONE";
  }
  llvm_unreachable("unknown pipeline");
}

} // namespace mlir::triton::gpu
