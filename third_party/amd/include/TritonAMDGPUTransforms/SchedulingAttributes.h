#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_SCHEDULINGATTRIBUTES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_SCHEDULINGATTRIBUTES_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::AMD {

constexpr llvm::StringLiteral kIntraWavePipelineWindowAttr =
    "triton.intra_wave_pipeline.window";

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_SCHEDULINGATTRIBUTES_H_
