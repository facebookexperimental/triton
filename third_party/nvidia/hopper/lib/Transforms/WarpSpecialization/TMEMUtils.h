#ifndef NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
// Generate code to reintepret the ttng::TMEMAllocOp by converting
// the provided dimension to 1. For example, this could interpret
// a [128, 128] TMEMAllocOp as [128, 1] with dim=1.
ttg::MemDescReinterpretOp
reinterpretTMEMBufferDroppingDim(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                                 size_t dim);
} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
