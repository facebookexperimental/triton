#ifndef NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
// Generate code to reintepret the ttng::TMEMAllocOp by converting
// the N dimension to 1 at the specified offset.
ttg::MemDescReinterpretOp
sliceAndReinterpretTMEMBuffer(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                              int offset);
} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
