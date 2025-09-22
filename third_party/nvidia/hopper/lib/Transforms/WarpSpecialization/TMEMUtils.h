#ifndef NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
// Generate code to reintepret the ttng::TMEMAllocOp by converting
// the N dimension to the given value that must be less the current size.
ttg::MemDescReinterpretOp
sliceAndReinterpretTMEMBuffer(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                              int offset, size_t blockN);
// Create a TMEM descriptor that is sufficient for the given
// TMEM Allocation Operator.
ttg::MemDescType createTMEMDesc(OpBuilder &builder, Type inputType,
                                int64_t blockM, int64_t blockN);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_TMEMUTILS_H_
