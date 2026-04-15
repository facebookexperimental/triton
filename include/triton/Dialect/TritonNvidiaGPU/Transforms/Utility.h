#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Value.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::nvidia_gpu {

LogicalResult verifyBarrierType(Operation *op,
                                mlir::triton::gpu::MemDescType barrierType);
int allocateTMemWithInterval(
    DenseMap<Operation *, Interval<int>> &allocToIntervals,
    SmallVector<Operation *> &allocOrder);

/// Like mlir::getBackwardSlice, but also traverses through
/// WarpSpecializePartitionsOp block arguments by mapping them back to the
/// corresponding WarpSpecializeOp operands.
void getBackwardSliceWithWS(Value target,
                            SetVector<Operation *> *backwardSlice);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
