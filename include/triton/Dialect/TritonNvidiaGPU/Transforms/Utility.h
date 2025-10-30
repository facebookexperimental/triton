#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

LogicalResult verifyBarrierType(Operation *op,
                                mlir::triton::gpu::MemDescType barrierType);
int allocateTMemWithInterval(
    DenseMap<Operation *, Interval<int>> &allocToIntervals,
    SmallVector<Operation *> &allocOrder);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
