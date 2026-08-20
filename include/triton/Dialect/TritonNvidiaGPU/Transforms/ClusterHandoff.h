#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERHANDOFF_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_CLUSTERHANDOFF_H_

#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

Value createPersistentMBarrierAlloc(ImplicitLocOpBuilder &builder,
                                    int arriveCount);

Value captureInWarpPartition(Value value, Operation *user);

ArriveBarrierOp createRemoteMBarrierArrive(OpBuilder &builder, Location loc,
                                           Value barrier, Value rank,
                                           Value pred = {});

} // namespace mlir::triton::nvidia_gpu

#endif
