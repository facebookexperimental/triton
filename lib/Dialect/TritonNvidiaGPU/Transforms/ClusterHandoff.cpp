#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterHandoff.h"

#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

namespace mlir::triton::nvidia_gpu {

namespace ttg = mlir::triton::gpu;

Value createPersistentMBarrierAlloc(ImplicitLocOpBuilder &builder,
                                    int arriveCount) {
  Value alloc = createScalarAlloc(builder, builder.getI64Type(), 1);
  Value barrier = createSingleBufferView(builder, alloc, 0);
  InitBarrierOp::create(builder, barrier, arriveCount);
  return alloc;
}

Value captureInWarpPartition(Value value, Operation *user) {
  auto wsOp = user->getParentOfType<ttg::WarpSpecializeOp>();
  if (!wsOp || wsOp.getDefaultRegion().isAncestor(user->getParentRegion()))
    return value;

  Value captured;
  auto partOp = wsOp.getPartitionOp();
  partOp->insertOperands(partOp.getNumOperands(), value);
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg = region->addArgument(value.getType(), value.getLoc());
    if (region->isAncestor(user->getParentRegion()))
      captured = arg;
  }
  assert(captured && "operation not found in a warp partition region");
  return captured;
}

ArriveBarrierOp createRemoteMBarrierArrive(OpBuilder &builder, Location loc,
                                           Value barrier, Value rank,
                                           Value pred) {
  auto localTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      localTy.getShape(), localTy.getElementType(), localTy.getEncoding(),
      SharedClusterMemorySpaceAttr::get(builder.getContext()),
      localTy.getMutableMemory(), localTy.getAllocShape());
  Value remote =
      MapToRemoteBufferOp::create(builder, loc, remoteTy, barrier, rank);
  return pred ? ArriveBarrierOp::create(builder, loc, remote, /*count=*/1, pred)
              : ArriveBarrierOp::create(builder, loc, remote, /*count=*/1);
}

} // namespace mlir::triton::nvidia_gpu
