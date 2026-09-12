#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::nvws {

namespace semaphore {

std::optional<PartitionWsTagIds> getPartitionWsTagIds(Operation *op) {
  std::optional<PartitionWsTagIds> partitionWsTagIds;
  if (hasPartition(op)) {
    partitionWsTagIds =
        PartitionWsTagIds{std::nullopt, triton::gpu::getPartitionIds(op)};
    if (auto wsTag = getWarpSpecializeTag(op)) {
      partitionWsTagIds->wsTag = *wsTag;
    }
  }
  return partitionWsTagIds;
}

void assignStageCluster(Operation *op,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        StageCluster stageCluster, OpBuilder &builder) {
  if (partitionWsTagIds) {
    setPartition(op, partitionWsTagIds->partitionIds);
    if (auto wsTag = partitionWsTagIds->wsTag) {
      setWarpSpecializeTag(op, *wsTag);
    }
    setStageCluster(builder, op, stageCluster);
  }
}

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

} // namespace semaphore

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return LocalAllocOp::create(builder, loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return TMEMAllocOp::create(builder, loc, memDescType, src);
  }
}

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return ArefCreateOp::create(builder, loc, arefTy, allocOps);
}

int getArefDepth(MemDescType bufTy) {
  auto shape = bufTy.getShape();
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding())
             ? 1
             : shape[0];
}

MemDescType getArefViewBufferType(MemDescType bufTy) {
  auto isScalesEnc =
      isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding());
  auto shape = bufTy.getShape();
  return gpu::MemDescType::get(isScalesEnc ? shape : shape.drop_front(),
                               bufTy.getElementType(), bufTy.getEncoding(),
                               bufTy.getMemorySpace(),
                               /*mutableMemory*/ true,
                               /*allocShape=*/bufTy.getAllocShape());
}

MemDescType getArefMultiBufferedType(MemDescType bufTy, int depth) {
  auto shape = bufTy.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(bufTy.getEncoding()))
    bufferShape.insert(bufferShape.begin(), depth);
  return gpu::MemDescType::get(bufferShape, bufTy.getElementType(),
                               bufTy.getEncoding(), bufTy.getMemorySpace(),
                               /*mutableMemory*/ true);
}

scf::ForOp getOuterWSLoop(scf::ForOp innerFor) {
  auto wsLoop = innerFor;
  while (wsLoop && !wsLoop->hasAttr(triton::kWarpSpecializeAttrName)) {
    wsLoop = wsLoop->getParentOfType<scf::ForOp>();
  }
  return wsLoop;
}

} // namespace mlir::triton::nvws
