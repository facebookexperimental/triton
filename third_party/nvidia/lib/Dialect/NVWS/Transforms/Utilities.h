#ifndef NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
#define NVIDIA_NVWS_TRANSFORMS_UTILITY_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir::triton::nvws {

namespace semaphore {

struct PartitionWsTagIds {
  std::optional<int> wsTag;
  SetVector<int> partitionIds;
};

std::optional<PartitionWsTagIds> getPartitionWsTagIds(Operation *op);

void assignStageCluster(Operation *op,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        gpu::StageCluster stageCluster, OpBuilder &builder);

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs);

} // namespace semaphore

Operation *createAlloc(OpBuilder &builder, Location loc,
                       gpu::MemDescType memDescType, Value src);

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc);

template <typename Range>
inline std::optional<int> findValuePosInRange(const Range &range,
                                              mlir::Value v) {
  for (auto [pos, arg] : llvm::enumerate(range)) {
    if (arg == v)
      return pos;
  }
  return {};
}

#if 0
struct PartitionId : std::pair<int, int> {
  PartitionId(int index, int tag) : std::pair<int, int>(index, tag) {}
  int &index() { return first; }
  int &tag() { return second; }
};

std::optional<PartitionId> getPartitionId(Operation *op);
#endif

gpu::MemDescType getArefViewBufferType(gpu::MemDescType arefBufType);
gpu::MemDescType getArefMultiBufferedType(gpu::MemDescType arefBufType,
                                          int depth);
int getArefDepth(gpu::MemDescType bufTy);

scf::ForOp getOuterWSLoop(scf::ForOp innerFor);
} // namespace mlir::triton::nvws

#endif // NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
