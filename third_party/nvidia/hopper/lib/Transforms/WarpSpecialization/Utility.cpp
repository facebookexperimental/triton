#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

namespace tt = mlir::triton;
namespace mlir {

static void
normalizeWSPartitionIds(SmallVectorImpl<WSPartitionId> &partitionIds) {
  llvm::sort(partitionIds);
  partitionIds.erase(std::unique(partitionIds.begin(), partitionIds.end()),
                     partitionIds.end());
}

//===----------------------------------------------------------------------===//
// Helper functions for partition
//===----------------------------------------------------------------------===//

SmallVector<WSPartitionId> getWSPartitionIds(Operation *op) {
  SmallVector<WSPartitionId> partitionIds;
  if (auto attr =
          op->getAttrOfType<DenseI32ArrayAttr>(tt::gpu::kPartitionAttrName)) {
    for (WSPartitionId partitionId : attr.asArrayRef()) {
      partitionIds.push_back(partitionId);
    }
  }
  normalizeWSPartitionIds(partitionIds);
  return partitionIds;
}

bool hasWSPartitionId(Operation *op, WSPartitionId partitionId) {
  return llvm::is_contained(getWSPartitionIds(op), partitionId);
}

void setWSPartitionIds(Operation *op, ArrayRef<WSPartitionId> partitionIds) {
  if (partitionIds.empty())
    return;
  SmallVector<WSPartitionId> sortedWSPartitionIds(partitionIds.begin(),
                                                  partitionIds.end());
  normalizeWSPartitionIds(sortedWSPartitionIds);
  op->setAttr(tt::gpu::kPartitionAttrName,
              DenseI32ArrayAttr::get(op->getContext(), sortedWSPartitionIds));
}

void setWSPartitionIds(Operation *op, WSPartitionId partitionId) {
  SmallVector<WSPartitionId, 1> partitionIds{partitionId};
  setWSPartitionIds(op, partitionIds);
}

void labelParentOps(Operation *op) {
  auto partitionIds = mlir::getWSPartitionIds(op);
  auto parent = op->getParentOp();
  while (parent && !isa<triton::FuncOp>(parent)) {
    addWSPartitionIds(parent, partitionIds);
    parent = parent->getParentOp();
  }
}

SmallVector<WSPartitionId> getNestedWSPartitionIds(Operation *op) {
  SetVector<WSPartitionId> partitionIds;
  op->walk([&](Operation *curOp) {
    partitionIds.insert_range(getWSPartitionIds(curOp));
  });
  SmallVector<WSPartitionId> res = partitionIds.takeVector();
  llvm::sort(res);
  return res;
}

void addWSPartitionIds(Operation *op, ArrayRef<WSPartitionId> partitionIds) {
  auto partitionIdsVec = getWSPartitionIds(op);
  DenseSet<WSPartitionId> partitionIdsSet(partitionIdsVec.begin(),
                                          partitionIdsVec.end());
  for (auto a : partitionIds) {
    if (!partitionIdsSet.contains(a)) {
      partitionIdsVec.push_back(a);
    }
  }
  if (partitionIdsVec.size() > 0) {
    setWSPartitionIds(op, partitionIdsVec);
  }
}

void removeWSPartitionId(Operation *op, WSPartitionId partitionId) {
  auto origWSPartitionIds = getWSPartitionIds(op);
  llvm::erase(origWSPartitionIds, partitionId);
  if (origWSPartitionIds.empty())
    op->removeAttr(tt::gpu::kPartitionAttrName);
  else
    setWSPartitionIds(op, origWSPartitionIds);
}

void removeWSPartitionIds(Operation *op) {
  op->removeAttr(tt::gpu::kPartitionAttrName);
}

void copyLoopScheduleInfo(Operation *newOp, Operation *oldOp) {
  // This assignment is optional because we may call this code
  // from sections outside the innermost loop.
  if (oldOp->hasAttr(tt::kLoopStageAttrName))
    newOp->setAttr(tt::kLoopStageAttrName,
                   oldOp->getAttr(tt::kLoopStageAttrName));
  if (oldOp->hasAttr(tt::kLoopClusterAttrName))
    newOp->setAttr(tt::kLoopClusterAttrName,
                   oldOp->getAttr(tt::kLoopClusterAttrName));
}
} // namespace mlir
