#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

namespace tt = mlir::triton;
namespace mlir {

static void normalizeAsyncTaskIds(SmallVectorImpl<AsyncTaskId> &asyncTaskIds) {
  llvm::sort(asyncTaskIds);
  asyncTaskIds.erase(std::unique(asyncTaskIds.begin(), asyncTaskIds.end()),
                     asyncTaskIds.end());
}

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kAsyncTaskIdAttrName)) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      asyncTaskIds.push_back(asyncTaskId);
    }
  } else if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(
                 tt::gpu::kPartitionAttrName)) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      asyncTaskIds.push_back(asyncTaskId);
    }
  }
  normalizeAsyncTaskIds(asyncTaskIds);
  return asyncTaskIds;
}

bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  return llvm::is_contained(getAsyncTaskIds(op), asyncTaskId);
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  if (asyncTaskIds.empty())
    return;
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                              asyncTaskIds.end());
  normalizeAsyncTaskIds(sortedAsyncTaskIds);
  op->setAttr(kAsyncTaskIdAttrName,
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

void labelParentOps(Operation *op) {
  auto asyncTaskIds = mlir::getAsyncTaskIds(op);
  auto parent = op->getParentOp();
  while (parent && !isa<triton::FuncOp>(parent)) {
    addAsyncTaskIds(parent, asyncTaskIds);
    parent = parent->getParentOp();
  }
}

SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op) {
  SetVector<AsyncTaskId> asyncTaskIds;
  op->walk([&](Operation *curOp) {
    asyncTaskIds.insert_range(getAsyncTaskIds(curOp));
  });
  SmallVector<AsyncTaskId> res = asyncTaskIds.takeVector();
  llvm::sort(res);
  return res;
}

void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks) {
  auto asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<AsyncTaskId> asyncTasksSet(asyncTasksVec.begin(),
                                      asyncTasksVec.end());
  for (auto a : asyncTasks) {
    if (!asyncTasksSet.contains(a)) {
      asyncTasksVec.push_back(a);
    }
  }
  if (asyncTasksVec.size() > 0) {
    setAsyncTaskIds(op, asyncTasksVec);
  }
}

void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  auto origAsyncTaskIds = getAsyncTaskIds(op);
  llvm::erase(origAsyncTaskIds, asyncTaskId);
  if (origAsyncTaskIds.empty())
    op->removeAttr(kAsyncTaskIdAttrName);
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) { op->removeAttr(kAsyncTaskIdAttrName); }

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
