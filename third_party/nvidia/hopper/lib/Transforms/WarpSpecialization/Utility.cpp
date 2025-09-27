#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"

namespace tt = mlir::triton;
namespace mlir {

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      // TODO(Arda): Remove this check once we figure out why we have duplicate
      // async task ids
      if (asyncTaskIds.empty() ||
          asyncTaskIds[asyncTaskIds.size() - 1] != asyncTaskId)
        asyncTaskIds.push_back(asyncTaskId);
    }
  } else if (auto attr = op->getAttrOfType<IntegerAttr>("ttg.partition")) {
    int64_t idx = attr.getInt();
    if (idx >= 0)
      asyncTaskIds.push_back(idx);
  }
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
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_task_id",
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
    op->removeAttr("async_task_id");
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) { op->removeAttr("async_task_id"); }

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

// Barrier naming utilities for creating descriptive NameLocs
std::string getChannelNameFromProducerConsumers(int producerId,
                                                ArrayRef<int> consumerIds) {
  std::string name = "ch_prod" + std::to_string(producerId);
  for (int consumerId : consumerIds) {
    name += "_cons" + std::to_string(consumerId);
  }
  return name;
}

std::string getBarrierName(llvm::StringRef barrierType,
                           llvm::StringRef bufferType,
                           llvm::StringRef channelName) {
  return (barrierType + "_" + bufferType + "_" + channelName).str();
}

std::string getInitBarrierName(llvm::StringRef bufferType, unsigned bufferIdx) {
  return ("init_barrier_" + bufferType + "_buf" + std::to_string(bufferIdx))
      .str();
}

// NameLoc creation utilities for barriers
Location createBarrierNameLoc(OpBuilder &builder, Operation *op,
                              llvm::StringRef barrierType,
                              llvm::StringRef bufferType) {
  auto loc = op->getLoc();

  std::string channelName = "unknown";
  if (auto attr = op->getAttrOfType<StringAttr>("channel_name")) {
    channelName = attr.getValue().str();
  }
  std::string barrierName =
      getBarrierName(barrierType, bufferType, channelName);
  return mlir::NameLoc::get(builder.getStringAttr(barrierName), loc);
}

Location createInitBarrierNameLoc(OpBuilder &builder, Location baseLoc,
                                  llvm::StringRef bufferType,
                                  unsigned bufferIdx) {
  std::string barrierName = getInitBarrierName(bufferType, bufferIdx);
  return mlir::NameLoc::get(builder.getStringAttr(barrierName), baseLoc);
}

// Buffer naming utilities for creating descriptive NameLocs
std::string getBufferName(llvm::StringRef bufferType,
                          llvm::StringRef channelName, unsigned numBuffers) {
  std::string name = bufferType.str() + "_" + channelName.str();

  // Add number of buffers for pipelining
  name += "_nbuf" + std::to_string(numBuffers);

  return name;
}

Location createBufferNameLoc(OpBuilder &builder, Location baseLoc,
                             llvm::StringRef bufferType,
                             llvm::StringRef channelName, unsigned numBuffers) {
  std::string bufferName = getBufferName(bufferType, channelName, numBuffers);
  return mlir::NameLoc::get(builder.getStringAttr(bufferName), baseLoc);
}

} // namespace mlir
