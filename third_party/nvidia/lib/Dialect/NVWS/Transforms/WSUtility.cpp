#include "WSUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

namespace mlir::triton::nvws {

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      if (asyncTaskIds.empty() || asyncTaskIds.back() != asyncTaskId)
        asyncTaskIds.push_back(asyncTaskId);
    }
  } else if (auto attr =
                 op->getAttrOfType<DenseI32ArrayAttr>(gpu::kPartitionAttrName)) {
    asyncTaskIds.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  }
  return asyncTaskIds;
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  if (asyncTaskIds.empty())
    return;
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                             asyncTaskIds.end());
  llvm::sort(sortedAsyncTaskIds);
  op->setAttr("async_task_id",
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks) {
  SmallVector<AsyncTaskId> asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<AsyncTaskId> asyncTasksSet(asyncTasksVec.begin(),
                                      asyncTasksVec.end());
  for (AsyncTaskId asyncTask : asyncTasks) {
    if (!asyncTasksSet.contains(asyncTask))
      asyncTasksVec.push_back(asyncTask);
  }
  if (!asyncTasksVec.empty())
    setAsyncTaskIds(op, asyncTasksVec);
}

void OpBuilderWithAsyncTaskIds::setLoopScheduleInfoFromOp(Operation *op) {
  IntegerAttr nextLoopStage = nullptr;
  IntegerAttr nextLoopCluster = nullptr;
  if (op->hasAttr(triton::kLoopStageAttrName))
    nextLoopStage = op->getAttrOfType<IntegerAttr>(triton::kLoopStageAttrName);
  if (op->hasAttr(triton::kLoopClusterAttrName))
    nextLoopCluster =
        op->getAttrOfType<IntegerAttr>(triton::kLoopClusterAttrName);
  setLoopScheduleInfoFromInfo({nextLoopStage, nextLoopCluster});
}

void OpBuilderWithAsyncTaskIds::setOpLoopScheduleInfo(Operation *op) {
  if (loopScheduleInfo.stage)
    op->setAttr(triton::kLoopStageAttrName, loopScheduleInfo.stage);
  if (loopScheduleInfo.cluster)
    op->setAttr(triton::kLoopClusterAttrName, loopScheduleInfo.cluster);
}

Location appendToNameLoc(Location loc, StringRef suffix, MLIRContext *ctx) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    auto newName = (nameLoc.getName().getValue() + suffix).str();
    return NameLoc::get(StringAttr::get(ctx, newName), nameLoc.getChildLoc());
  }
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    auto newCallee = appendToNameLoc(callSiteLoc.getCallee(), suffix, ctx);
    return CallSiteLoc::get(newCallee, callSiteLoc.getCaller());
  }
  return NameLoc::get(StringAttr::get(ctx, suffix), loc);
}

} // namespace mlir::triton::nvws
