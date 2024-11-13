
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <fstream>

namespace mlir {

namespace ttg = triton::gpu;

namespace {

bool knownSafeToIgnoreRegion(Operation *op) {
  return isa<triton::ReduceOp>(op);
}

// Assigns `dependentSet` and returns ok if the analysis is successful.
// We do not support dependency analysis across load/store, thus a failure will
// be returned if encountering such cases.
LogicalResult getDependentPointers(Value ptr, DenseSet<Value> &dependentSet,
                                   DenseSet<Value> &processedSet) {
  // early return if processed
  if (!processedSet.insert(ptr).second)
    return success();

  if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
    if (!blockArg.getOwner()->isEntryBlock())
      return failure();
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg.getArgNumber() >= forOp.getNumInductionVars()) {
        if (failed(getDependentPointers(forOp.getTiedLoopInit(blockArg)->get(),
                                        dependentSet, processedSet)))
          return failure();

        unsigned operandIdx =
            blockArg.getArgNumber() - forOp.getNumInductionVars();
        return getDependentPointers(
            forOp.getBody()->getTerminator()->getOperand(operandIdx),
            dependentSet, processedSet);
      }
    } else if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
      dependentSet.insert(ptr);
      return success();
    }
    // unknown ops, return failure for correctness.
    return failure();
  }

  auto definingOp = ptr.getDefiningOp();
  assert(definingOp);
  if (auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    return getDependentPointers(makeTensorPtrOp.getBase(), dependentSet,
                                processedSet);
  } else if (auto advanceOp = ptr.getDefiningOp<triton::AdvanceOp>()) {
    return getDependentPointers(advanceOp.getPtr(), dependentSet, processedSet);
  } else if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
    return getDependentPointers(addPtrOp.getPtr(), dependentSet, processedSet);
  } else if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    unsigned idx = cast<OpResult>(ptr).getResultNumber();
    return getDependentPointers(
        forOp.getBody()->getTerminator()->getOperand(idx), dependentSet,
        processedSet);
  } else if (auto ifOp = ptr.getDefiningOp<scf::IfOp>()) {
    unsigned idx = cast<OpResult>(ptr).getResultNumber();
    if (ifOp.elseBlock() &&
        failed(getDependentPointers(ifOp.elseYield()->getOperand(idx),
                                    dependentSet, processedSet)))
      return failure();
    return getDependentPointers(ifOp.thenYield()->getOperand(idx), dependentSet,
                                processedSet);
  } else if (!definingOp->getNumRegions() ||
             knownSafeToIgnoreRegion(definingOp)) {
    for (Value operand : definingOp->getOperands())
      if (failed(getDependentPointers(operand, dependentSet, processedSet)))
        return failure();
    return success();
  }
  // unknown ops, return failure for correctness.
  return failure();
}

} // namespace

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("async_task_id"))
    for (AsyncTaskId asyncTaskId : attr.getValues<AsyncTaskId>())
      asyncTaskIds.push_back(asyncTaskId);
  return asyncTaskIds;
}

bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  for (AsyncTaskId candidate : getAsyncTaskIds(op))
    if (candidate == asyncTaskId)
      return true;
  return false;
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(), asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_task_id", DenseIntElementsAttr::get(vecTy, sortedAsyncTaskIds));
}

SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op) {
  SetVector<AsyncTaskId> asyncTaskIds;
  op->walk([&](Operation *curOp) {
    for (AsyncTaskId asyncTaskId : getAsyncTaskIds(curOp))
      asyncTaskIds.insert(asyncTaskId);
  });
  SmallVector<AsyncTaskId> res(asyncTaskIds.begin(), asyncTaskIds.end());
  llvm::sort(res);
  return res;
}

void addAsyncTaskIds(Operation *op, ArrayRef<int> asyncTasks) {
  auto asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<int> asyncTasksSet(asyncTasksVec.begin(), asyncTasksVec.end());
  for (int a : asyncTasks) {
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
  auto end = std::remove(origAsyncTaskIds.begin(), origAsyncTaskIds.end(), asyncTaskId);
  origAsyncTaskIds.erase(end, origAsyncTaskIds.end());
  if (origAsyncTaskIds.empty())
    op->removeAttr("async_task_id");
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) {
  op->removeAttr("async_task_id");
}
//===----------------------------------------------------------------------===//
// Implementations for general auto WS
//===----------------------------------------------------------------------===//


} // namespace mlir
