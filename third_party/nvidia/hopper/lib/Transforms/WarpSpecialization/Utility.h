#ifndef NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace tt = mlir::triton;
namespace mlir {

typedef int AsyncTaskId;

// Retrieves the async task ids of the given operation.
SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op);

// Checks if the given operation has the given async task id.
bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Sets the async task ids of the given operation.
void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds);

// Propagate the async task ids of the given operation to its parent ops.
void labelParentOps(Operation *op);

// Retrieves the async task IDs of all operations nested within the given
// operation, including the operation itself.
SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op);

// Adds the given async task ids to the given operation.
void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks);

// Removes the given async task id from the given operation.
void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId);

// Removes all async task ids from the given operation.
void removeAsyncTaskIds(Operation *op);

struct LoopScheduleInfo {
  IntegerAttr stage;
  IntegerAttr cluster;
};

class OpBuilderWithAsyncTaskIds : public OpBuilder {
public:
  OpBuilderWithAsyncTaskIds(MLIRContext *context) : OpBuilder(context) {}

  explicit OpBuilderWithAsyncTaskIds(Operation *op) : OpBuilder(op) {
    setAsyncTaskIdsFromOp(op);
    setLoopScheduleInfoFromOp(op);
  }

  void setAsynTaskIdsFromArray(ArrayRef<AsyncTaskId> newAsyncTaskIds) {
    asyncTaskIds = SmallVector<AsyncTaskId>(newAsyncTaskIds.begin(),
                                            newAsyncTaskIds.end());
  }

  void setAsyncTaskIdsFromOp(Operation *op) {
    setAsynTaskIdsFromArray(mlir::getAsyncTaskIds(op));
  }

  void setAsyncTaskIdsFromValueUsers(Value value) {
    SetVector<AsyncTaskId> asyncTaskIdSet;
    for (Operation *user : value.getUsers())
      for (AsyncTaskId asyncTaskId : mlir::getAsyncTaskIds(user))
        asyncTaskIdSet.insert(asyncTaskId);
    setAsynTaskIdsFromArray(asyncTaskIdSet.getArrayRef());
  }

  SmallVector<AsyncTaskId> getAsyncTaskIds() { return asyncTaskIds; }

  template <typename OpTy, typename... Args>
  OpTy createWithAsyncTaskIds(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
    if (!asyncTaskIds.empty())
      setAsyncTaskIds(op, asyncTaskIds);
    setOpLoopScheduleInfo(op);
    return op;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = createWithAsyncTaskIds<OpTy>(std::forward<Args>(args)...);
    setOpLoopScheduleInfo(op);
    return op;
  }

  // Sets the loop schedule info (loop.stage, loop.cluster) of future
  // createWithAsyncTaskIds operations based on the `loop.stage` and
  // `loop.cluster` attributes of the given operation.
  void setLoopScheduleInfoFromInfo(LoopScheduleInfo newLoopScheduleInfo) {
    loopScheduleInfo = newLoopScheduleInfo;
  }

  void setLoopScheduleInfoFromOp(Operation *op) {
    IntegerAttr nextLoopStage = nullptr;
    IntegerAttr nextLoopCluster = nullptr;
    if (op->hasAttr(tt::kLoopStageAttrName)) {
      nextLoopStage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    }
    if (op->hasAttr(tt::kLoopClusterAttrName)) {
      nextLoopCluster =
          op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
    }
    setLoopScheduleInfoFromInfo({nextLoopStage, nextLoopCluster});
  }

  // Clears the loop schedule info (loop.stage, loop.cluster) for
  // future createWithAsyncTaskIds operations.
  void clearLoopScheduleInfo() { loopScheduleInfo = {nullptr, nullptr}; }

  LoopScheduleInfo getLoopScheduleInfo() { return loopScheduleInfo; }

private:
  void setOpLoopScheduleInfo(Operation *op) {
    if (loopScheduleInfo.stage) {
      op->setAttr(tt::kLoopStageAttrName, loopScheduleInfo.stage);
    }
    if (loopScheduleInfo.cluster) {
      op->setAttr(tt::kLoopClusterAttrName, loopScheduleInfo.cluster);
    }
  }

  SmallVector<AsyncTaskId> asyncTaskIds;
  LoopScheduleInfo loopScheduleInfo = {nullptr, nullptr};
};

// Copy any pipeline info (loop.stage, loop.cluster) from
// the oldOp to the newOp. This is needed for any operation
// where the dependency exists without a direct "user".
void copyLoopScheduleInfo(Operation *newOp, Operation *oldOp);

// Extract the outermost NameLoc name, unwrapping CallSiteLoc.
static std::string getOutermostNameFromLoc(Location loc) {
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc))
    return getOutermostNameFromLoc(callSiteLoc.getCallee());
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return "";
}

// Replace the outermost NameLoc name (or wrap with one), stripping any
// intermediate NameLoc layers. Preserves CallSiteLoc wrapping and the
// innermost non-NameLoc child (FileLineColLoc etc.).
static Location replaceOutermostNameLoc(Location loc, StringRef name) {
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    auto newCallee = replaceOutermostNameLoc(callSiteLoc.getCallee(), name);
    return CallSiteLoc::get(newCallee, callSiteLoc.getCaller());
  }
  Location child = loc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    child = nameLoc.getChildLoc();
    while (auto inner = dyn_cast<NameLoc>(child))
      child = inner.getChildLoc();
  }
  return NameLoc::get(StringAttr::get(loc.getContext(), name), child);
}

} // namespace mlir
#endif // NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
