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

class OpBuilderWithAsyncTaskIds : public OpBuilder {
public:
  OpBuilderWithAsyncTaskIds(MLIRContext *context) : OpBuilder(context) {}

  explicit OpBuilderWithAsyncTaskIds(Operation *op) : OpBuilder(op) {
    setAsyncTaskIdsFromOp(op);
    setLoopScheduleInfo(op);
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
  void setLoopScheduleInfoFromTuple(
      std::tuple<std::optional<IntegerAttr>, std::optional<IntegerAttr>>
          loopScheduleInfo) {
    loopStage = std::get<0>(loopScheduleInfo);
    loopCluster = std::get<1>(loopScheduleInfo);
  }

  void setLoopScheduleInfo(Operation *op) {
    std::optional<IntegerAttr> nextLoopStage = std::nullopt;
    std::optional<IntegerAttr> nextLoopCluster = std::nullopt;
    if (op->hasAttr(tt::kLoopStageAttrName)) {
      nextLoopStage = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
    }
    if (op->hasAttr(tt::kLoopClusterAttrName)) {
      nextLoopCluster =
          op->getAttrOfType<IntegerAttr>(tt::kLoopClusterAttrName);
    }
    setLoopScheduleInfoFromTuple({nextLoopStage, nextLoopCluster});
  }

  // Clears the loop schedule info (loop.stage, loop.cluster) for
  // future createWithAsyncTaskIds operations.
  void clearLoopScheduleInfo() {
    loopStage = std::nullopt;
    loopCluster = std::nullopt;
  }

  std::tuple<std::optional<IntegerAttr>, std::optional<IntegerAttr>>
  getLoopScheduleInfo() {
    return {loopStage, loopCluster};
  }

private:
  void setOpLoopScheduleInfo(Operation *op) {
    if (loopStage.has_value()) {
      op->setAttr(tt::kLoopStageAttrName, loopStage.value());
    }
    if (loopCluster.has_value()) {
      op->setAttr(tt::kLoopClusterAttrName, loopCluster.value());
    }
  }

  SmallVector<AsyncTaskId> asyncTaskIds;
  std::optional<IntegerAttr> loopStage = std::nullopt;
  std::optional<IntegerAttr> loopCluster = std::nullopt;
};

// Copy any pipeline info (loop.stage, loop.cluster) from
// the oldOp to the newOp. This is needed for any operation
// where the dependency exists without a direct "user".
void copyLoopScheduleInfo(Operation *newOp, Operation *oldOp);

} // namespace mlir
#endif // NV_DIALECT_HOPPER_TRANSFORMS_UTILITY_H_
