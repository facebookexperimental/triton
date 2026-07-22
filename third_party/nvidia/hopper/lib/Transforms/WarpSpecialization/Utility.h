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

typedef int WSPartitionId;

// Retrieves the partition ids of the given operation.
SmallVector<WSPartitionId> getWSPartitionIds(Operation *op);

// Checks if the given operation has the given partition id.
bool hasWSPartitionId(Operation *op, WSPartitionId partitionId);

// Sets the partition ids of the given operation.
void setWSPartitionIds(Operation *op, ArrayRef<WSPartitionId> partitionIds);
void setWSPartitionIds(Operation *op, WSPartitionId partitionId);

// Propagate the partition ids of the given operation to its parent ops.
void labelParentOps(Operation *op);

// Retrieves the partition IDs of all operations nested within the given
// operation, including the operation itself.
SmallVector<WSPartitionId> getNestedWSPartitionIds(Operation *op);

// Adds the given partition ids to the given operation.
void addWSPartitionIds(Operation *op, ArrayRef<WSPartitionId> partitionIds);

// Removes the given partition id from the given operation.
void removeWSPartitionId(Operation *op, WSPartitionId partitionId);

// Removes all partition ids from the given operation.
void removeWSPartitionIds(Operation *op);

struct LoopScheduleInfo {
  IntegerAttr stage;
  IntegerAttr cluster;
};

class OpBuilderWithPartitionIds : public OpBuilder {
public:
  OpBuilderWithPartitionIds(MLIRContext *context) : OpBuilder(context) {}

  explicit OpBuilderWithPartitionIds(Operation *op) : OpBuilder(op) {
    setPartitionIdsFromOp(op);
    setLoopScheduleInfoFromOp(op);
  }

  void setPartitionIdsFromArray(ArrayRef<WSPartitionId> newWSPartitionIds) {
    partitionIds = SmallVector<WSPartitionId>(newWSPartitionIds.begin(),
                                              newWSPartitionIds.end());
  }

  void setPartitionIdsFromOp(Operation *op) {
    setPartitionIdsFromArray(mlir::getWSPartitionIds(op));
  }

  void setPartitionIdsFromValueUsers(Value value) {
    SetVector<WSPartitionId> partitionIdSet;
    for (Operation *user : value.getUsers())
      for (WSPartitionId partitionId : mlir::getWSPartitionIds(user))
        partitionIdSet.insert(partitionId);
    setPartitionIdsFromArray(partitionIdSet.getArrayRef());
  }

  SmallVector<WSPartitionId> getWSPartitionIds() { return partitionIds; }

  template <typename OpTy, typename... Args>
  OpTy createWithPartitionIds(Args &&...args) {
    OpTy op = OpTy::create(*this, std::forward<Args>(args)...);
    if (!partitionIds.empty())
      setWSPartitionIds(op, partitionIds);
    setOpLoopScheduleInfo(op);
    return op;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = createWithPartitionIds<OpTy>(std::forward<Args>(args)...);
    setOpLoopScheduleInfo(op);
    return op;
  }

  // Sets the loop schedule info (loop.stage, loop.cluster) of future
  // createWithPartitionIds operations based on the `loop.stage` and
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
  // future createWithPartitionIds operations.
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

  SmallVector<WSPartitionId> partitionIds;
  LoopScheduleInfo loopScheduleInfo = {nullptr, nullptr};
};

// Copy any pipeline info (loop.stage, loop.cluster) from
// the oldOp to the newOp. This is needed for any operation
// where the dependency exists without a direct "user".
void copyLoopScheduleInfo(Operation *newOp, Operation *oldOp);

// Append a suffix to the innermost NameLoc in a Location hierarchy.
// Handles NameLoc, CallSiteLoc wrapping, and falls back to creating a new
// NameLoc if no NameLoc is found.
static Location appendToNameLoc(Location loc, StringRef suffix,
                                MLIRContext *ctx) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    auto newName = (nameLoc.getName().getValue() + suffix).str();
    return NameLoc::get(StringAttr::get(ctx, newName), nameLoc.getChildLoc());
  }
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    auto newCallee = appendToNameLoc(callSiteLoc.getCallee(), suffix, ctx);
    return CallSiteLoc::get(newCallee, callSiteLoc.getCaller());
  }
  // No NameLoc found — wrap with a new NameLoc.
  return NameLoc::get(StringAttr::get(ctx, suffix), loc);
}

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
