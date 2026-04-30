#ifndef NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
#define NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// Check if two WS barriers can be safely swapped by verifying their
// channelGraph sets are disjoint. Returns false if either barrier lacks
// a channelGraph constraint (conservative).
inline bool canAdvanceWSBarrier(std::optional<DictionaryAttr> constraintsA,
                                std::optional<DictionaryAttr> constraintsB) {
  if (!constraintsA || !constraintsB)
    return false;
  auto graphA = constraintsA->getAs<DenseI32ArrayAttr>("channelGraph");
  auto graphB = constraintsB->getAs<DenseI32ArrayAttr>("channelGraph");
  if (!graphA || !graphB)
    return false;
  DenseSet<int> setA(graphA.asArrayRef().begin(), graphA.asArrayRef().end());
  for (int id : graphB.asArrayRef())
    if (setA.contains(id))
      return false;
  return true;
}

// Push WS arrive barriers as far down as possible within a block.
// An arrive can freely move past non-barrier ops (it just delays the signal).
// An arrive can move past another arrive (always safe).
// An arrive can move past a wait only if canAdvanceWSBarrier says their
// channel graphs are disjoint.
inline bool sinkWSArrives(Block &block) {
  bool changed = false;
  SmallVector<ArriveBarrierOp> arrives;
  for (auto &op : block)
    if (auto arrive = dyn_cast<ArriveBarrierOp>(&op))
      arrives.push_back(arrive);

  for (auto arrive : arrives) {
    auto constraints = arrive.getConstraints();
    if (!constraints)
      continue;
    Operation *insertPt = arrive->getNextNode();
    for (auto *cur = insertPt; cur && !cur->hasTrait<OpTrait::IsTerminator>();
         cur = cur->getNextNode()) {
      if (auto wait = dyn_cast<WaitBarrierOp>(cur)) {
        if (!canAdvanceWSBarrier(constraints, wait.getConstraints()))
          break;
      }
      insertPt = cur->getNextNode();
    }
    if (insertPt != arrive->getNextNode()) {
      arrive->moveBefore(insertPt);
      changed = true;
    }
  }
  return changed;
}

// Pull WS wait barriers as far up as possible within a block.
// A wait can freely move past non-barrier ops (it just starts waiting sooner).
// A wait can move past another wait (always safe).
// A wait can move past an arrive only if canAdvanceWSBarrier says their
// channel graphs are disjoint.
inline bool raiseWSWaits(Block &block) {
  bool changed = false;
  SmallVector<WaitBarrierOp> waits;
  for (auto &op : block)
    if (auto wait = dyn_cast<WaitBarrierOp>(&op))
      waits.push_back(wait);

  for (auto wait : llvm::reverse(waits)) {
    auto constraints = wait.getConstraints();
    if (!constraints)
      continue;
    Operation *insertPt = wait.getOperation();
    for (auto *cur = wait->getPrevNode(); cur; cur = cur->getPrevNode()) {
      if (auto arrive = dyn_cast<ArriveBarrierOp>(cur)) {
        if (!canAdvanceWSBarrier(arrive.getConstraints(), constraints))
          break;
      }
      insertPt = cur;
    }
    if (insertPt != wait.getOperation()) {
      wait->moveBefore(insertPt);
      changed = true;
    }
  }
  return changed;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
