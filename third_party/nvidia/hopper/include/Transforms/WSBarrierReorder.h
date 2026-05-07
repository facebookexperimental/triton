#ifndef NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
#define NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

inline DictionaryAttr
getWSBarrierConstraints(std::optional<DictionaryAttr> constraints) {
  if (!constraints)
    return {};
  return constraints->getAs<DictionaryAttr>("WSBarrier");
}

inline bool hasWSBarrierConstraints(std::optional<DictionaryAttr> constraints) {
  return static_cast<bool>(getWSBarrierConstraints(constraints));
}

// Check if two WS barriers can be safely swapped by verifying their
// channelGraph sets are disjoint. Returns false if either barrier lacks
// a WSBarrier constraint or channelGraph constraint (conservative).
inline bool canAdvanceWSBarrier(std::optional<DictionaryAttr> constraintsA,
                                std::optional<DictionaryAttr> constraintsB) {
  auto wsBarrierA = getWSBarrierConstraints(constraintsA);
  auto wsBarrierB = getWSBarrierConstraints(constraintsB);
  if (!wsBarrierA || !wsBarrierB)
    return false;
  auto graphA = wsBarrierA.getAs<DenseI32ArrayAttr>("channelGraph");
  auto graphB = wsBarrierB.getAs<DenseI32ArrayAttr>("channelGraph");
  if (!graphA || !graphB)
    return false;
  DenseSet<int> setA(graphA.asArrayRef().begin(), graphA.asArrayRef().end());
  for (int id : graphB.asArrayRef())
    if (setA.contains(id))
      return false;
  return true;
}

inline bool hasArriveLikeSemantics(Operation *op) {
  // TODO: Refine this using WSBarrier metadata so independent arrive-like ops
  // can be reordered when their channel constraints prove it is safe.
  return isa<AsyncTMACopyGlobalToLocalOp, AsyncTMAGatherOp, TMAStoreWaitOp,
             TMAStoreTokenWaitOp, TCGen5CommitOp, MMAv5OpInterface>(op);
}

inline bool canAdvanceWSBarrier(std::optional<DictionaryAttr> constraints,
                                Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
    return canAdvanceWSBarrier(constraints, arrive.getConstraints());
  if (auto wait = dyn_cast<WaitBarrierOp>(op))
    return canAdvanceWSBarrier(constraints, wait.getConstraints());
  return !hasArriveLikeSemantics(op);
}

// Check whether moving `op` to just before `insertPt` would break SSA
// dominance for any of op's operands. Both must be in the same block.
inline bool wouldBreakOperandDominance(Operation *op, Operation *insertPt) {
  for (auto operand : op->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    if (defOp->getBlock() != op->getBlock())
      continue;
    if (!defOp->isBeforeInBlock(insertPt))
      return true;
  }
  return false;
}

// Return the latest same-block operation that an arrive must follow when it is
// restored near its associated memory op.
inline Operation *getArriveAnchorAfterOperands(ArriveBarrierOp arrive,
                                               Operation *memOp) {
  Operation *anchor = memOp;
  for (auto operand : arrive->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != arrive->getBlock())
      continue;
    if (anchor->isBeforeInBlock(defOp))
      anchor = defOp;
  }
  return anchor;
}

// Push WS arrive barriers as far down as possible within a block.
// An arrive can freely move past non-barrier ops (it just delays the signal).
// An arrive can move past another WSBarrier arrive (always safe).
// An arrive can move past a wait only if canAdvanceWSBarrier says their
// channel graphs are disjoint.
inline bool sinkWSArrives(Block &block) {
  bool changed = false;
  SmallVector<ArriveBarrierOp> arrives;
  for (auto &op : block)
    if (auto arrive = dyn_cast<ArriveBarrierOp>(&op);
        arrive && hasWSBarrierConstraints(arrive.getConstraints()))
      arrives.push_back(arrive);

  for (auto arrive : arrives) {
    auto constraints = arrive.getConstraints();
    Operation *insertPt = arrive->getNextNode();
    for (auto *cur = insertPt; cur && !cur->hasTrait<OpTrait::IsTerminator>();
         cur = cur->getNextNode()) {
      if (auto otherArrive = dyn_cast<ArriveBarrierOp>(cur)) {
        if (!hasWSBarrierConstraints(otherArrive.getConstraints()))
          break;
      } else if (!canAdvanceWSBarrier(constraints, cur)) {
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
// A wait can move past another WSBarrier wait (always safe).
// A wait can move past an arrive only if canAdvanceWSBarrier says their
// channel graphs are disjoint.
// Stops before moving past any op that defines an operand of the wait.
inline bool raiseWSWaits(Block &block) {
  bool changed = false;
  SmallVector<WaitBarrierOp> waits;
  for (auto &op : block)
    if (auto wait = dyn_cast<WaitBarrierOp>(&op);
        wait && hasWSBarrierConstraints(wait.getConstraints()))
      waits.push_back(wait);

  for (auto wait : llvm::reverse(waits)) {
    auto constraints = wait.getConstraints();
    Operation *insertPt = wait.getOperation();
    for (auto *cur = wait->getPrevNode(); cur; cur = cur->getPrevNode()) {
      // Don't raise past the definition of any of our operands.
      bool definesOperand = false;
      for (auto result : cur->getResults()) {
        if (llvm::is_contained(wait->getOperands(), result)) {
          definesOperand = true;
          break;
        }
      }
      if (definesOperand)
        break;
      if (auto otherWait = dyn_cast<WaitBarrierOp>(cur)) {
        if (!hasWSBarrierConstraints(otherWait.getConstraints()))
          break;
      } else if (!canAdvanceWSBarrier(constraints, cur)) {
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

// Build a map from each WS-annotated barrier to its nearest associated
// memory op. For arrives, scans backward; for waits, scans forward.
// Barrier ops and terminators are skipped when scanning.
inline DenseMap<Operation *, Operation *>
buildBarrierToMemoryOpMap(Block &block) {
  DenseMap<Operation *, Operation *> map;
  auto isMemoryOp = [](Operation *op) {
    return !isMemoryEffectFree(op) &&
           !isa<ArriveBarrierOp, WaitBarrierOp, InitBarrierOp>(op) &&
           !op->hasTrait<OpTrait::IsTerminator>();
  };

  for (auto &op : block) {
    if (auto arrive = dyn_cast<ArriveBarrierOp>(&op)) {
      if (!hasWSBarrierConstraints(arrive.getConstraints()))
        continue;
      for (auto *cur = arrive->getPrevNode(); cur; cur = cur->getPrevNode()) {
        if (isMemoryOp(cur)) {
          map[arrive] = cur;
          break;
        }
      }
    } else if (auto wait = dyn_cast<WaitBarrierOp>(&op)) {
      if (!hasWSBarrierConstraints(wait.getConstraints()))
        continue;
      for (auto *cur = wait->getNextNode(); cur; cur = cur->getNextNode()) {
        if (isMemoryOp(cur)) {
          map[wait] = cur;
          break;
        }
      }
    }
  }
  return map;
}

// After tmem_load sinking, relocate WS barriers back to optimal positions
// relative to their associated memory ops. Arrives go right after their memory
// op, or after later same-block operand definitions required by SSA. Waits go
// right before their memory op. Skips moves that would break SSA dominance.
inline void optimizeWSBarrierLocations(
    const DenseMap<Operation *, Operation *> &barrierToMemOp) {
  for (auto [barrier, memOp] : barrierToMemOp) {
    if (barrier->getBlock() != memOp->getBlock())
      continue;
    if (auto arrive = dyn_cast<ArriveBarrierOp>(barrier)) {
      Operation *anchor = getArriveAnchorAfterOperands(arrive, memOp);
      if (barrier->getPrevNode() != anchor) {
        Operation *target = anchor->getNextNode();
        if (!wouldBreakOperandDominance(barrier, target))
          barrier->moveAfter(anchor);
      }
    } else {
      if (barrier->getNextNode() != memOp) {
        if (!wouldBreakOperandDominance(barrier, memOp))
          barrier->moveBefore(memOp);
      }
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
