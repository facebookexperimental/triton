#ifndef NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
#define NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
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

inline bool areChannelGraphsDisjoint(DictionaryAttr wsBarrierA,
                                     DictionaryAttr wsBarrierB) {
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

inline bool hasOrderedWSBarrierInfo(DictionaryAttr wsBarrier) {
  auto parentId = wsBarrier.getAs<IntegerAttr>("parentId");
  auto minRegionId = wsBarrier.getAs<IntegerAttr>("minRegionId");
  auto maxRegionId = wsBarrier.getAs<IntegerAttr>("maxRegionId");
  return parentId && parentId.getInt() >= 0 && minRegionId &&
         minRegionId.getInt() >= 0 && maxRegionId && maxRegionId.getInt() >= 0;
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
  return areChannelGraphsDisjoint(wsBarrierA, wsBarrierB);
}

// Check whether an arrive can be delayed past a wait. V1 allows this when the
// channel graphs are disjoint. V2 also allows overlapping channel graphs when
// both barriers are in the same parent and the wait's ordered region is before
// the arrive's ordered region.
inline bool
canAdvanceWSBarrierArrivePastWait(std::optional<DictionaryAttr> arrive,
                                  std::optional<DictionaryAttr> wait) {
  auto arriveWS = getWSBarrierConstraints(arrive);
  auto waitWS = getWSBarrierConstraints(wait);
  if (!arriveWS || !waitWS)
    return false;
  if (areChannelGraphsDisjoint(arriveWS, waitWS))
    return true;
  if (!hasOrderedWSBarrierInfo(arriveWS) || !hasOrderedWSBarrierInfo(waitWS))
    return false;
  if (arriveWS.getAs<IntegerAttr>("parentId").getInt() !=
      waitWS.getAs<IntegerAttr>("parentId").getInt())
    return false;
  return waitWS.getAs<IntegerAttr>("maxRegionId").getInt() <
         arriveWS.getAs<IntegerAttr>("minRegionId").getInt();
}

inline bool hasArriveLikeSemantics(Operation *op) {
  if (auto wait = dyn_cast<TMAStoreTokenWaitOp>(op))
    return !wait.getBarriers().empty();
  // TODO: Refine this using WSBarrier metadata so independent arrive-like ops
  // can be reordered when their channel constraints prove it is safe.
  return isa<AsyncTMACopyGlobalToLocalOp, AsyncTMAGatherOp, TMAStoreWaitOp,
             TCGen5CommitOp, MMAv5OpInterface>(op);
}

inline bool isBarrierLikeOp(Operation *op) {
  return isa<ArriveBarrierOp, WaitBarrierOp, InitBarrierOp,
             NamedBarrierArriveOp, NamedBarrierWaitOp>(op);
}

inline bool isNamedBarrierOp(Operation *op) {
  return isa<NamedBarrierArriveOp, NamedBarrierWaitOp>(op);
}

inline bool canAdvanceWSBarrier(std::optional<DictionaryAttr> constraints,
                                Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
    return canAdvanceWSBarrier(constraints, arrive.getConstraints());
  if (auto wait = dyn_cast<WaitBarrierOp>(op))
    return canAdvanceWSBarrierArrivePastWait(constraints,
                                             wait.getConstraints());
  if (isNamedBarrierOp(op))
    return false;
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

// Check the barrier-specific part of raising `wait` before `crossedOp`.
// Profitability and non-barrier side-effect checks remain the caller's
// responsibility.
//
// Safety against crossing an unrelated barrier rests on the WSBarrier
// constraint metadata, not on comparing mbarrier values: a crossed arrive is
// admitted only when canAdvanceWSBarrierArrivePastWait proves disjoint channel
// graphs or a strict ordered-region precedence. An arrive on the very barrier
// this wait is waiting on shares that wait's channel graph, so it fails
// disjointness and, being in the same region, fails the ordering proof too.
//
// The isBarrierLikeOp rejection below is deliberately stricter than the
// hasArriveLikeSemantics fallthrough it replaced in raiseWSWaits: it also
// rejects InitBarrierOp, so a wait can never be raised above the
// initialization of its own barrier.
inline bool canRaiseWSWaitPast(WaitBarrierOp wait, Operation *crossedOp) {
  if (crossedOp->getNumRegions() != 0)
    return false;
  if (auto otherWait = dyn_cast<WaitBarrierOp>(crossedOp))
    return hasWSBarrierConstraints(otherWait.getConstraints());
  if (auto arrive = dyn_cast<ArriveBarrierOp>(crossedOp))
    return canAdvanceWSBarrierArrivePastWait(arrive.getConstraints(),
                                             wait.getConstraints());
  if (isBarrierLikeOp(crossedOp))
    return false;
  return !hasArriveLikeSemantics(crossedOp);
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
      } else if (auto wait = dyn_cast<WaitBarrierOp>(cur)) {
        if (!canAdvanceWSBarrierArrivePastWait(constraints,
                                               wait.getConstraints()))
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
      if (!canRaiseWSWaitPast(wait, cur)) {
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
    return !isMemoryEffectFree(op) && !isBarrierLikeOp(op) &&
           !op->hasTrait<OpTrait::IsTerminator>();
  };

  for (auto &op : block) {
    if (auto arrive = dyn_cast<ArriveBarrierOp>(&op)) {
      if (!hasWSBarrierConstraints(arrive.getConstraints()))
        continue;
      for (auto *cur = arrive->getPrevNode(); cur; cur = cur->getPrevNode()) {
        if (isNamedBarrierOp(cur))
          break;
        if (isMemoryOp(cur)) {
          map[arrive] = cur;
          break;
        }
      }
    } else if (auto wait = dyn_cast<WaitBarrierOp>(&op)) {
      if (!hasWSBarrierConstraints(wait.getConstraints()))
        continue;
      for (auto *cur = wait->getNextNode(); cur; cur = cur->getNextNode()) {
        if (isNamedBarrierOp(cur))
          break;
        if (isMemoryOp(cur)) {
          map[wait] = cur;
          break;
        }
      }
    }
  }
  return map;
}

// Whether two WS barrier endpoints in one partition may belong to the same
// channel. Endpoints with disjoint channel graphs, or with different peer
// partitions (dstTask), cannot, unless one of them uses a barrier array that
// endpoints with different peers share (a reuse group with more members than
// slots), where the peer no longer identifies the slot.
inline bool mayShareWSChannel(WaitBarrierOp wait, ArriveBarrierOp arrive,
                              const DenseSet<Value> &mixedBarrierRoots) {
  if (canAdvanceWSBarrier(wait.getConstraints(), arrive.getConstraints()))
    return false;
  auto wsA = getWSBarrierConstraints(wait.getConstraints());
  auto wsB = getWSBarrierConstraints(arrive.getConstraints());
  if (!wsA || !wsB)
    return true;
  Value waitRoot = mlir::getMemDescRoot(wait.getAlloc());
  Value arriveRoot = mlir::getMemDescRoot(arrive.getAlloc());
  if (waitRoot == arriveRoot || mixedBarrierRoots.contains(waitRoot) ||
      mixedBarrierRoots.contains(arriveRoot))
    return true;
  auto dstA = wsA.getAs<IntegerAttr>("dstTask");
  auto dstB = wsB.getAs<IntegerAttr>("dstTask");
  return !dstA || !dstB || dstA.getInt() == dstB.getInt();
}

// Check whether `wait` can be lowered below `crossedOp`. Lowering a wait
// moves `crossedOp` ahead of it, so an arrive may only be crossed when it
// cannot belong to the same channel: the release of the slot this wait
// acquires would otherwise signal before the acquire.
inline bool canLowerWSWaitPast(WaitBarrierOp wait, Operation *crossedOp,
                               const DenseSet<Value> &mixedBarrierRoots) {
  if (crossedOp->getNumRegions() != 0)
    return false;
  if (auto otherWait = dyn_cast<WaitBarrierOp>(crossedOp))
    return hasWSBarrierConstraints(otherWait.getConstraints());
  if (auto arrive = dyn_cast<ArriveBarrierOp>(crossedOp))
    return !mayShareWSChannel(wait, arrive, mixedBarrierRoots);
  if (isBarrierLikeOp(crossedOp))
    return false;
  return !hasArriveLikeSemantics(crossedOp);
}

// Check whether `arrive` can be raised above `crossedOp`. Raising an arrive
// signals before `crossedOp`, so a wait may only be crossed when it cannot
// belong to the same channel.
inline bool canRaiseWSArrivePast(ArriveBarrierOp arrive, Operation *crossedOp,
                                 const DenseSet<Value> &mixedBarrierRoots) {
  if (crossedOp->getNumRegions() != 0)
    return false;
  if (auto otherArrive = dyn_cast<ArriveBarrierOp>(crossedOp))
    return hasWSBarrierConstraints(otherArrive.getConstraints());
  if (auto wait = dyn_cast<WaitBarrierOp>(crossedOp))
    return !mayShareWSChannel(wait, arrive, mixedBarrierRoots);
  return !isBarrierLikeOp(crossedOp);
}

// Record each op's position in the block before any reordering, so that
// optimizeWSBarrierLocations can tell which ops a barrier originally guarded.
inline DenseMap<Operation *, unsigned> buildOriginalBlockOrder(Block &block) {
  DenseMap<Operation *, unsigned> order;
  unsigned idx = 0;
  for (auto &op : block)
    order[&op] = idx++;
  return order;
}

// After tmem_load sinking, relocate WS barriers back to optimal positions
// relative to their associated memory ops. Arrives go right after their memory
// op, or after later same-block operand definitions required by SSA. Waits go
// right before their memory op. Skips moves that would break SSA dominance.
//
// The associated memory op is only the nearest one when the map was built, and
// sinking can reorder the other ops the barrier guards. So a wait moving down
// stops at the first op on the same buffer that originally followed it, and an
// arrive moving up stops at the last op on the same buffer that originally
// preceded it. Otherwise, when a subtile is read twice and the two reads sink
// by different amounts, one read ends up outside the wait/arrive pair. "Same
// buffer" compares memdesc roots, so every slot and view of one allocation
// counts. When the memory op has no memdesc operand, any side-effecting op
// counts. Ops with regions, and ops missing from `originalOrder`, are treated
// as guarded.
//
// Lowering a wait toward its memory op, or raising an arrive toward its memory
// op, is only a scheduling choice: leaving the barrier where it is stays
// correct. So such a move also stops at the first barrier it may not cross,
// unless that only restores the original order. A wait lowered below an
// arrive (or an arrive raised above a wait) of the same channel would signal
// before the acquire it depends on. Without
// this, a wait whose nearest memory op belonged to another channel (a
// broadcast slot whose local_load was dead) was lowered below the release of
// its own slot, so the producer could overwrite the slot twice and the
// consumer's parity wait hung.
inline void optimizeWSBarrierLocations(
    const DenseMap<Operation *, Operation *> &barrierToMemOp,
    const DenseMap<Operation *, unsigned> &originalOrder) {
  auto memDescRoots = [](Operation *op) {
    DenseSet<Value> roots;
    for (Value operand : op->getOperands())
      if (isa<triton::gpu::MemDescType>(operand.getType()))
        roots.insert(mlir::getMemDescRoot(operand));
    return roots;
  };
  auto touchesBuffer = [&](Operation *op, const DenseSet<Value> &roots) {
    if (roots.empty() || op->getNumRegions() != 0)
      return true;
    return llvm::any_of(memDescRoots(op),
                        [&](Value root) { return roots.contains(root); });
  };
  auto isGuarded = [&](Operation *op, Operation *barrier, bool isWait,
                       const DenseSet<Value> &roots) {
    if (isMemoryEffectFree(op) || !touchesBuffer(op, roots))
      return false;
    auto it = originalOrder.find(op);
    if (it == originalOrder.end())
      return true;
    unsigned barrierPos = originalOrder.lookup(barrier);
    return isWait ? it->second > barrierPos : it->second < barrierPos;
  };
  // Moving a barrier back across an op it originally followed (or preceded)
  // only restores the original order, which is always legal.
  auto wasOriginallyBefore = [&](Operation *a, Operation *b) {
    auto itA = originalOrder.find(a), itB = originalOrder.find(b);
    return itA != originalOrder.end() && itB != originalOrder.end() &&
           itA->second < itB->second;
  };
  // Barrier arrays used by WS endpoints with more than one peer partition.
  // Every WS endpoint in the block votes, not just the ones the map holds.
  DenseMap<Value, std::optional<int64_t>> peerOfRoot;
  DenseSet<Value> mixedBarrierRoots;
  if (!barrierToMemOp.empty()) {
    for (Operation &op : *barrierToMemOp.begin()->first->getBlock()) {
      Value alloc;
      std::optional<DictionaryAttr> constraints;
      if (auto wait = dyn_cast<WaitBarrierOp>(&op)) {
        alloc = wait.getAlloc();
        constraints = wait.getConstraints();
      } else if (auto arrive = dyn_cast<ArriveBarrierOp>(&op)) {
        alloc = arrive.getAlloc();
        constraints = arrive.getConstraints();
      } else {
        continue;
      }
      auto ws = getWSBarrierConstraints(constraints);
      if (!ws)
        continue;
      std::optional<int64_t> peer;
      if (auto dst = ws.getAs<IntegerAttr>("dstTask"))
        peer = dst.getInt();
      Value root = mlir::getMemDescRoot(alloc);
      auto [it, inserted] = peerOfRoot.try_emplace(root, peer);
      if (!inserted && it->second != peer)
        mixedBarrierRoots.insert(root);
    }
  }
  // Visit barriers in block order so the result does not depend on pointer
  // hashing: a barrier's clamp depends on where earlier-moved barriers sit.
  SmallVector<std::pair<Operation *, Operation *>> barriers(
      barrierToMemOp.begin(), barrierToMemOp.end());
  llvm::sort(barriers, [&](const auto &a, const auto &b) {
    return originalOrder.lookup(a.first) < originalOrder.lookup(b.first);
  });
  for (auto [barrier, memOp] : barriers) {
    if (barrier->getBlock() != memOp->getBlock())
      continue;
    DenseSet<Value> roots = memDescRoots(memOp);
    if (auto arrive = dyn_cast<ArriveBarrierOp>(barrier)) {
      Operation *anchor = getArriveAnchorAfterOperands(arrive, memOp);
      if (anchor->isBeforeInBlock(barrier)) {
        for (auto *cur = barrier->getPrevNode(); cur != anchor;
             cur = cur->getPrevNode()) {
          if (isGuarded(cur, barrier, /*isWait=*/false, roots) ||
              (!wasOriginallyBefore(barrier, cur) &&
               !canRaiseWSArrivePast(arrive, cur, mixedBarrierRoots))) {
            anchor = cur;
            break;
          }
        }
      }
      if (barrier->getPrevNode() != anchor) {
        Operation *target = anchor->getNextNode();
        if (!wouldBreakOperandDominance(barrier, target))
          barrier->moveAfter(anchor);
      }
    } else {
      auto wait = cast<WaitBarrierOp>(barrier);
      Operation *target = memOp;
      if (barrier->isBeforeInBlock(memOp)) {
        for (auto *cur = barrier->getNextNode(); cur != memOp;
             cur = cur->getNextNode()) {
          if (isGuarded(cur, barrier, /*isWait=*/true, roots) ||
              (!wasOriginallyBefore(cur, barrier) &&
               !canLowerWSWaitPast(wait, cur, mixedBarrierRoots))) {
            target = cur;
            break;
          }
        }
      }
      if (barrier->getNextNode() != target) {
        if (!wouldBreakOperandDominance(barrier, target))
          barrier->moveBefore(target);
      }
    }
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#endif // NV_HOPPER_TRANSFORMS_WSBARRIERREORDER_H_
