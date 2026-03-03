#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPRUNEUNUSEDBARRIERSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Classify whether a barrier allocation is pruneable based on its transitive
/// uses. A barrier is pruneable if it has no wait-like uses and no unknown
/// (unrecognized) uses.
enum class UseKind {
  /// A wait-like use (e.g. wait_barrier).
  Wait,
  /// A pruneable use (init, arrive, expect, commit, etc.).
  Pruneable,
  /// An async op with a required barrier operand. The barrier itself can't be
  /// fully removed, but we can still prune explicit arrive/expect/commit ops.
  RequiredBarrier,
  /// An op we don't recognize — conservatively non-pruneable.
  Unknown,
};

/// Classify a single terminal use of a barrier value.
UseKind classifyUse(Operation *user) {
  // Wait-like uses.
  if (isa<ttng::WaitBarrierOp>(user))
    return UseKind::Wait;

  // Pure barrier lifecycle ops — always pruneable.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::ArriveBarrierOp,
          ttng::BarrierExpectOp, ttng::TCGen5CommitOp,
          ttng::AsyncCopyMbarrierArriveOp>(user))
    return UseKind::Pruneable;

  // Async ops where the barrier is optional or can be disconnected.
  if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(user))
    return UseKind::Pruneable;
  if (isa<ttng::TMEMCopyOp>(user))
    return UseKind::Pruneable;

  // Async ops with required barrier operands.
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp, ttng::AsyncTMAGatherOp,
          ttng::AsyncCLCTryCancelOp, ttg::AsyncRemoteShmemStoreOp>(user))
    return UseKind::RequiredBarrier;

  return UseKind::Unknown;
}

/// Recursively trace all transitive uses of a barrier value, following through
/// view ops and warp_specialize captures. Collects terminal (non-view) uses.
void traceBarrierUses(Value barrierVal,
                      SmallVectorImpl<OpOperand *> &terminalUses) {
  for (OpOperand &use : barrierVal.getUses()) {
    Operation *user = use.getOwner();

    // Follow through MemDescViewTrait ops (memdesc_index, memdesc_subslice,
    // etc.)
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      assert(user->getNumResults() == 1);
      traceBarrierUses(user->getResult(0), terminalUses);
      continue;
    }

    // Follow through warp_specialize captures.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      for (Region *region : wsOp.getPartitionRegions()) {
        Value blockArg = region->getArgument(operandIdx);
        traceBarrierUses(blockArg, terminalUses);
      }
      continue;
    }

    // Terminal use.
    terminalUses.push_back(&use);
  }
}

/// Check if a local_alloc is a barrier allocation: produces memdesc with i64
/// element type and has no src operand.
bool isBarrierAlloc(ttg::LocalAllocOp alloc) {
  auto memDescType = alloc.getType();
  if (!memDescType.getElementType().isInteger(64))
    return false;
  return !alloc.getSrc();
}

/// Remove a specific barrier from a TCGen5MMAOp or TCGen5MMAScaledOp's
/// variadic barriers list, along with the corresponding barrier_pred.
/// MutableOperandRange objects become stale after modifications on ops with
/// AttrSizedOperandSegments, so we snapshot values to keep first, then
/// rebuild from scratch using fresh getters.
template <typename MMAOpTy>
void removeBarrierFromMMAImpl(MMAOpTy mma, Value barrierVal) {
  // Snapshot values to keep.
  SmallVector<Value> keptBarriers, keptPreds;
  auto barriers = mma.getBarriers();
  auto barrierPreds = mma.getBarrierPreds();
  for (unsigned i = 0; i < barriers.size(); ++i) {
    if (barriers[i] != barrierVal) {
      keptBarriers.push_back(barriers[i]);
      keptPreds.push_back(barrierPreds[i]);
    }
  }
  // Clear preds first (later in operand list) to avoid stale ranges.
  mma.getBarrierPredsMutable().clear();
  // Re-fetch barriers mutable since the op changed.
  mma.getBarriersMutable().clear();
  // Re-add kept values using fresh getters each time.
  for (auto val : keptBarriers)
    mma.getBarriersMutable().append(val);
  for (auto val : keptPreds)
    mma.getBarrierPredsMutable().append(val);
  // If no barriers remain, unset is_async.
  if (mma.getBarriers().empty())
    mma.removeIsAsyncAttr();
}

void removeBarrierFromMMA(Operation *mmaOp, Value barrierVal) {
  if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(mmaOp))
    removeBarrierFromMMAImpl(mma, barrierVal);
  else if (auto mmaScaled = dyn_cast<ttng::TCGen5MMAScaledOp>(mmaOp))
    removeBarrierFromMMAImpl(mmaScaled, barrierVal);
}

/// Remove the optional barrier from a TMEMCopyOp.
void removeBarrierFromTMEMCopy(ttng::TMEMCopyOp tmemCopy) {
  tmemCopy.getBarrierMutable().clear();
}

/// Erase a barrier allocation and all its pruneable uses.
/// If the barrier has required-barrier async ops, do partial pruning:
/// keep the alloc and the async op, but erase explicit arrive/expect/commit.
void pruneBarrier(ttg::LocalAllocOp alloc,
                  SmallVectorImpl<OpOperand *> &terminalUses,
                  bool hasRequiredBarrierUses) {
  // Phase 1: Handle terminal uses.
  for (OpOperand *use : terminalUses) {
    Operation *user = use->getOwner();

    // Pure barrier ops — erase them.
    if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::ArriveBarrierOp,
            ttng::BarrierExpectOp, ttng::TCGen5CommitOp,
            ttng::AsyncCopyMbarrierArriveOp>(user)) {
      // Only erase if we're doing full pruning, or if partial pruning
      // (still erase arrive/expect/commit but keep init for required barriers).
      if (hasRequiredBarrierUses && isa<ttng::InitBarrierOp>(user))
        continue;
      user->erase();
      continue;
    }

    // Disconnect from MMA ops.
    if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(user)) {
      removeBarrierFromMMA(user, use->get());
      continue;
    }

    // Disconnect from TMEMCopy.
    if (auto tmemCopy = dyn_cast<ttng::TMEMCopyOp>(user)) {
      removeBarrierFromTMEMCopy(tmemCopy);
      continue;
    }

    // Required barrier ops — keep them.
    if (isa<ttng::AsyncTMACopyGlobalToLocalOp, ttng::AsyncTMAGatherOp,
            ttng::AsyncCLCTryCancelOp, ttg::AsyncRemoteShmemStoreOp>(user)) {
      continue;
    }
  }

  // Phase 2: Clean up warp_specialize captures. Walk the alloc's uses and
  // remove captures that are now unused in all partition regions.
  SmallVector<std::pair<ttg::WarpSpecializeOp, unsigned>> wsCaptures;
  std::function<void(Value)> collectWSCaptures = [&](Value val) {
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
        collectWSCaptures(user->getResult(0));
        continue;
      }
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        wsCaptures.push_back({wsOp, use.getOperandNumber()});
      }
    }
  };
  collectWSCaptures(alloc.getResult());

  for (auto [wsOp, idx] : wsCaptures) {
    bool allUnused = true;
    for (Region *region : wsOp.getPartitionRegions()) {
      if (!region->getArgument(idx).use_empty()) {
        allUnused = false;
        break;
      }
    }
    if (allUnused) {
      llvm::BitVector toRemove(wsOp.getNumOperands());
      toRemove.set(idx);
      for (Region *region : wsOp.getPartitionRegions())
        region->front().eraseArguments(toRemove);
      wsOp->eraseOperands(toRemove);
    }
  }

  // Phase 3: Clean up dead view ops (bottom-up: users before defs).
  std::function<void(Value)> eraseDeadViews = [&](Value val) {
    // Collect users first to avoid iterator invalidation.
    SmallVector<Operation *> users;
    for (OpOperand &use : val.getUses())
      users.push_back(use.getOwner());
    for (Operation *user : users) {
      if (user->hasTrait<OpTrait::MemDescViewTrait>() &&
          user->getResult(0).use_empty()) {
        user->erase();
      }
    }
  };
  eraseDeadViews(alloc.getResult());

  // Phase 4: Erase the alloc if it has no remaining uses.
  if (!hasRequiredBarrierUses && alloc.use_empty())
    alloc.erase();
}

} // anonymous namespace

class TritonNvidiaGPUPruneUnusedBarriersPass
    : public impl::TritonNvidiaGPUPruneUnusedBarriersPassBase<
          TritonNvidiaGPUPruneUnusedBarriersPass> {
public:
  using TritonNvidiaGPUPruneUnusedBarriersPassBase::
      TritonNvidiaGPUPruneUnusedBarriersPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Phase 1: Collect all barrier allocations.
    SmallVector<ttg::LocalAllocOp> barrierAllocs;
    mod.walk([&](ttg::LocalAllocOp alloc) {
      if (isBarrierAlloc(alloc))
        barrierAllocs.push_back(alloc);
    });

    // Phase 2-4: For each barrier, trace uses and prune if possible.
    for (auto alloc : barrierAllocs) {
      SmallVector<OpOperand *> terminalUses;
      traceBarrierUses(alloc.getResult(), terminalUses);

      // Classify all terminal uses.
      bool hasWaitUses = false;
      bool hasUnknownUses = false;
      bool hasRequiredBarrierUses = false;

      for (OpOperand *use : terminalUses) {
        UseKind kind = classifyUse(use->getOwner());
        switch (kind) {
        case UseKind::Wait:
          hasWaitUses = true;
          break;
        case UseKind::Unknown:
          hasUnknownUses = true;
          break;
        case UseKind::RequiredBarrier:
          hasRequiredBarrierUses = true;
          break;
        case UseKind::Pruneable:
          break;
        }
      }

      // A barrier is pruneable if it has no wait-like and no unknown uses.
      if (hasWaitUses || hasUnknownUses)
        continue;

      pruneBarrier(alloc, terminalUses, hasRequiredBarrierUses);
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
