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
          ttng::AsyncCopyMbarrierArriveOp>(user))
    return UseKind::Pruneable;

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

/// Erase a barrier allocation and all its pruneable uses.
void pruneBarrier(ttg::LocalAllocOp alloc,
                  SmallVectorImpl<OpOperand *> &terminalUses) {
  // Phase 1: Handle terminal uses.
  for (OpOperand *use : terminalUses) {
    Operation *user = use->getOwner();

    // Pure barrier ops — erase them.
    if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::ArriveBarrierOp,
            ttng::AsyncCopyMbarrierArriveOp>(user)) {
      user->erase();
      continue;
    }
  }

  // Phase 2: Clean up warp_specialize captures. Walk the alloc's uses and
  // remove captures that are now unused in all partition regions.
  SmallVector<std::pair<ttg::WarpSpecializePartitionsOp, unsigned>> wsCaptures;
  std::function<void(Value)> collectWSCaptures = [&](Value val) {
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
        collectWSCaptures(user->getResult(0));
        continue;
      }
      if (auto partOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
        wsCaptures.push_back({partOp, use.getOperandNumber()});
      }
    }
  };
  collectWSCaptures(alloc.getResult());

  for (auto [partOp, idx] : wsCaptures) {
    auto wsOp = partOp.getParentOp();
    bool allUnused = true;
    for (Region *region : wsOp.getPartitionRegions()) {
      if (!region->getArgument(idx).use_empty()) {
        allUnused = false;
        break;
      }
    }
    if (allUnused) {
      llvm::BitVector toRemove(partOp.getNumOperands());
      toRemove.set(idx);
      for (Region *region : wsOp.getPartitionRegions())
        region->front().eraseArguments(toRemove);
      partOp->eraseOperands(toRemove);
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
  if (alloc.use_empty())
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

      for (OpOperand *use : terminalUses) {
        UseKind kind = classifyUse(use->getOwner());
        switch (kind) {
        case UseKind::Wait:
          hasWaitUses = true;
          break;
        case UseKind::Unknown:
          hasUnknownUses = true;
          break;
        case UseKind::Pruneable:
          break;
        }
      }

      // A barrier is pruneable if it has no wait-like and no unknown uses.
      if (hasWaitUses || hasUnknownUses)
        continue;

      pruneBarrier(alloc, terminalUses);
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
