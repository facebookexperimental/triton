#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPOSTSUBTILEWAITREORDERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

#define DEBUG_TYPE "triton-nvidia-gpu-post-subtile-wait-reorder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static constexpr const char *kCanRotateByBufferCount =
    "can_rotate_by_buffer_count";

static AsyncTMACopyLocalToGlobalOp
getDefiningTMAStore(TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  if (auto defOp = token.getDefiningOp<AsyncTMACopyLocalToGlobalOp>())
    return defOp;

  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);
    return yieldedVal.getDefiningOp<AsyncTMACopyLocalToGlobalOp>();
  }

  return nullptr;
}

static bool validateAnnotatedWait(TMAStoreTokenWaitOp waitOp,
                                  scf::ForOp forOp) {
  auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
  if (!attr)
    return false;
  int k = attr.getInt();

  auto tmaStore = getDefiningTMAStore(waitOp);
  if (!tmaStore || tmaStore->getParentOp() != forOp) {
    LDBG("Wait validation failed: TMA store not in same loop");
    return false;
  }

  Value buffer = tmaStore.getSrc();
  auto allocOp = buffer.getDefiningOp<gpu::LocalAllocOp>();
  if (!allocOp) {
    LDBG("Wait validation failed: no LocalAllocOp for buffer");
    return false;
  }

  auto bufferCopy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
  if (!bufferCopy || bufferCopy.getInt() != k) {
    LDBG("Wait validation failed: buffer.copy mismatch (annotation="
         << k << ", actual=" << (bufferCopy ? bufferCopy.getInt() : -1) << ")");
    return false;
  }

  return true;
}

/// Schedule TMA store waits for pipelining. All ops are assigned to
/// stage 0 / cluster 0, then annotated waits are repositioned using
/// the same rotation logic as doTMAStoreWaitReorder. The schedule is
/// serialized onto the scf.for so the downstream pipeline pass
/// (add_pipeline) can expand it.
void postSubtileWaitReorder(FuncOp funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedFor = false;
    forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
    if (hasNestedFor)
      return;

    SmallVector<TMAStoreTokenWaitOp> annotatedWaits;
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<TMAStoreTokenWaitOp>(&op);
      if (!waitOp || !waitOp->hasAttr(kCanRotateByBufferCount))
        continue;
      annotatedWaits.push_back(waitOp);
    }
    if (annotatedWaits.empty())
      return;

    LDBG("Found " << annotatedWaits.size()
                  << " annotated TMA store waits in loop");

    for (auto waitOp : annotatedWaits) {
      if (!validateAnnotatedWait(waitOp, forOp)) {
        LDBG("Safety validation failed; skipping entire loop");
        for (auto w : annotatedWaits)
          w->removeAttr(kCanRotateByBufferCount);
        return;
      }
    }

    // Build schedule: all ops at stage 0, single cluster.
    triton::CoarseSchedule schedule;
    schedule.setNumStages(1);
    auto cluster = schedule.clusters.newAtBack();
    for (auto &op : forOp.getBody()->without_terminator())
      schedule.insert(&op, 0, cluster);

    bool changed = false;
    for (auto waitOp : annotatedWaits) {
      auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
      int k = attr.getInt();

      auto tmaStore = getDefiningTMAStore(waitOp);
      if (!tmaStore || !schedule.count(tmaStore)) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        continue;
      }

      auto it = schedule.linearized(forOp, tmaStore);
      it.setMaxStages(schedule.getNumStages() + k);
      ++it;

      Operation *insertionTarget = nullptr;
      int targetStage = 0;
      int copyCount = 0;

      while (!it.isEnd()) {
        Operation *op = *it;
        int stageAtOp = it.currStage();
        ++it;
        if (isa<AsyncTMACopyLocalToGlobalOp>(op)) {
          ++copyCount;
          if (copyCount == k) {
            insertionTarget = op;
            targetStage = stageAtOp;
            break;
          }
        }
      }

      if (!insertionTarget) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        continue;
      }

      // Look for a WaitBarrierOp before the insertion target.
      for (auto revIt = Block::reverse_iterator(insertionTarget->getIterator());
           revIt != insertionTarget->getBlock()->rend(); ++revIt) {
        if (isa<WaitBarrierOp>(&*revIt) && schedule.count(&*revIt)) {
          insertionTarget = &*revIt;
          break;
        }
      }

      auto targetCluster = schedule.splitClusterBefore(insertionTarget, forOp);
      auto waitCluster = schedule.clusters.newBefore(targetCluster);
      schedule.insert(waitOp, targetStage, waitCluster);

      if (targetStage + 1 > static_cast<int>(schedule.getNumStages()))
        schedule.setNumStages(targetStage + 1);

      waitOp->removeAttr(kCanRotateByBufferCount);
      changed = true;
    }

    if (!changed)
      return;

    // Serialize the schedule for the downstream pipeline pass to expand.
    schedule.serialize(forOp);

    LDBG("Scheduled TMA store waits for pipelining (" << schedule.getNumStages()
                                                      << " stages)");
  });
}

} // namespace

struct TritonNvidiaGPUPostSubtileWaitReorderPass
    : public impl::TritonNvidiaGPUPostSubtileWaitReorderPassBase<
          TritonNvidiaGPUPostSubtileWaitReorderPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](FuncOp funcOp) { postSubtileWaitReorder(funcOp); });
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
