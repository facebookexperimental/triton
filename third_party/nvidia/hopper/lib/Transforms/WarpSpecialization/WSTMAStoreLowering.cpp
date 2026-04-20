#include "CodePartitionUtility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/Debug.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-tma-store-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static void copyLoopScheduleAttrs(Operation *from, Operation *to) {
  if (auto attr = from->getAttr(tt::kLoopStageAttrName))
    to->setAttr(tt::kLoopStageAttrName, attr);
  if (auto attr = from->getAttr(tt::kLoopClusterAttrName))
    to->setAttr(tt::kLoopClusterAttrName, attr);
}

void doTMAStoreLowering(triton::FuncOp &funcOp) {
  SmallVector<tt::DescriptorStoreOp> storeOps;
  funcOp.walk([&](tt::DescriptorStoreOp op) {
    // Skip stores with non-trivial reduce semantics.
    if (op.getReduceKind() != tt::DescriptorReduceKind::NONE)
      return;
    storeOps.push_back(op);
  });

  if (storeOps.empty())
    return;

  LDBG("Lowering " << storeOps.size() << " DescriptorStoreOp(s)");

  MLIRContext *ctx = funcOp.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);

  for (auto storeOp : storeOps) {
    auto loc = storeOp.getLoc();
    auto asyncTaskIds = getAsyncTaskIds(storeOp);

    OpBuilderWithAsyncTaskIds builder(storeOp);
    builder.setInsertionPoint(storeOp);

    auto src = storeOp.getSrc();
    auto desc = storeOp.getDesc();
    auto tensorType = src.getType();

    // Compute shared encoding from the descriptor.
    auto encoding = ttng::getEncodingFromDescriptor(storeOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    // Allocate SMEM and copy register data into it in one step.
    auto alloc = builder.create<ttg::LocalAllocOp>(loc, memDescType, src);

    // Translate indices for TMA.
    auto indices = ttng::translateTMAIndices(
        builder, loc, desc.getType().getBlockType().getEncoding(),
        storeOp.getIndices());

    // Async TMA copy from local (SMEM) to global, producing a token.
    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaStore = builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, tokenType, desc, indices, alloc, tt::EvictionPolicy::NORMAL);
    copyLoopScheduleAttrs(storeOp, tmaStore);

    // Wait for this specific TMA store to finish reading from SMEM.
    auto waitOp = builder.create<ttng::TMAStoreTokenWaitOp>(
        loc, tmaStore.getToken(), ValueRange{}, ValueRange{}, ValueRange{},
        ValueRange{});
    copyLoopScheduleAttrs(storeOp, waitOp);

    storeOp.erase();
  }
}

// ---------------------------------------------------------------------------
// Standalone pass wrapper
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUWSTMASTORELOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

struct NVGPUWSTMAStoreLoweringPass
    : public impl::NVGPUWSTMAStoreLoweringBase<NVGPUWSTMAStoreLoweringPass> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!mod->hasAttr("ttg.early_tma_store_lowering"))
      return;
    mod->walk([&](triton::FuncOp funcOp) { doTMAStoreLowering(funcOp); });
  }
};

// ---------------------------------------------------------------------------

// Annotate TMA store waits with can_rotate_by_buffer_count
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTESTANNOTATETMASTOREWAITS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

static constexpr const char *kCanRotateByBufferCount =
    "can_rotate_by_buffer_count";

// Trace the token back to the defining AsyncTMACopyLocalToGlobalOp, handling
// both direct definitions and loop-carried block arguments.
static ttng::AsyncTMACopyLocalToGlobalOp
getDefiningTMAStore(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  // Direct case: token defined by AsyncTMACopyLocalToGlobalOp.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>())
    return defOp;

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);
    return yieldedVal.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>();
  }

  return nullptr;
}

void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  funcOp.walk([&](scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
      if (!waitOp)
        continue;

      auto tmaStore = getDefiningTMAStore(waitOp);
      if (!tmaStore)
        continue;

      Value buffer = tmaStore.getSrc();
      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp)
        continue;

      auto bufferCopy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
      if (!bufferCopy)
        continue;

      int k = bufferCopy.getInt();
      if (k <= 0)
        continue;

      waitOp->setAttr(kCanRotateByBufferCount,
                      IntegerAttr::get(IntegerType::get(ctx, 32), k));
    }
  });
}

struct NVGPUTestAnnotateTMAStoreWaitsPass
    : public impl::NVGPUTestAnnotateTMAStoreWaitsBase<
          NVGPUTestAnnotateTMAStoreWaitsPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doAnnotateTMAStoreWaits(funcOp); });
  }
};

// ---------------------------------------------------------------------------
// Validate TMA store annotations (safety checks)
// ---------------------------------------------------------------------------

void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
      if (!waitOp || !waitOp->hasAttr(kCanRotateByBufferCount))
        continue;

      auto tmaStore = getDefiningTMAStore(waitOp);
      if (!tmaStore) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        continue;
      }

      Value buffer = tmaStore.getSrc();
      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        continue;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Reschedule TMA store waits using the SWP CoarseSchedule
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTESTTMASTORETOKENWAITREORDER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

void doTMAStoreWaitReorder(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    // Deserialize the SWP schedule. If there is no schedule, create a basic
    // single-stage schedule so the reorder logic can still work.
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      schedule.setNumStages(1);
      auto cluster = schedule.clusters.newAtBack();
      for (auto &op : forOp.getBody()->without_terminator())
        schedule.insert(&op, 0, cluster);
    }

    // Bail out if the loop body contains any allocation ops. Reordering
    // waits in such loops would serialize a multi-stage schedule that
    // covers only a subset of the body ops, causing the pipeliner to fail
    // on the unscheduled allocations.
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op))
        return;
    }

    // Collect annotated TMA store waits that are direct children of this
    // loop and whose defining TMA store is in the same loop.
    SmallVector<ttng::TMAStoreTokenWaitOp> waits;
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
      if (!waitOp || !waitOp->hasAttr(kCanRotateByBufferCount))
        continue;
      auto tmaStore = getDefiningTMAStore(waitOp);
      if (!tmaStore || tmaStore->getParentOp() != forOp)
        continue;
      waits.push_back(waitOp);
    }
    if (waits.empty())
      return;

    bool changed = false;
    for (auto waitOp : waits) {
      auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
      if (!attr)
        continue;
      int k = attr.getInt();

      // Find the defining TMA store op.
      auto tmaStore = getDefiningTMAStore(waitOp);
      if (!tmaStore)
        continue;

      // The defining op must be in the schedule for the LinearizedIterator.
      if (!schedule.count(tmaStore))
        continue;

      // Walk the linearized schedule from the TMA store, counting K
      // AsyncTMACopyLocalToGlobalOp ops. The wait must be placed before
      // the K-th copy to ensure the buffer slot is not overwritten.
      auto it = schedule.linearized(forOp, tmaStore);
      it.setMaxStages(schedule.getNumStages() + k);

      // Skip past the starting TMA store itself.
      ++it;

      Operation *insertionTarget = nullptr;
      int targetStage = 0;
      int copyCount = 0;

      while (!it.isEnd()) {
        Operation *op = *it;
        int stageAtOp = it.currStage();
        ++it;
        if (auto copyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
          ++copyCount;
          if (copyCount == k) {
            insertionTarget = op;
            targetStage = stageAtOp;
            break;
          }
        }
      }

      if (insertionTarget) {
        // Look for a WaitBarrierOp before the insertion target in the same
        // block. If found, insert before the barrier wait instead.
        for (auto revIt =
                 Block::reverse_iterator(insertionTarget->getIterator());
             revIt != insertionTarget->getBlock()->rend(); ++revIt) {
          if (isa<ttng::WaitBarrierOp>(&*revIt) && schedule.count(&*revIt)) {
            insertionTarget = &*revIt;
            break;
          }
        }

        // Split the cluster at the insertion target: ops before it remain
        // in the original cluster, the target and subsequent ops stay in
        // the returned cluster.
        auto targetCluster =
            schedule.splitClusterBefore(insertionTarget, forOp);
        // Insert a new cluster for our wait between the split halves.
        auto waitCluster = schedule.clusters.newBefore(targetCluster);
        schedule.insert(waitOp, targetStage, waitCluster);
      } else {
        // Target not found; leave the schedule unchanged for this wait.
        continue;
      }

      waitOp->removeAttr(kCanRotateByBufferCount);
      changed = true;
    }

    if (changed)
      schedule.serialize(forOp);
  });
}

struct NVGPUTestTMAStoreTokenWaitReorderPass
    : public impl::NVGPUTestTMAStoreTokenWaitReorderBase<
          NVGPUTestTMAStoreTokenWaitReorderPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doTMAStoreWaitReorder(funcOp); });
  }
};

// ---------------------------------------------------------------------------
// Lower TMAStoreTokenWaitOp with barriers into TMAStoreWaitOp + ArriveBarrierOp
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITLOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

// Count AsyncTMACopyLocalToGlobalOp ops in [from, to) within a block.
static int countTMAStoresInRange(Block::iterator from, Block::iterator to) {
  int count = 0;
  for (auto it = from; it != to; ++it) {
    if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(&*it))
      ++count;
  }
  return count;
}

// Compute the pendings value for a TMAStoreTokenWaitOp.
// pendings = number of AsyncTMACopyLocalToGlobalOp ops issued after the token's
// defining store and before this wait, in program execution order.
static int computePendings(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  // Direct case: token defined by AsyncTMACopyLocalToGlobalOp in same block.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
    if (defOp->getBlock() == waitOp->getBlock()) {
      // Count TMA stores strictly between def and wait.
      return countTMAStoresInRange(std::next(defOp->getIterator()),
                                   waitOp->getIterator());
    }
    return 0;
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return 0;

    // Block args for scf.for body are [iv, iter_arg0, iter_arg1, ...].
    // The iter_arg index is blockArg.getArgNumber() - 1 (subtract the IV).
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;

    // Find the corresponding yield operand.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);

    // Trace the yielded value to its defining AsyncTMACopyLocalToGlobalOp.
    auto defOp = yieldedVal.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>();
    if (!defOp || defOp->getBlock() != forOp.getBody())
      return 0;

    Block *body = forOp.getBody();

    // Stores after the def until end of loop body (excluding yield).
    int storesAfterDef =
        countTMAStoresInRange(std::next(defOp->getIterator()), body->end());

    // Stores from start of loop body until the wait.
    int storesBeforeWait =
        countTMAStoresInRange(body->begin(), waitOp->getIterator());

    return storesAfterDef + storesBeforeWait;
  }

  // Fallback: unknown pattern, drain all stores.
  return 0;
}

struct NVGPUTMAStoreTokenWaitLoweringPass
    : public impl::NVGPUTMAStoreTokenWaitLoweringBase<
          NVGPUTMAStoreTokenWaitLoweringPass> {
  void runOnOperation() override {
    SmallVector<ttng::TMAStoreTokenWaitOp> opsToLower;
    getOperation()->walk(
        [&](ttng::TMAStoreTokenWaitOp op) { opsToLower.push_back(op); });
    for (auto op : opsToLower) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      int pendings = computePendings(op);
      ttng::TMAStoreWaitOp::create(builder, loc, pendings);
      for (auto barrier : op.getBarriers()) {
        ttng::ArriveBarrierOp::create(builder, loc, barrier, /*count=*/1);
      }
      op.erase();
    }
  }
};

} // namespace mlir
