#include "CodePartitionUtility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-tma-store-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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

    // Wait for this specific TMA store to finish reading from SMEM.
    builder.create<ttng::TMAStoreTokenWaitOp>(loc, tmaStore.getToken(),
                                              ValueRange{}, ValueRange{},
                                              ValueRange{}, ValueRange{});

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
// Reschedule TMA store waits using the SWP CoarseSchedule
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITREORDER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

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

// For a TMAStoreTokenWaitOp, return the SMEM memdesc that the corresponding
// TMA store reads from (i.e., the src operand of AsyncTMACopyLocalToGlobalOp).
static Value getTMAStoreBuffer(ttng::TMAStoreTokenWaitOp waitOp) {
  if (auto tmaStore = getDefiningTMAStore(waitOp))
    return tmaStore.getSrc();
  return nullptr;
}

static constexpr const char *kCanRotateByBufferCount =
    "can_rotate_by_buffer_count";

void doTMAStoreWaitReorder(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    // Deserialize the SWP schedule. If there is no schedule, skip this loop.
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      return;

    // Collect annotated TMA store waits.
    SmallVector<ttng::TMAStoreTokenWaitOp> waits;
    for (auto &op : forOp.getBody()->without_terminator()) {
      auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
      if (!waitOp || !waitOp->hasAttr(kCanRotateByBufferCount))
        continue;
      waits.push_back(waitOp);
    }
    if (waits.empty())
      return;

    bool changed = false;
    for (auto waitOp : waits) {
      Value buffer = getTMAStoreBuffer(waitOp);
      if (!buffer)
        continue;

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

      // Walk the linearized schedule from the TMA store, counting local_store
      // ops to the same buffer. Stop at the K-th one.
      auto it = schedule.linearized(forOp, tmaStore);
      int count = 0;
      Operation *kthWrite = nullptr;
      while (auto op = it.findNext([&](Operation *op) {
        auto storeOp = dyn_cast<ttg::LocalStoreOp>(op);
        return storeOp && storeOp.getDst() == buffer;
      })) {
        if (++count == k) {
          kthWrite = *op;
          break;
        }
      }

      if (kthWrite) {
        // Place the wait in a new cluster just before the K-th write's cluster
        // at the same stage. The cluster ordering ensures the wait executes
        // before the write within that pipeline time slot.
        auto [targetStage, targetCluster] = schedule[kthWrite];
        auto newCluster = schedule.clusters.newBefore(targetCluster);
        schedule.insert(waitOp, targetStage, newCluster);
      } else {
        // Fallback: place at the last stage in a new cluster at the back.
        auto lastCluster = schedule.clusters.newAtBack();
        schedule.insert(waitOp, schedule.getNumStages() - 1, lastCluster);
      }

      waitOp->removeAttr(kCanRotateByBufferCount);
      changed = true;
    }

    if (changed)
      schedule.serialize(forOp);
  });
}

struct NVGPUTMAStoreTokenWaitReorderPass
    : public impl::NVGPUTMAStoreTokenWaitReorderBase<
          NVGPUTMAStoreTokenWaitReorderPass> {
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
      builder.create<ttng::TMAStoreWaitOp>(loc, pendings);
      for (auto barrier : op.getBarriers()) {
        builder.create<ttng::ArriveBarrierOp>(loc, barrier, /*count=*/1);
      }
      op.erase();
    }
  }
};

} // namespace mlir
