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

    auto alloc = builder.create<ttg::LocalAllocOp>(loc, memDescType, src);

    // Async TMA copy from local (SMEM) to global, producing a token.
    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaStore = builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, tokenType, desc, storeOp.getIndices(), alloc,
        tt::EvictionPolicy::NORMAL);
    copyLoopScheduleAttrs(storeOp, tmaStore);

    // Wait for this specific TMA store to finish reading from SMEM.
    auto waitOp = builder.create<ttng::TMAStoreTokenWaitOp>(
        loc, tmaStore.getToken(), ValueRange{}, ValueRange{}, ValueRange{},
        ValueRange{});
    copyLoopScheduleAttrs(storeOp, waitOp);

    storeOp.erase();
  }

  // Also lower DescriptorReduceOp → local_alloc + AsyncTMAReduceOp (with token)
  // + TMAStoreTokenWaitOp, matching the early TMA store pattern.
  SmallVector<tt::DescriptorReduceOp> reduceOps;
  funcOp.walk([&](tt::DescriptorReduceOp op) { reduceOps.push_back(op); });

  if (!reduceOps.empty())
    LDBG("Lowering " << reduceOps.size() << " DescriptorReduceOp(s)");

  for (auto reduceOp : reduceOps) {
    auto loc = reduceOp.getLoc();
    OpBuilderWithAsyncTaskIds builder(reduceOp);
    builder.setInsertionPoint(reduceOp);

    auto src = reduceOp.getSrc();
    auto desc = reduceOp.getDesc();
    auto tensorType = src.getType();

    auto encoding = ttng::getEncodingFromDescriptor(reduceOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    auto alloc = builder.create<ttg::LocalAllocOp>(loc, memDescType, src);

    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaReduce = builder.create<ttng::AsyncTMAReduceOp>(
        loc, tokenType, reduceOp.getKind(), desc, reduceOp.getIndices(), alloc,
        tt::EvictionPolicy::NORMAL);
    copyLoopScheduleAttrs(reduceOp, tmaReduce);

    auto waitOp = builder.create<ttng::TMAStoreTokenWaitOp>(
        loc, tmaReduce.getToken(), ValueRange{}, ValueRange{}, ValueRange{},
        ValueRange{});
    copyLoopScheduleAttrs(reduceOp, waitOp);

    reduceOp.erase();
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

static bool isTMAStoreLikeOp(Operation *op) {
  return isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(op);
}

static Value getTMAStoreSource(Operation *op) {
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return copyOp.getSrc();
  if (auto reduceOp = dyn_cast<ttng::AsyncTMAReduceOp>(op))
    return reduceOp.getSrc();
  return {};
}

// Trace the token back to the defining TMA store-like op
// (AsyncTMACopyLocalToGlobalOp or AsyncTMAReduceOp), handling both direct
// definitions and loop-carried block arguments. Returns the SMEM source
// buffer and the defining op.
static Operation *getDefiningTMAStoreOp(ttng::TMAStoreTokenWaitOp waitOp,
                                        Value &buffer) {
  Value token = waitOp.getToken();

  // Direct case: token defined by AsyncTMACopyLocalToGlobalOp.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
    buffer = defOp.getSrc();
    return defOp;
  }

  // Direct case: token defined by AsyncTMAReduceOp.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMAReduceOp>()) {
    buffer = defOp.getSrc();
    return defOp;
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);
    if (auto defOp =
            yieldedVal.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
      buffer = defOp.getSrc();
      return defOp;
    }
    if (auto defOp = yieldedVal.getDefiningOp<ttng::AsyncTMAReduceOp>()) {
      buffer = defOp.getSrc();
      return defOp;
    }
  }

  return nullptr;
}

static bool sameMemDescValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef)
    return false;
  if (lhsDef->getName() != rhsDef->getName())
    return false;
  if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
           ttg::MemDescReinterpretOp>(lhsDef))
    return false;
  if (lhsDef->getNumOperands() != rhsDef->getNumOperands())
    return false;

  for (unsigned i = 0; i < lhsDef->getNumOperands(); ++i) {
    if (lhsDef->getOperand(i) != rhsDef->getOperand(i))
      return false;
  }
  return true;
}

static Operation *
findLocalStoreWritingBuffer(scf::ForOp forOp, Value buffer,
                            const tt::CoarseSchedule &schedule) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    auto localStore = dyn_cast<ttg::LocalStoreOp>(&op);
    if (!localStore || !schedule.count(&op))
      continue;
    if (sameMemDescValue(localStore.getDst(), buffer))
      return &op;
  }
  return nullptr;
}

void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  // Use walk to find TMAStoreTokenWaitOp ops inside ForOp bodies, including
  // those nested inside SubtiledRegionOp regions.
  funcOp.walk([&](scf::ForOp forOp) {
    forOp.walk([&](ttng::TMAStoreTokenWaitOp waitOp) {
      Value buffer;
      auto *tmaOp = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaOp)
        return;

      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp)
        return;

      // Only annotate buffers that have buffer.copy from the memory planner.
      // Buffers without buffer.copy were not planned and cannot be rotated.
      auto bufferCopy = allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
      if (!bufferCopy)
        return;

      int k = bufferCopy.getInt();
      if (k <= 0)
        return;

      waitOp->setAttr(kCanRotateByBufferCount,
                      IntegerAttr::get(IntegerType::get(ctx, 32), k));
    });
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
    forOp.walk([&](ttng::TMAStoreTokenWaitOp waitOp) {
      if (!waitOp->hasAttr(kCanRotateByBufferCount))
        return;

      Value buffer;
      auto *tmaOp = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaOp) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        return;
      }

      auto allocOp = buffer.getDefiningOp<ttg::LocalAllocOp>();
      if (!allocOp) {
        waitOp->removeAttr(kCanRotateByBufferCount);
        return;
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Reschedule TMA store waits using the SWP CoarseSchedule
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTESTTMASTORETOKENWAITREORDER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

static Operation *
findScheduledWaitBarrierBetween(Operation *producer, Operation *insertionTarget,
                                const tt::CoarseSchedule &schedule,
                                bool includeBarrierBeforeProducer) {
  auto findBefore = [&](Operation *op, Operation *stopOp) -> Operation * {
    for (auto revIt = Block::reverse_iterator(op->getIterator());
         revIt != op->getBlock()->rend(); ++revIt) {
      Operation *candidate = &*revIt;
      if (candidate == stopOp)
        return nullptr;
      if (isa<ttng::WaitBarrierOp>(candidate) && schedule.count(candidate))
        return candidate;
    }
    return nullptr;
  };

  if (producer->isBeforeInBlock(insertionTarget))
    return findBefore(insertionTarget, producer);

  if (!includeBarrierBeforeProducer)
    return nullptr;

  Operation *wraparoundTarget =
      insertionTarget->isBeforeInBlock(producer) ? insertionTarget : producer;
  for (auto revIt = Block::reverse_iterator(wraparoundTarget->getIterator());
       revIt != producer->getBlock()->rend(); ++revIt) {
    Operation *candidate = &*revIt;
    if (isTMAStoreLikeOp(candidate))
      return nullptr;
    if (isa<ttng::WaitBarrierOp>(candidate) && schedule.count(candidate))
      return candidate;
  }

  return nullptr;
}

LogicalResult doTMAStoreWaitReorder(triton::FuncOp &funcOp) {
  bool failedToReorder = false;
  funcOp.walk([&](scf::ForOp forOp) {
    if (failedToReorder)
      return;

    bool hasNestedFor = false;
    forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
    if (hasNestedFor)
      return;

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
      Value buffer;
      auto *tmaStore = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaStore || tmaStore->getParentOp() != forOp)
        continue;
      waits.push_back(waitOp);
    }
    if (waits.empty())
      return;

    int numTMAStores = 0;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isTMAStoreLikeOp(&op))
        ++numTMAStores;
    }

    bool changed = false;
    for (auto waitOp : waits) {
      auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
      if (!attr)
        continue;
      int k = attr.getInt();

      // Find the defining TMA store op.
      Value buffer;
      auto *tmaStore = getDefiningTMAStoreOp(waitOp, buffer);
      if (!tmaStore)
        continue;

      // The defining op must be in the schedule for the LinearizedIterator.
      if (!schedule.count(tmaStore))
        continue;

      // Walk the linearized schedule from the TMA store, counting K
      // TMA store-like ops. The wait must be placed before the K-th op to
      // ensure the buffer slot is not overwritten.
      auto it = schedule.linearized(forOp, tmaStore);
      it.setMaxStages(schedule.getNumStages() + k);

      // Skip past the starting TMA store itself.
      ++it;

      Operation *insertionTarget = nullptr;
      int targetStage = 0;
      int storeCount = 0;

      while (!it.isEnd()) {
        Operation *op = *it;
        int stageAtOp = it.currStage();
        ++it;
        if (isTMAStoreLikeOp(op)) {
          ++storeCount;
          if (storeCount == k) {
            insertionTarget = op;
            targetStage = stageAtOp;
            break;
          }
        }
      }

      if (insertionTarget) {
        Operation *targetTMAStore = insertionTarget;
        int numPrevTMAStores = 0;
        for (auto &op : forOp.getBody()->without_terminator()) {
          if (&op == tmaStore)
            break;
          if (isTMAStoreLikeOp(&op))
            ++numPrevTMAStores;
        }

        Value targetBuffer = getTMAStoreSource(targetTMAStore);
        Operation *targetWriter =
            targetBuffer
                ? findLocalStoreWritingBuffer(forOp, targetBuffer, schedule)
                : nullptr;
        if (targetWriter) {
          insertionTarget = targetWriter;
        } else {
          // If the buffer is updated by a different partition, the TMA store
          // must be guarded by that partition's wait_barrier. Reorder before
          // the barrier so the token wait completes before the target buffer
          // can be updated.
          Operation *waitBarrier = findScheduledWaitBarrierBetween(
              tmaStore, targetTMAStore, schedule,
              k >= (numTMAStores - numPrevTMAStores));
          if (!waitBarrier) {
            forOp.emitOpError(
                "failed to find wait_barrier guarding target TMA store");
            failedToReorder = true;
            return;
          }
          insertionTarget = waitBarrier;
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
  return failure(failedToReorder);
}

struct NVGPUTestTMAStoreTokenWaitReorderPass
    : public impl::NVGPUTestTMAStoreTokenWaitReorderBase<
          NVGPUTestTMAStoreTokenWaitReorderPass> {
  void runOnOperation() override {
    bool passFailed = false;
    getOperation()->walk([&](triton::FuncOp funcOp) {
      if (failed(doTMAStoreWaitReorder(funcOp)))
        passFailed = true;
    });
    if (passFailed)
      signalPassFailure();
  }
};

// ---------------------------------------------------------------------------
// Lower TMAStoreTokenWaitOp with barriers into TMAStoreWaitOp + ArriveBarrierOp
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITLOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

// Count TMA store-like ops (AsyncTMACopyLocalToGlobalOp and AsyncTMAReduceOp)
// in [from, to) within a block.
static int countTMAStoresInRange(Block::iterator from, Block::iterator to) {
  int count = 0;
  for (auto it = from; it != to; ++it) {
    if (isTMAStoreLikeOp(&*it))
      ++count;
  }
  return count;
}

// Compute the pendings value for a TMAStoreTokenWaitOp.
// pendings = number of TMA store-like ops issued after the token's defining
// store and before this wait, in program execution order.
static int computePendings(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  // Direct case: token defined by a TMA store-like op in same block.
  auto directDef = token.getDefiningOp();
  if (directDef && isTMAStoreLikeOp(directDef)) {
    if (directDef->getBlock() == waitOp->getBlock()) {
      return countTMAStoresInRange(std::next(directDef->getIterator()),
                                   waitOp->getIterator());
    }
    return 0;
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return 0;

    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);

    // Trace the yielded value to its defining TMA store-like op.
    auto defOp = yieldedVal.getDefiningOp();
    if (!defOp || !isTMAStoreLikeOp(defOp) ||
        defOp->getBlock() != forOp.getBody())
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
