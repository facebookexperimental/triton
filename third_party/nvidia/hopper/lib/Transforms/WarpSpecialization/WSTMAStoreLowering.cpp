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

    // Derive a NameLoc for the staging alloc from the descriptor's name.
    auto allocLoc = loc;
    Location descLoc = desc.getLoc();
    if (auto *defOp = desc.getDefiningOp())
      descLoc = defOp->getLoc();
    // Walk through CallSiteLoc/FusedLoc to find a NameLoc.
    auto findName = [](Location l, auto &self) -> StringRef {
      if (auto nameLoc = dyn_cast<NameLoc>(l))
        return nameLoc.getName().strref();
      if (auto callSiteLoc = dyn_cast<CallSiteLoc>(l))
        return self(callSiteLoc.getCallee(), self);
      if (auto fusedLoc = dyn_cast<FusedLoc>(l))
        for (Location sub : fusedLoc.getLocations()) {
          auto s = self(sub, self);
          if (!s.empty())
            return s;
        }
      return {};
    };
    auto descName = findName(descLoc, findName);
    if (!descName.empty()) {
      auto stagingName = (descName + "_staging").str();
      allocLoc = NameLoc::get(StringAttr::get(ctx, stagingName), loc);
    }
    auto alloc = builder.create<ttg::LocalAllocOp>(allocLoc, memDescType, src);

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

    // Derive a NameLoc for the staging alloc from the descriptor's name.
    auto allocLoc = loc;
    Location descLoc = desc.getLoc();
    if (auto *defOp = desc.getDefiningOp())
      descLoc = defOp->getLoc();
    auto findName = [](Location l, auto &self) -> StringRef {
      if (auto nameLoc = dyn_cast<NameLoc>(l))
        return nameLoc.getName().strref();
      if (auto callSiteLoc = dyn_cast<CallSiteLoc>(l))
        return self(callSiteLoc.getCallee(), self);
      if (auto fusedLoc = dyn_cast<FusedLoc>(l))
        for (Location sub : fusedLoc.getLocations()) {
          auto s = self(sub, self);
          if (!s.empty())
            return s;
        }
      return {};
    };
    auto descName = findName(descLoc, findName);
    if (!descName.empty()) {
      auto stagingName = (descName + "_reduce_staging").str();
      allocLoc = NameLoc::get(StringAttr::get(ctx, stagingName), loc);
    }
    auto alloc = builder.create<ttg::LocalAllocOp>(allocLoc, memDescType, src);

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

// (Previously a legacy AsyncTMACopyLocalToGlobalOp-only wrapper lived here;
// it silently dropped AsyncTMAReduceOp-based tokens such as dq's TMA reduce
// staging, defeating doTMAStoreWaitReorder for those waits. Callers now use
// getDefiningTMAStoreOp directly so both store flavors are handled.)

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

void doTMAStoreWaitReorder(triton::FuncOp &funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
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
    // loop and whose defining TMA store (or reduce) is in the same loop.
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

    bool changed = false;
    for (auto waitOp : waits) {
      auto attr = waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount);
      if (!attr)
        continue;
      int k = attr.getInt();

      // Find the defining TMA store-like op (copy_local_to_global or reduce).
      Value buffer;
      auto *tmaStore = getDefiningTMAStoreOp(waitOp, buffer);
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
        if (isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(
                op)) {
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
        // NOTE: we update the SWP schedule only. We do not physically
        // move the wait op in source order — the LinearizedIterator's
        // K-th forward TMA store can resolve to an insertion target that
        // is earlier than the defining store in source (the wrap-around
        // returns the same body op at a future stage), so a blind
        // moveBefore would violate SSA dominance. For WS partition loops
        // where SWP ExpandLoops does not source-reorder the body the net
        // effect today is that `NVGPUTMAStoreTokenWaitLowering` will see
        // 0 stores between the defining store and the wait and emit
        // `pendings = 0`. Realising the rotation in source order is
        // tracked as future work; until then the planner-side
        // `buffer.copy > 1` only adds extra slots without overlap.
      } else {
        // Target not found; leave the schedule unchanged for this wait.
        continue;
      }

      // Leave the kCanRotateByBufferCount annotation in place. The lowering
      // pass (computePendings) consumes it as a "trusted planner" signal when
      // the source-order TMA store count between the defining store and this
      // wait is less than K-1 (which happens when no source-level reorder
      // occurred, e.g., for WS partition loops where SWP ExpandLoops does
      // not peel based on the cluster/stage schedule update above).
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

// Count TMA store-like ops (AsyncTMACopyLocalToGlobalOp and AsyncTMAReduceOp)
// in [from, to) within a block.
static int countTMAStoresInRange(Block::iterator from, Block::iterator to) {
  int count = 0;
  for (auto it = from; it != to; ++it) {
    if (isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(&*it))
      ++count;
  }
  return count;
}

// Compute the pendings value for a TMAStoreTokenWaitOp.
//
// Primary signal: the number of AsyncTMACopyLocalToGlobalOp / AsyncTMAReduceOp
// ops issued between the token's defining store and this wait, in program
// execution order. That count directly maps to the CUDA "tma_store_wait N"
// semantics ("wait until at most N stores are still in flight") when source
// order matches execution order, which is the case after SWP ExpandLoops has
// peeled an outer loop.
//
// Fallback (Option C): when the source-order count is less than K-1 but the
// wait carries the planner's `can_rotate_by_buffer_count = K` annotation with
// K > 1, return K-1. This trusts the planner's intent: it allocated K rotating
// slots and verified the launch pattern is round-robin-safe, so K-1 stores
// from the same group may safely remain in flight when we wait on this token.
// This is what makes per-iteration dq subtile rotation actually pipeline in
// WS partition loops, where the cluster/stage schedule update from
// doTMAStoreWaitReorder does not translate into a source-order rewrite.
//
// Option C only fires when we successfully matched a defining store (direct
// or loop-carried via scf.for block-arg). When neither pattern matches (e.g.,
// kernel-exit drain whose token is an scf.for / cf.br result), we fall back
// to the conservative pendings = 0 (full drain) rather than trusting the
// annotation in a context we don't understand.
static int computePendings(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  int sourceCount = 0;
  bool matchedDefiningStore = false;

  // Direct case: token defined by a TMA store-like op in same block.
  auto directDef = token.getDefiningOp();
  if (directDef &&
      isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(
          directDef)) {
    if (directDef->getBlock() == waitOp->getBlock()) {
      sourceCount = countTMAStoresInRange(std::next(directDef->getIterator()),
                                          waitOp->getIterator());
      matchedDefiningStore = true;
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    // Loop-carried case: token is a block argument of an scf.for body.
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      unsigned iterArgIdx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value yieldedVal = yieldOp.getOperand(iterArgIdx);
      auto defOp = yieldedVal.getDefiningOp();
      if (defOp &&
          isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAReduceOp>(
              defOp) &&
          defOp->getBlock() == forOp.getBody()) {
        Block *body = forOp.getBody();
        int storesAfterDef = countTMAStoresInRange(
            std::next(defOp->getIterator()), body->end());
        int storesBeforeWait =
            countTMAStoresInRange(body->begin(), waitOp->getIterator());
        sourceCount = storesAfterDef + storesBeforeWait;
        matchedDefiningStore = true;
      }
    }
  }

  // Option C: if the planner annotated this wait with
  // can_rotate_by_buffer_count = K and the source-order count is too low to
  // realize the rotation (because no upstream pass peeled / interleaved the
  // source order), trust the planner and emit K-1. Only fires when we
  // matched a defining store — kernel-exit drains whose tokens flow through
  // an scf.for result or cf.br block-arg keep the conservative pendings = 0.
  if (matchedDefiningStore) {
    if (auto attr =
            waitOp->getAttrOfType<IntegerAttr>(kCanRotateByBufferCount)) {
      int k = attr.getInt();
      if (k > 1 && sourceCount < k - 1)
        return k - 1;
    }
  }

  return sourceCount;
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
