#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static bool hasTCGen5CommitCrossCTA(Operation *op) {
  SmallVector<Value> descs;
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    descs = mma.getCompletionDescs();
  else if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    llvm::append_range(descs, commit.getDescs());
  else
    return false;
  return !ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), descs).empty();
}

// An op reaches another CTA for one of two unrelated reasons, and which reasons
// are even possible depends on which cluster model the kernel uses (see
// `triton::gpu::isPhysicalCluster`). Keeping the two apart is what lets the two
// passes below agree about the same op.

// Reaches another CTA because a distributed tensor layout spreads the operand
// over the logical `num_ctas` cluster. Necessarily false under `ctas_per_cga`,
// where `num_ctas` is one and every tensor is CTA-local.
static bool opCrossesCTAsByLayout(Operation *op, bool isRead) {
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = cvt.getSrc().getType();
    auto dstTy = cvt.getType();
    auto kBlock = StringAttr::get(op->getContext(), "block");
    return !isCvtDimSync(ttg::toLinearLayout(srcTy), ttg::toLinearLayout(dstTy),
                         kBlock);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    if (!isRead)
      return false;
    auto srcTy = reduce.getInputTypes()[0];
    auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
    return splitNum[reduce.getAxis()] > 1;
  }
  // `async_shared_store` writes a distributed tensor, so like the two above it
  // reaches another CTA only when the logical cluster spreads that tensor.
  if (isa<ttng::AsyncSharedStoreOp>(op))
    return ttg::lookupNumCTAs(op) > 1;
  return false;
}

// Reaches another CTA because the op says so -- a `two_ctas` or multicast
// attribute, a non-zero `ctaMask`, or a remote buffer operand. These are the
// only cross-CTA ops reachable under the physical model, and they cross under
// the logical model too.
static bool opCrossesCTAsByConstruction(Operation *op) {
  if (isa<ttng::TMEMCopyOp>(op))
    return ttng::getModuleTwoCTAs(op);
  // Checked ahead of the generic interface below: only this op carries an
  // explicit `multicastTargets` mask, which the interface cannot express.
  if (auto tma = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return tma.getMulticast() || tma.getMulticastTargets();
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
    return tma.getMulticast();
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
    return arrive.isMulticast();
  if (hasTCGen5CommitCrossCTA(op))
    return true;
  return isa<ttng::CLCTryCancelOp, ttng::MapToRemoteBufferOp>(op);
}

bool isDistributedMultiCTAOp(Operation *op, bool isRead) {
  return opCrossesCTAsByLayout(op, isRead) || opCrossesCTAsByConstruction(op);
}

// The shared entry condition for both passes. `lookupPhysicalNumCTAs` is the
// no-cluster test under either model, so the model is only consulted where it
// changes the answer: a physical cluster reaches other CTAs solely through ops
// that say so, and a kernel with none of them needs no cluster sync at all.
static bool clusterCanCrossCTAs(ModuleOp mod) {
  if (ttg::lookupPhysicalNumCTAs(mod) == 1)
    return false;
  if (!ttg::isPhysicalCluster(mod))
    return true;
  return mod
      ->walk([](Operation *op) {
        return opCrossesCTAsByConstruction(op) ? WalkResult::interrupt()
                                               : WalkResult::advance();
      })
      .wasInterrupted();
}

namespace {

static bool isPreAllocAliasSliceFilter(const AllocationSlice &lhsSlice,
                                       const AllocationSlice &rhsSlice,
                                       bool /*lhsIsRead*/, bool /*rhsIsRead*/,
                                       Allocation *allocation) {
  auto bufferId = lhsSlice.getBufferId();
  return bufferId != Allocation::InvalidBufferId &&
         bufferId == rhsSlice.getBufferId() &&
         allocation->isExplicitBuffer(bufferId);
}

// Under the logical model a barrier array carries one slot per CTA, so a size
// that deviates from `num_ctas` means the barrier is not the plain per-CTA one.
// That is a statement about the barrier's layout and carries no information
// under `ctas_per_cga`, where a CTA-local barrier and a remotely completed one
// are both size one.
static bool mbarrierShapeDeviatesFromNumCTAs(ttng::InitBarrierOp initBarrierOp,
                                             int numCTAs) {
  auto barrierTy = cast<ttg::MemDescType>(initBarrierOp.getAlloc().getType());
  return barrierTy.getShape()[0] != numCTAs;
}

static bool valueAliasesTrackedBuffers(Value value,
                                       const Allocation::BufferIdSetT &tracked,
                                       Allocation *allocation) {
  for (auto bufferId : allocation->getAllBufferIdsWithAliases(value)) {
    if (bufferId != Allocation::InvalidBufferId && tracked.contains(bufferId))
      return true;
  }
  return false;
}

static bool
usesTrackedBarrierInCrossCTAConsumerOp(Operation *op,
                                       const Allocation::BufferIdSetT &tracked,
                                       Allocation *allocation) {
  auto aliasesTracked = [&](Value value) {
    return value && valueAliasesTrackedBuffers(value, tracked, allocation);
  };

  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto barrierOp = cast<ttg::MBarrierOpInterface>(op);
    return hasTCGen5CommitCrossCTA(op) &&
           llvm::any_of(barrierOp.getBarriers(), aliasesTracked);
  }
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    return hasTCGen5CommitCrossCTA(op) && aliasesTracked(commit.getBarrier());
  }
  // Checked ahead of the generic interface below: only this op carries an
  // explicit `multicastTargets` mask, which the interface cannot express.
  if (auto tma = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    return (tma.getMulticast() || tma.getMulticastTargets()) &&
           aliasesTracked(tma.getBarrier());
  }
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    return tma.getMulticast() && aliasesTracked(tma.getBarrier());
  }
  if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op)) {
    return aliasesTracked(clc.getMbarrier());
  }
  if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op)) {
    return aliasesTracked(store.getMbarrier());
  }
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op)) {
    return arrive.isMulticast() && aliasesTracked(arrive.getAlloc());
  }
  // Addressing a barrier in another CTA is what makes it cluster-visible, so
  // the mapping op counts even though the arrive lands on its result. The op is
  // Pure, though, so a dead result is never itself found by the memory-effect
  // use scan -- tracking one would strand the init with no insertion window.
  if (auto remote = dyn_cast<ttng::MapToRemoteBufferOp>(op)) {
    return !remote.getResult().use_empty() && aliasesTracked(remote.getSrc());
  }
  return false;
}

// Can a CTA other than the one that ran `init` complete or observe this
// mbarrier? If so its initialization has to be made visible cluster-wide before
// anyone touches it. The consumer walk answers this under either model; the
// logical model additionally encodes the answer in the barrier's own shape,
// which is why `num_ctas` appears here and nowhere else in this pass.
static bool mbarrierIsClusterVisible(ttng::InitBarrierOp initBarrierOp,
                                     FunctionOpInterface funcOp,
                                     Allocation *allocation) {
  auto mod = initBarrierOp->getParentOfType<ModuleOp>();
  if (!ttg::isPhysicalCluster(mod) &&
      mbarrierShapeDeviatesFromNumCTAs(initBarrierOp,
                                       ttg::TritonGPUDialect::getNumCTAs(mod)))
    return true;

  Allocation::BufferIdSetT initBarrierBuffers;
  for (auto bufferId :
       allocation->getAllBufferIdsWithAliases(initBarrierOp.getAlloc())) {
    assert(bufferId != Allocation::InvalidBufferId);
    initBarrierBuffers.insert(bufferId);
  }

  return funcOp
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (usesTrackedBarrierInCrossCTAConsumerOp(op, initBarrierBuffers,
                                                   allocation)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool nestedOpUsesTrackedMBarrier(Operation *op,
                                        const Allocation::BufferIdSetT &tracked,
                                        Allocation *allocation) {
  if (isa<ttng::InitBarrierOp, ttg::LocalAllocOp>(op))
    return false;

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    memEffects.getEffects(effects);
    for (const auto &effect : effects) {
      Value value = effect.getValue();
      if (value && valueAliasesTrackedBuffers(value, tracked, allocation))
        return true;
    }
  }
  return false;
}

static bool opUsesTrackedMBarrier(Operation *op,
                                  const Allocation::BufferIdSetT &tracked,
                                  Allocation *allocation) {
  return op
      ->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        if (nestedOpUsesTrackedMBarrier(nestedOp, tracked, allocation))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool hasWarpSpecializeOp(FunctionOpInterface funcOp) {
  return funcOp
      ->walk([](ttg::WarpSpecializeOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static LogicalResult insertCrossCTAMBarrierInitSyncForFunction(
    FunctionOpInterface funcOp, Allocation *allocation, OpBuilder &builder) {
  if (!funcOp || funcOp->getNumRegions() != 1) {
    return funcOp.emitOpError(
        "cross-CTA mbarrier init sync insertion requires a single function "
        "top-level region");
  }
  Region &topLevelRegion = funcOp->getRegion(0);
  llvm::SetVector<Operation *> crossCTAInitAnchors;
  Allocation::BufferIdSetT trackedBarrierBuffers;

  // Find all cross-CTA mbarrier.init ops and map each
  // one to the containing top-level op that bounds the insertion window.
  funcOp.walk([&](ttng::InitBarrierOp initBarrierOp) {
    if (!mbarrierIsClusterVisible(initBarrierOp, funcOp, allocation))
      return;
    Operation *topLevelAnchor =
        topLevelRegion.findAncestorOpInRegion(*initBarrierOp.getOperation());
    assert(topLevelAnchor && "init op must be inside the function region");
    crossCTAInitAnchors.insert(topLevelAnchor);
    for (auto bufferId :
         allocation->getAllBufferIdsWithAliases(initBarrierOp.getAlloc())) {
      assert(bufferId != Allocation::InvalidBufferId);
      trackedBarrierBuffers.insert(bufferId);
    }
  });
  // Nothing to do
  if (crossCTAInitAnchors.empty())
    return success();

  llvm::SetVector<Operation *> trackedUseAnchors;
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      if (opUsesTrackedMBarrier(&op, trackedBarrierBuffers, allocation))
        trackedUseAnchors.insert(&op);
    }
  }
  if (trackedUseAnchors.empty()) {
    return funcOp.emitOpError("found at least one mbarrier.init op but could "
                              "not find any mbarrier use");
  }

  // Find the earliest insertion point that postdominates every tracked init.
  PostDominanceInfo postDomInfo(funcOp);
  llvm::SmallPtrSet<Block *, 8> initBlocks;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors)
    initBlocks.insert(crossCTAInitAnchor->getBlock());
  Block *firstInsertionBlock =
      postDomInfo.findNearestCommonDominator(initBlocks);
  if (!firstInsertionBlock) {
    return funcOp.emitOpError(
        "could not find a common post-dominating insertion block for "
        "cross-CTA mbarrier.init");
  }

  Operation *lastInitInInsertionBlock = nullptr;
  for (Operation *crossCTAInitAnchor : crossCTAInitAnchors) {
    if (crossCTAInitAnchor->getBlock() != firstInsertionBlock)
      continue;
    if (!lastInitInInsertionBlock ||
        lastInitInInsertionBlock->isBeforeInBlock(crossCTAInitAnchor)) {
      lastInitInInsertionBlock = crossCTAInitAnchor;
    }
  }
  Operation *firstInsertionAnchor =
      lastInitInInsertionBlock ? lastInitInInsertionBlock->getNextNode()
                               : &firstInsertionBlock->front();

  // Find the latest insertion point that still dominates every tracked use.
  DominanceInfo domInfo(funcOp);
  llvm::SmallPtrSet<Block *, 8> useBlocks;
  for (Operation *trackedUseAnchor : trackedUseAnchors)
    useBlocks.insert(trackedUseAnchor->getBlock());
  Block *lastInsertionBlock = domInfo.findNearestCommonDominator(useBlocks);
  if (!lastInsertionBlock) {
    return funcOp.emitOpError(
        "could not find a common insertion block that dominates all tracked "
        "mbarrier uses");
  }

  Operation *firstTrackedUseInInsertionBlock = nullptr;
  for (Operation *trackedUseAnchor : trackedUseAnchors) {
    if (trackedUseAnchor->getBlock() != lastInsertionBlock)
      continue;
    if (!firstTrackedUseInInsertionBlock ||
        trackedUseAnchor->isBeforeInBlock(firstTrackedUseInInsertionBlock)) {
      firstTrackedUseInInsertionBlock = trackedUseAnchor;
    }
  }
  Operation *lastInsertionAnchor = firstTrackedUseInInsertionBlock
                                       ? firstTrackedUseInInsertionBlock
                                       : lastInsertionBlock->getTerminator();

  if (!domInfo.dominates(firstInsertionAnchor, lastInsertionAnchor)) {
    return funcOp.emitOpError(
        "could not find an insertion point between cross-CTA mbarrier.init "
        "ops and tracked mbarrier uses");
  }

  // Reuse the latest cluster barrier that lies between the init-side and
  // use-side insertion boundaries.
  ttng::ClusterBarrierOp reusedClusterBarrier;
  for (Block &block : topLevelRegion) {
    for (Operation &op : block) {
      auto clusterBarrier = dyn_cast<ttng::ClusterBarrierOp>(&op);
      if (!clusterBarrier)
        continue;
      if (!postDomInfo.postDominates(clusterBarrier.getOperation(),
                                     firstInsertionAnchor))
        continue;
      if (!domInfo.dominates(clusterBarrier.getOperation(),
                             lastInsertionAnchor))
        continue;
      if (!reusedClusterBarrier ||
          domInfo.properlyDominates(reusedClusterBarrier.getOperation(),
                                    clusterBarrier.getOperation())) {
        reusedClusterBarrier = clusterBarrier;
      }
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  Operation *fenceInsertionPoint =
      reusedClusterBarrier && reusedClusterBarrier.getRelaxed()
          ? reusedClusterBarrier.getOperation()
          : lastInsertionAnchor;
  builder.setInsertionPoint(fenceInsertionPoint);
  Location loc = lastInitInInsertionBlock
                     ? lastInitInInsertionBlock->getLoc()
                     : crossCTAInitAnchors.front()->getLoc();
  ttng::FenceMBarrierInitReleaseClusterOp::create(builder, loc);
  if (!reusedClusterBarrier)
    ttng::ClusterBarrierOp::create(builder, loc, /*relaxed=*/true);
  return success();
}

class ClusterBarrierAnalysis : public MembarOrFenceAnalysis {
public:
  ClusterBarrierAnalysis() = default;
  explicit ClusterBarrierAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override;

  void insertClusterBarrier(Operation *op, OpBuilder *builder);
};

void ClusterBarrierAnalysis::insertClusterBarrier(Operation *op,
                                                  OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  ttng::ClusterArriveOp::create(*builder, op->getLoc(), /*relaxed=*/false);
  ttng::ClusterWaitOp::create(*builder, op->getLoc());
}

void ClusterBarrierAnalysis::update(Operation *op, BlockInfo *blockInfo,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  if (isa<ttng::ClusterWaitOp>(op)) {
    blockInfo->sync();
    return;
  }

  // Any path from distributed shared memory use to kernel exit must include a
  // cluster barrier. A return-site barrier is only reached by default warps in
  // warp-specialized kernels; their lowering must provide any terminal sync.
  if (op->hasTrait<OpTrait::ReturnLike>() &&
      isa<FunctionOpInterface>(op->getParentOp())) {
    auto funcOp = cast<FunctionOpInterface>(op->getParentOp());
    if (triton::isKernel(funcOp) && !hasWarpSpecializeOp(funcOp)) {
      builder->setInsertionPoint(op);
      ttng::ClusterBarrierOp::create(*builder, op->getLoc());
      blockInfo->sync();
    }
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable())) {
      auto calleeBlockInfo = funcBlockInfoMap->lookup(callee);
      auto callBufferId = allocation->getBufferId(op);
      size_t callOffset = 0;
      if (callBufferId != Allocation::InvalidBufferId)
        callOffset = allocation->getAllocatedInterval(callBufferId).start();
      curBlockInfo = translateBlockInfoToCallsite(calleeBlockInfo, callOffset);
    }
  } else {
    if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memEffects.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              auto interval = allocation->getAllocatedInterval(bufferId);
              auto slice = AllocationSlice(value, interval, bufferId);
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo.syncWriteSlices[slice].insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo.syncReadSlices[slice].insert(op);
            }
          }
        }
      }
    }
    scratchBufferId = allocation->getBufferId(op);
  }

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
    if (!curBlockInfo.syncReadSlices.empty() ||
        !curBlockInfo.syncWriteSlices.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }

    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    auto scratchSlice = AllocationSlice(interval);
    curBlockInfo.syncWriteSlices[scratchSlice].insert(op);

    auto insertClusterBarrierNeeded = blockInfo->isIntersected(
        curBlockInfo, filter, allocation, isPreAllocAliasSliceFilter);
    if (insertClusterBarrierNeeded) {
      builder->setInsertionPoint(op);
      insertClusterBarrier(op, builder);
    }

    // Clear prior distributed dependencies if we have inserted a cluster
    // barrier, or if the scratch op itself performs a cluster-level sync.
    bool hasClusterSync = isDistributedMultiCTAOp(op, /*isRead=*/true);
    if (insertClusterBarrierNeeded || hasClusterSync)
      blockInfo->sync();

    curBlockInfo.syncReadSlices[scratchSlice].insert(op);
  } else if (blockInfo->isIntersected(curBlockInfo, filter, allocation,
                                      isPreAllocAliasSliceFilter)) {
    builder->setInsertionPoint(op);
    insertClusterBarrier(op, builder);
    blockInfo->sync();
  }

  blockInfo->join(curBlockInfo);
}

} // namespace

void runClusterBarrierInsertion(ModuleAllocation &moduleAllocation,
                                int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return;
  if (!clusterCanCrossCTAs(mod))
    return;

  MembarFilterFn filterFn = [](Operation *lhs, Operation *rhs, bool lhsIsRead,
                               bool rhsIsRead, Allocation * /*allocation*/) {
    // Filter ops that do not touch distributed shared memory. Whether the
    // aliasing was already present in TTGIR is handled per-allocation slice.
    bool lhsDist = isDistributedMultiCTAOp(lhs, lhsIsRead);
    bool rhsDist = isDistributedMultiCTAOp(rhs, rhsIsRead);
    if (!lhsDist && !rhsDist)
      return true;
    return false;
  };

  ModuleMembarOrFenceAnalysis<ClusterBarrierAnalysis> analysis(
      &moduleAllocation, filterFn);
  analysis.run();
}

LogicalResult
runCrossCTAMBarrierInitSyncInsertion(ModuleAllocation &moduleAllocation,
                                     int computeCapability) {
  ModuleOp mod = moduleAllocation.getModuleOp();
  if (computeCapability < 90)
    return success();
  // This pass owns the logical model only, so bail before `clusterCanCrossCTAs`
  // rather than after -- under `ctas_per_cga` that predicate would walk the
  // whole module and the result would be discarded here anyway.
  //
  // Not threaded through the Meta warp specialization paths yet. A cluster
  // barrier at function scope only reaches the default warps, and
  // `ConvertWarpSpecializeToLLVM` emits exactly one compensating
  // `@!isDefault barrier.cluster.arrive` in the function header -- which
  // `maybeInsertClusterSync`'s entry rendezvous already claims. A second
  // rendezvous from here is arrived at by the default warps alone and never
  // completes.
  //
  // Standing down for entry-block inits alone is not enough. Narrowing this to
  // "every classified init is in the entry block" was tried and reverted: on
  // physical-cluster kernels whose cluster-visible init sits inside a region,
  // this pass then runs and silently miscomputes -- 14 warp-specialized
  // tutorial09 matmuls returned wrong results (3-11% of elements) rather than
  // hanging or erroring. So under `ctas_per_cga` leave the sync to
  // `maybeInsertClusterSync` unconditionally.
  //
  // Known gap: that pass only scans the entry block, so a cluster-visible init
  // nested in a region now gets no sync and no diagnostic. Covering it needs a
  // mechanism neither pass has yet.
  //
  // TODO: unify the two. They emit the same fence + relaxed cluster barrier and
  // differ only in how they decide it is needed -- `maybeInsertClusterSync` on
  // a module-wide predicate over entry-block inits, this pass on a per-barrier
  // classification anywhere in the function. One pass should own the emission
  // and take the decision from the other, which also removes the "exactly one
  // rendezvous" hazard this stand-down is working around.
  if (ttg::isPhysicalCluster(mod))
    return success();
  if (!clusterCanCrossCTAs(mod))
    return success();

  LogicalResult status = success();
  moduleAllocation.walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        if (failed(status))
          return;
        auto *allocation = moduleAllocation.getFuncData(funcOp);
        OpBuilder builder(funcOp);
        if (failed(insertCrossCTAMBarrierInitSyncForFunction(funcOp, allocation,
                                                             builder))) {
          status = failure();
        }
      });
  return status;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
