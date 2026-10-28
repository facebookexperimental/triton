#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>
#include <optional>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMBARRIERINSERTIONPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUTMEMWAITINSERTIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

enum class TMemAccessKind { None, Load, Store, MMA };

// Fine grain modeling of TMEM ops as pipelining behavior is not fully
// represented in ops attributes.
static bool isWritingAlloc(Operation *op) {
  auto alloc = dyn_cast<TMEMAllocOp>(op);
  return alloc && alloc.getSrc();
}

static bool isMMALikeOp(Operation *op) {
  return isa<TCGen5MMAOp, TCGen5MMAScaledOp, TMEMCopyOp, TMEMShiftOp>(op);
}

static TMemAccessKind getTMemAccessKind(Operation *op) {
  if (isa<TMEMLoadOp>(op))
    return TMemAccessKind::Load;
  if (isa<TMEMStoreOp>(op) || isWritingAlloc(op))
    return TMemAccessKind::Store;
  if (isMMALikeOp(op))
    return TMemAccessKind::MMA;
  return TMemAccessKind::None;
}

struct TMemAccessInfo {
  Region *warpScope;
  SmallVector<DenseSet<uint32_t>> addressesByWarp;
};

static Region *getWarpScope(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (isa_and_nonnull<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(
            parent))
      return region;
    region = parent ? parent->getParentRegion() : nullptr;
  }
  return nullptr;
}

// Resolve a tensor-memory descriptor to one statically known address. Control
// flow merges are deliberately not unified here: if an access can name more
// than one allocation, keep the CTA barrier.
static std::optional<uint32_t> getTMemBaseAddress(Value value) {
  DenseSet<Value> seen;
  uint32_t offset = 0;
  while (seen.insert(value).second) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
          arg.getOwner()->getParentOp());
      if (!partitions)
        return std::nullopt;
      value = partitions.getExplicitCaptures()[arg.getArgNumber()];
      continue;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    if (auto alloc = dyn_cast<TMEMAllocOp>(defOp)) {
      auto col = alloc->getAttrOfType<IntegerAttr>("tensor_memory_col_offset");
      auto row = alloc->getAttrOfType<IntegerAttr>("tensor_memory_row_offset");
      if (!col || !row)
        return std::nullopt;
      return offset + static_cast<uint32_t>(col.getInt()) +
             (static_cast<uint32_t>(row.getInt()) << 16);
    }
    if (auto slice = dyn_cast<TMEMSubSliceOp>(defOp)) {
      offset += getTMemSubSliceOffset(slice.getSrc().getType(),
                                      slice.getOffset(), slice.getDim());
      value = slice.getSrc();
      continue;
    }
    if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(defOp)) {
      value = reinterpret.getSrc();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

struct PerWarpTMemOpInfo {
  RankedTensorType regType;
  ttg::MemDescType memDescType;
  Value memDesc;
};

// These operations lower to one tcgen05.ld/st instruction stream per warp.
// MMA and copy operations use an elected issuer or multicast and are excluded.
static std::optional<PerWarpTMemOpInfo> getPerWarpTMemOpInfo(Operation *op) {
  if (auto load = dyn_cast<TMEMLoadOp>(op))
    return PerWarpTMemOpInfo{load.getType(), load.getSrc().getType(),
                             load.getSrc()};
  if (auto store = dyn_cast<TMEMStoreOp>(op))
    return PerWarpTMemOpInfo{store.getSrc().getType(), store.getDst().getType(),
                             store.getDst()};
  if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    if (alloc.getSrc())
      return PerWarpTMemOpInfo{alloc.getSrc().getType(), alloc.getType(),
                               alloc.getResult()};
  }
  return std::nullopt;
}

static std::optional<TMemAccessInfo> getTMemAccessInfo(Operation *op) {
  auto accessOp = getPerWarpTMemOpInfo(op);
  std::optional<int> numWarps = ttg::maybeLookupNumWarps(op);
  if (!accessOp || !numWarps)
    return std::nullopt;

  std::optional<uint32_t> baseAddress = getTMemBaseAddress(accessOp->memDesc);
  if (!baseAddress)
    return std::nullopt;

  auto addressesByWarp =
      getTMemLdStWarpAddresses(accessOp->regType, accessOp->memDescType,
                               getContextualMaxNReg(op), *baseAddress);
  if (failed(addressesByWarp) || addressesByWarp->size() != *numWarps)
    return std::nullopt;

  TMemAccessInfo access{getWarpScope(op), std::move(*addressesByWarp)};
  if (llvm::any_of(access.addressesByWarp,
                   [](const auto &addresses) { return addresses.empty(); }))
    return std::nullopt;
  return access;
}

using TMemAccessCache = DenseMap<Operation *, std::optional<TMemAccessInfo>>;

static void cacheTMemAccessInfo(Operation *op, TMemAccessCache &cache) {
  auto [it, inserted] = cache.try_emplace(op);
  if (inserted)
    it->second = getTMemAccessInfo(op);
}

static bool isWarpLocalHazard(Operation *lhs, Operation *rhs,
                              TMemAccessCache &cache) {
  cacheTMemAccessInfo(lhs, cache);
  cacheTMemAccessInfo(rhs, cache);
  const std::optional<TMemAccessInfo> &lhsAccess = cache.find(lhs)->second;
  const std::optional<TMemAccessInfo> &rhsAccess = cache.find(rhs)->second;
  if (!lhsAccess || !rhsAccess ||
      lhsAccess->warpScope != rhsAccess->warpScope ||
      lhsAccess->addressesByWarp.size() != rhsAccess->addressesByWarp.size())
    return false;

  for (auto [lhsWarp, lhsAddresses] :
       llvm::enumerate(lhsAccess->addressesByWarp)) {
    for (auto [rhsWarp, rhsAddresses] :
         llvm::enumerate(rhsAccess->addressesByWarp)) {
      if (lhsWarp == rhsWarp)
        continue;
      const auto &smaller = lhsAddresses.size() < rhsAddresses.size()
                                ? lhsAddresses
                                : rhsAddresses;
      const auto &larger = lhsAddresses.size() < rhsAddresses.size()
                               ? rhsAddresses
                               : lhsAddresses;
      if (llvm::any_of(smaller, [&](uint32_t address) {
            return larger.contains(address);
          }))
        return false;
    }
  }
  return true;
}

static bool filterFn(Operation *lhs, Operation *rhs, bool /*lhsIsRead*/,
                     bool /*rhsIsRead*/, Allocation * /*allocation*/,
                     TMemAccessCache &cache) {
  TMemAccessKind lhsKind = getTMemAccessKind(lhs);
  TMemAccessKind rhsKind = getTMemAccessKind(rhs);

  bool war =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::Store;
  bool raw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Load;
  bool waw =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::Store;

  // Elected async TMEM ops are special cases: we care about load/store -> op
  // dependencies, but op -> load/store doesn't require a barrier
  // since it would need a mbarrier wait that will ensure the op is finished
  // before any thread can reach the load/store.
  bool loadToMma =
      lhsKind == TMemAccessKind::Load && rhsKind == TMemAccessKind::MMA;
  bool storeToMma =
      lhsKind == TMemAccessKind::Store && rhsKind == TMemAccessKind::MMA;

  // A hazard between two per-warp accesses needs no CTA rendezvous when every
  // physical address is owned by the same warp at both endpoints. Program
  // order plus the tcgen05.wait::{ld,st} emitted by lowering is sufficient.
  if ((war || raw || waw) && isWarpLocalHazard(lhs, rhs, cache))
    return true;

  bool requiresBarrier = war || raw || waw || loadToMma || storeToMma;
  return !requiresBarrier;
}

static bool isTensorMemory(Value value) {
  auto memDescType = dyn_cast<gpu::MemDescType>(value.getType());
  return memDescType &&
         isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace());
}

static BlockInfo getTMemAccesses(Operation *op, BufferRegionAnalysis &regions) {
  BlockInfo accesses;
  for (const MemoryAccess &access : getMemoryAccesses(op)) {
    if (!isTensorMemory(access.value))
      continue;
    // Footprints preserve TMEM rows and columns; a full interval makes unknown
    // footprints alias all TMEM.
    AllocationSlice slice(Interval<size_t>{});
    slice.physicalFootprint = regions.getFootprint(access.value);
    if (access.isRead)
      accesses.syncReadSlices[slice].insert(op);
    if (access.isWrite)
      accesses.syncWriteSlices[slice].insert(op);
  }
  return accesses;
}

class TMemBarrierAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;
};

void TMemBarrierAnalysis::update(Operation *op, MembarInfo *membarInfo,
                                 FuncMapT *funcMap, OpBuilder *builder) {
  // Per-thread arrives are not local barriers, so they fall through to the
  // generic path untouched.
  if (auto arrive = dyn_cast<ArriveBarrierOp>(op);
      arrive && !arrive.getPerThread()) {
    // An mbarrier arrive signals another task; it orders nothing across this
    // task's own warps. Drop pending writes (same-task WAW needs no barrier)
    // but keep pending reads: a later store to the same TMEM is an intra-task
    // WAR hazard that still needs a barrier.
    membarInfo->pending.syncWriteSlices.clear();
    return;
  }
  auto stages = mlir::getLocalBarrierStages(op, &allocation);
  if (stages.beforeMemoryEffects)
    membarInfo->sync();

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    if (auto summary = getCallSummary(call, *funcMap)) {
      // Tensor-memory allocation attributes are physical addresses assigned
      // across the whole module, so callee slices need no call-frame offset.
      syncIfNeeded(op, summary->entryBlockInfo, membarInfo, builder);
      membarInfo->applyCallSummary(*summary);
    }
    return;
  }

  BlockInfo curBlockInfo = getTMemAccesses(op, regions);
  syncIfNeeded(op, curBlockInfo, membarInfo, builder);

  membarInfo->addBlockInfo(curBlockInfo);
  // A leading or interior rendezvous must preserve the operation's own effects.
  if (stages.afterMemoryEffects ||
      (stages.betweenMemoryEffects && curBlockInfo.syncReadSlices.empty() &&
       curBlockInfo.syncWriteSlices.empty()))
    membarInfo->sync();
}

// Ops before which we 100% need to flush reads and writes
bool isWaitBoundary(Operation *op) {
  // Defer waits at CTA barriers until a hazard or boundary requires them.
  if (isa<gpu::BarrierOp>(op))
    return false;

  // Flush before starting or ending WS
  if (isa<gpu::WarpSpecializeOp, gpu::WarpSpecializePartitionsOp,
          gpu::WarpYieldOp, gpu::WarpReturnOp, CallOpInterface>(op))
    return true;
  // A non-relaxed cluster barrier could be followed by a 2CTA MMA, in which
  // case we need to put the wait before the cluster barrier.
  // We could track barriers and cluster barriers separately
  // and place the wait before the previous relevant op, but
  // I don't think there are many use cases for that.
  if (auto barrier = dyn_cast<ClusterBarrierOp>(op))
    return !barrier.getRelaxed();
  // Atomic ops may synchronise as well
  if (auto atomic = dyn_cast<AtomicOpInterface>(op))
    if (atomic.getMemSemantic() == MemSemantic::RELEASE ||
        atomic.getMemSemantic() == MemSemantic::ACQUIRE_RELEASE)
      return true;
  // Mbarriers may signal other WS partitions.
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op))
    if (!barrier.getBarriers().empty())
      return true;

  // Carry pending accesses through control-flow joins and backedges.
  if (isa<BranchOpInterface, RegionBranchOpInterface>(op) ||
      (isa<RegionBranchTerminatorOpInterface>(op) &&
       isa<RegionBranchOpInterface>(op->getParentOp())))
    return false;
  if (op->hasTrait<OpTrait::ReturnLike>())
    return true;

  // Volatile loads add an opaque write effect to prevent optimization, but
  // still access only global memory.
  if (isa<triton::LoadOp>(op))
    return false;

  // Wait if we can't prove that it's memory-effect free
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return !isMemoryEffectFree(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    Value value = effect.getValue();
    // - An unbound DefaultResource effect may hide synchronization
    //   in impure inline asm or extern calls.
    if (!value)
      return isa<SideEffects::DefaultResource>(effect.getResource());
    return isTensorMemory(value) &&
           isa<MemoryEffects::Free>(effect.getEffect());
  });
}

class TMemWaitAnalysis : public MembarAnalysis {
public:
  using MembarAnalysis::MembarAnalysis;

private:
  static BlockInfo::SliceMapT &accesses(BlockInfo &info, TMEMWaitKind kind) {
    return kind == TMEMWaitKind::LOAD ? info.syncReadSlices
                                      : info.syncWriteSlices;
  }

  bool hasHazard(const BlockInfo &pending, const BlockInfo &effects,
                 TMEMWaitKind kind) {
    return pending.isIntersected(
        effects,
        [kind](Operation *, Operation *, bool isRead, bool, Allocation *) {
          return isRead != (kind == TMEMWaitKind::LOAD);
        },
        &allocation);
  }

  void waitBeforeBarrier(TMEMWaitKind kind, BlockInfo &pending,
                         OpBuilder *builder) {
    if (accesses(beforeBarrier, kind).empty())
      return;
    assert(lastBarrier && "publication needs a remembered rendezvous");
    builder->setInsertionPoint(lastBarrier);
    TMEMWaitOp::create(*builder, lastBarrier->getLoc(), kind);
    // The unconditional wait completes all preceding accesses of this kind.
    // Preserve later accesses, even if the same operation also has pending
    // accesses from an earlier loop iteration.
    accesses(beforeBarrier, kind).clear();
    accesses(pending, kind).clear();
    pending.join(afterBarrier);
  }

  void waitNow(Operation *op, TMEMWaitKind kind, BlockInfo &pending,
               OpBuilder *builder) {
    if (accesses(pending, kind).empty())
      return;
    builder->setInsertionPoint(op);
    TMEMWaitOp::create(*builder, op->getLoc(), kind);
    accesses(pending, kind).clear();
    accesses(afterBarrier, kind).clear();
  }

  void rememberBarrier(Operation *op, const BlockInfo &pending) {
    lastBarrier = op;
    beforeBarrier = pending;
    afterBarrier.sync();
  }

  void finishBlock(BlockInfo &pending, OpBuilder *builder) {
    // Complete accesses preceding lastBarrier before forgetting it.
    // Leave later accesses in pending for joins and backedges.
    for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
      waitBeforeBarrier(kind, pending, builder);
    lastBarrier = nullptr;
    afterBarrier.sync();
  }

  void update(Operation *op, MembarInfo *membarInfo, FuncMapT *,
              OpBuilder *builder) override {
    BlockInfo &pending = membarInfo->pending;
    if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
      accesses(pending, wait.getKind()).clear();
      accesses(afterBarrier, wait.getKind()).clear();
      return;
    }

    BlockInfo effects = getTMemAccesses(op, regions);
    // Include internal scratch barriers even without descriptor memory effects.
    auto stages = getLocalBarrierStages(op, &allocation);
    // Reuse this barrier for incoming hazards only if it precedes the op's TMEM
    // effects, or the op has no TMEM effects.
    bool beforeEffects =
        stages.beforeMemoryEffects ||
        (effects.syncReadSlices.empty() && effects.syncWriteSlices.empty());
    if (stages.hasBarrier() && beforeEffects)
      rememberBarrier(op, pending);

    if (isWaitBoundary(op)) {
      finishBlock(pending, builder);
      for (TMEMWaitKind kind : {TMEMWaitKind::LOAD, TMEMWaitKind::STORE})
        waitNow(op, kind, pending, builder);
    }

    bool loadHazard = hasHazard(pending, effects, TMEMWaitKind::LOAD) ||
                      hasHazard(beforeBarrier, effects, TMEMWaitKind::LOAD);
    bool storeHazard = hasHazard(pending, effects, TMEMWaitKind::STORE) ||
                       hasHazard(beforeBarrier, effects, TMEMWaitKind::STORE);
    if (loadHazard || storeHazard) {
      // Reuse the last barrier if no later access conflicts with this
      // operation. Otherwise, insert a wait and a new barrier before the
      // operation.
      if (lastBarrier &&
          !afterBarrier.isIntersected(effects, nullptr, &allocation)) {
        if (loadHazard)
          waitBeforeBarrier(TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitBeforeBarrier(TMEMWaitKind::STORE, pending, builder);
      } else {
        if (loadHazard)
          waitNow(op, TMEMWaitKind::LOAD, pending, builder);
        if (storeHazard)
          waitNow(op, TMEMWaitKind::STORE, pending, builder);
        builder->setInsertionPoint(op);
        insertBarrier(op, builder);
        rememberBarrier(op->getPrevNode(), pending);
      }
    }

    // Handle this operation's hazards before recording its internal barrier.
    // Waits go before the operation, so its own accesses remain in
    // afterBarrier.
    if (stages.hasBarrier() && !beforeEffects)
      rememberBarrier(op, pending);

    // Track loads and stores, including allocation initializers. An allocation
    // without an initializer preserves pending accesses to reused storage.
    auto kind = getTMemAccessKind(op);
    if (kind == TMemAccessKind::Load || kind == TMemAccessKind::Store) {
      pending.join(effects);
      afterBarrier.join(effects);
    }

    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<RegionBranchOpInterface>(op))
      finishBlock(pending, builder);
  }

  // These snapshots track accesses before and after lastBarrier within one
  // virtual block. Only MembarInfo::pending flows through joins and backedges.
  Operation *lastBarrier = nullptr;
  BlockInfo beforeBarrier;
  BlockInfo afterBarrier;
};

template <typename AnalysisT>
LogicalResult runTMemAnalysis(ModuleOp mod, MembarFilterFn filter = nullptr) {
  auto solver = createDataFlowSolver();
  auto *regions = solver->load<BufferRegionAnalysis>(
      BufferRegionAnalysis::Mode::TensorMemoryOnly);
  if (failed(solver->initializeAndRun(mod)))
    return failure();
  ModuleAllocation allocation(mod);
  ModuleMembarAnalysis analysis(allocation, std::move(filter));
  analysis.runAnalysis<AnalysisT>(*regions);
  return success();
}

} // namespace

struct TMemWaitInsertionPass
    : public impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
          TMemWaitInsertionPass> {
  using impl::TritonNvidiaGPUTMemWaitInsertionPassBase<
      TMemWaitInsertionPass>::TritonNvidiaGPUTMemWaitInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              auto kind = getTMemAccessKind(op);
              return kind == TMemAccessKind::Load ||
                             kind == TMemAccessKind::Store
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
             .wasInterrupted())
      return;
    // Run after TMEM and shared-memory barrier insertion. Calls and returns
    // complete pending accesses, so this pass needs no call summaries.
    if (failed(runTMemAnalysis<TMemWaitAnalysis>(mod)))
      return signalPassFailure();
  }
};

struct TMemBarrierInsertionPass
    : public impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
          TMemBarrierInsertionPass> {
  using impl::TritonNvidiaGPUTMemBarrierInsertionPassBase<
      TMemBarrierInsertionPass>::TritonNvidiaGPUTMemBarrierInsertionPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod.walk([](Operation *op) {
              return getTMemAccessKind(op) == TMemAccessKind::None
                         ? WalkResult::advance()
                         : WalkResult::interrupt();
            })
             .wasInterrupted())
      return;
    TMemAccessCache cache;
    MembarFilterFn filter = [&](Operation *lhs, Operation *rhs, bool lhsIsRead,
                                bool rhsIsRead, Allocation *allocation) {
      return filterFn(lhs, rhs, lhsIsRead, rhsIsRead, allocation, cache);
    };
    if (failed(runTMemAnalysis<TMemBarrierAnalysis>(mod, filter)))
      return signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
