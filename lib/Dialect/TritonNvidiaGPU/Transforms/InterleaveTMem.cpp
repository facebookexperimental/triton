#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/WSBarrierReorder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>

#define DEBUG_TYPE "triton-nvidia-interleave-tmem"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

inline bool isWSBarrierReorderEnabled() {
  auto disableReorder =
      triton::tools::getBoolEnv("TRITON_DISABLE_WSBARRIER_REORDER");
  return !disableReorder;
}

#define GEN_PASS_DEF_TRITONNVIDIAGPUINTERLEAVETMEMPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// If we don't know the effects of the op, we add all possible effects.
void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

bool collectEffects(Operation *op,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

struct AccessRange {
  SmallVector<std::optional<llvm::AddressRange>> ranges;
  unsigned rankOffset = 0;
};

std::pair<Value, AccessRange> findBufferAccess(Value a);

// Resolve the subview's per-dimension start offsets. An empty entry means the
// offset is not a compile-time constant, so the whole dimension may be
// accessed.
//
// This must not build IR. It runs as alias analysis underneath `sinkOps`,
// which re-queries it while walking the very block it is deciding how to
// reorder; materializing an `arith.constant` here inserts an op into that
// block and breaks the caller's contiguity check, so the sinking loop never
// reaches a fixed point.
std::pair<Value, AccessRange>
findBufferAccessMemdescSubview(Operation *subview) {
  TypedValue<ttg::MemDescType> src;
  SmallVector<int64_t> shape;
  SmallVector<std::optional<int64_t>> offsets;
  if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(subview)) {
    src = indexOp.getSrc();
    shape = to_vector(indexOp.getType().getShape());
    APInt index;
    offsets.push_back(matchPattern(indexOp.getIndex(), m_ConstantInt(&index))
                          ? std::optional<int64_t>(index.getSExtValue())
                          : std::nullopt);
    // An index view selects one slice of the outermost dim; the dims it keeps
    // all start at 0.
    for (int i = 0, e = std::max<int>(0, shape.size() - 1); i < e; ++i)
      offsets.push_back(0);
  } else {
    auto subsliceOp = cast<ttg::MemDescSubsliceOp>(subview);
    src = subsliceOp.getSrc();
    shape = to_vector(subsliceOp.getType().getShape());
    // Subslice offsets are static attributes.
    for (auto offset : subsliceOp.getOffsets())
      offsets.push_back(offset);
  }
  auto [alloc, parentAccess] = findBufferAccess(src);
  if (!alloc)
    return {};
  // Handle subview of a subview. The first `rankOffset` access sizes are
  // the same as in the parent access.
  AccessRange childAccess;
  for (auto i : llvm::seq(parentAccess.rankOffset))
    childAccess.ranges.push_back(parentAccess.ranges[i]);

  // The subview may have a smaller rank, in which case its access size is
  // just 1 for the higher dims.
  childAccess.rankOffset = src.getType().getRank() - shape.size();
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    auto parentRange = parentAccess.ranges[i + parentAccess.rankOffset];
    if (!parentRange) {
      childAccess.ranges.push_back({});
      continue;
    }

    // If the offset is not known, then the entire dim may be accessed.
    if (!offset) {
      childAccess.ranges.push_back({});
      continue;
    }

    uint64_t accessStart = parentRange->start() + *offset;
    uint64_t accessSize = 1;
    if (i >= childAccess.rankOffset)
      accessSize = shape[i - childAccess.rankOffset];
    childAccess.ranges.push_back({{accessStart, accessStart + accessSize}});
  }
  return {alloc, std::move(childAccess)};
}

// Simple local alias analysis that looks for a single underlying allocation and
// an access subrange.
std::pair<Value, AccessRange> findBufferAccess(Value a) {
  // Handle block arguments.
  if (auto arg = dyn_cast<BlockArgument>(a)) {
    Operation *parentOp = arg.getOwner()->getParentOp();

    // Look through `ttg.warp_specialize` explicit captures.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
      return findBufferAccess(wsOp.getExplicitCaptures()[arg.getArgNumber()]);
    }

    // Unknown block argument.
    return {};
  }

  Operation *defOp = a.getDefiningOp();
  // Accessing the alloc accesses the whole buffer.
  if (auto alloc = dyn_cast<TMEMAllocOp>(defOp)) {
    AccessRange access;
    for (uint64_t dim : alloc.getType().getShape())
      access.ranges.push_back({{0, dim}});
    return {a, std::move(access)};
  }

  // Trans and Reshape views don't change the access size.
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(defOp)) {
    return findBufferAccess(defOp->getOperand(0));
  }

  // Subviews can reduce the access sizes.
  if (isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(defOp)) {
    return findBufferAccessMemdescSubview(defOp);
  }

  // Subslice is a subview on one tensor dimension.
  if (auto subslice = dyn_cast<TMEMSubSliceOp>(defOp)) {
    auto [alloc, parentAccess] = findBufferAccess(subslice.getSrc());
    if (!alloc)
      return {};
    unsigned dim =
        subslice.getSrc().getType().getRank() - 2 + subslice.getDim();
    if (!parentAccess.ranges[dim])
      return {alloc, parentAccess};
    uint64_t start = parentAccess.ranges[dim]->start() + subslice.getOffset();
    uint64_t size = subslice.getType().getShape()[dim];
    AccessRange childAccess = parentAccess;
    childAccess.ranges[dim] = {{start, start + size}};
    return {alloc, std::move(childAccess)};
  }

  // Unknown defining op.
  return {};
}

bool tmemMayAlias(Value a, Value b) {
  auto [aAlloc, aRanges] = findBufferAccess(a);
  auto [bAlloc, bRanges] = findBufferAccess(b);
  // If the underlying buffer was not identified, assume mayalias.
  if (!aAlloc || !bAlloc)
    return true;
  // If the buffers are different, they don't alias.
  if (aAlloc != bAlloc)
    return false;
  // If the access ranges along any dimension are known to not overlap, then the
  // accesses don't alias.
  for (auto [aRange, bRange] : llvm::zip(aRanges.ranges, bRanges.ranges)) {
    // If either access range at this dim is unknown, we can't determine if they
    // don't overlap.
    if (!aRange || !bRange)
      continue;
    // The access ranges are known and don't overlap.
    if (!aRange->intersects(*bRange))
      return false;
  }
  return true;
}

// Trace SMEM memdesc views to a stable root, looking through
// `ttg.warp_specialize` explicit captures the same way findBufferAccess does.
// Resolving captures matters for aliasing: a partition that captures one buffer
// twice presents it as two distinct block arguments, which must not be mistaken
// for two distinct buffers. Only distinct local allocations are known not to
// alias; every other root, including a block argument that does not resolve to
// a capture, is treated conservatively by smemMayAlias below.
Value findSMemBase(Value value) {
  while (true) {
    if (Operation *def = value.getDefiningOp()) {
      if (!isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
               ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
               ttg::MemDescReshapeOp>(def))
        break;
      value = def->getOperand(0);
      continue;
    }
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg)
      break;
    auto wsOp = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
        arg.getOwner()->getParentOp());
    if (!wsOp)
      break;
    value = wsOp.getExplicitCaptures()[arg.getArgNumber()];
  }
  return value;
}

bool isKnownSMemBase(Value value) {
  return value.getDefiningOp<ttg::LocalAllocOp>() != nullptr;
}

bool smemMayAlias(Value a, Value b) {
  Value aBase = findSMemBase(a);
  Value bBase = findSMemBase(b);
  if (aBase == bBase)
    return true;
  return !(isKnownSMemBase(aBase) && isKnownSMemBase(bBase));
}

bool mayWriteOrFreeSMem(Operation *op, Value buffer) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  collectEffects(op, effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
      continue;
    if (isa<SideEffects::DefaultResource>(effect.getResource()))
      return true;
    if (effect.getResource() == ttg::SharedMemory::get() &&
        (!effect.getValue() || smemMayAlias(effect.getValue(), buffer)))
      return true;
  }
  return false;
}

// A token wait is "plain" when it carries no barrier operands, i.e. it only
// waits for the TMA store queue to drain past its own token. "Plain" is about
// barriers specifically: a wait that also gates an mbarrier publishes a signal
// other partitions observe, so it is not free to move.
bool isPlainTMAStoreTokenWait(Operation *op) {
  auto wait = dyn_cast<TMAStoreTokenWaitOp>(op);
  return wait && wait.getBarriers().empty();
}

bool isTMAStoreLike(Operation *op);

// Sink each barrier-free TMA store token wait as late as the staging buffer
// allows: down to just before the first following store that may clobber the
// buffer the store is still reading. Nothing between the store and that point
// needs the transfer to have completed, so delaying the wait lets the TMEM
// load and layout conversion for the next subtile overlap the in-flight TMA
// store instead of stalling behind it. The walk stops early at another async
// reader or at any operation whose memory effects cannot be proven independent
// of the staging buffer.
void delayPlainTMAStoreTokenWaits(Block &block) {
  SmallVector<TMAStoreTokenWaitOp> waits;
  for (Operation &op : block)
    if (isPlainTMAStoreTokenWait(&op))
      waits.push_back(cast<TMAStoreTokenWaitOp>(&op));

  for (TMAStoreTokenWaitOp wait : waits) {
    Operation *tmaStore = wait.getToken().getDefiningOp();
    if (!tmaStore ||
        !isa<AsyncTMACopyLocalToGlobalOp, AsyncTMAReduceOp>(tmaStore) ||
        tmaStore->getBlock() != &block)
      continue;
    Value staging;
    if (auto copy = dyn_cast<AsyncTMACopyLocalToGlobalOp>(tmaStore))
      staging = copy.getSrc();
    else
      staging = cast<AsyncTMAReduceOp>(tmaStore).getSrc();
    for (auto it = std::next(wait->getIterator()); it != block.end(); ++it) {
      auto localStore = dyn_cast<ttg::LocalStoreOp>(&*it);
      if (localStore) {
        if (!smemMayAlias(localStore.getDst(), staging))
          continue;
        wait->moveBefore(localStore);
        break;
      }

      // Do not make this queue-wide wait cover another async reader, or cross
      // an operation whose memory effects we cannot prove independent.
      if (isa<ttg::LocalAllocOp>(&*it))
        continue;
      if (isTMAStoreLike(&*it) || mayWriteOrFreeSMem(&*it, staging))
        break;
    }
  }
}

// Check whether a movable chain can sink past `next`. When opConstraints is
// provided, use canAdvanceWSBarrier to decide whether the chain can sink past
// barriers from independent channels.
bool canSinkUseChainPast(Value buffer, ArrayRef<Operation *> useChain,
                         Operation *next,
                         std::optional<DictionaryAttr> opConstraints) {
  bool dep = false;
  for (auto operand : getNestedOperands(next)) {
    if (llvm::any_of(useChain, [&](Operation *op) {
          return llvm::is_contained(op->getResults(), operand);
        })) {
      dep = true;
      break;
    }
  }
  // A TMEM epilogue load can carry the constraints from its own arrive
  // barrier.  Use those constraints even when the global barrier-reorder
  // knob is disabled: this moves only the load/use chain (and its matching
  // arrive), rather than globally raising and sinking every WS barrier in the
  // block.
  if (opConstraints || isWSBarrierReorderEnabled()) {
    // Once the load chain has picked up its own arrive, it may pass another
    // WS arrive: both operations only delay signals.  This mirrors
    // sinkWSArrives, but is deliberately limited to the one epilogue chain.
    // Both arrives must carry WS barrier constraints; an unconstrained arrive
    // has no channel to prove independence against, so it stays subject to
    // canAdvanceWSBarrier.
    auto chainArrive = dyn_cast<ArriveBarrierOp>(useChain.back());
    auto nextArrive = dyn_cast<ArriveBarrierOp>(next);
    bool arrivesCanSwap =
        chainArrive && nextArrive &&
        hasWSBarrierConstraints(chainArrive.getConstraints()) &&
        hasWSBarrierConstraints(nextArrive.getConstraints());
    if (!arrivesCanSwap && !canAdvanceWSBarrier(opConstraints, next))
      return false;
  } else {
    // Legacy safe behavior: don't sink past barrier signals, since they may
    // guard the liverange of the buffer.
    if (isa<ArriveBarrierOp>(next))
      return false;
  }
  if (!isMemoryEffectFree(next) && !isPlainTMAStoreTokenWait(next)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    collectEffects(next, effects);
    for (auto effect : effects) {
      // Look for potentially aliasing write or free effects.
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
        continue;
      if (isa<SideEffects::DefaultResource>(effect.getResource())) {
        dep = true;
        break;
      }
      if (isa<TensorMemory>(effect.getResource()) &&
          (!effect.getValue() || tmemMayAlias(effect.getValue(), buffer))) {
        dep = true;
        break;
      }
    }
  }
  return !dep;
}

// An arrive that releases the buffer this chain is reading must stay behind
// the chain, so it cannot be sunk past. Absorbing it instead lets both move
// together: the load keeps its position ahead of the release it signals, and
// the release is only ever delayed. Matching the chain's own constraints is
// what identifies the arrive as belonging to this channel; an arrive from
// another channel stays subject to canSinkUseChainPast.
bool isAbsorbableArrive(Operation *op,
                        std::optional<DictionaryAttr> opConstraints) {
  if (!opConstraints)
    return false;
  auto arrive = dyn_cast<ArriveBarrierOp>(op);
  if (!arrive)
    return false;
  auto constraints = arrive.getConstraints();
  return constraints && *constraints == *opConstraints;
}

// True when the chain already sits contiguously just before `insertBefore`,
// i.e. moving it would be a no-op. Absorbing an arrive can leave the chain
// non-contiguous even when its last op is already in place, so contiguity has
// to be checked rather than just the final position.
bool isChainInPlace(ArrayRef<Operation *> useChain, Operation *insertBefore) {
  for (auto [op, nextOp] :
       llvm::zip(useChain.drop_back(), useChain.drop_front()))
    if (op->getNextNode() != nextOp)
      return false;
  return useChain.back()->getNextNode() == insertBefore;
}

// Sink ops as close to their use as possible to reduce register pressure.
// When opConstraints is provided, uses canAdvanceWSBarrier to decide whether
// the op can sink past barriers from independent channels.
bool sinkOps(Value buffer, SmallVectorImpl<Operation *> &useChain,
             std::optional<DictionaryAttr> opConstraints) {
  Operation *insertBefore = nullptr;
  Operation *next = useChain.back()->getNextNode();
  while (next && !next->hasTrait<OpTrait::IsTerminator>()) {
    // The walk starts after the chain, so only an arrive that already follows
    // the load is ever a candidate; one that precedes it is never visited.
    if (isAbsorbableArrive(next, opConstraints)) {
      useChain.push_back(next);
      // Any insertion point found before the arrive is now stale: reusing it
      // would hoist the arrive above ops it currently follows.
      insertBefore = nullptr;
      next = next->getNextNode();
      continue;
    }
    insertBefore = next;
    bool dep = false;
    if (!canSinkUseChainPast(buffer, useChain, next, opConstraints))
      dep = true;
    if (dep)
      break;
    next = next->getNextNode();
  }
  if (!insertBefore)
    insertBefore = useChain.back()->getNextNode();
  if (!insertBefore || isChainInPlace(useChain, insertBefore))
    return false;
  for (Operation *op : useChain)
    op->moveBefore(insertBefore);
  return true;
}

SmallVector<Operation *> getMovableUseChain(Operation *op) {
  SmallVector<Operation *> useChain{op};
  while (useChain.back()->hasOneUse() &&
         isPure(*useChain.back()->user_begin()) &&
         useChain.back()->getNextNode() == *useChain.back()->user_begin()) {
    useChain.push_back(*useChain.back()->user_begin());
  }
  return useChain;
}

// Try to sink a load and a collection of its users.
bool trySinkOp(Operation *op, Value buffer,
               std::optional<DictionaryAttr> opConstraints) {
  SmallVector<Operation *> useChain = getMovableUseChain(op);
  return sinkOps(buffer, useChain, opConstraints);
}

bool isTMAStoreLike(Operation *op) {
  return isa<AsyncTMACopyLocalToGlobalOp, AsyncTMAReduceOp>(op);
}

struct BlockInterleaveInfo {
  Block *block;
  SmallVector<std::pair<Operation *, Value>> opsToSink;
};

BlockInterleaveInfo collectBlockInterleaveInfo(Block *block) {
  BlockInterleaveInfo info;
  info.block = block;
  for (Operation &op : *block) {
    if (auto load = dyn_cast<TMEMLoadOp>(&op)) {
      info.opsToSink.emplace_back(load, load.getSrc());
    } else if (auto alloc = dyn_cast<TMEMAllocOp>(&op)) {
      info.opsToSink.emplace_back(alloc, alloc.getResult());
    }
  }
  return info;
}

DenseMap<Operation *, DictionaryAttr> buildTMemLoadConstraints(Block &block) {
  DenseMap<Operation *, DictionaryAttr> memOpConstraints;

  // For each arrive barrier with constraints, scan backward and assign its
  // constraints to ALL tmem_loads in its channel region (between the arrive and
  // the preceding same-channel wait or block start). This ensures all split
  // tmem_loads inherit the channelGraph, not just the one nearest to the
  // arrive.
  for (Operation &op : block) {
    auto arrive = dyn_cast<ArriveBarrierOp>(&op);
    if (!arrive)
      continue;
    auto constraints = arrive.getConstraints();
    if (!hasWSBarrierConstraints(constraints))
      continue;
    DictionaryAttr dict = *constraints;
    for (auto *cur = arrive->getPrevNode(); cur; cur = cur->getPrevNode()) {
      if (!canAdvanceWSBarrier(constraints, cur))
        break;
      if (isa<TMEMLoadOp>(cur))
        memOpConstraints[cur] = dict;
    }
  }

  return memOpConstraints;
}

void processBlock(BlockInterleaveInfo &info) {
  Block &block = *info.block;
  bool reorderWSBarriers = isWSBarrierReorderEnabled();

  // Step 1: Record which memory op each WS barrier guards.
  DenseMap<Operation *, Operation *> barrierMap;
  if (reorderWSBarriers)
    barrierMap = buildBarrierToMemoryOpMap(block);

  // Step 2: Reorder WS barriers. Pushes arrives down and pulls waits up past
  // barriers from independent channels, unblocking tmem_load sinking.
  if (reorderWSBarriers) {
    sinkWSArrives(block);
    raiseWSWaits(block);
  }

  // Step 3: Move TMEM allocs close to their uses, then sink tmem_loads as far
  // as legality allows.
  // Constraint-guided TMEM epilogue sinking is safe independently of the
  // global WS-barrier normalization.  Build the mapping unconditionally so a
  // load can carry its own arrive across barriers from independent channels.
  DenseMap<Operation *, DictionaryAttr> memOpConstraints =
      buildTMemLoadConstraints(block);
  auto sinkGreedily = [&](Operation *op, Value buffer) {
    auto it = memOpConstraints.find(op);
    std::optional<DictionaryAttr> constraints =
        it != memOpConstraints.end() ? std::optional<DictionaryAttr>(it->second)
                                     : std::nullopt;
    while (trySinkOp(op, buffer, constraints)) {
    }
  };
  for (auto [op, buffer] : info.opsToSink)
    if (!isa<TMEMLoadOp>(op))
      sinkGreedily(op, buffer);
  for (auto [op, buffer] : info.opsToSink)
    if (isa<TMEMLoadOp>(op))
      sinkGreedily(op, buffer);

  // Step 4: Restore barriers to optimal positions near their memory ops.
  if (reorderWSBarriers)
    optimizeWSBarrierLocations(barrierMap);
  // Barrier restoration and TMEM-load sinking may move a load across a token
  // wait that was already positioned before staging reuse. Re-establish that
  // canonical placement so the load/conversion overlaps the prior TMA store.
  delayPlainTMAStoreTokenWaits(block);
}

} // anonymous namespace

struct TritonNvidiaGPUInterleaveTMemPass
    : public impl::TritonNvidiaGPUInterleaveTMemPassBase<
          TritonNvidiaGPUInterleaveTMemPass> {
  using impl::TritonNvidiaGPUInterleaveTMemPassBase<
      TritonNvidiaGPUInterleaveTMemPass>::TritonNvidiaGPUInterleaveTMemPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<BlockInterleaveInfo> blocksToProcess;
    m.walk([&](Block *block) {
      BlockInterleaveInfo info = collectBlockInterleaveInfo(block);
      // One movable op is enough: distancing a single load from its producing
      // MMA is worthwhile on its own, and alloc sinking does not depend on
      // loads at all. The gate stays because processBlock's barrier steps are
      // block-scoped rather than driven by opsToSink, so an empty worklist
      // would still reorder every WS barrier and TMA store token wait here.
      if (info.opsToSink.empty())
        return;
      blocksToProcess.push_back(std::move(info));
    });
    for (auto &info : blocksToProcess)
      processBlock(info);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
