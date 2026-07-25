#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/WSBarrierReorder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/SmallPtrSet.h"
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

std::pair<Value, AccessRange>
findBufferAccessMemdescSubview(Operation *subview) {
  OpBuilder builder(subview);
  Location loc = subview->getLoc();
  TypedValue<ttg::MemDescType> src;
  SmallVector<int64_t> shape;
  SmallVector<Value> offsets;
  if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(subview)) {
    src = indexOp.getSrc();
    shape = to_vector(indexOp.getType().getShape());
    offsets = {indexOp.getIndex()};
    for (auto i : llvm::seq(std::max<int>(0, shape.size() - 1)))
      offsets.push_back(arith::ConstantIntOp::create(builder, loc, 0, 32));
  } else {
    auto subsliceOp = cast<ttg::MemDescSubsliceOp>(subview);
    src = subsliceOp.getSrc();
    shape = to_vector(subsliceOp.getType().getShape());
    for (auto offset : subsliceOp.getOffsets())
      offsets.push_back(arith::ConstantIntOp::create(builder, loc, offset, 32));
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
    APInt value;
    if (!matchPattern(offset, m_ConstantInt(&value))) {
      childAccess.ranges.push_back({});
      continue;
    }

    uint64_t accessStart = parentRange->start() + value.getSExtValue();
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

  // Subslice is a subview only on the N dimension.
  if (auto subslice = dyn_cast<TMEMSubSliceOp>(defOp)) {
    auto [alloc, parentAccess] = findBufferAccess(subslice.getSrc());
    if (!alloc)
      return {};
    if (!parentAccess.ranges[1])
      return {alloc, parentAccess};
    uint64_t mStart = parentAccess.ranges[1]->start() + subslice.getN();
    uint64_t mSize = subslice.getType().getShape()[1];
    AccessRange childAccess = parentAccess;
    childAccess.ranges[1] = {{mStart, mStart + mSize}};
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

bool isPlainTMAStoreTokenWait(Operation *op) {
  auto wait = dyn_cast<TMAStoreTokenWaitOp>(op);
  return wait && wait.getBarriers().empty();
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
  if (isWSBarrierReorderEnabled()) {
    if (!canAdvanceWSBarrier(opConstraints, next))
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

void moveUseChainAfter(ArrayRef<Operation *> useChain, Operation *op) {
  Operation *insertBefore = op->getNextNode();
  assert(insertBefore && "expected op before block terminator");
  for (Operation *chainOp : useChain)
    chainOp->moveBefore(insertBefore);
}

// Sink ops as close to their use as possible to reduce register pressure.
// When opConstraints is provided, uses canAdvanceWSBarrier to decide whether
// the op can sink past barriers from independent channels.
bool sinkOps(Value buffer, ArrayRef<Operation *> useChain,
             std::optional<DictionaryAttr> opConstraints) {
  Operation *insertBefore = nullptr;
  Operation *next = useChain.back()->getNextNode();
  while (next && !next->hasTrait<OpTrait::IsTerminator>()) {
    insertBefore = next;
    bool dep = false;
    if (!canSinkUseChainPast(buffer, useChain, next, opConstraints))
      dep = true;
    if (dep)
      break;
    next = next->getNextNode();
  }
  if (insertBefore && insertBefore != useChain.back()->getNextNode()) {
    for (Operation *op : useChain)
      op->moveBefore(insertBefore);
    return true;
  }
  return false;
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

struct TMemLoadGroup {
  DictionaryAttr constraints;
  Value alloc;
  SmallVector<Operation *> loads;
};

DenseMap<Operation *, unsigned> getBlockOpPositions(Block &block) {
  DenseMap<Operation *, unsigned> opToPosition;
  unsigned position = 0;
  for (Operation &op : block)
    opToPosition[&op] = position++;
  return opToPosition;
}

Operation *
getTMemLoadLiveRangeEnd(Operation *load,
                        const DenseMap<Operation *, unsigned> &opToPosition) {
  SmallVector<Operation *> useChain = getMovableUseChain(load);
  Operation *tail = useChain.back();
  Operation *end = tail;
  unsigned endPos = opToPosition.lookup(tail);

  for (Value result : tail->getResults()) {
    for (Operation *user : result.getUsers()) {
      auto userIt = opToPosition.find(user);
      if (userIt != opToPosition.end() && userIt->second > endPos) {
        end = user;
        endPos = userIt->second;
      }
    }
  }
  return end;
}

bool isAfter(Operation *op, Operation *boundary,
             const DenseMap<Operation *, unsigned> &opToPosition) {
  auto opIt = opToPosition.find(op);
  auto boundaryIt = opToPosition.find(boundary);
  if (opIt == opToPosition.end() || boundaryIt == opToPosition.end())
    return false;
  return opIt->second > boundaryIt->second;
}

bool sinkTMemLoadAfter(Operation *load, Operation *boundary, Value buffer,
                       std::optional<DictionaryAttr> opConstraints) {
  bool changed = false;
  while (true) {
    DenseMap<Operation *, unsigned> opToPosition =
        getBlockOpPositions(*load->getBlock());
    if (isAfter(load, boundary, opToPosition))
      return changed;

    SmallVector<Operation *> useChain = getMovableUseChain(load);
    std::optional<DictionaryAttr> effectiveConstraints = opConstraints;
    Operation *next = useChain.back()->getNextNode();
    auto arrive = dyn_cast_or_null<ArriveBarrierOp>(next);
    if (arrive && arrive.getConstraints()) {
      DictionaryAttr arriveConstraints = *arrive.getConstraints();
      if (!effectiveConstraints || arriveConstraints == *effectiveConstraints) {
        useChain.push_back(next);
        effectiveConstraints = arriveConstraints;
      }
    }
    next = useChain.back()->getNextNode();
    if (!next || next->hasTrait<OpTrait::IsTerminator>())
      return changed;
    if (!canSinkUseChainPast(buffer, useChain, next, effectiveConstraints))
      return changed;

    moveUseChainAfter(useChain, next);
    changed = true;
  }
}

bool sinkTMemLoadsToFreshLiveRanges(
    const TMemLoadGroup &group,
    const DenseMap<Operation *, DictionaryAttr> &memOpConstraints) {
  bool changed = false;
  Operation *previousLoad = group.loads.front();
  for (Operation *load : llvm::drop_begin(group.loads)) {
    DenseMap<Operation *, unsigned> opToPosition =
        getBlockOpPositions(*load->getBlock());
    Operation *previousEnd =
        getTMemLoadLiveRangeEnd(previousLoad, opToPosition);
    auto loadOp = cast<TMEMLoadOp>(load);
    auto it = memOpConstraints.find(load);
    std::optional<DictionaryAttr> constraints =
        it != memOpConstraints.end() ? std::optional<DictionaryAttr>(it->second)
                                     : std::nullopt;
    changed |=
        sinkTMemLoadAfter(load, previousEnd, loadOp.getSrc(), constraints);
    previousLoad = load;
  }
  return changed;
}

struct BlockInterleaveInfo {
  Block *block;
  unsigned tmemLoadCount = 0;
  SmallVector<Operation *> tmemLoads;
  SmallVector<std::pair<Operation *, Value>> opsToSink;
};

struct OverlapLiveness {
  SmallVector<unsigned> numLiveTMEMLoads;
  SmallVector<unsigned> overlapProfile;
};

BlockInterleaveInfo collectBlockInterleaveInfo(Block *block) {
  BlockInterleaveInfo info;
  info.block = block;
  for (Operation &op : *block) {
    if (auto load = dyn_cast<TMEMLoadOp>(&op)) {
      info.tmemLoadCount++;
      info.tmemLoads.push_back(load);
      info.opsToSink.emplace_back(load, load.getSrc());
    } else if (auto alloc = dyn_cast<TMEMAllocOp>(&op)) {
      info.opsToSink.emplace_back(alloc, alloc.getResult());
    }
  }
  return info;
}

SmallVector<TMemLoadGroup> buildTMemLoadGroups(
    ArrayRef<Operation *> tmemLoads,
    const DenseMap<Operation *, DictionaryAttr> &memOpConstraints) {
  SmallVector<TMemLoadGroup> groups;
  for (Operation *op : tmemLoads) {
    auto load = cast<TMEMLoadOp>(op);
    DictionaryAttr constraints;
    if (auto it = memOpConstraints.find(op); it != memOpConstraints.end())
      constraints = it->second;
    Value alloc = findBufferAccess(load.getSrc()).first;

    auto groupIt = llvm::find_if(groups, [&](const TMemLoadGroup &group) {
      return group.constraints == constraints && group.alloc == alloc;
    });
    if (groupIt == groups.end()) {
      groups.push_back({constraints, alloc, {}});
      groupIt = std::prev(groups.end());
    }
    groupIt->loads.push_back(op);
  }

  llvm::erase_if(groups, [](const TMemLoadGroup &group) {
    return group.loads.size() < 2;
  });
  return groups;
}

SmallVector<Operation *> getBlockOpOrder(Block &block) {
  SmallVector<Operation *> order;
  for (Operation &op : block) {
    if (!op.hasTrait<OpTrait::IsTerminator>())
      order.push_back(&op);
  }
  return order;
}

void restoreBlockOpOrder(Block &block, ArrayRef<Operation *> order) {
  llvm::SmallPtrSet<Operation *, 32> originalOps(order.begin(), order.end());
  SmallVector<Operation *> addedOps;
  for (Operation &op : block) {
    if (!op.hasTrait<OpTrait::IsTerminator>() && !originalOps.contains(&op))
      addedOps.push_back(&op);
  }
  for (Operation *op : addedOps) {
    if (op->use_empty() && isMemoryEffectFree(op))
      op->erase();
  }

  Operation *insertPt = block.getTerminator();
  for (Operation *op : llvm::reverse(order)) {
    if (op->getBlock() != &block)
      continue;
    op->moveBefore(insertPt);
    insertPt = op;
  }
}

// Computes the live range (start and end positions) for each TMEM load
// operation within a block's operation order.
DenseMap<Operation *, std::pair<unsigned, unsigned>>
computeLoadLiveRanges(ArrayRef<Operation *> order,
                      ArrayRef<Operation *> tmemLoads) {
  DenseMap<Operation *, unsigned> opToPosition;
  unsigned position = 0;
  for (Operation *op : order)
    opToPosition[op] = position++;

  DenseMap<Operation *, std::pair<unsigned, unsigned>> liveRanges;
  for (Operation *load : tmemLoads) {
    auto startIt = opToPosition.find(load);
    if (startIt == opToPosition.end())
      continue;

    SmallVector<Operation *> useChain = getMovableUseChain(load);
    Operation *tail = useChain.back();
    auto tailIt = opToPosition.find(tail);
    unsigned end =
        tailIt != opToPosition.end() ? tailIt->second : startIt->second;

    for (Value result : tail->getResults()) {
      for (Operation *user : result.getUsers()) {
        auto userIt = opToPosition.find(user);
        if (userIt != opToPosition.end())
          end = std::max(end, userIt->second);
      }
    }
    liveRanges[load] = {startIt->second, end};
  }

  return liveRanges;
}

// Computes overlapping liveness occupancy across the given TMEM loads. Walks
// the union of their live ranges and records, for each contiguous span, how
// many of the candidate loads are simultaneously live. The resulting
// `overlapProfile` (counts > 1, sorted descending) is the rollback acceptance
// key: a transformation is kept only when this profile improves.
OverlapLiveness computeOverlapLiveness(
    const DenseMap<Operation *, std::pair<unsigned, unsigned>> &liveRanges,
    ArrayRef<Operation *> tmemLoads) {
  OverlapLiveness liveness;
  SmallVector<std::pair<unsigned, unsigned>> groupLiveRanges;
  for (Operation *load : tmemLoads) {
    auto it = liveRanges.find(load);
    if (it != liveRanges.end())
      groupLiveRanges.push_back(it->second);
  }
  if (groupLiveRanges.empty())
    return liveness;

  unsigned minStart = groupLiveRanges.front().first;
  unsigned maxEnd = groupLiveRanges.front().second;
  for (auto [start, end] : groupLiveRanges) {
    minStart = std::min(minStart, start);
    maxEnd = std::max(maxEnd, end);
  }

  unsigned lastCount = 0;
  for (unsigned pos = minStart; pos <= maxEnd; ++pos) {
    unsigned count = 0;
    for (auto [start, end] : groupLiveRanges) {
      if (start <= pos && pos <= end)
        count++;
    }
    if (count == 0) {
      lastCount = 0;
      continue;
    }
    if (count == lastCount)
      continue;
    liveness.numLiveTMEMLoads.push_back(count);
    lastCount = count;
  }

  for (unsigned count : liveness.numLiveTMEMLoads) {
    if (count > 1)
      liveness.overlapProfile.push_back(count);
  }
  llvm::sort(liveness.overlapProfile,
             [](unsigned lhs, unsigned rhs) { return lhs > rhs; });
  return liveness;
}

bool isOverlapProfileImproved(ArrayRef<unsigned> before,
                              ArrayRef<unsigned> after) {
  for (auto [beforeCount, afterCount] : llvm::zip(before, after)) {
    if (afterCount < beforeCount)
      return true;
    if (afterCount > beforeCount)
      return false;
  }
  return after.size() < before.size();
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
  SmallVector<Operation *> originalOrder = getBlockOpOrder(block);
  SmallVector<Operation *> originalLivenessOrder = originalOrder;
  originalLivenessOrder.push_back(block.getTerminator());
  auto beforeLiveRanges =
      computeLoadLiveRanges(originalLivenessOrder, info.tmemLoads);
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

  // Step 3: Move TMEM allocs close to their uses, then sink tmem_loads only
  // far enough to start after the previous load's live range.
  DenseMap<Operation *, DictionaryAttr> memOpConstraints;
  if (reorderWSBarriers)
    memOpConstraints = buildTMemLoadConstraints(block);
  SmallVector<TMemLoadGroup> loadGroups =
      buildTMemLoadGroups(info.tmemLoads, memOpConstraints);
  for (auto [op, buffer] : info.opsToSink) {
    if (isa<TMEMLoadOp>(op))
      continue;
    auto it = memOpConstraints.find(op);
    std::optional<DictionaryAttr> constraints =
        it != memOpConstraints.end() ? std::optional<DictionaryAttr>(it->second)
                                     : std::nullopt;
    while (trySinkOp(op, buffer, constraints)) {
    }
  }
  for (const TMemLoadGroup &group : loadGroups)
    sinkTMemLoadsToFreshLiveRanges(group, memOpConstraints);

  // Step 4: Restore barriers to optimal positions near their memory ops.
  if (reorderWSBarriers)
    optimizeWSBarrierLocations(barrierMap);

  SmallVector<Operation *> currentLivenessOrder = getBlockOpOrder(block);
  currentLivenessOrder.push_back(block.getTerminator());
  auto afterLiveRanges =
      computeLoadLiveRanges(currentLivenessOrder, info.tmemLoads);
  bool hasOverlappingGroup = false;
  bool allOverlappingGroupsImproved = true;
  for (const TMemLoadGroup &group : loadGroups) {
    OverlapLiveness before =
        computeOverlapLiveness(beforeLiveRanges, group.loads);
    if (before.overlapProfile.empty())
      continue;
    hasOverlappingGroup = true;
    OverlapLiveness after =
        computeOverlapLiveness(afterLiveRanges, group.loads);
    if (!isOverlapProfileImproved(before.overlapProfile,
                                  after.overlapProfile)) {
      allOverlappingGroupsImproved = false;
      break;
    }
  }

  if (!hasOverlappingGroup || !allOverlappingGroupsImproved)
    restoreBlockOpOrder(block, originalOrder);
}

} // anonymous namespace

struct TritonNvidiaGPUInterleaveTMemPass
    : public impl::TritonNvidiaGPUInterleaveTMemPassBase<
          TritonNvidiaGPUInterleaveTMemPass> {
  using impl::TritonNvidiaGPUInterleaveTMemPassBase<
      TritonNvidiaGPUInterleaveTMemPass>::TritonNvidiaGPUInterleaveTMemPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    SmallVector<BlockInterleaveInfo> blocksToProcess;
    m.walk([&](Block *block) {
      BlockInterleaveInfo info = collectBlockInterleaveInfo(block);
      if (info.tmemLoadCount < 2)
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
