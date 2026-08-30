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
    for (int i = 0, e = std::max<int>(0, shape.size() - 1); i < e; ++i)
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

    // Partition outlining turns captured TMEM allocations into block
    // arguments. Within one partition the argument remains a stable alias root
    // for all of its views, so retain its full range for local analysis.
    if (auto type = dyn_cast<ttg::MemDescType>(arg.getType());
        type && isa<TensorMemorySpaceAttr>(type.getMemorySpace())) {
      AccessRange access;
      for (uint64_t dim : type.getShape())
        access.ranges.push_back({{0, dim}});
      return {arg, std::move(access)};
    }
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
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp,
          ttg::MemDescReinterpretOp>(defOp)) {
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
  // If the buffers are different, they don't alias -- but only a distinct
  // allocation proves that. findBufferAccess also roots a TMEM access at a
  // block argument so the reuse repair can compare views within one outlined
  // partition; two such arguments (for example memdesc iter args of the same
  // loop) may still be the same underlying allocation, so they are not
  // evidence of disjointness. Mirrors the rule findSMemBase documents.
  if (aAlloc != bAlloc)
    return isa<BlockArgument>(aAlloc) || isa<BlockArgument>(bAlloc);
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

Operation *findEarliestTMAStoreUser(Value staging, Operation *after) {
  Operation *earliest = nullptr;
  for (Operation *user : staging.getUsers()) {
    if (!isTMAStoreLike(user) || user->getBlock() != after->getBlock() ||
        !after->isBeforeInBlock(user))
      continue;
    if (!earliest || user->isBeforeInBlock(earliest))
      earliest = user;
  }
  return earliest;
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

// Trace the value produced by a TMEM load through a single-use pure chain to
// the local store that materializes it in SMEM. This is the common output
// epilogue shape: tmem_load -> convert/scale -> local_store -> TMA store.
ttg::LocalStoreOp findEpilogueLocalStore(Operation *load) {
  if (load->getNumResults() != 1)
    return {};
  Value value = load->getResult(0);
  while (value.hasOneUse()) {
    Operation *user = *value.user_begin();
    if (auto localStore = dyn_cast<ttg::LocalStoreOp>(user))
      return localStore;
    if (!isPure(user) || user->getNumResults() != 1)
      return {};
    value = user->getResult(0);
  }
  return {};
}

bool isTMAStoreLike(Operation *op) {
  return isa<AsyncTMACopyLocalToGlobalOp, AsyncTMAReduceOp>(op);
}

Operation *findEpilogueTMAStore(Operation *load) {
  auto localStore = findEpilogueLocalStore(load);
  if (!localStore)
    return nullptr;
  return findEarliestTMAStoreUser(localStore.getDst(), localStore);
}

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
  // A TMA output epilogue does not become reusable at local_store: the TMA
  // engine still has to begin reading the staging buffer. Extend the boundary
  // through that launch so the next TMEM slice can overlap the asynchronous
  // transfer without extending both register live ranges.
  if (Operation *tmaStore = findEpilogueTMAStore(load)) {
    auto tmaIt = opToPosition.find(tmaStore);
    if (tmaIt != opToPosition.end() && tmaIt->second > endPos) {
      end = tmaStore;
      endPos = tmaIt->second;
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
    // Output-epilogue loads may come from distinct TMEM allocations (dV and
    // dK), but they compete for registers and feed a single ordered TMA-store
    // stream. Group them together when their WS constraints match so the
    // interleaver can realize load -> store launch -> next load across tensor
    // boundaries. Other loads retain allocation-local grouping.
    Value alloc = findEpilogueTMAStore(op)
                      ? Value{}
                      : findBufferAccess(load.getSrc()).first;

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

// Prefer materializing a TMEM operand before an independent SMEM operand when
// both feed the same pure operation.  Code partitioning places each channel
// consumer immediately before its first use, which can otherwise leave the
// SMEM load/broadcast live across the TMEM load.  Delay the complete SMEM
// consumer channel until the TMEM channel has completed.  Both the wait and
// release are allowed to cross the intervening channel only when the WS
// ordered-region metadata proves that reordering safe.
//
// This does not shorten a live range on its own; it swaps which of the two
// values is live across the other's channel.  It pays off when the SMEM value
// is the cheaper one to hold, which is the shape this was written for (a
// narrow scalar broadcast against a wide TMEM tile).  The selection below is
// purely structural and does not compare the two footprints or chain lengths,
// so a wide SMEM operand or a long SMEM consumer chain can raise peak
// register pressure instead of lowering it.  Adding a width or length guard
// needs a policy decision and ideally evidence that the adverse shape occurs
// in practice; until then the qualifying conditions are the structural ones
// only, and this comment states the intent rather than an enforced invariant.
bool prioritizeTMemOperand(Block &block) {
  bool changed = false;
  SmallVector<TMEMLoadOp> tmemLoads;
  for (Operation &op : block)
    if (auto load = dyn_cast<TMEMLoadOp>(&op))
      tmemLoads.push_back(load);

  for (TMEMLoadOp tmemLoad : tmemLoads) {
    Value tmemValue = tmemLoad.getResult();
    Operation *commonUser = nullptr;
    while (tmemValue.hasOneUse()) {
      Operation *user = *tmemValue.user_begin();
      if (!isPure(user))
        break;
      if (user->getNumOperands() != 1 || user->getNumResults() != 1) {
        commonUser = user;
        break;
      }
      tmemValue = user->getResult(0);
    }
    if (!commonUser || !isPure(commonUser) || commonUser->getBlock() != &block)
      continue;

    for (Value operand : commonUser->getOperands()) {
      if (operand == tmemValue)
        continue;

      SmallVector<Operation *> reverseChain;
      Value current = operand;
      ttg::LocalLoadOp localLoad;
      while (Operation *def = current.getDefiningOp()) {
        if (def->getBlock() != &block || !def->hasOneUse())
          break;
        if (auto load = dyn_cast<ttg::LocalLoadOp>(def)) {
          localLoad = load;
          break;
        }
        if (!isPure(def) || def->getNumOperands() != 1 ||
            def->getNumResults() != 1)
          break;
        reverseChain.push_back(def);
        current = def->getOperand(0);
      }
      if (!localLoad || !localLoad->isBeforeInBlock(tmemLoad))
        continue;

      auto acquire = dyn_cast_or_null<WaitBarrierOp>(localLoad->getPrevNode());
      if (!acquire || !hasWSBarrierConstraints(acquire.getConstraints()))
        continue;
      SmallVector<Operation *> acquirePrefix;
      llvm::SmallPtrSet<Operation *, 8> movingOps{acquire, localLoad};
      for (Operation *op = acquire->getPrevNode(); op && isPure(op);
           op = op->getPrevNode()) {
        bool usedOnlyByMovingOps =
            llvm::all_of(op->getUsers(), [&](Operation *user) {
              return movingOps.contains(user);
            });
        if (!usedOnlyByMovingOps)
          break;
        acquirePrefix.push_back(op);
        movingOps.insert(op);
      }

      SmallVector<Operation *> releasePrefix;
      Operation *releaseCandidate = localLoad->getNextNode();
      while (releaseCandidate && isPure(releaseCandidate)) {
        releasePrefix.push_back(releaseCandidate);
        releaseCandidate = releaseCandidate->getNextNode();
      }
      auto release = dyn_cast_or_null<ArriveBarrierOp>(releaseCandidate);
      if (!release || !hasWSBarrierConstraints(release.getConstraints()))
        continue;

      bool safe = true;
      for (Operation *op = release->getNextNode(); op && op != commonUser;
           op = op->getNextNode()) {
        if (auto arrive = dyn_cast<ArriveBarrierOp>(op)) {
          if (!hasWSBarrierConstraints(arrive.getConstraints()) ||
              !canAdvanceWSBarrierArrivePastWait(arrive.getConstraints(),
                                                 acquire.getConstraints())) {
            safe = false;
            break;
          }
          continue;
        }
        if (auto wait = dyn_cast<WaitBarrierOp>(op)) {
          // The release is an arrive that ends up after this wait. Delaying
          // an arrive past a wait is the cycle sinkWSArrives guards with the
          // same predicate; requiring only that the wait carries constraints
          // would let the SMEM empty-arrive sink past a wait whose producer
          // is blocked on it.
          if (!hasWSBarrierConstraints(wait.getConstraints()) ||
              !canAdvanceWSBarrierArrivePastWait(release.getConstraints(),
                                                 wait.getConstraints())) {
            safe = false;
            break;
          }
          continue;
        }
        if (!canAdvanceWSBarrier(release.getConstraints(), op)) {
          safe = false;
          break;
        }
      }
      if (!safe)
        continue;

      for (Operation *op : llvm::reverse(acquirePrefix))
        op->moveBefore(commonUser);
      acquire->moveBefore(commonUser);
      localLoad->moveBefore(commonUser);
      for (Operation *op : releasePrefix)
        op->moveBefore(commonUser);
      release->moveBefore(commonUser);
      for (Operation *op : llvm::reverse(reverseChain))
        op->moveBefore(commonUser);
      changed = true;
      break;
    }
  }
  return changed;
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
  // Constraint-guided TMEM epilogue sinking is safe independently of the
  // global WS-barrier normalization.  Build the mapping unconditionally so a
  // load can carry its own arrive across barriers from independent channels.
  DenseMap<Operation *, DictionaryAttr> memOpConstraints =
      buildTMemLoadConstraints(block);
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
  // Barrier restoration and TMEM-load sinking may move a load across a token
  // wait that was already positioned before staging reuse. Re-establish that
  // canonical placement so the load/conversion overlaps the prior TMA store.
  delayPlainTMAStoreTokenWaits(block);

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
    ModuleOp m = getOperation();

    SmallVector<BlockInterleaveInfo> blocksToProcess;
    m.walk([&](Block *block) {
      BlockInterleaveInfo info = collectBlockInterleaveInfo(block);
      if (info.tmemLoadCount > 0)
        prioritizeTMemOperand(*block);
      if (info.tmemLoadCount < 2)
        return;
      blocksToProcess.push_back(std::move(info));
    });
    for (auto &info : blocksToProcess)
      processBlock(info);
    // WS code partitioning keeps a structurally redundant P-publication wait
    // long enough for loop scheduling and TMEM interleaving to observe its
    // cross-partition ordering edge.  Erase it only after those transformations
    // have fixed the operation order; removing it in WSCodePartition changes
    // the schedule and does not reproduce the proven final-IR ablation.
    SmallVector<WaitBarrierOp> redundantPublicationWaits;
    m.walk([&](WaitBarrierOp wait) {
      if (wait->hasAttr("ttng.redundant_publication_wait"))
        redundantPublicationWaits.push_back(wait);
    });
    for (WaitBarrierOp wait : redundantPublicationWaits)
      wait.erase();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
