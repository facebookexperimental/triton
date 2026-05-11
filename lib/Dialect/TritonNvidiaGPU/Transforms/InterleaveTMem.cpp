#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/WSBarrierReorder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-nvidia-interleave-tmem"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

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

// Sink tmem_loads as close to their use as possible to reduce register
// pressure. When opConstraints is provided, uses canAdvanceWSBarrier to
// decide whether the op can sink past barriers from independent channels.
bool sinkOps(Value buffer, ArrayRef<Operation *> useChain,
             std::optional<DictionaryAttr> opConstraints) {
  Operation *insertBefore = nullptr;
  Operation *next = useChain.back()->getNextNode();
  while (next && !next->hasTrait<OpTrait::IsTerminator>()) {
    insertBefore = next;
    bool dep = false;
    for (auto operand : getNestedOperands(next)) {
      if (llvm::any_of(useChain, [&](Operation *op) {
            return llvm::is_contained(op->getResults(), operand);
          })) {
        dep = true;
        break;
      }
    }
    if (!canAdvanceWSBarrier(opConstraints, next))
      break;
    if (!isMemoryEffectFree(next)) {
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

// Try to sink a load and a collection of its users.
bool trySinkOp(Operation *op, Value buffer,
               std::optional<DictionaryAttr> opConstraints) {
  SmallVector<Operation *> useChain{op};
  while (useChain.back()->hasOneUse() &&
         isPure(*useChain.back()->user_begin()) &&
         useChain.back()->getNextNode() == *useChain.back()->user_begin()) {
    useChain.push_back(*useChain.back()->user_begin());
  }
  return sinkOps(buffer, useChain, opConstraints);
}

struct BlockInterleaveInfo {
  Block *block;
  unsigned tmemLoadCount = 0;
  SmallVector<std::pair<Operation *, Value>> opsToSink;
};

BlockInterleaveInfo collectBlockInterleaveInfo(Block *block) {
  BlockInterleaveInfo info;
  info.block = block;
  for (Operation &op : *block) {
    if (auto load = dyn_cast<TMEMLoadOp>(&op)) {
      info.tmemLoadCount++;
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

  // Step 1: Record which memory op each WS barrier guards.
  auto barrierMap = buildBarrierToMemoryOpMap(block);

  // Step 2: Reorder WS barriers. Pushes arrives down and pulls waits up past
  // barriers from independent channels, unblocking tmem_load sinking.
  sinkWSArrives(block);
  raiseWSWaits(block);

  // Step 3: Sink tmem_loads closer to their uses.
  auto memOpConstraints = buildTMemLoadConstraints(block);
  for (auto [op, buffer] : info.opsToSink) {
    auto it = memOpConstraints.find(op);
    std::optional<DictionaryAttr> constraints =
        it != memOpConstraints.end() ? std::optional<DictionaryAttr>(it->second)
                                     : std::nullopt;
    while (trySinkOp(op, buffer, constraints)) {
    }
  }

  // Step 4: Restore barriers to optimal positions near their memory ops.
  optimizeWSBarrierLocations(barrierMap);
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
