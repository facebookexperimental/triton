#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTESTGENERATESUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Get the async task IDs from an operation.
static SmallVector<int32_t> getOpAsyncTaskIds(Operation *op) {
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id"))
    return SmallVector<int32_t>(attr.asArrayRef());
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(gpu::kPartitionAttrName))
    return SmallVector<int32_t>(attr.asArrayRef());
  return {};
}

/// A segment of structurally equivalent per-tile chain ops with a uniform
/// async task set.
struct ChainSegment {
  SmallVector<Operation *> ops0; // from chain0
  SmallVector<Operation *> ops1; // from chain1
  SmallVector<int32_t> taskIds;
};

/// Strip convert_layout ops wrapping a value.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<gpu::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

/// Trace the setup chain backward from a SplitOp:
///   split <- trans{[0,2,1]} <- reshape <- (convert_layout)* <- tmem_load
/// Returns the tmem_load op, or nullptr if the pattern doesn't match.
static TMEMLoadOp traceSetupChain(triton::SplitOp splitOp) {
  Value src = stripConvertLayout(splitOp.getSrc());
  auto transOp = src.getDefiningOp<triton::TransOp>();
  if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
    return nullptr;
  auto reshapeOp = transOp.getSrc().getDefiningOp<triton::ReshapeOp>();
  if (!reshapeOp)
    return nullptr;
  Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
  return reshapeSrc.getDefiningOp<TMEMLoadOp>();
}

/// Check if two per-tile op chains are structurally equivalent.
/// Returns a list of (operand_from_chain0, operand_from_chain1) pairs
/// for operands that differ between the two chains.
/// Returns nullopt if the chains are not structurally equivalent.
static std::optional<SmallVector<std::pair<Value, Value>>>
checkStructuralEquivalence(ArrayRef<Operation *> chain0,
                           ArrayRef<Operation *> chain1) {
  if (chain0.size() != chain1.size())
    return std::nullopt;

  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> differingOperands;

  for (auto [op0, op1] : llvm::zip(chain0, chain1)) {
    if (op0->getName() != op1->getName())
      return std::nullopt;
    if (op0->getNumOperands() != op1->getNumOperands())
      return std::nullopt;
    if (op0->getNumResults() != op1->getNumResults())
      return std::nullopt;
    for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults())) {
      if (r0.getType() != r1.getType())
        return std::nullopt;
    }
    if (op0->getAttrDictionary() != op1->getAttrDictionary())
      return std::nullopt;

    for (auto [v0, v1] : llvm::zip(op0->getOperands(), op1->getOperands())) {
      auto it = valueMap.find(v0);
      if (it != valueMap.end()) {
        if (it->second != v1)
          return std::nullopt;
        continue;
      }
      if (v0 == v1)
        continue;
      differingOperands.push_back({v0, v1});
      valueMap[v0] = v1;
    }

    for (auto [r0, r1] : llvm::zip(op0->getResults(), op1->getResults()))
      valueMap[r0] = r1;
  }

  return differingOperands;
}

/// Collect the per-tile op chain for a split result: all ops in the block
/// that transitively depend on `splitResult`, plus any ops needed exclusively
/// by those (e.g., offset computations).
static SmallVector<Operation *>
collectPerTileChain(Value splitResult, Operation *splitOp, Block *block) {
  SmallVector<Operation *> chain;
  llvm::DenseSet<Operation *> visited;
  SmallVector<Value> worklist;
  worklist.push_back(splitResult);

  // Forward walk: find all transitive users of the split result.
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user->getBlock() != block)
        continue;
      if (user->isBeforeInBlock(splitOp) || user == splitOp)
        continue;
      if (!visited.insert(user).second)
        continue;
      chain.push_back(user);
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }

  // Also collect ops that are needed by the chain but don't depend on the
  // split result (e.g., offset computations like arith.addi %col, %half_n).
  llvm::DenseSet<Operation *> chainSet(chain.begin(), chain.end());
  for (Operation *op : chain) {
    for (Value operand : op->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != block)
        continue;
      if (defOp->isBeforeInBlock(splitOp) || defOp == splitOp)
        continue;
      if (!chainSet.contains(defOp) && visited.insert(defOp).second)
        chainSet.insert(defOp);
    }
  }

  // Rebuild as sorted list.
  SmallVector<Operation *> fullChain;
  for (Operation *op : chainSet)
    fullChain.push_back(op);
  llvm::sort(fullChain,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return fullChain;
}

/// Group structurally equivalent chain ops by contiguous async task set.
/// Ops without task IDs are merged into the current segment.
/// Returns nullopt if corresponding ops in chain0/chain1 have different task
/// sets.
static std::optional<SmallVector<ChainSegment>>
groupByContiguousTaskSet(ArrayRef<Operation *> chain0,
                         ArrayRef<Operation *> chain1) {
  assert(chain0.size() == chain1.size());
  if (chain0.empty())
    return std::nullopt;

  SmallVector<ChainSegment> segments;
  for (size_t i = 0; i < chain0.size(); ++i) {
    auto taskIds0 = getOpAsyncTaskIds(chain0[i]);
    auto taskIds1 = getOpAsyncTaskIds(chain1[i]);

    // Corresponding ops must have the same task set.
    if (taskIds0 != taskIds1)
      return std::nullopt;

    // Ops without task IDs join the current segment.
    if (taskIds0.empty() && !segments.empty()) {
      segments.back().ops0.push_back(chain0[i]);
      segments.back().ops1.push_back(chain1[i]);
      continue;
    }

    // Start a new segment if the task set changes.
    if (segments.empty() || segments.back().taskIds != taskIds0)
      segments.push_back({{}, {}, taskIds0});

    segments.back().ops0.push_back(chain0[i]);
    segments.back().ops1.push_back(chain1[i]);
  }
  return segments;
}

/// Build a single SubtiledRegionOp (existing logic).
static void
buildSingleSubtiledRegion(OpBuilder &builder, Location loc,
                          ArrayRef<Operation *> setupOps, Value lhs, Value rhs,
                          ArrayRef<Operation *> chain0,
                          ArrayRef<std::pair<Value, Value>> differing) {
  // Tile arg types and mappings.
  SmallVector<Type> tileArgTypes;
  SmallVector<int32_t> tile0Mapping, tile1Mapping;

  // Tile arg 0: split result.
  tileArgTypes.push_back(lhs.getType());
  tile0Mapping.push_back(0);
  tile1Mapping.push_back(1);

  // Additional tile args from differing operands.
  for (auto [i, pair] : llvm::enumerate(differing)) {
    tileArgTypes.push_back(pair.first.getType());
    tile0Mapping.push_back(2 + 2 * i);
    tile1Mapping.push_back(2 + 2 * i + 1);
  }

  auto tileMappingsAttr = builder.getArrayAttr(
      {DenseI32ArrayAttr::get(builder.getContext(), tile0Mapping),
       DenseI32ArrayAttr::get(builder.getContext(), tile1Mapping)});
  auto barrierAnnotationsAttr = builder.getArrayAttr({});

  auto regionOp = SubtiledRegionOp::create(
      builder, loc, /*resultTypes=*/TypeRange{},
      /*barriers=*/ValueRange{}, /*barrierPhases=*/ValueRange{},
      tileMappingsAttr, barrierAnnotationsAttr);

  // --- Setup Region ---
  Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
  OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);
  IRMapping setupMapping;
  for (Operation *op : setupOps)
    setupBuilder.clone(*op, setupMapping);

  SmallVector<Value> setupYieldValues;
  setupYieldValues.push_back(setupMapping.lookupOrDefault(lhs));
  setupYieldValues.push_back(setupMapping.lookupOrDefault(rhs));
  for (auto [v0, v1] : differing) {
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v0));
    setupYieldValues.push_back(setupMapping.lookupOrDefault(v1));
  }
  SubtiledRegionYieldOp::create(setupBuilder, loc, setupYieldValues);

  // --- Tile Region ---
  Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
  for (Type ty : tileArgTypes)
    tileBlock->addArgument(ty, loc);

  OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
  IRMapping tileMapping;
  tileMapping.map(lhs, tileBlock->getArgument(0));
  for (auto [i, pair] : llvm::enumerate(differing))
    tileMapping.map(pair.first, tileBlock->getArgument(1 + i));

  for (Operation *op : chain0)
    tileBuilder.clone(*op, tileMapping);
  SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

  // --- Teardown Region ---
  Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
  OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
  SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
}

/// Build multiple SubtiledRegionOps for a chain that spans multiple contiguous
/// async task sets, with explicit memory stores (local_alloc with data) at the
/// transitions.
///
/// At each transition, the local_alloc is split into:
///   - An empty local_alloc in outer scope (one per tile)
///   - A local_store in the first segment's tile body
/// The outer-scope SMEM memdescs are captured by both the producing and
/// consuming SubtiledRegionOps.
static void buildMultiTaskSubtiledRegions(OpBuilder &outerBuilder, Location loc,
                                          ArrayRef<Operation *> setupOps,
                                          Value lhs, Value rhs,
                                          ArrayRef<ChainSegment> segments) {
  MLIRContext *ctx = outerBuilder.getContext();

  // Step 1: Create outer-scope SMEM allocations for each transition.
  // Transition i is between segments[i] and segments[i+1].
  // segments[i] ends with a local_alloc-with-data op.
  struct TransitionInfo {
    gpu::LocalAllocOp alloc0; // from chain0
    gpu::LocalAllocOp alloc1; // from chain1
    Value smem0;              // outer-scope empty alloc for tile 0
    Value smem1;              // outer-scope empty alloc for tile 1
  };
  SmallVector<TransitionInfo> transitions;

  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    auto alloc0 = cast<gpu::LocalAllocOp>(segments[i].ops0.back());
    auto alloc1 = cast<gpu::LocalAllocOp>(segments[i].ops1.back());

    // Ensure the memdesc type is mutable for the empty alloc.
    auto memDescType = cast<gpu::MemDescType>(alloc0.getType());
    if (!memDescType.getMutableMemory()) {
      memDescType = gpu::MemDescType::get(
          memDescType.getShape(), memDescType.getElementType(),
          memDescType.getEncoding(), memDescType.getMemorySpace(),
          /*mutableMemory=*/true, memDescType.getAllocShape());
    }

    auto emptyAlloc0 =
        gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});
    auto emptyAlloc1 =
        gpu::LocalAllocOp::create(outerBuilder, loc, memDescType, Value{});

    transitions.push_back(
        {alloc0, alloc1, emptyAlloc0.getResult(), emptyAlloc1.getResult()});
  }

  // Step 2: Generate a SubtiledRegionOp for each segment.
  for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx) {
    const auto &seg = segments[segIdx];
    bool isFirstSegment = (segIdx == 0);
    bool hasTransitionStore = (segIdx < transitions.size());

    // Build the sub-chain for structural equivalence, excluding the
    // transition local_alloc (it will be replaced by local_store).
    SmallVector<Operation *> subOps0(seg.ops0);
    SmallVector<Operation *> subOps1(seg.ops1);
    if (hasTransitionStore) {
      subOps0.pop_back(); // remove local_alloc
      subOps1.pop_back();
    }

    // Compute per-segment differing operands.
    auto segDiff = checkStructuralEquivalence(subOps0, subOps1);
    if (!segDiff) // shouldn't happen since full chain passed
      return;

    // For cross-segment operands (e.g., memdescs from previous segments),
    // replace original values with outer-scope SMEM values.
    // Track the original chain0 values for tileMapping.
    struct DiffEntry {
      Value chain0Val; // original value in chain0 ops (for tileMapping)
      Value setupVal0; // value to yield in setup for tile 0
      Value setupVal1; // value to yield in setup for tile 1
    };
    SmallVector<DiffEntry> resolvedDiff;
    for (auto &[v0, v1] : *segDiff) {
      Value setupV0 = v0, setupV1 = v1;
      for (auto &tr : transitions) {
        if (v0 == tr.alloc0.getResult())
          setupV0 = tr.smem0;
        if (v1 == tr.alloc1.getResult())
          setupV1 = tr.smem1;
      }
      resolvedDiff.push_back({v0, setupV0, setupV1});
    }

    // Build tile arg types and mappings.
    SmallVector<Type> tileArgTypes;
    SmallVector<int32_t> tile0Map, tile1Map;
    int32_t yieldIdx = 0;

    if (isFirstSegment) {
      // First segment: tile arg 0 is the split result.
      tileArgTypes.push_back(lhs.getType());
      tile0Map.push_back(0);
      tile1Map.push_back(1);
      yieldIdx = 2;
    }

    // Differing operands.
    for (auto &entry : resolvedDiff) {
      tileArgTypes.push_back(entry.chain0Val.getType());
      tile0Map.push_back(yieldIdx);
      tile1Map.push_back(yieldIdx + 1);
      yieldIdx += 2;
    }

    // SMEM arg for local_store (if this segment has a transition store).
    if (hasTransitionStore) {
      tileArgTypes.push_back(transitions[segIdx].smem0.getType());
      tile0Map.push_back(yieldIdx);
      tile1Map.push_back(yieldIdx + 1);
      yieldIdx += 2;
    }

    auto tileMappingsAttr =
        outerBuilder.getArrayAttr({DenseI32ArrayAttr::get(ctx, tile0Map),
                                   DenseI32ArrayAttr::get(ctx, tile1Map)});
    auto barrierAnnotationsAttr = outerBuilder.getArrayAttr({});

    auto regionOp = SubtiledRegionOp::create(
        outerBuilder, loc, TypeRange{}, ValueRange{}, ValueRange{},
        tileMappingsAttr, barrierAnnotationsAttr);

    // --- Setup Region ---
    Block *setupBlock = &regionOp.getSetupRegion().emplaceBlock();
    OpBuilder setupBuilder = OpBuilder::atBlockEnd(setupBlock);

    SmallVector<Value> setupYields;
    if (isFirstSegment) {
      // Clone the setup chain (tmem_load → split) into the setup region.
      IRMapping setupMapping;
      for (Operation *op : setupOps)
        setupBuilder.clone(*op, setupMapping);
      setupYields.push_back(setupMapping.lookupOrDefault(lhs));
      setupYields.push_back(setupMapping.lookupOrDefault(rhs));

      // Yield differing operands (pass through from outer scope via mapping).
      for (auto &entry : resolvedDiff) {
        setupYields.push_back(setupMapping.lookupOrDefault(entry.setupVal0));
        setupYields.push_back(setupMapping.lookupOrDefault(entry.setupVal1));
      }
    } else {
      // Non-first segments: just yield outer-scope values.
      for (auto &entry : resolvedDiff) {
        setupYields.push_back(entry.setupVal0);
        setupYields.push_back(entry.setupVal1);
      }
    }

    // Yield SMEM values for transition store.
    if (hasTransitionStore) {
      setupYields.push_back(transitions[segIdx].smem0);
      setupYields.push_back(transitions[segIdx].smem1);
    }

    SubtiledRegionYieldOp::create(setupBuilder, loc, setupYields);

    // --- Tile Region ---
    Block *tileBlock = &regionOp.getTileRegion().emplaceBlock();
    for (Type ty : tileArgTypes)
      tileBlock->addArgument(ty, loc);

    OpBuilder tileBuilder = OpBuilder::atBlockEnd(tileBlock);
    IRMapping tileMapping;
    unsigned argIdx = 0;

    if (isFirstSegment)
      tileMapping.map(lhs, tileBlock->getArgument(argIdx++));

    for (auto &entry : resolvedDiff)
      tileMapping.map(entry.chain0Val, tileBlock->getArgument(argIdx++));

    Value smemTileArg;
    if (hasTransitionStore)
      smemTileArg = tileBlock->getArgument(argIdx++);

    // Clone segment ops into the tile body.
    for (Operation *op : subOps0)
      tileBuilder.clone(*op, tileMapping);

    // Emit local_store at the end (replacing the local_alloc).
    if (hasTransitionStore) {
      auto allocOp = transitions[segIdx].alloc0;
      Value data = tileMapping.lookupOrDefault(allocOp.getSrc());
      gpu::LocalStoreOp::create(tileBuilder, loc, data, smemTileArg);
    }

    SubtiledRegionYieldOp::create(tileBuilder, loc, ValueRange{});

    // --- Teardown Region ---
    Block *teardownBlock = &regionOp.getTeardownRegion().emplaceBlock();
    OpBuilder teardownBuilder = OpBuilder::atBlockEnd(teardownBlock);
    SubtiledRegionYieldOp::create(teardownBuilder, loc, ValueRange{});
  }
}

} // anonymous namespace

void tryGenerateForSplit(triton::SplitOp splitOp) {
  auto tmemLoad = traceSetupChain(splitOp);
  if (!tmemLoad)
    return;

  Block *block = splitOp->getBlock();
  Value lhs = splitOp.getOutLHS();
  Value rhs = splitOp.getOutRHS();

  SmallVector<Operation *> chain0 = collectPerTileChain(lhs, splitOp, block);
  SmallVector<Operation *> chain1 = collectPerTileChain(rhs, splitOp, block);

  if (chain0.empty() || chain1.empty())
    return;

  auto differing = checkStructuralEquivalence(chain0, chain1);
  if (!differing)
    return;

  // Group by contiguous async task set.
  auto segments = groupByContiguousTaskSet(chain0, chain1);
  if (!segments || segments->empty())
    return;

  // Collect setup ops: everything from tmemLoad to splitOp (inclusive).
  SmallVector<Operation *> setupOps;
  for (auto it = Block::iterator(tmemLoad); it != Block::iterator(splitOp);
       ++it)
    setupOps.push_back(&*it);
  setupOps.push_back(splitOp);

  OpBuilder builder(tmemLoad);
  Location loc = splitOp.getLoc();

  if (segments->size() == 1) {
    // Single task set: generate one SubtiledRegionOp.
    buildSingleSubtiledRegion(builder, loc, setupOps, lhs, rhs, chain0,
                              *differing);
  } else {
    // Multiple task sets: verify transitions are at explicit memory stores.
    for (size_t i = 0; i + 1 < segments->size(); ++i) {
      Operation *lastOp = (*segments)[i].ops0.back();
      auto alloc = dyn_cast<gpu::LocalAllocOp>(lastOp);
      if (!alloc || !alloc.getSrc())
        return; // Not an explicit memory store, bail out.

      // Verify the local_alloc result is not used within the same segment.
      for (size_t j = 0; j + 1 < (*segments)[i].ops0.size(); ++j) {
        for (Value operand : (*segments)[i].ops0[j]->getOperands()) {
          if (operand == alloc.getResult())
            return; // alloc result used before the alloc, bail out.
        }
      }
    }

    buildMultiTaskSubtiledRegions(builder, loc, setupOps, lhs, rhs, *segments);
  }

  // Erase original ops (reverse program order).
  for (Operation *op : llvm::reverse(chain1))
    op->erase();
  for (Operation *op : llvm::reverse(chain0))
    op->erase();
  for (Operation *op : llvm::reverse(setupOps)) {
    if (op->use_empty())
      op->erase();
  }
}

namespace {

class TritonNvidiaGPUTestGenerateSubtiledRegionPass
    : public impl::TritonNvidiaGPUTestGenerateSubtiledRegionPassBase<
          TritonNvidiaGPUTestGenerateSubtiledRegionPass> {
public:
  using TritonNvidiaGPUTestGenerateSubtiledRegionPassBase::
      TritonNvidiaGPUTestGenerateSubtiledRegionPassBase;

  void runOnOperation() override {
    SmallVector<triton::SplitOp> splitOps;
    getOperation().walk([&](triton::SplitOp op) { splitOps.push_back(op); });
    for (auto splitOp : splitOps)
      tryGenerateForSplit(splitOp);
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
