#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERSUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

/// Emit a barrier operation based on the annotation kind.
static void emitBarrierOp(OpBuilder &builder, Location loc,
                          BarrierAnnotationAttr annotation, ValueRange barriers,
                          ValueRange phases) {
  unsigned idx = annotation.getBarrierIdx();
  StringRef kind = annotation.getBarrierOpKind().getValue();
  if (kind == "wait_barrier") {
    WaitBarrierOp::create(builder, loc, barriers[idx], phases[idx]);
  } else {
    assert(kind == "arrive_barrier");
    ArriveBarrierOp::create(builder, loc, barriers[idx], annotation.getCount());
  }
}

/// Emit barrier ops for a list of annotations at a given op index in a
/// region block, using the provided builder.
static void emitBarriersForRegion(
    OpBuilder &builder, Location loc, Block &block,
    llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> &beforeMap,
    llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> &afterMap,
    ValueRange barriers, ValueRange phases, IRMapping &mapping) {
  unsigned opIdx = 0;
  for (Operation &op : block.without_terminator()) {
    auto itBefore = beforeMap.find(opIdx);
    if (itBefore != beforeMap.end()) {
      for (auto &annotation : itBefore->second)
        emitBarrierOp(builder, loc, annotation, barriers, phases);
    }

    builder.clone(op, mapping);

    auto itAfter = afterMap.find(opIdx);
    if (itAfter != afterMap.end()) {
      for (auto &annotation : itAfter->second)
        emitBarrierOp(builder, loc, annotation, barriers, phases);
    }
    ++opIdx;
  }
}

void lowerSubtiledRegion(SubtiledRegionOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  ValueRange barriers = op.getBarriers();
  ValueRange phases = op.getBarrierPhases();

  // Pre-process barrier annotations by region and target op index.
  // For TILE region: beforeFirst[opIdx] / afterLast[opIdx]
  // For SETUP/TEARDOWN: before[opIdx] / after[opIdx]
  llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> tileBeforeFirst,
      tileAfterLast;
  llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> setupBefore,
      setupAfter, teardownBefore, teardownAfter;

  for (Attribute attr : op.getBarrierAnnotations()) {
    auto annotation = cast<BarrierAnnotationAttr>(attr);
    unsigned opIdx = annotation.getTargetOpIdx();
    BarrierRegion region = annotation.getRegion();

    if (region == BarrierRegion::SETUP) {
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        setupBefore[opIdx].push_back(annotation);
      else
        setupAfter[opIdx].push_back(annotation);
    } else if (region == BarrierRegion::TEARDOWN) {
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        teardownBefore[opIdx].push_back(annotation);
      else
        teardownAfter[opIdx].push_back(annotation);
    } else {
      // TILE region: existing behavior (BEFORE first tile, AFTER last tile).
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        tileBeforeFirst[opIdx].push_back(annotation);
      else
        tileAfterLast[opIdx].push_back(annotation);
    }
  }

  // 1. Clone setup region ops (except yield), emitting setup barriers.
  Block &setupBlock = op.getSetupRegion().front();
  IRMapping setupMapping;
  if (setupBefore.empty() && setupAfter.empty()) {
    // Fast path: no setup barriers.
    for (Operation &setupOp : setupBlock.without_terminator())
      builder.clone(setupOp, setupMapping);
  } else {
    emitBarriersForRegion(builder, loc, setupBlock, setupBefore, setupAfter,
                          barriers, phases, setupMapping);
  }

  // 2. Collect remapped setup outputs from the cloned yield operands.
  auto yieldOp = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());
  SmallVector<Value> setupOutputs;
  for (Value v : yieldOp.getResults())
    setupOutputs.push_back(setupMapping.lookupOrDefault(v));

  ArrayAttr tileMappings = op.getTileMappings();
  unsigned numTiles = tileMappings.size();
  Block &tileBlock = op.getTileRegion().front();

  // 3. For each tile, clone tile region ops with substitution.
  for (unsigned tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
    auto indices = cast<DenseI32ArrayAttr>(tileMappings[tileIdx]);
    IRMapping tileMapping;

    // Build mapping: tile block arg -> setup output
    for (auto [j, idx] : llvm::enumerate(indices.asArrayRef()))
      tileMapping.map(tileBlock.getArgument(j), setupOutputs[idx]);

    unsigned opIdx = 0;
    for (Operation &tileOp : tileBlock.without_terminator()) {
      // Before first tile: emit BEFORE annotations for this op index
      if (tileIdx == 0) {
        auto it = tileBeforeFirst.find(opIdx);
        if (it != tileBeforeFirst.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, phases);
        }
      }

      builder.clone(tileOp, tileMapping);

      // After last tile: emit AFTER annotations for this op index
      if (tileIdx == numTiles - 1) {
        auto it = tileAfterLast.find(opIdx);
        if (it != tileAfterLast.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, phases);
        }
      }
      ++opIdx;
    }
  }

  // 4. Clone teardown region ops (except terminator), emitting teardown
  // barriers.
  Block &teardownBlock = op.getTeardownRegion().front();
  IRMapping teardownMapping;
  if (teardownBefore.empty() && teardownAfter.empty()) {
    for (Operation &teardownOp : teardownBlock.without_terminator())
      builder.clone(teardownOp, teardownMapping);
  } else {
    emitBarriersForRegion(builder, loc, teardownBlock, teardownBefore,
                          teardownAfter, barriers, phases, teardownMapping);
  }

  // 5. Replace op results with teardown yield values.
  auto teardownTerminator =
      cast<SubtiledRegionYieldOp>(teardownBlock.getTerminator());
  for (auto [opResult, teardownVal] :
       llvm::zip(op.getResults(), teardownTerminator.getResults()))
    opResult.replaceAllUsesWith(teardownMapping.lookupOrDefault(teardownVal));

  // 6. Erase the SubtiledRegionOp.
  op.erase();
}

namespace {

class TritonNvidiaGPULowerSubtiledRegionPass
    : public impl::TritonNvidiaGPULowerSubtiledRegionPassBase<
          TritonNvidiaGPULowerSubtiledRegionPass> {
public:
  using TritonNvidiaGPULowerSubtiledRegionPassBase::
      TritonNvidiaGPULowerSubtiledRegionPassBase;

  void runOnOperation() override {
    SmallVector<SubtiledRegionOp> ops;
    getOperation().walk([&](SubtiledRegionOp op) { ops.push_back(op); });

    for (auto op : ops)
      lowerSubtiledRegion(op);
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
