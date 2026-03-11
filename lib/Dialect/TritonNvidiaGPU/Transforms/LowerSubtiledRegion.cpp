#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERSUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Emit a barrier operation based on the annotation kind.
static void emitBarrierOp(OpBuilder &builder, Location loc,
                          BarrierAnnotationAttr annotation, ValueRange barriers,
                          ValueRange phases) {
  unsigned idx = annotation.getBarrierIdx();
  StringRef kind = annotation.getBarrierOpKind().getValue();
  if (kind == "wait_barrier") {
    builder.create<WaitBarrierOp>(loc, barriers[idx], phases[idx]);
  } else {
    assert(kind == "arrive_barrier");
    builder.create<ArriveBarrierOp>(loc, barriers[idx], annotation.getCount());
  }
}

class TritonNvidiaGPULowerSubtiledRegionPass
    : public impl::TritonNvidiaGPULowerSubtiledRegionPassBase<
          TritonNvidiaGPULowerSubtiledRegionPass> {
public:
  using TritonNvidiaGPULowerSubtiledRegionPassBase::
      TritonNvidiaGPULowerSubtiledRegionPassBase;

  void runOnOperation() override {
    // Collect all SubtiledRegionOps first to avoid modifying while iterating.
    SmallVector<SubtiledRegionOp> ops;
    getOperation().walk([&](SubtiledRegionOp op) { ops.push_back(op); });

    for (auto op : ops)
      lowerSubtiledRegion(op);
  }

private:
  void lowerSubtiledRegion(SubtiledRegionOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();

    // 1. Clone setup region ops (except yield) before the op.
    Block &setupBlock = op.getSetupRegion().front();
    IRMapping setupMapping;
    for (Operation &setupOp : setupBlock.without_terminator())
      builder.clone(setupOp, setupMapping);

    // 2. Collect remapped setup outputs from the cloned yield operands.
    auto yieldOp = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());
    SmallVector<Value> setupOutputs;
    for (Value v : yieldOp.getResults())
      setupOutputs.push_back(setupMapping.lookupOrDefault(v));

    // 3. Pre-process barrier annotations.
    //    beforeFirst[opIdx] -> annotations with placement=BEFORE
    //    afterLast[opIdx]   -> annotations with placement=AFTER
    DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> beforeFirst,
        afterLast;
    for (Attribute attr : op.getBarrierAnnotations()) {
      auto annotation = cast<BarrierAnnotationAttr>(attr);
      unsigned opIdx = annotation.getTargetOpIdx();
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        beforeFirst[opIdx].push_back(annotation);
      else
        afterLast[opIdx].push_back(annotation);
    }

    Block &tileBlock = op.getTileRegion().front();
    unsigned numTileArgs = tileBlock.getNumArguments();
    unsigned numTiles = setupOutputs.size() / numTileArgs;

    ValueRange barriers = op.getBarriers();
    ValueRange phases = op.getBarrierPhases();

    // 4. For each tile, clone tile region ops with substitution.
    //    Collect yielded values from each tile for teardown.
    auto tileYieldOp = cast<SubtiledRegionYieldOp>(tileBlock.getTerminator());
    unsigned numTileYieldOperands = tileYieldOp.getResults().size();
    // Flat vector of K*N values: tileYieldValues[tileIdx * N + yieldPos]
    SmallVector<Value> tileYieldValues;

    for (unsigned tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
      IRMapping tileMapping;

      // Build mapping: tile block arg -> sequential setup output
      for (unsigned j = 0; j < numTileArgs; ++j)
        tileMapping.map(tileBlock.getArgument(j),
                        setupOutputs[tileIdx * numTileArgs + j]);

      unsigned opIdx = 0;
      for (Operation &tileOp : tileBlock.without_terminator()) {
        // Before first tile: emit BEFORE annotations
        if (tileIdx == 0) {
          auto it = beforeFirst.find(opIdx);
          if (it != beforeFirst.end()) {
            for (auto &annotation : it->second)
              emitBarrierOp(builder, loc, annotation, barriers, phases);
          }
        }

        builder.clone(tileOp, tileMapping);

        // After last tile: emit AFTER annotations
        if (tileIdx == numTiles - 1) {
          auto it = afterLast.find(opIdx);
          if (it != afterLast.end()) {
            for (auto &annotation : it->second)
              emitBarrierOp(builder, loc, annotation, barriers, phases);
          }
        }
        ++opIdx;
      }

      // Collect remapped tile yield operands for this tile
      for (Value v : tileYieldOp.getResults())
        tileYieldValues.push_back(tileMapping.lookupOrDefault(v));
    }

    // 5. If teardown region is present, inline it and replace op results.
    if (!op.getTeardownRegion().empty()) {
      Block &teardownBlock = op.getTeardownRegion().front();
      IRMapping teardownMapping;

      // Map teardown block args to collected tile yield values (in order)
      for (auto [arg, val] :
           llvm::zip(teardownBlock.getArguments(), tileYieldValues))
        teardownMapping.map(arg, val);

      // Clone teardown ops (except yield)
      for (Operation &tdOp : teardownBlock.without_terminator())
        builder.clone(tdOp, teardownMapping);

      // Replace op results with remapped teardown yield operands
      auto teardownYieldOp =
          cast<SubtiledRegionYieldOp>(teardownBlock.getTerminator());
      for (auto [result, yieldVal] :
           llvm::zip(op.getResults(), teardownYieldOp.getResults()))
        result.replaceAllUsesWith(teardownMapping.lookupOrDefault(yieldVal));
    }

    // 6. Erase the SubtiledRegionOp.
    op.erase();
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
