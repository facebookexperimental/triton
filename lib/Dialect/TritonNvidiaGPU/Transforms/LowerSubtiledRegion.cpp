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
    //    beforeFirst[opName] -> annotations with placement=BEFORE
    //    afterLast[opName]   -> annotations with placement=AFTER
    llvm::StringMap<SmallVector<BarrierAnnotationAttr>> beforeFirst, afterLast;
    for (Attribute attr : op.getBarrierAnnotations()) {
      auto annotation = cast<BarrierAnnotationAttr>(attr);
      StringRef opName = annotation.getTargetOpName().getValue();
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        beforeFirst[opName].push_back(annotation);
      else
        afterLast[opName].push_back(annotation);
    }

    ArrayAttr tileMappings = op.getTileMappings();
    unsigned numTiles = tileMappings.size();
    Block &tileBlock = op.getTileRegion().front();

    ValueRange barriers = op.getBarriers();
    ValueRange phases = op.getBarrierPhases();

    // 4. For each tile, clone tile region ops with substitution.
    for (unsigned tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
      auto indices = cast<DenseI32ArrayAttr>(tileMappings[tileIdx]);
      IRMapping tileMapping;

      // Build mapping: tile block arg -> setup output
      for (auto [j, idx] : llvm::enumerate(indices.asArrayRef()))
        tileMapping.map(tileBlock.getArgument(j), setupOutputs[idx]);

      for (Operation &tileOp : tileBlock.without_terminator()) {
        StringRef opName = tileOp.getName().getStringRef();

        // Before first tile: emit BEFORE annotations
        if (tileIdx == 0) {
          auto it = beforeFirst.find(opName);
          if (it != beforeFirst.end()) {
            for (auto &annotation : it->second)
              emitBarrierOp(builder, loc, annotation, barriers, phases);
          }
        }

        builder.clone(tileOp, tileMapping);

        // After last tile: emit AFTER annotations
        if (tileIdx == numTiles - 1) {
          auto it = afterLast.find(opName);
          if (it != afterLast.end()) {
            for (auto &annotation : it->second)
              emitBarrierOp(builder, loc, annotation, barriers, phases);
          }
        }
      }
    }

    // 5. Erase the SubtiledRegionOp.
    op.erase();
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
