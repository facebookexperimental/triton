#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERSUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

/// Compute the phase from an accumulation count and number of buffers:
///   phase = (accumCnt / numBuffers) & 1
/// Returns an i32 value.
static Value computePhase(OpBuilder &builder, Location loc, Value accumCnt,
                          unsigned numBuffers) {
  Value numBuf = arith::ConstantOp::create(
      builder, loc, builder.getI64IntegerAttr(numBuffers));
  Value div = arith::DivUIOp::create(builder, loc, accumCnt, numBuf);
  Value one64 =
      arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1));
  Value andOp = arith::AndIOp::create(builder, loc, div, one64);
  return arith::TruncIOp::create(builder, loc, builder.getI32Type(), andOp);
}

/// Emit a barrier operation based on the annotation kind.
/// For wait_barrier, computes the phase from the accumulation count.
/// `resolvedBarrierIdx` overrides the annotation's barrierIdx when
/// bufferIdxArgIdx is used (per-tile dynamic barrier selection).
static void emitBarrierOp(OpBuilder &builder, Location loc,
                          BarrierAnnotationAttr annotation, ValueRange barriers,
                          ValueRange accumCnts, unsigned resolvedBarrierIdx) {
  StringRef kind = annotation.getBarrierOpKind().getValue();
  if (kind == "wait_barrier") {
    Value phase = computePhase(builder, loc, accumCnts[resolvedBarrierIdx],
                               annotation.getNumBuffers());
    WaitBarrierOp::create(builder, loc, barriers[resolvedBarrierIdx], phase);
  } else {
    assert(kind == "arrive_barrier");
    ArriveBarrierOp::create(builder, loc, barriers[resolvedBarrierIdx],
                            annotation.getCount());
  }
}

/// Resolve the barrier index for an annotation. If bufferIdxArgIdx is set,
/// look up the tile arg's value from the setup outputs (via tileMappings).
/// The setup yield value must be a constant integer.
static unsigned resolveBarrierIdx(BarrierAnnotationAttr annotation,
                                  ArrayRef<Value> setupOutputs,
                                  ArrayRef<int32_t> tileIndices) {
  int argIdx = annotation.getBufferIdxArgIdx();
  if (argIdx < 0)
    return annotation.getBarrierIdx();

  // Look up the setup yield value for this tile's buffer index arg.
  unsigned setupYieldIdx = tileIndices[argIdx];
  Value setupVal = setupOutputs[setupYieldIdx];

  // The setup yield must be a constant integer (produced by arith.constant).
  auto constOp = setupVal.getDefiningOp<arith::ConstantIntOp>();
  assert(constOp && "bufferIdxArgIdx must resolve to a constant integer");
  return constOp.value();
}

/// Emit barrier ops for a list of annotations at a given op index in a
/// region block, using the provided builder. Uses static barrierIdx
/// (no tile-mapped resolution — for setup/teardown regions).
static void emitBarriersForRegion(
    OpBuilder &builder, Location loc, Block &block,
    llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> &beforeMap,
    llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> &afterMap,
    ValueRange barriers, ValueRange accumCnts, IRMapping &mapping) {
  unsigned opIdx = 0;
  for (Operation &op : block.without_terminator()) {
    auto itBefore = beforeMap.find(opIdx);
    if (itBefore != beforeMap.end()) {
      for (auto &annotation : itBefore->second)
        emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                      annotation.getBarrierIdx());
    }

    builder.clone(op, mapping);

    auto itAfter = afterMap.find(opIdx);
    if (itAfter != afterMap.end()) {
      for (auto &annotation : itAfter->second)
        emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                      annotation.getBarrierIdx());
    }
    ++opIdx;
  }
}

void lowerSubtiledRegion(SubtiledRegionOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  ValueRange barriers = op.getBarriers();
  ValueRange accumCnts = op.getAccumCnts();

  // Pre-process barrier annotations by region and target op index.
  // For TILE region with BEFORE/AFTER: first/last tile only.
  // For TILE region with TILE_START/TILE_END: every tile.
  // For SETUP/TEARDOWN: placed in the respective region.
  llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> tileBeforeFirst,
      tileAfterLast, tileEveryStart, tileEveryEnd;
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
      // TILE region.
      switch (annotation.getPlacement()) {
      case BarrierPlacement::BEFORE:
        tileBeforeFirst[opIdx].push_back(annotation);
        break;
      case BarrierPlacement::AFTER:
        tileAfterLast[opIdx].push_back(annotation);
        break;
      case BarrierPlacement::TILE_START:
        tileEveryStart[opIdx].push_back(annotation);
        break;
      case BarrierPlacement::TILE_END:
        tileEveryEnd[opIdx].push_back(annotation);
        break;
      }
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
                          barriers, accumCnts, setupMapping);
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

    // Helper to resolve barrier index for this tile's annotations.
    auto resolve = [&](BarrierAnnotationAttr annotation) -> unsigned {
      return resolveBarrierIdx(annotation, setupOutputs, indices.asArrayRef());
    };

    unsigned opIdx = 0;
    for (Operation &tileOp : tileBlock.without_terminator()) {
      // BEFORE first tile only.
      if (tileIdx == 0) {
        auto it = tileBeforeFirst.find(opIdx);
        if (it != tileBeforeFirst.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                          resolve(annotation));
        }
      }

      // TILE_START: before every tile.
      {
        auto it = tileEveryStart.find(opIdx);
        if (it != tileEveryStart.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                          resolve(annotation));
        }
      }

      builder.clone(tileOp, tileMapping);

      // TILE_END: after every tile.
      {
        auto it = tileEveryEnd.find(opIdx);
        if (it != tileEveryEnd.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                          resolve(annotation));
        }
      }

      // AFTER last tile only.
      if (tileIdx == numTiles - 1) {
        auto it = tileAfterLast.find(opIdx);
        if (it != tileAfterLast.end()) {
          for (auto &annotation : it->second)
            emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                          resolve(annotation));
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
                          teardownAfter, barriers, accumCnts, teardownMapping);
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
