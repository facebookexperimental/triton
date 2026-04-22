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

/// Compute tileAccumCnt = outerAccumCnt + tileIdx (as i64).
static Value computeTileAccumCnt(OpBuilder &builder, Location loc,
                                 Value outerAccumCnt, unsigned tileIdx) {
  Value tileIdxVal = arith::ConstantOp::create(
      builder, loc, builder.getI64IntegerAttr(tileIdx));
  return arith::AddIOp::create(builder, loc, outerAccumCnt, tileIdxVal);
}

/// Emit a barrier operation based on the annotation kind.
/// For tile region annotations with a tileMask, `tileIdx` is used to compute
/// the per-tile buffer index and phase. For setup/teardown annotations,
/// the static barrierIdx is used directly.
static void emitBarrierOp(OpBuilder &builder, Location loc,
                          BarrierAnnotationAttr annotation, ValueRange barriers,
                          ValueRange accumCnts, unsigned tileIdx) {
  unsigned numBuffers = annotation.getNumBuffers();
  StringRef kind = annotation.getBarrierOpKind().getValue();

  // For tile region annotations, compute bufferIdx from tileIdx.
  // For setup/teardown, use the static barrierIdx.
  unsigned barrierIdx;
  if (annotation.getRegion() == BarrierRegion::TILE) {
    barrierIdx = tileIdx % numBuffers;
  } else {
    barrierIdx = annotation.getBarrierIdx();
  }

  if (kind == "wait_barrier") {
    Value outerAccumCnt = accumCnts[barrierIdx];
    Value tileAccumCnt =
        computeTileAccumCnt(builder, loc, outerAccumCnt, tileIdx);
    Value phase = computePhase(builder, loc, tileAccumCnt, numBuffers);
    WaitBarrierOp::create(builder, loc, barriers[barrierIdx], phase);
  } else {
    assert(kind == "arrive_barrier");
    ArriveBarrierOp::create(builder, loc, barriers[barrierIdx],
                            annotation.getCount());
  }
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

/// Check if a tile annotation should fire for a given tile index.
/// Empty tileMask means fire on all tiles.
static bool isTileEnabled(BarrierAnnotationAttr annotation, unsigned tileIdx) {
  auto mask = annotation.getTileMask();
  if (!mask || mask.empty())
    return true;
  return tileIdx < static_cast<unsigned>(mask.size()) && mask[tileIdx] != 0;
}

void lowerSubtiledRegion(SubtiledRegionOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  ValueRange barriers = op.getBarriers();
  ValueRange accumCnts = op.getAccumCnts();

  // Pre-process barrier annotations by region and target op index.
  // Tile annotations are stored per (opIdx, placement) and filtered by
  // tileMask at each tile replication.
  llvm::DenseMap<unsigned, SmallVector<BarrierAnnotationAttr>> tileBefore,
      tileAfter;
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
      // TILE region — placement is BEFORE or AFTER, filtered by tileMask.
      if (annotation.getPlacement() == BarrierPlacement::BEFORE)
        tileBefore[opIdx].push_back(annotation);
      else
        tileAfter[opIdx].push_back(annotation);
    }
  }

  // 1. Clone setup region ops (except yield), emitting setup barriers.
  Block &setupBlock = op.getSetupRegion().front();
  IRMapping setupMapping;
  if (setupBefore.empty() && setupAfter.empty()) {
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

    for (auto [j, idx] : llvm::enumerate(indices.asArrayRef()))
      tileMapping.map(tileBlock.getArgument(j), setupOutputs[idx]);

    unsigned opIdx = 0;
    for (Operation &tileOp : tileBlock.without_terminator()) {
      // BEFORE: emit if tileMask allows this tile.
      {
        auto it = tileBefore.find(opIdx);
        if (it != tileBefore.end()) {
          for (auto &annotation : it->second) {
            if (isTileEnabled(annotation, tileIdx))
              emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                            tileIdx);
          }
        }
      }

      builder.clone(tileOp, tileMapping);

      // AFTER: emit if tileMask allows this tile.
      {
        auto it = tileAfter.find(opIdx);
        if (it != tileAfter.end()) {
          for (auto &annotation : it->second) {
            if (isTileEnabled(annotation, tileIdx))
              emitBarrierOp(builder, loc, annotation, barriers, accumCnts,
                            tileIdx);
          }
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
