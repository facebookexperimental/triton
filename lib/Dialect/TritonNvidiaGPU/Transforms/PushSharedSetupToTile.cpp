#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPUSHSHAREDSETUPTOTILEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// For each SubtiledRegionOp whose setup region contains tmem_subslice ops,
/// extract the per-tile N offsets as i32 constants, yield them from setup,
/// and add per-tile mapped args to the tile body.  This makes the subtile
/// offset explicitly available in the tile body for address computations.
void addSubsliceRangeToSetup(SubtiledRegionOp op) {
  ArrayAttr tileMappings = op.getTileMappings();
  unsigned numTiles = tileMappings.size();
  if (numTiles <= 1)
    return;

  Block &setupBlock = op.getSetupRegion().front();
  auto setupYield = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());

  // Collect tmem_subslice ops in the setup, grouped by source.
  // We expect exactly numTiles subslice ops from the same source.
  SmallVector<TMEMSubSliceOp> subsliceOps;
  for (Operation &op : setupBlock.without_terminator()) {
    if (auto subslice = dyn_cast<TMEMSubSliceOp>(&op))
      subsliceOps.push_back(subslice);
  }

  if (subsliceOps.size() != numTiles)
    return;

  // Verify they all share the same source.
  Value commonSrc = subsliceOps[0].getSrc();
  for (auto subslice : subsliceOps) {
    if (subslice.getSrc() != commonSrc)
      return;
  }

  // Extract per-tile N offsets and create constants in setup.
  OpBuilder setupBuilder(setupYield);
  Location loc = op.getLoc();
  SmallVector<Value> offsetConstants;
  for (auto subslice : subsliceOps) {
    int32_t nOffset = subslice.getN();
    Value c = arith::ConstantOp::create(
        setupBuilder, loc, setupBuilder.getI32IntegerAttr(nOffset));
    offsetConstants.push_back(c);
  }

  // Add offset constants to the setup yield.
  SmallVector<Value> newYieldValues(setupYield.getResults());
  unsigned rangeYieldBase = newYieldValues.size();
  for (Value c : offsetConstants)
    newYieldValues.push_back(c);

  SubtiledRegionYieldOp::create(setupBuilder, setupYield.getLoc(),
                                newYieldValues);
  setupYield.erase();

  // Add a new tile arg (i32) and extend tile mappings.
  Block &tileBlock = op.getTileRegion().front();
  bool hasTileIndex =
      (tileBlock.getNumArguments() >
       cast<DenseI32ArrayAttr>(tileMappings[0]).asArrayRef().size());

  // Insert the new arg before the tile index arg (if present), otherwise
  // append.
  unsigned insertPos = hasTileIndex ? tileBlock.getNumArguments() - 1
                                    : tileBlock.getNumArguments();
  tileBlock.insertArgument(insertPos, setupBuilder.getI32Type(), loc);

  // Extend tile mappings with the per-tile offset yield index.
  MLIRContext *ctx = op.getContext();
  SmallVector<Attribute> newMappingAttrs;
  for (unsigned t = 0; t < numTiles; ++t) {
    SmallVector<int32_t> indices(
        cast<DenseI32ArrayAttr>(tileMappings[t]).asArrayRef());
    indices.push_back(static_cast<int32_t>(rangeYieldBase + t));
    newMappingAttrs.push_back(DenseI32ArrayAttr::get(ctx, indices));
  }
  op.setTileMappingsAttr(ArrayAttr::get(ctx, newMappingAttrs));
}

void pushSharedSetupToTile(SubtiledRegionOp op) {
  ArrayAttr tileMappings = op.getTileMappings();
  unsigned numTiles = tileMappings.size();
  if (numTiles <= 1)
    return;

  Block &tileBlock = op.getTileRegion().front();
  unsigned numArgs = tileBlock.getNumArguments();
  Block &setupBlock = op.getSetupRegion().front();
  auto setupYield = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());

  // Detect optional tile index argument (last arg, not in tileMappings).
  unsigned mappingSize =
      cast<DenseI32ArrayAttr>(tileMappings[0]).asArrayRef().size();
  unsigned numMappedArgs = mappingSize;

  // Step 1: Find shared arg positions — all tiles map to the same yield index.
  // Only scan mapped args (skip trailing tile index arg if present).
  struct SharedArg {
    unsigned argPosition;
    unsigned yieldIndex;
  };
  SmallVector<SharedArg> sharedArgs;

  for (unsigned p = 0; p < numMappedArgs; ++p) {
    int32_t yIdx = cast<DenseI32ArrayAttr>(tileMappings[0]).asArrayRef()[p];
    bool isShared = true;
    for (unsigned t = 1; t < numTiles; ++t) {
      if (cast<DenseI32ArrayAttr>(tileMappings[t]).asArrayRef()[p] != yIdx) {
        isShared = false;
        break;
      }
    }
    if (isShared)
      sharedArgs.push_back({p, static_cast<unsigned>(yIdx)});
  }

  if (sharedArgs.empty())
    return;

  // Step 2: Determine which shared args are movable.
  // A shared value is movable if it and all its setup-internal dependencies
  // are defined outside the SubtiledRegionOp or only depend on values from
  // outside.
  DenseSet<Operation *> opsToMove;
  SmallVector<SharedArg> movableArgs;

  for (auto &sa : sharedArgs) {
    Value yieldVal = setupYield.getResults()[sa.yieldIndex];
    Operation *defOp = yieldVal.getDefiningOp();

    if (!defOp || defOp->getBlock() != &setupBlock) {
      // Defined outside setup — directly usable in tile body.
      movableArgs.push_back(sa);
      continue;
    }

    // Backward slice within setup to find all internal dependencies.
    SmallVector<Operation *> slice;
    SmallVector<Operation *> worklist = {defOp};
    DenseSet<Operation *> visited;
    bool allMovable = true;

    while (!worklist.empty() && allMovable) {
      Operation *curr = worklist.pop_back_val();
      if (!visited.insert(curr).second)
        continue;
      slice.push_back(curr);

      for (Value operand : curr->getOperands()) {
        Operation *operandDef = operand.getDefiningOp();
        if (!operandDef || operandDef->getBlock() != &setupBlock)
          continue;
        worklist.push_back(operandDef);
      }
    }

    if (allMovable) {
      movableArgs.push_back(sa);
      for (auto *o : slice)
        opsToMove.insert(o);
    }
  }

  if (movableArgs.empty())
    return;

  // Step 3: Clone ops into the tile body, sinking each shared arg's
  // dependency chain to right before its first use. This keeps tmem_load
  // close to its consumer rather than hoisting it above barrier waits.

  // Sort ops in program order for correct cloning.
  SmallVector<Operation *> sortedOps(opsToMove.begin(), opsToMove.end());
  llvm::sort(sortedOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  // Record original tile ops so we can distinguish them from cloned ops
  // when adjusting barrier annotations.
  DenseSet<Operation *> originalTileOps;
  for (Operation &tileOp : tileBlock.without_terminator())
    originalTileOps.insert(&tileOp);

  // For each movable arg, find the earliest op in the tile body that uses
  // it. This is where we will sink the shared dependency chain.
  Operation *earliestUser = nullptr;
  for (auto &sa : movableArgs) {
    BlockArgument arg = tileBlock.getArgument(sa.argPosition);
    for (Operation *user : arg.getUsers()) {
      if (!earliestUser || user->isBeforeInBlock(earliestUser))
        earliestUser = user;
    }
  }

  // Clone the dependency chain right before the earliest consumer.
  IRMapping cloneMapping;
  if (earliestUser) {
    OpBuilder tileBuilder(&tileBlock, Block::iterator(earliestUser));
    for (Operation *o : sortedOps)
      tileBuilder.clone(*o, cloneMapping);
  }

  // Replace tile block args with cloned values (or external values).
  for (auto &sa : movableArgs) {
    Value yieldVal = setupYield.getResults()[sa.yieldIndex];
    Value replacement = cloneMapping.lookupOrDefault(yieldVal);
    tileBlock.getArgument(sa.argPosition).replaceAllUsesWith(replacement);
  }

  // Step 4: Remove shared args from tile block and rebuild tileMappings/yield.
  DenseSet<unsigned> sharedPositions;
  for (auto &sa : movableArgs)
    sharedPositions.insert(sa.argPosition);

  // Determine which yield indices are still needed by non-shared args.
  DenseSet<unsigned> usedYieldIndices;
  SmallVector<SmallVector<int32_t>> newMappingsRaw(numTiles);

  for (unsigned p = 0; p < numMappedArgs; ++p) {
    if (sharedPositions.contains(p))
      continue;
    for (unsigned t = 0; t < numTiles; ++t) {
      int32_t yIdx = cast<DenseI32ArrayAttr>(tileMappings[t]).asArrayRef()[p];
      newMappingsRaw[t].push_back(yIdx);
      usedYieldIndices.insert(yIdx);
    }
  }

  // Build compacted yield and index remapping.
  unsigned numYieldValues = setupYield.getResults().size();
  SmallVector<int32_t> oldToNew(numYieldValues, -1);
  SmallVector<Value> newYieldValues;
  int32_t newIdx = 0;
  for (unsigned i = 0; i < numYieldValues; ++i) {
    if (usedYieldIndices.contains(i)) {
      oldToNew[i] = newIdx++;
      newYieldValues.push_back(setupYield.getResults()[i]);
    }
  }

  // Remap indices in new mappings.
  for (auto &mapping : newMappingsRaw) {
    for (auto &idx : mapping)
      idx = oldToNew[idx];
  }

  // Erase shared block args (reverse order to preserve indices).
  SmallVector<unsigned> toRemove(sharedPositions.begin(),
                                 sharedPositions.end());
  llvm::sort(toRemove, std::greater<unsigned>());
  for (unsigned p : toRemove)
    tileBlock.eraseArgument(p);

  // Update tileMappings attribute.
  MLIRContext *ctx = op.getContext();
  SmallVector<Attribute> mappingAttrs;
  for (auto &mapping : newMappingsRaw)
    mappingAttrs.push_back(DenseI32ArrayAttr::get(ctx, mapping));
  op.setTileMappingsAttr(ArrayAttr::get(ctx, mappingAttrs));

  // Rebuild setup yield with only used values.
  OpBuilder setupBuilder(setupYield);
  SubtiledRegionYieldOp::create(setupBuilder, setupYield.getLoc(),
                                newYieldValues);
  setupYield.erase();

  // Step 5: Update barrier annotations — for each TILE annotation, count
  // how many cloned ops were inserted before its target op.
  unsigned numCloned = sortedOps.size();
  if (numCloned > 0) {
    SmallVector<unsigned> clonedBeforeOrigOp;
    unsigned clonedSoFar = 0;
    for (Operation &tileOp : tileBlock.without_terminator()) {
      if (!originalTileOps.contains(&tileOp))
        clonedSoFar++;
      else
        clonedBeforeOrigOp.push_back(clonedSoFar);
    }

    SmallVector<Attribute> newAnnotations;
    for (Attribute attr : op.getBarrierAnnotations()) {
      auto annotation = cast<BarrierAnnotationAttr>(attr);
      if (annotation.getRegion() == BarrierRegion::TILE) {
        unsigned origTarget = annotation.getTargetOpIdx();
        unsigned offset = origTarget < clonedBeforeOrigOp.size()
                              ? clonedBeforeOrigOp[origTarget]
                              : clonedSoFar;
        auto newAnnotation = BarrierAnnotationAttr::get(
            ctx, annotation.getBarrierIdx(), annotation.getPlacement(),
            origTarget + offset, annotation.getBarrierOpKind(),
            annotation.getCount(), annotation.getRegion(),
            annotation.getNumBuffers(), annotation.getTileMask());
        newAnnotations.push_back(newAnnotation);
      } else {
        newAnnotations.push_back(attr);
      }
    }
    op.setBarrierAnnotationsAttr(ArrayAttr::get(ctx, newAnnotations));
  }
}

class TritonNvidiaGPUPushSharedSetupToTilePass
    : public impl::TritonNvidiaGPUPushSharedSetupToTilePassBase<
          TritonNvidiaGPUPushSharedSetupToTilePass> {
public:
  using TritonNvidiaGPUPushSharedSetupToTilePassBase::
      TritonNvidiaGPUPushSharedSetupToTilePassBase;

  void runOnOperation() override {
    SmallVector<SubtiledRegionOp> ops;
    getOperation().walk([&](SubtiledRegionOp op) { ops.push_back(op); });

    for (auto op : ops)
      addSubsliceRangeToSetup(op);
    for (auto op : ops)
      pushSharedSetupToTile(op);
  }
};

} // namespace
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
