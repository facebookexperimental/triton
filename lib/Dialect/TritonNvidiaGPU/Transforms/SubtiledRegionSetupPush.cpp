#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-nvidia-gpu-subtiled-region-setup-push"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUSUBTILEDREGIONSETUPPUSHPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

/// Analyze whether the setup region has a pattern where each tile's yield
/// values come from independent TMEMLoadOp chains that can be pushed into the
/// tile body.
///
/// The expected pattern (after TMemSplitLoadPattern):
///   %sub0 = tmem_subslice %base -> memdesc<128x32xf32>
///   %load0, %tok0 = tmem_load %sub0[%input_token]
///   %sub1 = tmem_subslice %base -> memdesc<128x32xf32>
///   %load1, %tok1 = tmem_load %sub1[%input_token]
///   ...
///   yield %load0, %load1, ... (other args per tile)
///
/// After setup push:
///   yield %sub0, %sub1, ... (memdesc + other captured args per tile)
///
/// Tile body gains tmem_load at the start:
///   ^bb0(%memdesc: ..., ...other_args...):
///     %load, %tok = tmem_load %memdesc[%captured_token]
///     ... original tile body ops ...
struct SetupPushCandidate {
  /// For each tile, the TMEMLoadOp in the setup region.
  SmallVector<TMEMLoadOp> perTileLoads;
  /// The index into the tile block args where the load result appears.
  unsigned loadArgIdx;
  /// The token input to the tmem_load (same for all tiles).
  Value sharedToken;
};

/// Check if a setup region yield value is produced by a TMEMLoadOp whose
/// source is a TMEMSubSliceOp.
static TMEMLoadOp getSubslicedLoad(Value yieldVal) {
  auto loadOp = yieldVal.getDefiningOp<TMEMLoadOp>();
  if (!loadOp)
    return nullptr;
  auto subslice = loadOp.getSrc().getDefiningOp<TMEMSubSliceOp>();
  if (!subslice)
    return nullptr;
  return loadOp;
}

/// Analyze the setup region of a SubtiledRegionOp to find pushable loads.
static std::optional<SetupPushCandidate>
analyzeSetupForPush(SubtiledRegionOp op) {
  Block &setupBlock = op.getSetupRegion().front();
  auto yieldOp = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());
  Block &tileBlock = op.getTileRegion().front();

  unsigned numTileArgs = tileBlock.getNumArguments();
  unsigned numYields = yieldOp.getResults().size();
  if (numYields == 0 || numTileArgs == 0)
    return std::nullopt;
  unsigned numTiles = numYields / numTileArgs;
  if (numTiles < 2)
    return std::nullopt;

  // For each tile arg position, check if it's a TMEMLoadOp from a subslice.
  for (unsigned argIdx = 0; argIdx < numTileArgs; ++argIdx) {
    SmallVector<TMEMLoadOp> loads;
    Value sharedToken;
    bool allMatch = true;

    for (unsigned t = 0; t < numTiles; ++t) {
      Value yieldVal = yieldOp.getResults()[t * numTileArgs + argIdx];
      auto loadOp = getSubslicedLoad(yieldVal);
      if (!loadOp) {
        allMatch = false;
        break;
      }

      // Check that the load's result has exactly one use (the yield).
      // The token result may have other uses.
      if (!loadOp.getResult().hasOneUse()) {
        allMatch = false;
        break;
      }

      // Verify all loads use the same token input.
      Value token = loadOp.getToken();
      if (t == 0) {
        sharedToken = token;
      } else if (token != sharedToken) {
        allMatch = false;
        break;
      }

      loads.push_back(loadOp);
    }

    if (!allMatch || loads.empty())
      continue;

    // Verify all loads produce the same result type.
    Type resultType = loads[0].getResult().getType();
    bool typesMatch = true;
    for (unsigned t = 1; t < numTiles; ++t) {
      if (loads[t].getResult().getType() != resultType) {
        typesMatch = false;
        break;
      }
    }
    if (!typesMatch)
      continue;

    LDBG("Found pushable TMEMLoad at tile arg index " << argIdx);
    SetupPushCandidate candidate;
    candidate.perTileLoads = std::move(loads);
    candidate.loadArgIdx = argIdx;
    candidate.sharedToken = sharedToken;
    return candidate;
  }

  return std::nullopt;
}

/// Push the TMEMLoadOp from the setup region into the tile body.
static void pushSetupLoads(SubtiledRegionOp op, SetupPushCandidate &candidate) {
  Block &setupBlock = op.getSetupRegion().front();
  auto yieldOp = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());
  Block &tileBlock = op.getTileRegion().front();

  unsigned numTileArgs = tileBlock.getNumArguments();
  unsigned numYields = yieldOp.getResults().size();
  unsigned numTiles = numYields / numTileArgs;
  unsigned loadArgIdx = candidate.loadArgIdx;

  // Step 1: Change the setup yields at loadArgIdx from loaded tensor values
  // to subsliced memdesc values.
  SmallVector<Value> newYields;
  for (unsigned t = 0; t < numTiles; ++t) {
    for (unsigned a = 0; a < numTileArgs; ++a) {
      unsigned yieldIdx = t * numTileArgs + a;
      if (a == loadArgIdx) {
        // Replace loaded tensor value with the subslice memdesc.
        auto loadOp = candidate.perTileLoads[t];
        auto subslice = loadOp.getSrc().getDefiningOp<TMEMSubSliceOp>();
        newYields.push_back(subslice->getResult(0));
      } else {
        newYields.push_back(yieldOp.getResults()[yieldIdx]);
      }
    }
  }

  // Step 2: Update the setup yield.
  OpBuilder setupBuilder(yieldOp);
  auto newYieldOp =
      setupBuilder.create<SubtiledRegionYieldOp>(yieldOp.getLoc(), newYields);
  // Copy attributes from the old yield.
  for (auto attr : yieldOp->getAttrs())
    newYieldOp->setAttr(attr.getName(), attr.getValue());
  yieldOp.erase();

  // Step 3: Change the tile block arg type at loadArgIdx from tensor to
  // memdesc.
  Value memdescArg = tileBlock.getArgument(loadArgIdx);
  Type newArgType = candidate.perTileLoads[0].getSrc().getType();
  memdescArg.setType(newArgType);

  // Step 4: Insert tmem_load at the beginning of the tile body.
  OpBuilder tileBuilder(&tileBlock, tileBlock.begin());
  Location loc = candidate.perTileLoads[0].getLoc();

  // Create the tmem_load in the tile body.
  auto repLoad = candidate.perTileLoads[0];
  SmallVector<Type> loadResultTypes;
  for (Type t : repLoad->getResultTypes())
    loadResultTypes.push_back(t);

  // Build the tmem_load using the memdesc block arg and the shared token.
  auto newLoad = tileBuilder.create<TMEMLoadOp>(
      loc, repLoad.getResult().getType(), repLoad.getToken().getType(),
      memdescArg, candidate.sharedToken);

  // Copy attributes from the original load.
  for (auto attr : repLoad->getAttrs())
    newLoad->setAttr(attr.getName(), attr.getValue());

  // Replace all uses of the old tile block arg (which used to be the loaded
  // tensor) with the new load result. But skip the load itself (its operand
  // is the memdesc block arg, not the old tensor type).
  Value newLoadResult = newLoad.getResult();
  memdescArg.replaceAllUsesExcept(memdescArg, newLoad);
  // Actually, we need to replace uses of memdescArg that expect the OLD type
  // (tensor type) with the new load result. The memdescArg now has memdesc
  // type and is consumed by newLoad. All other uses should use newLoadResult.
  SmallVector<OpOperand *> usesToReplace;
  for (auto &use : memdescArg.getUses()) {
    if (use.getOwner() != newLoad)
      usesToReplace.push_back(&use);
  }
  for (auto *use : usesToReplace)
    use->set(newLoadResult);

  // Step 5: Erase the original TMEMLoadOps from the setup region.
  // Their results are no longer yielded. The subslice ops remain.
  for (auto loadOp : candidate.perTileLoads) {
    if (loadOp.getResult().use_empty() && loadOp.getToken().use_empty())
      loadOp.erase();
  }
}

class TritonNvidiaGPUSubtiledRegionSetupPushPass
    : public impl::TritonNvidiaGPUSubtiledRegionSetupPushPassBase<
          TritonNvidiaGPUSubtiledRegionSetupPushPass> {
public:
  using TritonNvidiaGPUSubtiledRegionSetupPushPassBase::
      TritonNvidiaGPUSubtiledRegionSetupPushPassBase;

  void runOnOperation() override {
    SmallVector<SubtiledRegionOp> ops;
    getOperation().walk([&](SubtiledRegionOp op) { ops.push_back(op); });

    for (auto op : ops) {
      auto candidate = analyzeSetupForPush(op);
      if (!candidate)
        continue;

      LDBG("Pushing setup loads for SubtiledRegionOp at " << op.getLoc());
      pushSetupLoads(op, *candidate);
    }
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
