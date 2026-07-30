#include "PartitionAttrs.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace tlx = mlir::triton::tlx;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyWarpSpecializationPartitions)

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kPartitionStagesAttrName))
        return WalkResult::advance();
      if (failed(verifyPartitionedLoop(loop))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

struct AutomaticWarpSpecialization
    : triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  using TritonGPUAutomaticWarpSpecializationBase::
      TritonGPUAutomaticWarpSpecializationBase;

  // META_WS_CHANGE: Do not run the upstream autoWS pipeline on modules that
  // already contain manually warp-specialized TLX operations.
  bool shouldBail(ModuleOp &mod) const {
    auto hasManualWarpSpec =
        mod->getAttrOfType<BoolAttr>(tlx::AttrHasWarpSpecOpsName);
    return hasManualWarpSpec != nullptr && hasManualWarpSpec.getValue() == true;
  }

  void runOnOperation() override;
};

void multiBufferTMADescriptors(ModuleOp mod, int numStages) {
  SetVector<scf::ForOp> descUpdateLoops;
  mod.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      loop.walk([&](triton::MakeTensorDescOp op) {
        if (auto forOp = op->getParentOfType<scf::ForOp>()) {
          descUpdateLoops.insert(forOp);
        }
      });
    }
  });

  // +1 to make sure that overlapping of the next desc update and the oldest
  // inflight TMA load is safe
  const int numDescs = numStages + 1;
  // CoarseSchedule's notion of numStages is the maximuim loop-pipelining
  // stage + 1, see CoarseSchedule::deSerialize(). So if we want n buffers,
  // we need to pass n + 1 as numStages.
  triton::CoarseSchedule schedule(numDescs + 1);

  for (auto loop : descUpdateLoops) {
    triton::lowerTMADescriptors(loop, schedule);
  }
}

void clearInternalWarpSpecializationAttrs(ModuleOp mod) {
  mod.walk([](Operation *op) {
    op->removeAttr(kPartitionAttrName);
    op->removeAttr(kPartitionOutputsAttrName);
    op->removeAttr(kPartitionStagesAttrName);
    op->removeAttr(kWarpSpecializeTagAttrName);
  });
}

std::unique_ptr<Pass> createVerifyWarpSpecializationPartitionsPass() {
  return std::make_unique<VerifyWarpSpecializationPartitions>();
}

} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  // META_WS_CHANGE: Preserve Meta's manual-WS bailout.
  ModuleOp m = getOperation();
  if (shouldBail(m))
    return;
  OpPassManager pm;
  auto addPassWithPartitionVerifier = [&](std::unique_ptr<Pass> pass) {
    pm.addPass(std::move(pass));
    pm.addPass(createVerifyWarpSpecializationPartitionsPass());
  };

  addPassWithPartitionVerifier(createTritonGPUPartitionScheduling());
  addPassWithPartitionVerifier(createNVWSHoistTmemStore());
  addPassWithPartitionVerifier(createNVWSInsertAref());
  // META_WS_CHANGE: InsertTmemAref fails with Meta's partition layout
  // (getInitialSchedule + schedulePostLoopOps). Keep disabled until partition
  // scheduling is aligned with upstream. LoadMMASpecialization is retained
  // locally as the fallback.
#if 0
  addPassWithPartitionVerifier(createNVWSInsertTmemAref());
#else
  addPassWithPartitionVerifier(
      createTritonGPULoadMMASpecialization({numStages}));
#endif
  // `int-range-optimizations` and SCCP are good at cleaning up loop arithmetic.
  // FIXME: Re-enable integer range analysis once it is fixed.
  // pm.addPass(arith::createIntRangeOptimizationsPass());
  addPassWithPartitionVerifier(createSCCPPass());
  addPassWithPartitionVerifier(createCSEPass());
  addPassWithPartitionVerifier(createNVWSLowerAref({numStages}));
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createNVWSLowerWarpGroup());
  pm.addPass(createTritonGPUScheduleLoops());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  // META_WS_CHANGE: Preserve Meta's post-WS cleanup and partition-warp
  // optimization pipeline, which is not present in upstream.
  // Cleanup code generated by warp specialization.
  runDeadIterArgElimination(getOperation());
  RewritePatternSet patterns(&getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
  WarpSpecializeOp::getCanonicalizationPatterns(patterns, &getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  pm.clear();
  pm.addPass(createTritonGPUOptimizePartitionWarps());
  pm.addPass(createTritonGPUScheduleLoops());
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
  // desc updates in nested loops.
  multiBufferTMADescriptors(getOperation(), numStages);
  clearInternalWarpSpecializationAttrs(getOperation());
}
