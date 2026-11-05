#include "PartitionAttrs.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/hopper/include/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <cstdlib>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace tlx = mlir::triton::tlx;

namespace mlir {
void removeRedundantTmemZeroStores(triton::FuncOp funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp funcOp);
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
static bool hasWarpSpecializeLoop(FuncOp func) {
  bool found = false;
  func.walk([&](scf::ForOp loop) {
    if (!loop->hasAttr(kWarpSpecializeAttrName))
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

// These two canonical Meta prefix steps expose function implementations but
// not standalone pass factories. Wrap them here so AutomaticWarpSpecialization
// still owns an explicit pass pipeline and MetaToNVWSConvert remains a pure
// representation conversion.
struct MetaRemoveRedundantTmemZeroStores
    : PassWrapper<MetaRemoveRedundantTmemZeroStores,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MetaRemoveRedundantTmemZeroStores)

  void runOnOperation() override {
    getOperation().walk([&](FuncOp func) {
      if (hasWarpSpecializeLoop(func))
        removeRedundantTmemZeroStores(func);
    });
  }
};

struct MetaValidateTMAStoreAnnotations
    : PassWrapper<MetaValidateTMAStoreAnnotations, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MetaValidateTMAStoreAnnotations)

  void runOnOperation() override {
    getOperation().walk([&](FuncOp func) {
      if (hasWarpSpecializeLoop(func))
        doValidateTMAStoreAnnotations(func);
    });
  }
};

struct VerifyWarpSpecializationPartitions
    : PassWrapper<VerifyWarpSpecializationPartitions, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyWarpSpecializationPartitions)

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](LoopLikeOpInterface loop) {
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
  LogicalResult runSemaphorePipeline();
};

void multiBufferTMADescriptors(ModuleOp mod, int numStages) {
  SetVector<LoopLikeOpInterface> descUpdateLoops;
  mod.walk([&](LoopLikeOpInterface loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      loop.walk([&](triton::MakeTensorDescOp op) {
        if (auto loopOp = op->getParentOfType<LoopLikeOpInterface>()) {
          descUpdateLoops.insert(loopOp);
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

static bool useMetaNVWSAllocas() {
  const char *value = std::getenv("TRITON_NVWS_USE_META_NVWS_ALLOCAS");
  return value && StringRef(value) == "1";
}

} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  // META_WS_CHANGE: Preserve Meta's manual-WS bailout.
  ModuleOp m = getOperation();
  if (shouldBail(m))
    return;
  if (useSemaphores || useMetaPartitioner) {
    if (failed(runSemaphorePipeline()))
      signalPassFailure();
    return;
  }

  OpPassManager pm;
  auto addPassWithPartitionVerifier = [&](std::unique_ptr<Pass> pass) {
    pm.addPass(std::move(pass));
    pm.addPass(createVerifyWarpSpecializationPartitionsPass());
  };

  pm.addPass(createTritonGPUNormalizeWSWhileLoops());
  addPassWithPartitionVerifier(createTritonGPUPartitionScheduling());
  addPassWithPartitionVerifier(createNVWSHoistTmemStore());
  addPassWithPartitionVerifier(createNVWSInsertAref());
  addPassWithPartitionVerifier(createNVWSInsertTmemAref());
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

LogicalResult AutomaticWarpSpecialization::runSemaphorePipeline() {
  bool enabled = false;
  getOperation().walk([&](FuncOp func) {
    enabled |= hasWarpSpecializeLoop(func);
  });
  if (!enabled)
    return success();

  OpPassManager pm;
  auto addPassWithPartitionVerifier = [&](std::unique_ptr<Pass> pass) {
    pm.addPass(std::move(pass));
    pm.addPass(createVerifyWarpSpecializationPartitionsPass());
  };

  if (useMetaPartitioner) {
    // Canonical Meta temporarily leaves verifier-incomplete ownership metadata
    // between PartitionSchedulingMeta and task-id propagation. Run the actual
    // Meta pass objects as one atomic planning prefix, convert its completed
    // result, and expose only verifier-valid NVWS IR to the mechanism suffix.
    PassManager metaPM(getOperation().getContext());
    metaPM.enableVerifier(false);
    metaPM.addPass(createNVGPUPartitionSchedulingMeta());

    NVGPUTestWSTaskIdPropagateOptions taskIdOptions;
    taskIdOptions.numWarpGroups = 2;
    metaPM.addPass(createNVGPUTestWSTaskIdPropagate(taskIdOptions));
    metaPM.addPass(std::make_unique<MetaRemoveRedundantTmemZeroStores>());

    if (useMetaNVWSAllocas()) {
      // Change only allocation materialization: convert Meta task ownership to
      // NVWS, then use the existing NVWS InsertAllocas implementation.
      metaPM.addPass(createNVWSMetaToNVWSConvert());
      metaPM.addPass(createNVWSInsertAllocas());
    } else {
      // Original Meta allocation materialization.
      // This existing pass converts ordinary descriptor loads before
      // allocating their buffers. Meta gather conversion is separate work.
      metaPM.addPass(createNVGPUTestWSBufferAllocation());
    }

    // Everything after allocation materialization is identical for both paths.
    metaPM.addPass(createNVGPUTestWSHoistTMEMStore());
    // Pack each data-partitioned merged-epilogue producer/store slice before
    // planning so MemoryPlanner sees the shortened register live ranges.
    metaPM.addPass(createNVWSPackEpilogueSlices());

    NVGPUTestWSMemoryPlannerOptions memoryPlannerOptions;
    memoryPlannerOptions.numBuffers = numStages;
    memoryPlannerOptions.smemAllocAlgo = 1;
    memoryPlannerOptions.smemBudget = std::max<int32_t>(smemBudget, 0);
    memoryPlannerOptions.smemCircularReuse = false;
    // The Meta-NVWS suffix materializes its own synchronization storage, which
    // the final shared-memory allocator accounts exactly. Do not also apply
    // Meta code partition's conservative auxiliary reservation here: it can
    // force a different data-buffer plan from an estimate that does not model
    // the NVWS suffix, including unsupported allocation.reuseTarget relations.
    memoryPlannerOptions.reserveAuxiliarySmem = false;
    metaPM.addPass(createNVGPUTestWSMemoryPlanner(memoryPlannerOptions));
    metaPM.addPass(createNVGPUTestAnnotateTMAStoreWaits());
    metaPM.addPass(std::make_unique<MetaValidateTMAStoreAnnotations>());
    if (failed(metaPM.run(getOperation())))
      return failure();

    // Convert the completed physical plan. The optional NVWS allocation
    // path has run memory planning since its earlier ownership conversion.
    addPassWithPartitionVerifier(createNVWSMetaToNVWSConvert());
    // InsertSemas emits co-located blocking waits in buffer-group discovery
    // order. Make that order an explicit latency policy rather than inheriting
    // the Meta memory planner's allocation order.
    addPassWithPartitionVerifier(createNVWSOrderBufferGroups());
  } else {
    TritonGPUPartitionSchedulingOptions schedulingOptions;
    schedulingOptions.useSemaphores = true;
    addPassWithPartitionVerifier(
        createTritonGPUPartitionScheduling(schedulingOptions));
    NVWSHoistTmemStoreOptions hoistOptions;
    hoistOptions.useSemaphores = true;
    addPassWithPartitionVerifier(createNVWSHoistTmemStore(hoistOptions));
    addPassWithPartitionVerifier(createNVWSInsertAllocas());
  }

  NVWSInsertSemasOptions insertSemasOptions;
  insertSemasOptions.useMetaPartitioner = useMetaPartitioner;
  insertSemasOptions.numStages = numStages;
  addPassWithPartitionVerifier(createNVWSInsertSemas(insertSemasOptions));
  NVWSSemaphoreOptimizeOptions semaphoreOptimizeOptions;
  semaphoreOptimizeOptions.numStages = numStages;
  addPassWithPartitionVerifier(
      createNVWSSemaphoreOptimize(semaphoreOptimizeOptions));
  addPassWithPartitionVerifier(createNVWSAssignSemaphoreStagePhase());
  addPassWithPartitionVerifier(createNVWSLowerSemaphore({numStages}));
  TritonGPUScheduleLoopsOptions scheduleLoopsOptions;
  TritonGPUPartitionLoopsOptions partitionOptions;
  partitionOptions.useSemaphores = true;
  pm.addPass(createTritonGPUPartitionLoops(partitionOptions));
  pm.addPass(createNVWSLowerWarpGroup());
  scheduleLoopsOptions.numStages = numStages;
  scheduleLoopsOptions.useMetaWS = useMetaPartitioner;
  pm.addPass(createTritonGPUScheduleLoops(scheduleLoopsOptions));
  if (failed(runPipeline(pm, getOperation())))
    return failure();

  // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
  // desc updates in nested loops.
  multiBufferTMADescriptors(getOperation(), numStages);
  clearInternalWarpSpecializationAttrs(getOperation());
  return success();
}
