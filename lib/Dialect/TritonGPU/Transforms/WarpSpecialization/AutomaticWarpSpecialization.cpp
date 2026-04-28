#include "PartitionAttrs.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/hopper/include/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <cstdlib>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir {
void removeRedundantTmemZeroStores(triton::FuncOp &funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp);
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

static bool useMetaNVWSAllocas() {
  const char *value = std::getenv("TRITON_NVWS_USE_META_NVWS_ALLOCAS");
  return value && StringRef(value) == "1";
}

} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  // The default and Meta-NVWS sub-pipelines, including each pass handoff, are
  // documented in sema-docs/nvws-aws-overview.md.
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
    metaPM.addPass(createNVGPUTestWSMemoryPlanner(memoryPlannerOptions));
    metaPM.addPass(createNVGPUTestAnnotateTMAStoreWaits());
    metaPM.addPass(std::make_unique<MetaValidateTMAStoreAnnotations>());
    if (failed(metaPM.run(getOperation())))
      return signalPassFailure();

    // Convert the completed physical plan. With NVWS allocas enabled this is
    // the second, idempotent conversion; otherwise it is the original one.
    addPassWithPartitionVerifier(createNVWSMetaToNVWSConvert());
    // InsertSemas emits co-located blocking waits in buffer-group discovery
    // order. Make that order an explicit latency policy rather than inheriting
    // the Meta memory planner's allocation order.
    addPassWithPartitionVerifier(createNVWSOrderBufferGroups());
  } else {
    addPassWithPartitionVerifier(createTritonGPUPartitionScheduling());
    addPassWithPartitionVerifier(createNVWSHoistTmemStore());
    addPassWithPartitionVerifier(createNVWSInsertAllocas());
  }

  NVWSInsertSemasOptions insertSemasOptions;
  insertSemasOptions.useMetaPartitioner = useMetaPartitioner;
  insertSemasOptions.numStages = numStages;
  addPassWithPartitionVerifier(createNVWSInsertSemas(insertSemasOptions));
  addPassWithPartitionVerifier(createNVWSLowerSemaphore({numStages}));
  TritonGPUScheduleLoopsOptions scheduleLoopsOptions;
  pm.addPass(createTritonGPUPartitionLoops());
  pm.addPass(createNVWSLowerWarpGroup());
  scheduleLoopsOptions.numStages = numStages;
  scheduleLoopsOptions.useMetaWS = useMetaPartitioner;
  pm.addPass(createTritonGPUScheduleLoops(scheduleLoopsOptions));
  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  // Multi-buffer TMA descriptors. We cannot rely on SWP to do it, to support
  // desc updates in nested loops.
  multiBufferTMADescriptors(getOperation(), numStages);
  clearInternalWarpSpecializationAttrs(getOperation());
}
