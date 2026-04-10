// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass B: Schedule Integration (Warp Specialization Reconstruction)
//
// Configures IR attributes so downstream passes (schedule_loops,
// warp_specialize, pipeline) use the modulo schedule from Pass A.
//
// Buffer depths (tt.num_buffers) are already computed by Pass A (Steps 3-4.5).
// This pass reads them and sets tt.num_stages, tt.scheduled_max_stage,
// and strips tt.latency attrs for WS loops.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-modulo-ws-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

static void processScheduledLoop(scf::ForOp loop) {
  auto ctx = loop.getContext();
  bool isWS = loop->hasAttr(tt::kWarpSpecializeAttrName);

  // Read num_stages if already set by Pass A Step 3 (computeBufferDepths).
  int numStages = 0;
  if (auto ns = loop->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName))
    numStages = ns.getInt();

  if (isWS || loop->hasAttr("tt.modulo_ii")) {
    // WS loops or modulo-scheduled loops: keep loop.stage/loop.cluster attrs.
    // For modulo-scheduled non-WS loops, the schedule must survive to
    // downstream ScheduleLoops (which skips them via tt.modulo_ii check).
    int maxStage = 0;
    for (auto &op : loop.getBody()->without_terminator()) {
      if (auto stageAttr =
              op.getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName))
        maxStage = std::max(maxStage, (int)stageAttr.getInt());
    }

    // Derive num_stages from the schedule when Pass A Step 3 found no
    // LocalAllocOp (e.g. outer tile loops of persistent kernels where
    // SMEM buffers are allocated outside the loop).
    if (numStages == 0 && maxStage > 0) {
      numStages = maxStage + 1;
      LDBG("Derived num_stages=" << numStages
                                 << " from maxStage=" << maxStage);
    }

    if (numStages > 0) {
      loop->setAttr(tt::kNumStagesAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), numStages));
      // scheduled_max_stage reflects the actual schedule, not buffer depth.
      loop->setAttr(tt::kScheduledMaxStageAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));
      LDBG("Set num_stages=" << numStages
                             << " scheduled_max_stage=" << maxStage);
    }
    LDBG("Modulo/WS loop: kept loop.stage/loop.cluster");
  } else {
    for (auto &op : loop.getBody()->without_terminator()) {
      op.removeAttr(tt::kLoopStageAttrName);
      op.removeAttr(tt::kLoopClusterAttrName);
      op.walk([](Operation *nestedOp) {
        nestedOp->removeAttr(triton::kLoopStageAttrName);
        nestedOp->removeAttr(triton::kLoopClusterAttrName);
      });
    }
    LDBG("Non-WS loop: stripped loop.stage/loop.cluster");
  }
  // Keep tt.modulo_ii on the loop so downstream ScheduleLoops (inside AutoWS)
  // knows to skip re-scheduling this loop and its partition clones.
}

struct ModuloWSPartitionPass
    : public PassWrapper<ModuloWSPartitionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloWSPartitionPass)

  StringRef getArgument() const override {
    return "nvgpu-modulo-ws-partition";
  }

  StringRef getDescription() const override {
    return "Schedule integration for warp specialization (Pass B)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp.walk([&](scf::ForOp loop) {
      bool hasMMAv5 = false;
      bool hasTMALoad = false;
      bool hasSchedule = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
          hasMMAv5 = true;
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp,
                ttng::AsyncTMACopyGlobalToLocalOp>(op))
          hasTMALoad = true;
        if (op->hasAttr(tt::kLoopStageAttrName))
          hasSchedule = true;
        if (hasSchedule && (hasMMAv5 || hasTMALoad))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (!hasSchedule || (!hasMMAv5 && !hasTMALoad))
        return;

      processScheduledLoop(loop);
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloWSPartition() {
  return std::make_unique<ModuloWSPartitionPass>();
}

void registerNVGPUModuloWSPartition() {
  PassRegistration<ModuloWSPartitionPass>();
}
} // namespace mlir
