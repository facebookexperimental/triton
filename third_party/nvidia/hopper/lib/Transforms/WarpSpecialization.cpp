#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

// Helper to get OpPrintingFlags with NameLoc prefix enabled via env var
static OpPrintingFlags getWSPrintingFlags() {
  OpPrintingFlags flags;
  if (triton::tools::getBoolEnv("TRITON_USE_NAMELOC_PREFIX")) {
    flags.printNameLocAsPrefix(true);
  }
  return flags;
}

// Helper to dump module with appropriate printing flags
static void dumpModuleIR(StringRef label, ModuleOp moduleOp) {
  llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: " << label
               << "\n";
  moduleOp.print(llvm::dbgs(), getWSPrintingFlags());
  llvm::dbgs() << "\n\n\n";
}

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
void doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doBufferAllocation(triton::FuncOp &funcOp);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability);

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp, int defaultNumStages) {
    bool enabled = false;
    funcOp->walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id"))
        enabled = true;
      if (auto attr = op->getAttrOfType<IntegerAttr>("ttg.partition"))
        enabled = true;
    });
    if (!enabled) {
      SmallVector<scf::ForOp> loops;
      funcOp->walk([&](scf::ForOp forOp) {
        if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
          loops.push_back(forOp);
      });
      if (!loops.empty())
        enabled = true;
    }
    if (!enabled)
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4)
      return;

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
          if (!isa<scf::YieldOp>(&op))
            hasElse = true;
        }
      }
    });
    if (hasElse)
      return;

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    // FIXME: skip data partitioning with on-host TMA.
    // FIXME: skip data partitioning for Blackwell.
    bool ForBlackWell = (capability / 10) > 9;
    unsigned numWarpGroups = ForBlackWell ? 2 : 3;
    if (!ForBlackWell) {
      bool success = false;
      for (; numWarpGroups >= 2; numWarpGroups--) {
        // Partition key ops into multiple async tasks.
        doTaskPartition(funcOp, numWarpGroups);
        if (dumpIntermediateSteps) {
          dumpModuleIR("doTaskPartition", moduleOp);
        }
        // Propagate taskId.
        int retCode = doTaskIdPropagate(funcOp);
        if (retCode == -1)
          continue;
        if (dumpIntermediateSteps) {
          dumpModuleIR("doTaskIdPropagate", moduleOp);
        }

        // Partition ops into parallel sub ops.
        if (doDataPartition(funcOp, numWarpGroups - 1)) {
          if (dumpIntermediateSteps) {
            dumpModuleIR("doDataPartition", moduleOp);
          }
          success = true;
          break;
        }
        // Clear async_task.
      }
      if (!success)
        signalPassFailure();
    } else {
      int retCode = doTaskIdPropagate(funcOp);
      if (retCode == -1)
        signalPassFailure();
      if (dumpIntermediateSteps) {
        dumpModuleIR("doTaskIdPropagate", moduleOp);
      }
    }

    // Canonicalize the SMEM/TEM buffers.
    // Create buffers for register channels.
    doBufferAllocation(funcOp);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doBufferAllocation", moduleOp);
    }

    doMemoryPlanner(funcOp, numStages);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doMemoryPlanner", moduleOp);
    }

    doCodePartitionPost(funcOp, numStages);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doCodePartition", moduleOp);
    }

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      if (dumpIntermediateSteps) {
        dumpModuleIR("doPingPongSync", moduleOp);
      }
    }

    doTokenLowering(funcOp, numWarpGroups - 1);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doTokenLowering", moduleOp);
    }

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doLoopSchedulePreprocessing", moduleOp);
    }
    triton::gpu::scheduleLoops(moduleOp, defaultNumStages, true);
    if (dumpIntermediateSteps) {
      dumpModuleIR("doLoopSchedule", moduleOp);
    }
  }

  void runOnOperation() override {
    assert(numStages >= 1 && "numStages must be at least 1");
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { runOnFuncOp(funcOp, numStages); });

    // Cleanup code generated by warp specialization.
    RewritePatternSet patterns(&getContext());
    populateForOpDeadArgumentElimination(patterns);
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    mlir::triton::gpu::WarpSpecializeOp::getCanonicalizationPatterns(
        patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace mlir
