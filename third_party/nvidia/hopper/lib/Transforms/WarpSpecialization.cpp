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

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
void doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doBufferAllocation(triton::FuncOp &funcOp);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups, int capability);

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
          llvm::dbgs()
              << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
              << moduleOp << "\n\n\n";
        }
        // Propagate taskId.
        int retCode = doTaskIdPropagate(funcOp);
        if (retCode == -1)
          continue;
        if (dumpIntermediateSteps) {
          llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                          "doTaskIdPropagate\n"
                       << moduleOp << "\n\n\n";
        }

        // Partition ops into parallel sub ops.
        if (doDataPartition(funcOp, numWarpGroups - 1)) {
          if (dumpIntermediateSteps) {
            llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                            "doDataPartition\n"
                         << moduleOp << "\n\n\n";
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
        llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                        "doTaskIdPropagate\n"
                     << moduleOp << "\n\n\n";
      }
    }

    // Canonicalize the SMEM/TEM buffers.
    // Create buffers for register channels.
    doBufferAllocation(funcOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doBufferAllocation\n"
          << moduleOp << "\n\n\n";
    }

    doMemoryPlanner(funcOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doMemoryPlanner\n"
          << moduleOp << "\n\n\n";
    }

    doCodePartitionPost(funcOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doPingPongSync\n"
            << moduleOp << "\n\n\n";
      }
    }

    doTokenLowering(funcOp, numWarpGroups - 1);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doTokenLowering\n"
          << moduleOp << "\n\n\n";
    }

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doLoopSchedulePreprocessing\n"
                   << moduleOp << "\n\n\n";
    }
    triton::gpu::scheduleLoops(moduleOp, defaultNumStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doLoopSchedule\n"
                   << moduleOp << "\n\n\n";
    }
  }

  void runOnOperation() override {
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
