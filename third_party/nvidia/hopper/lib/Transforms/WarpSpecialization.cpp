#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

// Helper to get printing flags with location info enabled
static OpPrintingFlags getOpPrintingFlagsWithLoc() {
  OpPrintingFlags flags;
  flags.enableDebugInfo();
  flags.printNameLocAsPrefix(true);
  return flags;
}

int doTaskIdPropagate(triton::FuncOp &funcOp);
LogicalResult doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers,
                              StringRef readDecisionFile = "",
                              StringRef writeDecisionFile = "",
                              int smemAllocAlgo = 0, unsigned smemBudget = 0,
                              bool smemCircularReuse = false);
void doBufferAllocation(triton::FuncOp &funcOp);
void doHoistLoopInvariantTMEMStore(triton::FuncOp &funcOp);
void removeRedundantTmemZeroStores(triton::FuncOp &funcOp);
void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doInsert2CTASync(triton::FuncOp funcOp);
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability, int defaultNumStages);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability);
void doTMAStoreWaitReorder(triton::FuncOp &funcOp);
void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp);
void doLowerSubtiledRegionsWithNVWSOps(triton::FuncOp &funcOp) {
  namespace ttng = triton::nvidia_gpu;
  namespace nvws = triton::nvws;
  SmallVector<ttng::SubtiledRegionOp> toInline;
  funcOp.walk([&](ttng::SubtiledRegionOp op) {
    Block &tileBlock = op.getTileRegion().front();
    for (Operation &tileOp : tileBlock.without_terminator()) {
      if (isa<nvws::ProducerAcquireOp, nvws::ProducerCommitOp,
              nvws::ConsumerWaitOp, nvws::ConsumerReleaseOp>(&tileOp)) {
        toInline.push_back(op);
        break;
      }
    }
  });
  for (auto op : toInline)
    ttng::lowerSubtiledRegion(op);
}

void doLowerRemainingSubtiledRegions(triton::FuncOp &funcOp) {
  namespace ttng = triton::nvidia_gpu;
  SmallVector<ttng::SubtiledRegionOp> remaining;
  funcOp.walk([&](ttng::SubtiledRegionOp op) { remaining.push_back(op); });
  for (auto op : remaining)
    ttng::lowerSubtiledRegion(op);
}

void doGenerateSubtiledRegion(triton::FuncOp &funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  PassManager pm(moduleOp.getContext());
  pm.addPass(triton::nvidia_gpu::
                 createTritonNvidiaGPUTestGenerateSubtiledRegionPass());
  // OptimizeTMemLayouts runs later via add_optimize_tmem_layouts in
  // compiler.py. This avoids transforming bare splits into tmem_subslice
  // ops that lack async_task_id and would crash createChannelPost.
  (void)pm.run(moduleOp);
}

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  // Remove the warp_specialize attribute from all loops in the function, plus
  // any partition metadata that the earlier `tritongpu-partition-scheduling`
  // pass may have written. The two passes form a pair: when this pass takes
  // an early-exit and skips warp specialization (e.g. else-block fallback),
  // leaving `ttg.partition` / `ttg.partition.stages` /
  // `ttg.warp_specialize.tag` behind on ops + loops produces a half-tagged
  // state — the downstream `tritongpu-pipeline` pass treats partition-tagged
  // regions as WS regions and crashes when sibling ops in an scf.if/else aren't
  // tagged. Stripping everything ensures downstream sees a plain (non-WS) loop.
  void removeWarpSpecializeAttr(triton::FuncOp funcOp) {
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->removeAttr(mlir::triton::kWarpSpecializeAttrName);
      forOp->removeAttr(mlir::triton::gpu::kPartitionStagesAttrName);
      forOp->removeAttr(mlir::triton::gpu::kWarpSpecializeTagAttrName);
    });
    funcOp->walk([&](Operation *op) {
      op->removeAttr(mlir::triton::gpu::kPartitionAttrName);
      op->removeAttr(mlir::triton::gpu::kPartitionOutputsAttrName);
    });
  }

  void runOnFuncOp(triton::FuncOp funcOp, int defaultNumStages) {
    bool enabled = false;
    funcOp->walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id"))
        enabled = true;
      if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(
              triton::gpu::kPartitionAttrName))
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
    if (numWarps != 4) {
      LDBG("Warp specialization requires num_warps=4, but got "
           << numWarps << ". Skipping.");
      removeWarpSpecializeAttr(funcOp);
      return;
    }

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
    if (hasElse) {
      LDBG("Warp specialization does not support else blocks. Skipping.");
      removeWarpSpecializeAttr(funcOp);
      return;
    }

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    // FIXME: skip data partitioning for Blackwell.
    bool ForBlackWell = (capability / 10) > 9;
    unsigned numWarpGroups = ForBlackWell ? 2 : 3;

    int retCode = doTaskIdPropagate(funcOp);
    if (retCode == -1) {
      signalPassFailure();
      return;
    }
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doTaskIdPropagate\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    if (pingpongAutoWS) {
      doPingPongPrep(funcOp, numWarpGroups, capability, defaultNumStages);
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doPingPongPrep\n";
        moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
        llvm::dbgs() << "\n\n\n";
      }
    }

    // Remove redundant TMEM zeroing stores before buffer allocation.
    // When a TMEMAllocOp is used as operand D of a TCGen5MMAOp with
    // useAccumulator=false (on the first iteration), any preceding
    // tmem_store of zeros is redundant — the MMA's useD=false already
    // zeros the accumulator. Removing the store prevents the autoWS
    // compiler from creating a cross-partition channel for it, which
    // would otherwise cause a race condition between the reduction
    // partition (zeroing) and the computation partition (reading) in
    // persistent kernels.
    removeRedundantTmemZeroStores(funcOp);

    // Canonicalize the SMEM/TEM buffers.
    // Create buffers for register channels.
    doBufferAllocation(funcOp);

    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doBufferAllocation\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    doHoistLoopInvariantTMEMStore(funcOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doHoistLoopInvariantTMEMStore\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    if (failed(doMemoryPlanner(funcOp, numStages, /*readDecisionFile=*/"",
                               /*writeDecisionFile=*/"",
                               /*smemAllocAlgo=*/0, smemBudget))) {
      signalPassFailure();
      return;
    }
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doMemoryPlanner\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    if (generateSubtiledRegion) {
      doGenerateSubtiledRegion(funcOp);
      if (dumpIntermediateSteps) {
        llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                        "doGenerateSubtiledRegion\n";
        moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
        llvm::dbgs() << "\n\n\n";
      }
    }

    doAnnotateTMAStoreWaits(funcOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doAnnotateTMAStoreWaits\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    doValidateTMAStoreAnnotations(funcOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doValidateTMAStoreAnnotations\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    doCodePartitionPost(funcOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doPingPongSync\n";
        moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
        llvm::dbgs() << "\n\n\n";
      }
    }

    doLowerSubtiledRegionsWithNVWSOps(funcOp);
    doTokenLowering(funcOp, numWarpGroups - 1);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doTokenLowering\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doLoopSchedulePreprocessing\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }
    triton::gpu::scheduleLoops(moduleOp, defaultNumStages, true);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doLoopSchedule\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
    }

    doLowerRemainingSubtiledRegions(funcOp);
    doTMAStoreWaitReorder(funcOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: "
                      "doTMAStoreWaitReorder\n";
      moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
      llvm::dbgs() << "\n\n\n";
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
