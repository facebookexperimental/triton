#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/WarpSpecializationPipeline.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/Dump.h"
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

static LogicalResult cleanupWarpSpecializedLoops(Operation *op) {
  runDeadIterArgElimination(op);
  RewritePatternSet patterns(op->getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, op->getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, op->getContext());
  mlir::triton::gpu::WarpSpecializeOp::getCanonicalizationPatterns(
      patterns, op->getContext());
  return applyPatternsGreedily(op, std::move(patterns));
}

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
    auto stripLoop = [](Operation *loop) {
      loop->removeAttr(mlir::triton::kWarpSpecializeAttrName);
      loop->removeAttr(mlir::triton::gpu::kPartitionStagesAttrName);
      loop->removeAttr(mlir::triton::gpu::kWarpSpecializeTagAttrName);
      loop->removeAttr(kPartitionTypesAttrName);
    };
    funcOp->walk([&](scf::ForOp forOp) { stripLoop(forOp); });
    funcOp->walk([&](scf::WhileOp whileOp) { stripLoop(whileOp); });
    funcOp->walk([&](Operation *op) {
      // Strip both the partition id (`ttg.partition`) and the task id
      // (`async_task_id`). The task id is only present once `doTaskIdPropagate`
      // has run (e.g. the atomic-broadcast reject path bails after
      // propagation); for the earlier bail-outs it simply does not exist yet
      // and this is a no-op. Leaving either behind produces a half-tagged state
      // that the downstream `tritongpu-pipeline` pass mis-treats as a WS
      // region. Use the shared helper so the attr name lives in one place.
      removeAsyncTaskIds(op);
      op->removeAttr(mlir::triton::gpu::kPartitionAttrName);
      op->removeAttr(mlir::triton::gpu::kPartitionOutputsAttrName);
    });
  }

  // Dump the whole module to llvm::dbgs() after a pipeline step, gated on the
  // `dump-intermediate-steps` pass option. Collapses the identical dump blocks
  // that otherwise dominate runOnFuncOp; the emitted text is unchanged.
  void dumpAfter(ModuleOp moduleOp, StringRef stepName) {
    if (!dumpIntermediateSteps)
      return;
    llvm::dbgs() << "// -----// WarpSpec internal IR Dump After: " << stepName
                 << "\n";
    moduleOp.print(llvm::dbgs(), getOpPrintingFlagsWithLoc());
    llvm::dbgs() << "\n\n\n";
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
    if (numWarps < 4) {
      LDBG("Warp specialization requires at least 4 warps. Skipping.");
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
    dumpAfter(moduleOp, "doTaskIdPropagate");

    // Cross-partition run-once, loop-carried "claim the next tile" support for
    // dynamic-persistent kernels. Handles both the `tt.atomic_rmw` tile counter
    // and the CLC tile-scheduler fetch (`ttng.clc_read`) with the same idea:
    // run the claim once in the owner/producer partition and broadcast the
    // loop-carried result(s) to every partition through SMEM, or gracefully
    // bail out of warp specialization (unsupported shape). On a reject we strip
    // all WS metadata via removeWarpSpecializeAttr (which also clears the
    // `async_task_id`s that doTaskIdPropagate materialized above) so downstream
    // sees a plain, compilable non-WS kernel. The broadcast channel depth comes
    // from the `tile-prefetch-depth` pass option (a Python knob), not an env
    // var.
    if (failed(doDynamicTileBroadcast(funcOp, tilePrefetchDepth))) {
      LDBG("Dynamic tile broadcast rejected warp specialization. Skipping.");
      removeWarpSpecializeAttr(funcOp);
      return;
    }
    dumpAfter(moduleOp, "doDynamicTileBroadcast");

    if (pingpongAutoWS) {
      doPingPongPrep(funcOp, numWarpGroups, capability, defaultNumStages);
      dumpAfter(moduleOp, "doPingPongPrep");
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
    dumpAfter(moduleOp, "doBufferAllocation");

    doHoistLoopInvariantTMEMStore(funcOp);
    dumpAfter(moduleOp, "doHoistLoopInvariantTMEMStore");

    if (failed(doMemoryPlanner(funcOp, numStages, /*readDecisionFile=*/"",
                               /*writeDecisionFile=*/"",
                               /*smemAllocAlgo=*/1, smemBudget))) {
      signalPassFailure();
      return;
    }
    dumpAfter(moduleOp, "doMemoryPlanner");

    if (generateSubtiledRegion) {
      doGenerateSubtiledRegion(funcOp);
      dumpAfter(moduleOp, "doGenerateSubtiledRegion");
    }

    doAnnotateTMAStoreWaits(funcOp);
    dumpAfter(moduleOp, "doAnnotateTMAStoreWaits");

    doValidateTMAStoreAnnotations(funcOp);
    dumpAfter(moduleOp, "doValidateTMAStoreAnnotations");

    doCodePartitionPost(funcOp, numStages);
    // Label kept as "doCodePartition" for output stability (see WS-15).
    dumpAfter(moduleOp, "doCodePartition");

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      dumpAfter(moduleOp, "doPingPongSync");
    }

    doLowerSubtiledRegionsWithNVWSOps(funcOp);
    doTokenLowering(funcOp, numWarpGroups - 1);
    invalidateWarpSpecializeBarriers(funcOp);
    dumpAfter(moduleOp, "doTokenLowering");

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    dumpAfter(moduleOp, "doLoopSchedulePreprocessing");

    triton::gpu::scheduleLoops(moduleOp, defaultNumStages, true);
    dumpAfter(moduleOp, "doLoopSchedule");

    doLowerRemainingSubtiledRegions(funcOp);
    if (failed(cleanupWarpSpecializedLoops(funcOp))) {
      signalPassFailure();
      return;
    }
    dumpAfter(moduleOp, "cleanupWarpSpecializedLoops");

    doTMAStoreWaitReorder(funcOp);
    dumpAfter(moduleOp, "doTMAStoreWaitReorder");
  }

  void runOnOperation() override {
    assert(numStages >= 1 && "numStages must be at least 1");
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { runOnFuncOp(funcOp, numStages); });

    // Cleanup code generated by warp specialization.
    if (failed(cleanupWarpSpecializedLoops(getOperation())))
      return signalPassFailure();
  }
};

} // namespace mlir
