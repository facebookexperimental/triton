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

// A GPU is Blackwell-class (sm_100+) when its major compute capability is >= 10
// (capability is encoded as major*10 + minor, e.g. 90 for Hopper, 100 for
// Blackwell).
static bool capabilityIsBlackwell(int capability) {
  return capability / 10 > 9;
}

// Warp-group split for warp specialization: one producer (load) group plus N
// consumer (compute) groups. Blackwell needs fewer groups than Hopper because
// its MMA is issued by a single warp.
static constexpr unsigned kNumWarpGroupsBlackwell = 2;
static constexpr unsigned kNumWarpGroupsHopper = 3;

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

void doLowerSubtiledRegionsWithNVWSOps(triton::FuncOp funcOp) {
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

void doLowerRemainingSubtiledRegions(triton::FuncOp funcOp) {
  namespace ttng = triton::nvidia_gpu;
  SmallVector<ttng::SubtiledRegionOp> remaining;
  funcOp.walk([&](ttng::SubtiledRegionOp op) { remaining.push_back(op); });
  for (auto op : remaining)
    ttng::lowerSubtiledRegion(op);
}

void doGenerateSubtiledRegion(triton::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  PassManager pm(moduleOp.getContext());
  pm.addPass(triton::nvidia_gpu::
                 createTritonNvidiaGPUTestGenerateSubtiledRegionPass());
  // OptimizeTMemLayouts runs later via add_optimize_tmem_layouts in
  // compiler.py. This avoids transforming bare splits into tmem_subslice
  // ops that lack async_task_id and would crash createAllocChannel.
  (void)pm.run(moduleOp);
}

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  // Reject warp specialization for this function: strip the WS metadata so the
  // downstream pipeline sees a plain, compilable non-WS kernel. This is the
  // single reject epilogue shared by the early-exit paths in runOnFuncOp; use
  // it as `return bailOut(funcOp);`. The canonical WS-metadata set lives in the
  // shared `removeWarpSpecMetadata` (CodePartitionUtility.h).
  void bailOut(triton::FuncOp funcOp) { removeWarpSpecMetadata(funcOp); }

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

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Warp specialization is enabled for this function if any op carries WS
    // metadata: an `async_task_id` or `ttg.partition` tag, or a loop marked
    // with the warp-specialize attribute. A single walk with early-out
    // suffices.
    bool enabled = false;
    funcOp->walk([&](Operation *op) {
      if (op->getAttrOfType<DenseI32ArrayAttr>(kAsyncTaskIdAttrName) ||
          op->getAttrOfType<DenseI32ArrayAttr>(
              triton::gpu::kPartitionAttrName) ||
          (isa<scf::ForOp>(op) &&
           op->hasAttr(mlir::triton::kWarpSpecializeAttrName))) {
        enabled = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!enabled)
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps < 4) {
      LDBG("Warp specialization requires at least 4 warps. Skipping.");
      return bailOut(funcOp);
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
      return bailOut(funcOp);
    }

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    // FIXME: skip data partitioning for Blackwell.
    bool isBlackwell = capabilityIsBlackwell(capability);
    unsigned numWarpGroups =
        isBlackwell ? kNumWarpGroupsBlackwell : kNumWarpGroupsHopper;

    if (failed(doTaskIdPropagate(funcOp))) {
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
    // all WS metadata via removeWarpSpecMetadata (which also clears the
    // `async_task_id`s that doTaskIdPropagate materialized above) so downstream
    // sees a plain, compilable non-WS kernel. The broadcast channel depth comes
    // from the `tile-prefetch-depth` pass option (a Python knob), not an env
    // var.
    if (failed(doDynamicTileBroadcast(funcOp, tilePrefetchDepth))) {
      LDBG("Dynamic tile broadcast rejected warp specialization. Skipping.");
      return bailOut(funcOp);
    }
    dumpAfter(moduleOp, "doDynamicTileBroadcast");

    if (pingpongAutoWS) {
      doPingPongPrep(funcOp, numWarpGroups, capability, numStages);
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

    if (failed(doMemoryPlanner(funcOp, numStages, smemBudget))) {
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

    doCodePartition(funcOp, numStages);
    dumpAfter(moduleOp, "doCodePartition");

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      dumpAfter(moduleOp, "doPingPongSync");
    }

    doLowerSubtiledRegionsWithNVWSOps(funcOp);
    // One producer (load) group; the remaining groups are consumers.
    unsigned numConsumerGroups = numWarpGroups - 1;
    doTokenLowering(funcOp, numConsumerGroups);
    invalidateWarpSpecializeBarriers(funcOp);
    dumpAfter(moduleOp, "doTokenLowering");

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    dumpAfter(moduleOp, "doLoopSchedulePreprocessing");

    triton::gpu::scheduleLoops(moduleOp, numStages, true);
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
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });

    // Cleanup code generated by warp specialization.
    if (failed(cleanupWarpSpecializedLoops(getOperation())))
      return signalPassFailure();
  }
};

} // namespace mlir
