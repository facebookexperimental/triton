// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Modulo Loop Expansion Pass (Phase 2 + Phase 3 combined)
//
// This pass takes the modulo-scheduled loop (with loop.stage attrs from
// ModuloSchedulePass) and performs the full software pipelining
// transformation:
//   1. lowerLoops() — transform loads into async copies, insert barriers,
//      allocate multi-buffered SMEM/TMEM (same as existing Pipeline pass)
//   2. expandLoops() — generate prologue/kernel/epilogue via PipelineExpander
//
// The key difference from the standard Pipeline pass is that our schedule
// comes from Rau's iterative modulo scheduling (Phase 0) rather than
// the heuristic-based assign_latencies + schedule_loops.
//
// NOTE: lowerLoops() processes ALL loops in the module, not just
// modulo-scheduled ones. When integrating with the standard Pipeline pass,
// ensure they don't both run lowerLoops() on the same module.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-modulo-expand"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

/// Check if the loop has MMAv5 waits in its last stage — if so, we need
/// custom epilogue peeling (same logic as SoftwarePipeliner.cpp).
static bool hasMMAv5WaitsInLastStage(scf::ForOp forOp,
                                     triton::CoarseSchedule &schedule) {
  int maxStage = schedule.getNumStages() - 1;
  bool hasMMAv5 = false;
  bool hasWaitInLastStage = false;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<ttng::WaitBarrierOp>(op) && schedule[&op].first == maxStage)
      hasWaitInLastStage = true;
    if (isa<ttng::MMAv5OpInterface>(op))
      hasMMAv5 = true;
  }
  return hasMMAv5 && hasWaitInLastStage;
}

/// Replicate the expandLoops() logic from SoftwarePipeliner.cpp.
/// Deserializes the schedule, calls pipelineForLoop(), handles epilogue
/// peeling for MMAv5 loops.
static void moduloExpandLoops(ModuleOp moduleOp) {
  DenseSet<ttg::MaskOp> peeledMaskOps;
  auto processPeeledEpilogueOp = [&](RewriterBase &rewriter, Operation *op,
                                     bool isEpilogue) -> Operation * {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (auto predOp = dyn_cast<ttg::PredicateStageOp>(op)) {
      if (isEpilogue) {
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 0);
      }
      if (predOp.getStage() == predOp.getMaxStage() - 1) {
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 1);
      }
      return triton::emitPredicateForStage(
                 rewriter, predOp.getIv(), predOp.getUb(), predOp.getStep(),
                 predOp.getMaxStage(), predOp.getStage())
          .getDefiningOp();
    }
    if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
      if (isEpilogue)
        peeledMaskOps.insert(maskOp);
    }
    return op;
  };

  // Collect loops with their nesting depth. We must expand inner loops first
  // (bottom-up) so that after inner expansion, the inner loop is a "black box"
  // for outer expansion. moduleOp->walk uses pre-order (outer before inner),
  // so we explicitly sort by descending depth.
  SmallVector<std::pair<scf::ForOp, unsigned>> loopsWithDepth;
  moduleOp->walk([&](scf::ForOp forOp) {
    unsigned depth = 0;
    for (auto *parent = forOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent))
        ++depth;
    }
    loopsWithDepth.push_back({forOp, depth});
  });
  // Sort by descending depth — innermost loops first.
  llvm::sort(loopsWithDepth,
             [](const auto &a, const auto &b) { return a.second > b.second; });

  for (auto &[forOp, depth] : loopsWithDepth) {
    // Safety: inner loop expansion may have erased or replaced this op.
    if (!forOp || !forOp->getBlock())
      continue;

    triton::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      continue;

    // Skip loops with only 1 stage — no pipelining needed.
    if (schedule.getNumStages() <= 1) {
      LDBG("Skipping loop at depth " << depth << " with "
                                     << schedule.getNumStages()
                                     << " stage(s) — no expansion needed");
      continue;
    }

    LDBG("Expanding loop at depth " << depth << " with "
                                    << schedule.getNumStages() << " stages");

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    triton::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = false;
    options.predicateFn = triton::wrapInMaskOp;
    options.getScheduleFn =
        [&](scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &sched) {
          sched = finalSchedule;
        };

    bool customEpiloguePeeling =
        hasMMAv5WaitsInLastStage(forOp, schedule) &&
        !forOp->getParentOfType<ttg::WarpSpecializeOp>();
    if (customEpiloguePeeling) {
      options.emitPredicateStageFn = [](RewriterBase &rewriter,
                                        Value inductionVar, Value upperBound,
                                        Value step, uint64_t maxStage,
                                        uint64_t stage) {
        return ttg::PredicateStageOp::create(rewriter, inductionVar.getLoc(),
                                             inductionVar, upperBound, step,
                                             maxStage, stage);
      };
    }

    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        triton::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp)) {
      LDBG("pipelineForLoop FAILED for a loop");
      continue;
    }
    forOp = *newForOp;
    if (customEpiloguePeeling)
      triton::peelLoopEpilogue(forOp, processPeeledEpilogueOp);
  }

  assert(moduleOp.getOps<ttg::PredicateStageOp>().empty() &&
         "PredicateStageOp should be resolved after pipeline expansion");
  LLVM_DEBUG({
    if (failed(verify(moduleOp)))
      DBGS() << "WARNING: IR verification failed after expansion\n";
  });
  triton::resolveMaskOp(moduleOp);
}

struct ModuloExpandPass
    : public PassWrapper<ModuloExpandPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloExpandPass)

  StringRef getArgument() const override { return "nvgpu-modulo-expand"; }

  StringRef getDescription() const override {
    return "Modulo loop expansion (lowerLoops + pipelineForLoop)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    LDBG("=== Phase 2+3: Loop Expansion ===");

    LDBG("Step 1: lowerLoops");
    triton::gpu::lowerLoops(moduleOp);

    LDBG("Step 2: expandLoops");
    moduloExpandLoops(moduleOp);

    LDBG("Expansion complete");
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloExpand() {
  return std::make_unique<ModuloExpandPass>();
}

void registerNVGPUModuloExpand() {
  PassRegistration<ModuloExpandPass>();
}
} // namespace mlir
