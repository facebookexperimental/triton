// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Modulo Lowering Pass (post-expansion cleanup)
//
// Runs after ModuloExpandPass. Performs the same post-expansion steps
// as the standard PipelinePass:
//   1. removePipeliningAttributes — strip loop.stage/loop.cluster attrs
//   2. asyncLaunchDots — pipeline wgmma ops (mark async, insert waits)
//   3. updateWaits — adjust AsyncWaitOp pending counts
//   4. pipelineTMAStores — pipeline TMA store operations
//   5. arith canonicalization — clean up arithmetic

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-modulo-lower"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;

namespace {

struct ModuloLowerPass
    : public PassWrapper<ModuloLowerPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloLowerPass)

  StringRef getArgument() const override { return "nvgpu-modulo-lower"; }

  StringRef getDescription() const override {
    return "Post-expansion cleanup (asyncLaunchDots, updateWaits, TMA stores)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    LDBG("=== Post-expansion cleanup ===");

    // Step 1: Remove pipelining attributes (loop.stage, loop.cluster, etc.)
    LDBG("Step 1: removePipeliningAttributes");
    tt::removePipeliningAttributes(moduleOp);

    // Verify all loop.stage attrs were consumed and removed.
    LLVM_DEBUG({
      bool hasStaleAttrs = false;
      moduleOp->walk([&](Operation *op) {
        if (op->hasAttr(tt::kLoopStageAttrName)) {
          hasStaleAttrs = true;
          DBGS() << "WARNING: stale loop.stage on: " << *op << "\n";
        }
      });
      if (hasStaleAttrs)
        DBGS() << "WARNING: loop.stage attributes remain after "
               << "removePipeliningAttributes\n";
    });

    // Step 2: Pipeline wgmma ops — mark dots as async, insert waits.
    LDBG("Step 2: asyncLaunchDots");
    SmallVector<scf::ForOp> loops;
    moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      tt::asyncLaunchDots(forOp);

    // Step 3: Update wait ops with correct pending counts.
    LDBG("Step 3: updateWaits");
    tt::updateWaits(moduleOp);

    // Step 4: Canonicalize arith to simplify index arithmetic from expansion.
    auto *arithDialect =
        moduleOp.getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(moduleOp.getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(moduleOp, std::move(patterns)).failed())
      return signalPassFailure();

    // Step 5: Pipeline TMA stores.
    LDBG("Step 5: pipelineTMAStores");
    loops.clear();
    moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      tt::pipelineTMAStores(forOp);

    LDBG("Post-expansion cleanup complete");
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloLower() {
  return std::make_unique<ModuloLowerPass>();
}

void registerNVGPUModuloLower() { PassRegistration<ModuloLowerPass>(); }
} // namespace mlir
