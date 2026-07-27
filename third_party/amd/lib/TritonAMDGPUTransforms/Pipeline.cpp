#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h" // IWYU pragma: keep
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define DEBUG_TYPE "tritonamdgpu-pipeline-expand-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                             Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, dotOp->getResult(0).getType(),
                                  pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = scf::YieldOp::create(thenB, loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    auto ifOpBuilder = ifOp.getElseBodyBuilder();
    scf::YieldOp::create(ifOpBuilder, loc, dotOp->getOperand(2));
    return ifOp;
  }
  if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
    auto loc = op->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, op->getResultTypes(), pred,
                                  /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = scf::YieldOp::create(thenB, loc, op->getResults());
    op->moveBefore(yield);

    auto elseB = ifOp.getElseBodyBuilder();
    SmallVector<Value> zeroValues;
    zeroValues.reserve(op->getNumResults());
    for (Type resultType : op->getResultTypes()) {
      zeroValues.push_back(
          arith::ConstantOp::create(elseB, loc, elseB.getZeroAttr(resultType)));
    }
    scf::YieldOp::create(elseB, loc, zeroValues);
    return ifOp;
  }
  if (auto copyOp = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(copyOp);
    // TDM requires the mask as I32
    auto predI32 = arith::ExtUIOp::create(rewriter, copyOp->getLoc(),
                                          copyOp.getPred().getType(), pred);
    Value mask = arith::AndIOp::create(rewriter, copyOp->getLoc(),
                                       copyOp.getPred(), predI32);
    copyOp.getPredMutable().assign(mask);
    return op;
  } else if (auto gatherOp = dyn_cast<triton::amdgpu::AsyncTDMGatherOp>(op)) {
    rewriter.setInsertionPoint(gatherOp);
    // TDM requires the mask as I32
    auto predI32 = arith::ExtUIOp::create(rewriter, gatherOp->getLoc(),
                                          gatherOp.getPred().getType(), pred);
    Value mask = arith::AndIOp::create(rewriter, gatherOp->getLoc(),
                                       gatherOp.getPred(), predI32);
    gatherOp.getPredMutable().assign(mask);
    return op;
  } else if (auto prefetchOp = dyn_cast<triton::amdgpu::TDMPrefetchOp>(op)) {
    rewriter.setInsertionPoint(prefetchOp);
    Value mask = arith::AndIOp::create(rewriter, prefetchOp->getLoc(),
                                       prefetchOp.getPred(), pred);
    prefetchOp.getPredMutable().assign(mask);
    return op;
  } else if (auto waitOp = dyn_cast<triton::amdgpu::AsyncTDMWait>(op)) {
    return op;
  } else if (isa<tt::DescriptorStoreLikeOpInterface>(op)) {
    auto loc = op->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, pred,
                                  /*withElseRegion=*/false);
    op->moveBefore(ifOp.thenYield());
    return ifOp;
  }
  return tt::wrapInMaskOp(rewriter, op, pred);
}
} // namespace

// Exposed for reuse by the decompose+modulo pipeline (change #4): deserialize
// the CoarseSchedule the caller serialized and run the general pipeline
// expander.
void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      continue;

    // Create the final schedule for the kernel loop. This will dictate the
    // stages and order of operations to the pipeline expander.
    auto coarseSchedule = schedule.createFinalSchedule(forOp);

    tt::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
    // Annotate loadOp in prologue for further moving up
    options.annotateFn = [](Operation *op,
                            tt::PipeliningOption::PipelinerPart part,
                            unsigned stage) {
      if (part != tt::PipeliningOption::PipelinerPart::Prologue)
        return;

      auto annotateLoad = [](Operation *loadOp) {
        loadOp->setAttr("amd.pipeliner_part",
                        StringAttr::get(loadOp->getContext(), "prologue"));
      };

      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        annotateLoad(loadOp);
        return;
      }
      // loadOp may be wrapped by a MaskOp as predicateFn execution
      // precedes annotation
      if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
        for (auto &innerOp : maskOp.getBody()->without_terminator()) {
          if (auto loadOp = dyn_cast<tt::LoadOp>(&innerOp)) {
            annotateLoad(loadOp);
            return;
          }
        }
      }
    };
    // Set the final schedule as our scheduling function
    options.getScheduleFn =
        [coarseSchedule](scf::ForOp,
                         std::vector<std::pair<Operation *, unsigned>> &s) {
          s = std::move(coarseSchedule);
        };

    LDBG("Loop before sending to expander:\n" << *forOp);

    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        tt::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp))
      continue;
  }

  tt::resolveMaskOp(moduleOp);
}
// Fold consecutive waits of the same kind into a single wait.
void combineWaitOps(ModuleOp moduleOp, bool useAsyncCopy) {
  llvm::SmallSetVector<Operation *, 8> asyncWaitOps;
  llvm::SmallSetVector<Operation *, 8> tdmWaitOps;
  moduleOp.walk([&](Operation *op) {
    if (useAsyncCopy && isa<ttg::AsyncWaitOp>(op))
      asyncWaitOps.insert(op);
    else if (isa<triton::amdgpu::AsyncTDMWait>(op))
      tdmWaitOps.insert(op);
  });

  if (useAsyncCopy) {
    tt::combineRedundantWaitOps(
        asyncWaitOps,
        [](Operation *op) { return isa<ttg::AsyncCommitGroupOp>(op); },
        [](OpBuilder &b, Location loc, ValueRange operands,
           unsigned num) -> Operation * {
          return ttg::AsyncWaitOp::create(b, loc, operands, num);
        });
  }

  tt::combineRedundantWaitOps(
      tdmWaitOps,
      [](Operation *op) { return isa<triton::amdgpu::TDMOpInterface>(op); },
      [](OpBuilder &b, Location loc, ValueRange operands,
         unsigned num) -> Operation * {
        return triton::amdgpu::AsyncTDMWait::create(b, loc, operands, num);
      });
}

struct PipelinePass : impl::TritonAMDGPUPipelineBase<PipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    lowerLoops(moduleOp, useAsyncCopy, usePingpong);
    expandLoops(moduleOp);

    if (useAsyncCopy) {
      auto arch = getAMDArch(moduleOp);
      auto family =
          arch ? tt::AMD::deduceISAFamily(*arch) : tt::AMD::ISAFamily::Unknown;
      // Only asyncmark targets (CDNA3/CDNA4) need updateWaits here: their
      // lowering reads ttg.async_wait's `num` directly into wait.asyncmark(N),
      // and PR #9883 made UpdateAsyncWaitCount a no-op on those archs, so
      // without this call the pipeliner-authored num=0 would serialize the
      // SWP. Every other family keeps the prior combineRedundantWaitOps-only
      // path: their num is re-derived downstream by UpdateAsyncWaitCount.
      if (family == tt::AMD::ISAFamily::CDNA3 ||
          family == tt::AMD::ISAFamily::CDNA4) {
        mlir::triton::updateWaits(moduleOp);
      }
    }
    combineWaitOps(moduleOp, useAsyncCopy);

    tt::removePipeliningAttributes(moduleOp);
  }
};
} // namespace mlir
