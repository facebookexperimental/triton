#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
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
  // Gate the copy by chaining a pred-only update_tensor_descriptor onto its
  // descriptor: the chained update inherits the positioning and narrows pred to
  // the loop predicate.
  if (auto copyOp = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(rewriter, op->getLoc(),
                                          rewriter.getI32Type(), pred);
    auto updated = triton::amdgpu::UpdateTensorDescriptorOp::create(
        rewriter, op->getLoc(), copyOp.getDesc().getType(), copyOp.getDesc(),
        /*add_offsets=*/ValueRange{}, /*set_bounds=*/ValueRange{},
        /*pred=*/predI32);
    copyOp.getDescMutable().assign(updated.getResult());
    return op;
  }
  // Descriptor updates are scheduled with their fused consumer. Predicate
  // them in place so the pipeliner does not hide the descriptor behind a
  // ttg.mask result before the fused operation can recover its member pred.
  if (auto updateOp = dyn_cast<triton::amdgpu::UpdateTensorDescriptorOp>(op)) {
    Value memberPred = updateOp.getPred();
    Value desc = updateOp.getDesc();
    while (!memberPred) {
      auto baseUpdate =
          desc.getDefiningOp<triton::amdgpu::UpdateTensorDescriptorOp>();
      if (!baseUpdate)
        break;
      memberPred = baseUpdate.getPred();
      desc = baseUpdate.getDesc();
    }
    if (!memberPred && !desc.getDefiningOp<tt::MakeTensorDescOp>()) {
      updateOp.emitError(
          "cannot pipeline a tensor descriptor update whose inherited "
          "predicate is opaque");
      return nullptr;
    }

    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(rewriter, op->getLoc(),
                                          rewriter.getI32Type(), pred);
    if (!memberPred)
      memberPred = arith::ConstantIntOp::create(rewriter, op->getLoc(), 1, 32);
    Value narrowedPred =
        arith::AndIOp::create(rewriter, op->getLoc(), memberPred, predI32);
    updateOp.getPredMutable().assign(narrowedPred);
    return op;
  }
  // Fused TDM descriptors already carry their member predicates. Append a
  // pred-only update so the pipeline predicate is ANDed with (rather than
  // replacing) the effective member predicate.
  if (auto fusedOp =
          dyn_cast<triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(op)) {
    SmallVector<Value> memberPreds;
    memberPreds.reserve(fusedOp.getDescs().size());
    for (Value desc : fusedOp.getDescs()) {
      Value memberPred;
      while (
          auto update =
              desc.getDefiningOp<triton::amdgpu::UpdateTensorDescriptorOp>()) {
        if ((memberPred = update.getPred()))
          break;
        desc = update.getDesc();
      }
      if (!memberPred && !desc.getDefiningOp<tt::MakeTensorDescOp>()) {
        fusedOp.emitError(
            "cannot pipeline a fused TDM load whose descriptor predicate "
            "is opaque");
        return nullptr;
      }
      memberPreds.push_back(memberPred);
    }

    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(rewriter, op->getLoc(),
                                          rewriter.getI32Type(), pred);
    Value truePred;
    for (auto [desc, memberPred] :
         llvm::zip(fusedOp.getDescsMutable(), memberPreds)) {
      if (!memberPred) {
        if (!truePred)
          truePred =
              arith::ConstantIntOp::create(rewriter, op->getLoc(), 1, 32);
        memberPred = truePred;
      }
      Value narrowedPred =
          arith::AndIOp::create(rewriter, op->getLoc(), memberPred, predI32);
      auto narrowed = triton::amdgpu::UpdateTensorDescriptorOp::create(
          rewriter, op->getLoc(), desc.get().getType(), desc.get(),
          /*add_offsets=*/ValueRange{}, /*set_bounds=*/ValueRange{},
          narrowedPred);
      desc.assign(narrowed);
    }
    return op;
  }
  // Pure gather inherits pred from its descriptor; gate it the same way as the
  // copy by chaining a pred-only update_tensor_descriptor onto its descriptor.
  if (auto gatherOp = dyn_cast<triton::amdgpu::AsyncTDMGatherOp>(op)) {
    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(rewriter, op->getLoc(),
                                          rewriter.getI32Type(), pred);
    auto updated = triton::amdgpu::UpdateTensorDescriptorOp::create(
        rewriter, op->getLoc(), gatherOp.getDesc().getType(),
        gatherOp.getDesc(),
        /*add_offsets=*/ValueRange{}, /*set_bounds=*/ValueRange{},
        /*pred=*/predI32);
    gatherOp.getDescMutable().assign(updated.getResult());
    return op;
  } else if (isa<tt::DescriptorStoreLikeOpInterface>(op)) {
    auto loc = op->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, pred,
                                  /*withElseRegion=*/false);
    op->moveBefore(ifOp.thenYield());
    return ifOp;
  }
  if (isa<triton::amdgpu::AsyncTDMWait>(op))
    return op;
  return tt::wrapInMaskOp(rewriter, op, pred);
}
} // namespace

// Exposed for reuse by the decompose+modulo pipeline (change #4): deserialize
// the CoarseSchedule the caller serialized and run the general pipeline
// expander.
LogicalResult expandLoops(ModuleOp moduleOp) {
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
    bool tdmPredicationFailed = false;
    options.predicateFn = [&](RewriterBase &rewriter, Operation *op,
                              Value pred) {
      Operation *predicated = streamPredication(rewriter, op, pred);
      if (!predicated && isa<triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp,
                             triton::amdgpu::UpdateTensorDescriptorOp>(op))
        tdmPredicationFailed = true;
      return predicated;
    };
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

    if (failed(newForOp) && tdmPredicationFailed)
      return failure();
    if (failed(newForOp))
      continue;
  }

  tt::resolveMaskOp(moduleOp);
  return success();
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
      [](Operation *op) {
        return isa<triton::amdgpu::TDMOpInterface,
                   triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(op);
      },
      [](OpBuilder &b, Location loc, ValueRange operands,
         unsigned num) -> Operation * {
        return triton::amdgpu::AsyncTDMWait::create(b, loc, operands, num);
      });
}

struct PipelinePass : impl::TritonAMDGPUPipelineBase<PipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto targetFeatures = tt::amdgpu::TargetFeatures::fromModuleOp(moduleOp);
    if (moduleOp->hasAttr(tt::kSkipGenericPipelineAttrName) &&
        !targetFeatures.isGFX1250())
      return;

    lowerLoops(moduleOp, useAsyncCopy, usePingpong);
    if (failed(expandLoops(moduleOp)))
      return signalPassFailure();

    if (useAsyncCopy) {
      // Only asyncmark targets (CDNA3/CDNA4) need updateWaits here: their
      // lowering reads ttg.async_wait's `num` directly into wait.asyncmark(N),
      // and PR #9883 made UpdateAsyncWaitCount a no-op on those archs, so
      // without this call the pipeliner-authored num=0 would serialize the
      // SWP. Every other family keeps the prior combineRedundantWaitOps-only
      // path: their num is re-derived downstream by UpdateAsyncWaitCount.
      if (targetFeatures.isCDNA3() || targetFeatures.isCDNA4()) {
        mlir::triton::updateWaits(moduleOp);
      }
    }
    combineWaitOps(moduleOp, useAsyncCopy);

    tt::removePipeliningAttributes(moduleOp);
  }
};
} // namespace mlir
