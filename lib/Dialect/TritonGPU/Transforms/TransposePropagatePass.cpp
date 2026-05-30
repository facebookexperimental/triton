//===- TransposePropagatePass.cpp ------------------------------*- C++ -*-===//
//
// Driver for the algebraic transpose propagation pass. Walks every FuncOp
// in the module, scans for tt.dot ops annotated with
// tt.transpose_propagate_root, and runs plan -> score -> commit for each
// annotated root. D0 scaffolding: with no rules registered yet, the
// pass is a structural no-op.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TransposePropagate.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUTRANSPOSEPROPAGATE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct TransposePropagatePass
    : public impl::TritonGPUTransposePropagateBase<TransposePropagatePass> {
  using impl::TritonGPUTransposePropagateBase<
      TransposePropagatePass>::TritonGPUTransposePropagateBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    bool verbose = std::getenv("TRITON_TRANSPOSE_PROPAGATE_DEBUG") != nullptr;
    bool autoAnnotateFirstDot = std::getenv(
        "TRITON_TRANSPOSE_PROPAGATE_AUTOANNOTATE_FIRST_DOT") != nullptr;
    m.walk([&](mlir::triton::FuncOp funcOp) {
      // Optional zero-source-change experimentation knob: if no dot in
      // the func is annotated and the env var is set, annotate the
      // FIRST tt.dot (in program order) as a propagation root. Useful
      // for benching the pass on kernels we don't want to modify.
      if (autoAnnotateFirstDot) {
        bool anyAnnotated = false;
        Operation *firstDot = nullptr;
        funcOp.walk([&](mlir::triton::DotOp dotOp) {
          if (dotOp->hasAttr(kTransposePropagateRootAttrName))
            anyAnnotated = true;
          if (!firstDot)
            firstDot = dotOp.getOperation();
        });
        if (!anyAnnotated && firstDot) {
          OpBuilder b(firstDot->getContext());
          firstDot->setAttr(kTransposePropagateRootAttrName, b.getUnitAttr());
          if (verbose)
            firstDot->emitRemark()
                << "transpose-propagate: auto-annotated first dot as root";
        }
      }

      TransposePlan plan = planTransposePropagation(funcOp);
      if (plan.rejected()) {
        if (verbose && plan.rejectedAt)
          plan.rejectedAt->emitRemark()
              << "transpose-propagate: rejected: " << plan.rejectReason;
        return;
      }
      if (plan.empty())
        return;
      if (verbose)
        funcOp.emitRemark()
            << "transpose-propagate: plan roots=" << plan.roots.size()
            << " ops=" << plan.ops.size()
            << " boundary=" << plan.boundaryOps.size();
      TransposeScore score = scoreTransposePlan(plan);
      if (!score.feasible)
        return;
      // Dry-run guard: clone funcOp, simulate commit, verify, discard.
      // Skip the real commit if the simulated IR doesn't verify.
      if (!dryRunCommit(funcOp)) {
        if (verbose)
          funcOp.emitRemark()
              << "transpose-propagate: dry-run failed, skipping commit";
        return;
      }
      commitTransposePlan(plan);
    });
  }
};

} // namespace

} // namespace mlir::triton::gpu
