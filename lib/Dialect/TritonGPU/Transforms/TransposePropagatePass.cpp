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
    m.walk([](mlir::triton::FuncOp funcOp) {
      // D0: skeleton. Later commits will:
      //   1. Locate every tt.dot with kTransposePropagateRootAttrName.
      //   2. For each: plan -> score -> commit (with dry-run safety net).
      //   3. Strip the attribute after a successful commit.
      // For now, just call planTransposePropagation so the function
      // is exercised by tests; with no rules registered it returns
      // an empty plan and we bail.
      TransposePlan plan = planTransposePropagation(funcOp);
      if (plan.empty())
        return;
      TransposeScore score = scoreTransposePlan(plan);
      if (!score.feasible)
        return;
      commitTransposePlan(plan);
    });
  }
};

} // namespace

} // namespace mlir::triton::gpu
