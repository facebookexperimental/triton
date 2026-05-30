//===- TransposePropagate.cpp ----------------------------------*- C++ -*-===//
//
// D0 scaffolding: empty rule registry + stub plan/score/commit. The pass
// driver in TransposePropagatePass.cpp invokes these but with no rules
// registered the plan is always empty and commit is a no-op. This commit
// exists to land the file layout, header API, and pass registration so
// later commits can fill in the rules incrementally.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/TransposePropagate.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

llvm::ArrayRef<TransposeRule> getDefaultTransposeRules() {
  // D0: empty registry. Future commits populate this.
  return {};
}

TransposePlan planTransposePropagation(Operation *root) {
  TransposePlan plan;
  if (!root)
    return plan;
  // D0: no rules registered -> nothing to plan. Future D2+ commits walk
  // the use chain and populate plan.ops via the registry.
  return plan;
}

TransposeScore scoreTransposePlan(const TransposePlan &plan) {
  TransposeScore score;
  // D0: stub. Future D9 commit adds dry-run + real cost.
  score.value = 0.0;
  score.feasible = false;
  return score;
}

void commitTransposePlan(const TransposePlan &plan) {
  // D0: no-op. Future D5+ commits materialize the plan.
  (void)plan;
}

} // namespace mlir::triton::gpu
