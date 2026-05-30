//===- TransposePropagate.cpp ----------------------------------*- C++ -*-===//
//
// D1: real engine. Rule registry still empty; the DFS walks every annotated
// root's use chain and classifies each user via the registry, defaulting
// unknown ops to BoundaryInsert (insert tt.trans, don't recurse). Score
// and commit are still stubs.
//
// Engine invariant: every entry in plan.ops or plan.boundaryOps represents
// a use of a value that is logically transposed (relative to what the
// original IR would have produced). plan.ops are ops we will rewrite;
// plan.boundaryOps are sites where we insert a real tt.trans to undo the
// transposition before a consumer that can't handle it.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/TransposePropagate.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::gpu {

namespace {

//===----------------------------------------------------------------------===//
// Rule: elementwise (arith + math + truncf/extf).
//
// Original:    C[i,j] = f(A[i,j], B[i,j], ...)
// Rewritten:   C'[j,i] = f(A'[j,i], B'[j,i], ...)  where X' = X^T
// Therefore:   C'[j,i] = f(A[i,j], B[i,j], ...) = C[i,j]
// i.e.         C' = C^T  ✓
//
// The transform clones the op with the same op kind and discardable
// attrs; result type is the original result type with shape reversed
// (encoding preserved -- a later cleanup will pick that up via the
// existing AccelerateAMDMatmul + RemoveLayoutConversions paths).
//===----------------------------------------------------------------------===//

bool matchElementwise(Operation *op, unsigned /*opIdx*/) {
  // Recognize commonly-occurring scalar/elementwise ops on tensors. Limit
  // to single-result, ranked-tensor result, and ops that have the same
  // operands-and-result shape semantics. Conservative list rather than a
  // generic trait scan -- avoids accidentally matching SCF ops, ops with
  // region-side effects, etc.
  if (!op || op->getNumResults() != 1)
    return false;
  auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resTy)
    return false;
  if (resTy.getRank() != 2)
    return false;
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
             arith::MinimumFOp, arith::NegFOp, arith::TruncFOp,
             arith::ExtFOp, arith::SelectOp, math::ExpOp, math::Exp2Op,
             math::LogOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
             math::FmaOp>(op);
}

Value transformElementwise(OpBuilder &builder, Operation *op,
                           llvm::ArrayRef<Value> transposedOperands) {
  // Result type: same encoding, swapped shape.
  auto origTy = cast<RankedTensorType>(op->getResult(0).getType());
  auto origShape = origTy.getShape();
  if (origShape.size() != 2)
    return nullptr;
  llvm::SmallVector<int64_t, 2> newShape{origShape[1], origShape[0]};
  auto newTy = RankedTensorType::get(newShape, origTy.getElementType(),
                                     origTy.getEncoding());

  // Build the new op by cloning with operands replaced + result type
  // replaced. cloneWithoutRegions preserves the discardable attrs.
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(transposedOperands);
  state.addTypes({newTy});
  state.attributes = op->getAttrs();
  Operation *cloned = builder.create(state);
  return cloned->getResult(0);
}

const TransposeRule kDefaultRules[] = {
    {"elementwise", TransposeRuleKind::Rewrite, &matchElementwise,
     &transformElementwise},
};

} // namespace

llvm::ArrayRef<TransposeRule> getDefaultTransposeRules() {
  return llvm::ArrayRef<TransposeRule>(kDefaultRules);
}

namespace {

const TransposeRule *findRule(Operation *op, unsigned opIdx) {
  for (const TransposeRule &rule : getDefaultTransposeRules()) {
    if (rule.match && rule.match(op, opIdx))
      return &rule;
  }
  return nullptr;
}

} // namespace

TransposePlan planTransposePropagation(Operation *root) {
  TransposePlan plan;
  if (!root)
    return plan;

  // Step 1: find every annotated tt.dot under `root`. Multiple roots per
  // FuncOp is supported (HSTU-bwd has 2: qk and dqk).
  root->walk([&](Operation *op) {
    if (!isa<DotOp, DotScaledOp>(op))
      return;
    if (op->hasAttr(kTransposePropagateRootAttrName))
      plan.roots.push_back(op);
  });
  if (plan.roots.empty())
    return plan;

  // Step 2: forward DFS from each root's result through its use chain.
  // visited prevents revisiting an op (each op gets one classification).
  llvm::DenseSet<Operation *> visited;
  llvm::SmallVector<Value, 16> worklist;

  // Each root's *result* is the first transposed value in the closure --
  // the engine treats it as if the root dot has already been DotFlipped
  // (commit phase will do the actual flip). The root op itself is added
  // to plan.ops so commit can find and rewrite it.
  for (Operation *rootOp : plan.roots) {
    if (rootOp->getNumResults() != 1)
      continue;
    visited.insert(rootOp);
    plan.ops.push_back(rootOp);
    // The root's rule entry is nullptr (engine recognises it as a root
    // via membership in plan.roots, not via the rule registry).
    plan.ruleFor[rootOp] = nullptr;
    worklist.push_back(rootOp->getResult(0));
  }

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second)
        continue;

      unsigned opIdx = use.getOperandNumber();
      const TransposeRule *rule = findRule(user, opIdx);
      if (!rule || rule->kind == TransposeRuleKind::Reject) {
        if (rule && rule->kind == TransposeRuleKind::Reject) {
          plan.rejectedAt = user;
          plan.rejectReason = rule->name;
          return plan;
        }
        // Conservative default: BoundaryInsert. Insert tt.trans before
        // this consumer at this operand index, don't recurse.
        plan.boundaryOps.push_back({user, opIdx});
        continue;
      }
      plan.ops.push_back(user);
      plan.ruleFor[user] = rule;

      // Recurse on the user's transposed-output value(s). For most rule
      // kinds this is user.getResult(0). TransElide rules are recorded
      // but produce no new "transposed" value (they undo). For DotFlip,
      // BroadcastSwap, ExpandDimsSwap, Rewrite, ReduceAxisSwap: the
      // result is in the closure.
      if (rule->kind != TransposeRuleKind::TransElide &&
          rule->kind != TransposeRuleKind::BoundaryInsert &&
          user->getNumResults() == 1) {
        worklist.push_back(user->getResult(0));
      }
    }
  }
  return plan;
}

TransposeScore scoreTransposePlan(const TransposePlan &plan) {
  TransposeScore score;
  score.value = 0.0;
  // D1: stub. Feasibility decided by D9's dry-run.
  score.feasible = !plan.empty() && !plan.rejected();
  return score;
}

void commitTransposePlan(const TransposePlan &plan) {
  // D1: no-op. D5+ materializes the plan.
  (void)plan;
}

} // namespace mlir::triton::gpu
