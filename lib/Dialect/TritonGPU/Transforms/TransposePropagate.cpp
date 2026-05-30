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

//===----------------------------------------------------------------------===//
// Rule: tt.reduce (ReduceAxisSwap).
//
// Original input:    X : <MxN>             (in non-transposed orientation)
//                    Y[i] = reduce_j X[i, j]   where axis = 1
// Transposed input:  X' : <NxM>            (X' = X^T, in closure)
//                    Y'[i] = reduce_j X'[j, i] where axis = 0
//                          = reduce_j X[i, j] = Y[i]
// Output Y' == Y (1-D vector, same elements). Encoding may flip
// (slice<dim=1> vs slice<dim=0>), but the math is unchanged.
//===----------------------------------------------------------------------===//

bool matchReduce(Operation *op, unsigned /*opIdx*/) {
  auto reduceOp = dyn_cast<triton::ReduceOp>(op);
  if (!reduceOp)
    return false;
  // Engine assumes single-input single-result reduces over rank-2 input.
  if (reduceOp.getNumOperands() != 1 || reduceOp.getNumResults() != 1)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(reduceOp.getOperand(0).getType());
  if (!srcTy || srcTy.getRank() != 2)
    return false;
  // axis must be 0 or 1.
  unsigned axis = reduceOp.getAxis();
  return axis == 0 || axis == 1;
}

Value transformReduce(OpBuilder &builder, Operation *op,
                      llvm::ArrayRef<Value> transposedOperands) {
  auto reduceOp = cast<triton::ReduceOp>(op);
  uint32_t newAxis = reduceOp.getAxis() ^ 1u;
  auto newReduce = triton::ReduceOp::create(builder, op->getLoc(),
                                            ValueRange(transposedOperands),
                                            newAxis, /*ordering=*/StringAttr{});
  newReduce.getRegion().takeBody(reduceOp.getRegion());
  return newReduce.getResult().front();
}

//===----------------------------------------------------------------------===//
// Rule: tt.expand_dims (ExpandDimsSwap).
//
// Original:    X : <M>          (1-D)
//              Y : <Mx1>        when axis=1   (Y[i,0] = X[i])
// Transposed:  X' : <M>         (still 1-D, no transpose on 1-D)
//              Y' : <1xM>       when axis=0   (Y'[0,i] = X'[i] = X[i])
// Output Y' = Y^T.
//===----------------------------------------------------------------------===//

bool matchExpandDims(Operation *op, unsigned /*opIdx*/) {
  auto expandOp = dyn_cast<triton::ExpandDimsOp>(op);
  if (!expandOp)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
  return srcTy && srcTy.getRank() == 1;
}

Value transformExpandDims(OpBuilder &builder, Operation *op,
                          llvm::ArrayRef<Value> transposedOperands) {
  auto expandOp = cast<triton::ExpandDimsOp>(op);
  unsigned newAxis = expandOp.getAxis() ^ 1u;
  auto newExpand = triton::ExpandDimsOp::create(
      builder, op->getLoc(), transposedOperands.front(), newAxis);
  return newExpand.getResult();
}

//===----------------------------------------------------------------------===//
// Rule: tt.broadcast (BroadcastSwap).
//
// Original:    X : <Mx1>, Y : <MxN>          Y[i,j] = X[i,0]
// Transposed:  X' : <1xM>, Y' : <NxM>        Y'[j,i] = X'[0,i] = X[i,0]
// So Y'[j,i] = Y[i,j], i.e. Y' = Y^T.
//===----------------------------------------------------------------------===//

bool matchBroadcast(Operation *op, unsigned /*opIdx*/) {
  auto bcastOp = dyn_cast<triton::BroadcastOp>(op);
  if (!bcastOp)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(bcastOp.getSrc().getType());
  auto resTy = dyn_cast<RankedTensorType>(bcastOp.getType());
  return srcTy && resTy && srcTy.getRank() == 2 && resTy.getRank() == 2;
}

Value transformBroadcast(OpBuilder &builder, Operation *op,
                         llvm::ArrayRef<Value> transposedOperands) {
  auto bcastOp = cast<triton::BroadcastOp>(op);
  auto srcTy = cast<RankedTensorType>(transposedOperands.front().getType());
  auto resTy = cast<RankedTensorType>(bcastOp.getType());
  // Swap the result shape.
  auto resShape = resTy.getShape();
  llvm::SmallVector<int64_t, 2> newShape{resShape[1], resShape[0]};
  // Encoding for new result: same parent encoding as original output, but
  // the broadcast op preserves the operand's encoding type. Use the
  // transposed operand's encoding (it was set by the upstream rule).
  auto newResTy = RankedTensorType::get(newShape, resTy.getElementType(),
                                        srcTy.getEncoding());
  auto newBcast = triton::BroadcastOp::create(
      builder, op->getLoc(), newResTy, transposedOperands.front());
  return newBcast.getResult();
}

//===----------------------------------------------------------------------===//
// Rule: tt.trans (TransElide).
//
// Original:    Y = tt.trans(X)  (order=[1,0] on rank-2)  =>  Y = X^T
// Transposed:  X' = X^T  is already in closure.
//              Then trans(X') = (X^T)^T = X.
// So the elided result Y is the *original* X (not in transposed
// closure). transformTrans returns the operand directly; the engine
// does not recurse on TransElide outputs, so downstream uses of Y are
// untouched.
//===----------------------------------------------------------------------===//

bool matchTrans(Operation *op, unsigned /*opIdx*/) {
  auto transOp = dyn_cast<triton::TransOp>(op);
  if (!transOp)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(transOp.getSrc().getType());
  if (!srcTy || srcTy.getRank() != 2)
    return false;
  // Only [1,0] (i.e., a true 2-D matrix transpose) cancels with a
  // closure-transposed value. Other orders aren't 2-D->2-D transposes.
  auto order = transOp.getOrder();
  return order.size() == 2 && order[0] == 1 && order[1] == 0;
}

Value transformTrans(OpBuilder & /*builder*/, Operation * /*op*/,
                     llvm::ArrayRef<Value> transposedOperands) {
  // Return the operand verbatim; commit will replaceAllUsesWith on the
  // trans's result. Downstream consumers receive the un-transposed value.
  return transposedOperands.front();
}

//===----------------------------------------------------------------------===//
// Rule: tt.dot (DotFlip).
//
// Algebraic identity for a downstream dot consuming a transposed value:
//
//   Original:  out = dot(A, B, C)        out:[M,N], A:[M,K], B:[K,N], C:[M,N]
//   Target:    out_t = dot(B^T, A^T, C_t)   where out_t = out^T
//   Proof:     out^T = (A·B + C)^T = B^T · A^T + C^T
//
// In the propagation engine: when an in-closure value flows into a dot at
// some operand index, the entire dot is flipped. Operands NOT in the
// closure get wrapped with tt.trans (commit-phase responsibility). The
// dot's result is placed in the closure with swapped shape.
//
// D5: rule classification only. Transform factory is stubbed; the actual
// rewrite lands in a later commit when the commit engine is wired up.
//===----------------------------------------------------------------------===//

bool matchDot(Operation *op, unsigned /*opIdx*/) {
  // Recognise both plain tt.dot and tt.dot_scaled. For now both are
  // structurally similar from the engine's perspective.
  if (!isa<triton::DotOp, triton::DotScaledOp>(op))
    return false;
  if (op->getNumResults() != 1)
    return false;
  return isa<RankedTensorType>(op->getResult(0).getType());
}

Value transformDot(OpBuilder & /*builder*/, Operation * /*op*/,
                   llvm::ArrayRef<Value> /*transposedOperands*/) {
  // D5: stub. The real DotFlip materialisation lands in a follow-up
  // commit alongside the commit engine + boundary trans insertion.
  return nullptr;
}

const TransposeRule kDefaultRules[] = {
    {"elementwise", TransposeRuleKind::Rewrite, &matchElementwise,
     &transformElementwise},
    {"reduce", TransposeRuleKind::ReduceAxisSwap, &matchReduce,
     &transformReduce},
    {"expand-dims", TransposeRuleKind::ExpandDimsSwap, &matchExpandDims,
     &transformExpandDims},
    {"broadcast", TransposeRuleKind::BroadcastSwap, &matchBroadcast,
     &transformBroadcast},
    {"trans-elide", TransposeRuleKind::TransElide, &matchTrans,
     &transformTrans},
    {"dot-flip", TransposeRuleKind::DotFlip, &matchDot, &transformDot},
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
