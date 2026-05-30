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

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

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
  // Pick the target encoding from the first in-closure operand. Operands
  // that aren't in the closure get wrapped with tt.trans (to swap shape)
  // and then ttg.convert_layout (to match the target encoding).
  auto origTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!origTy || origTy.getRank() != 2)
    return nullptr;

  Attribute targetEnc;
  bool anyInClosure = false;
  for (size_t i = 0; i < transposedOperands.size(); ++i) {
    if (transposedOperands[i] != op->getOperand(i)) {
      anyInClosure = true;
      if (auto t = dyn_cast<RankedTensorType>(transposedOperands[i].getType()))
        targetEnc = t.getEncoding();
      break;
    }
  }
  if (!anyInClosure)
    return nullptr;

  // Result shape (swapped) and type using targetEnc.
  llvm::SmallVector<int64_t, 2> newShape{origTy.getDimSize(1),
                                          origTy.getDimSize(0)};
  auto newTy = RankedTensorType::get(newShape, origTy.getElementType(),
                                      targetEnc);

  // Bring each operand into (transposed shape, target encoding).
  llvm::SmallVector<Value, 4> coercedOperands;
  for (size_t i = 0; i < transposedOperands.size(); ++i) {
    Value v = transposedOperands[i];
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!vTy)
      return nullptr;
    if (v == op->getOperand(i)) {
      // Out-of-closure: wrap with tt.trans to swap shape.
      if (vTy.getRank() != 2)
        return nullptr;
      OpBuilder b(op);
      auto trans = triton::TransOp::create(
          b, op->getLoc(), v, b.getDenseI32ArrayAttr({1, 0}));
      v = trans.getResult();
      vTy = cast<RankedTensorType>(v.getType());
    }
    if (vTy.getEncoding() != targetEnc) {
      auto coercedTy = RankedTensorType::get(vTy.getShape(),
                                              vTy.getElementType(), targetEnc);
      OpBuilder b(op);
      v = ConvertLayoutOp::create(b, op->getLoc(), coercedTy, v).getResult();
    }
    coercedOperands.push_back(v);
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands(coercedOperands);
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

Value transformDot(OpBuilder &builder, Operation *op,
                   llvm::ArrayRef<Value> transposedOperands) {
  // Algebraic identity (out_t = out^T):
  //   out = dot(A, B, C)   out:[M,N], A:[M,K], B:[K,N], C:[M,N]
  //   out_t = dot(B^T, A^T, C^T)   out_t:[N,M]
  //                                 newA = B^T : [N,K]  with opIdx=0
  //                                 newB = A^T : [K,M]  with opIdx=1
  //                                 newC = C^T : [N,M]
  //
  // For each new-operand position:
  //   * If the original operand is already in closure (transposed via an
  //     upstream rule), reuse it.
  //   * Otherwise wrap the original with tt.trans to swap its shape.
  //   * Then insert ttg.convert_layout to bring it into the correct
  //     DotOperand<opIdx, parent=accEnc, kWidth=k> form. kWidth is
  //     preserved from the original A/B's DotOperand encoding.
  //
  // The result encoding (= newC's encoding, since dot's result type must
  // equal C's type) is determined by transposedOperands[2]: if C was in
  // closure, take its encoding as-is; otherwise insert tt.trans on C to
  // derive a swapped-shape encoding.
  if (transposedOperands.size() != 3)
    return nullptr;
  auto dotOp = dyn_cast<triton::DotOp>(op);
  if (!dotOp)
    return nullptr;

  Value origA = op->getOperand(0);
  Value origB = op->getOperand(1);
  Value origC = op->getOperand(2);

  auto origATy = dyn_cast<RankedTensorType>(origA.getType());
  auto origBTy = dyn_cast<RankedTensorType>(origB.getType());
  auto origCTy = dyn_cast<RankedTensorType>(origC.getType());
  if (!origATy || !origBTy || !origCTy)
    return nullptr;
  if (origATy.getRank() != 2 || origBTy.getRank() != 2 ||
      origCTy.getRank() != 2)
    return nullptr;

  // Preserve kWidth from the original DotOperand encodings of A and B.
  unsigned kWidthA = 0;
  unsigned kWidthB = 0;
  if (auto enc = dyn_cast<DotOperandEncodingAttr>(origATy.getEncoding()))
    kWidthA = enc.getKWidth();
  if (auto enc = dyn_cast<DotOperandEncodingAttr>(origBTy.getEncoding()))
    kWidthB = enc.getKWidth();

  // newC: either the in-closure value or trans(origC).
  Value newC = transposedOperands[2];
  if (newC == origC) {
    OpBuilder b(op);
    auto trans = triton::TransOp::create(
        b, op->getLoc(), origC, b.getDenseI32ArrayAttr({1, 0}));
    newC = trans.getResult();
  }
  auto newCTy = dyn_cast<RankedTensorType>(newC.getType());
  if (!newCTy)
    return nullptr;
  Attribute accEnc = newCTy.getEncoding();

  // Helper: bring `v` into (transposed shape, DotOperand encoding with
  // opIdx + kWidth + parent=accEnc).
  auto buildOperand = [&](Value v, Value orig, unsigned opIdx,
                          unsigned kWidth) -> Value {
    Value transposed = v;
    if (v == orig) {
      auto srcTy = dyn_cast<RankedTensorType>(orig.getType());
      if (!srcTy || srcTy.getRank() != 2)
        return nullptr;
      OpBuilder b(op);
      auto trans = triton::TransOp::create(
          b, op->getLoc(), orig, b.getDenseI32ArrayAttr({1, 0}));
      transposed = trans.getResult();
    }
    auto transTy = dyn_cast<RankedTensorType>(transposed.getType());
    if (!transTy)
      return nullptr;
    auto targetEnc = DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, accEnc, kWidth);
    if (transTy.getEncoding() == targetEnc)
      return transposed;
    auto targetTy = RankedTensorType::get(transTy.getShape(),
                                           transTy.getElementType(), targetEnc);
    OpBuilder b(op);
    return ConvertLayoutOp::create(b, op->getLoc(), targetTy, transposed)
        .getResult();
  };

  // newA = (orig B at opIdx=0 in new dot); newB = (orig A at opIdx=1).
  Value newA = buildOperand(transposedOperands[1], origB, 0, kWidthB);
  Value newB = buildOperand(transposedOperands[0], origA, 1, kWidthA);
  if (!newA || !newB)
    return nullptr;

  // Result type: shape-swapped, encoding = accEnc (matches newC).
  llvm::SmallVector<int64_t, 2> newShape{origCTy.getDimSize(1),
                                          origCTy.getDimSize(0)};
  auto newResTy = RankedTensorType::get(newShape, origCTy.getElementType(),
                                         accEnc);

  auto newDot = triton::DotOp::create(
      builder, op->getLoc(), newResTy, newA, newB, newC,
      dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
  return newDot.getResult();
}

Value transformConvertLayout(OpBuilder &builder, Operation *op,
                             llvm::ArrayRef<Value> transposedOperands) {
  auto cvt = dyn_cast<ConvertLayoutOp>(op);
  if (!cvt || transposedOperands.size() != 1)
    return nullptr;
  Value newSrc = transposedOperands.front();
  // New result type: shape-swapped from original cvt result, encoding
  // preserved (cleanup passes can adjust later).
  auto origResTy = dyn_cast<RankedTensorType>(cvt.getType());
  if (!origResTy || origResTy.getRank() != 2)
    return nullptr;
  llvm::SmallVector<int64_t, 2> newShape{origResTy.getDimSize(1),
                                          origResTy.getDimSize(0)};
  auto newResTy = RankedTensorType::get(newShape, origResTy.getElementType(),
                                         origResTy.getEncoding());
  auto newCvt = ConvertLayoutOp::create(builder, op->getLoc(), newResTy, newSrc);
  return newCvt.getResult();
}

//===----------------------------------------------------------------------===//
// Rule: ttg.convert_layout (ConvertLayoutAdjust).
//
// A convert_layout consuming an in-closure value: build a new convert
// from the transposed src type to the transposed dst type.
//
// Original:   Y = convert(X : T_src) -> T_dst
// Transposed: Y' = convert(X' : T_src^T) -> T_dst^T
//             where T_src^T / T_dst^T have shape swapped, encoding
//             adjusted via the dialect's layout-inference machinery.
//
// D6: rule classification only. Transform factory is stubbed.
//===----------------------------------------------------------------------===//

bool matchConvertLayout(Operation *op, unsigned /*opIdx*/) {
  return isa<ConvertLayoutOp>(op);
}

//===----------------------------------------------------------------------===//
// Rule: scf.yield (SCFCarryRetype).
//
// When scf.yield's value at position k is in the closure, the parent
// scf.for's iter_arg at position k carries a transposed value across
// iterations. Commit phase will:
//   * retype iterArg(k) to the transposed type;
//   * insert tt.trans on the init_arg at position k (loop entry);
//   * the iter_arg's downstream uses are already classified by the
//     engine (extended DFS at SCFCarryRetype).
//===----------------------------------------------------------------------===//

bool matchSCFYield(Operation *op, unsigned /*opIdx*/) {
  return isa<scf::YieldOp>(op) && isa_and_nonnull<scf::ForOp>(op->getParentOp());
}

//===----------------------------------------------------------------------===//
// Rule: shared-mem boundary (ttg.local_alloc / ttg.local_load).
//
// Shared-memory ops (MemDescType operands/results) can't carry a
// "transposed orientation" through their LDS chains in a layout-neutral
// way; we insert a tt.trans before them and stop the propagation.
//
// D7: explicit classification only. Behaviour is identical to the
// conservative-default BoundaryInsert; the distinction is for cost-
// model accounting (an LDS-bouncing boundary is more expensive than
// a tt.return boundary, which a future Score phase can charge for).
//===----------------------------------------------------------------------===//

bool matchSharedMem(Operation *op, unsigned /*opIdx*/) {
  return isa<LocalAllocOp, LocalLoadOp>(op);
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
    {"convert-layout", TransposeRuleKind::ConvertLayoutAdjust,
     &matchConvertLayout, &transformConvertLayout},
    {"shared-mem", TransposeRuleKind::SharedMemBoundary, &matchSharedMem,
     /*transform=*/nullptr},
    {"scf-yield", TransposeRuleKind::SCFCarryRetype, &matchSCFYield,
     /*transform=*/nullptr},
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
      // SharedMemBoundary explicitly behaves like BoundaryInsert (and
      // is counted in boundaryOps for cost-model accounting), but it's
      // distinguished from the conservative-default fallthrough.
      if (rule->kind == TransposeRuleKind::SharedMemBoundary ||
          rule->kind == TransposeRuleKind::BoundaryInsert) {
        plan.boundaryOps.push_back({user, opIdx});
        continue;
      }
      plan.ops.push_back(user);
      plan.ruleFor[user] = rule;

      // SCFCarryRetype: also walk the parent scf.for's iter_arg at the
      // corresponding position so its in-loop uses get classified. The
      // commit phase will retype both the iter_arg and the initial
      // value (with a tt.trans inserted on the init).
      if (rule->kind == TransposeRuleKind::SCFCarryRetype) {
        auto yieldOp = cast<scf::YieldOp>(user);
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          if (opIdx < forOp.getRegionIterArgs().size()) {
            Value iterArg = forOp.getRegionIterArgs()[opIdx];
            worklist.push_back(iterArg);
          }
        }
        continue;
      }

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
  // D9: real commit engine. Lazy-root variant: insert a tt.trans on each
  // annotated root's result (no algebraic flip yet), then materialize the
  // downstream plan via rule transforms; insert boundary back-trans;
  // replaceAllUsesWith + erase originals. SCFCarryRetype is not yet
  // commit-implemented (its transform stays nullptr -> commit aborts on
  // plans that include it). DotFlip and ConvertLayoutAdjust are real.
  if (plan.empty() || plan.rejected())
    return;

  llvm::DenseMap<Value, Value> seedMap;
  llvm::SmallVector<Operation *> createdOps;
  llvm::DenseSet<Operation *> rootSet(plan.roots.begin(), plan.roots.end());

  auto rollback = [&]() {
    for (Operation *op : llvm::reverse(createdOps)) {
      if (op->use_empty())
        op->erase();
    }
  };

  // Lazy root: tt.trans on each root's result.
  for (Operation *root : plan.roots) {
    if (root->getNumResults() != 1) {
      rollback();
      return;
    }
    Value rootRes = root->getResult(0);
    auto rootTy = dyn_cast<RankedTensorType>(rootRes.getType());
    if (!rootTy || rootTy.getRank() != 2) {
      rollback();
      return;
    }
    // Insert tt.trans just after the root op.
    OpBuilder builder(root->getBlock(),
                      std::next(Block::iterator(root)));
    auto trans = triton::TransOp::create(builder, root->getLoc(), rootRes,
                                          builder.getDenseI32ArrayAttr({1, 0}));
    createdOps.push_back(trans);
    seedMap[rootRes] = trans.getResult();
  }

  // Topo-sort plan.ops, materialize each non-root rule.
  llvm::SetVector<Operation *> opSet(plan.ops.begin(), plan.ops.end());
  auto sorted = mlir::topologicalSort(opSet);

  for (Operation *op : sorted) {
    if (rootSet.contains(op))
      continue;
    auto ruleIt = plan.ruleFor.find(op);
    if (ruleIt == plan.ruleFor.end() || !ruleIt->second)
      continue;
    const TransposeRule *rule = ruleIt->second;
    if (!rule->transform) {
      // Rule has no commit-side transform (e.g. SCFCarryRetype, not yet
      // implemented). Abort the whole plan.
      rollback();
      return;
    }
    if (op->getNumResults() != 1) {
      rollback();
      return;
    }

    llvm::SmallVector<Value, 4> operands;
    for (Value v : op->getOperands()) {
      auto it = seedMap.find(v);
      operands.push_back(it != seedMap.end() ? it->second : v);
    }
    OpBuilder builder(op);
    Value newResult = rule->transform(builder, op, operands);
    if (!newResult) {
      rollback();
      return;
    }
    if (auto *defOp = newResult.getDefiningOp())
      createdOps.push_back(defOp);
    seedMap[op->getResult(0)] = newResult;
  }

  // Boundary back-trans inserts. For each (user, opIdx), look up the
  // operand in seedMap; if found (in-closure), insert tt.trans to bring
  // the value back to original orientation, and rewire.
  for (auto &boundary : plan.boundaryOps) {
    Operation *user = boundary.first;
    unsigned opIdx = boundary.second;
    Value orig = user->getOperand(opIdx);
    auto it = seedMap.find(orig);
    if (it == seedMap.end())
      continue;
    auto srcTy = dyn_cast<RankedTensorType>(it->second.getType());
    if (!srcTy || srcTy.getRank() != 2) {
      rollback();
      return;
    }
    OpBuilder builder(user);
    auto backTrans = triton::TransOp::create(
        builder, user->getLoc(), it->second,
        builder.getDenseI32ArrayAttr({1, 0}));
    createdOps.push_back(backTrans);
    user->setOperand(opIdx, backTrans.getResult());
  }

  // ReplaceAllUsesWith for each non-root seedMap entry, then erase the
  // originals. Roots are left in place (the inserted lazy tt.trans
  // depends on them).
  llvm::SmallVector<Operation *> toErase;
  for (auto &kv : seedMap) {
    Value orig = kv.first;
    Value newV = kv.second;
    Operation *defOp = orig.getDefiningOp();
    if (!defOp || rootSet.contains(defOp))
      continue;
    orig.replaceAllUsesWith(newV);
    toErase.push_back(defOp);
  }
  // Erase in reverse-topological order (uses before defs).
  for (Operation *op : llvm::reverse(toErase)) {
    if (op->use_empty())
      op->erase();
  }

  // Strip annotation from roots so the pass is idempotent.
  for (Operation *root : plan.roots)
    root->removeAttr(kTransposePropagateRootAttrName);
}

} // namespace mlir::triton::gpu
