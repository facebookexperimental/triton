//===- TransposePropagate.h -------------------------------------*- C++ -*-===//
//
// Algebraic transpose propagation for chained-dot patterns.
//
// Annotate a `tt.dot` with `tt.transpose_propagate_root` (UnitAttr). The
// pass treats the dot's output as if it were the mathematical transpose of
// the original (DotFlip: swap A and B, insert/remove `tt.trans` as needed)
// and propagates that "transposed-orientation" invariant forward through
// the use chain. Per-op rules decide whether each downstream op can be
// Rewritten (elementwise: output type just becomes the transpose, no
// arithmetic change), undergo a ReduceAxisSwap / BroadcastSwap /
// ExpandDimsSwap, fire a TransElide (consume `tt.trans` of a transposed
// value), fold into another DotFlip, undergo a ConvertLayoutAdjust, hit
// a SharedMemBoundary / SCFCarryRetype, or trigger BoundaryInsert (insert
// a real `tt.trans` and stop propagation here). Reject bails the whole
// plan.
//
// Three pure phases:
//   * plan  : DFS the use chain, classify via rule registry. No IR change.
//   * score : Cost-model the plan. No IR change.
//   * commit: Topo-sort plan actions, apply factories, retype scf carries,
//             insert boundary trans ops.
//
// Distinct from `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`
// (which propagates layout *encodings* — register partitions); this pass
// propagates *mathematical transpose orientation* through ops, changing
// shapes and reshape/reduce axes accordingly.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRANSPOSEPROPAGATE_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRANSPOSEPROPAGATE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu {

/// Kinds of per-op rewrites the propagation engine understands. Each one
/// preserves the invariant "the value produced by the rewritten op is the
/// mathematical transpose of what the original op would have produced".
enum class TransposeRuleKind {
  /// Elementwise (e.g. arith.mulf, math.exp2, arith.truncf). Clone the op
  /// with new (transposed) operand types; result type is the transpose of
  /// the original.
  Rewrite,
  /// tt.reduce: flip the `axis` attribute. Result is 1-D, mathematically
  /// unchanged. The 1-D output is recursed-on so 1-D consumers can be
  /// classified.
  ReduceAxisSwap,
  /// tt.expand_dims: flip the `axis` attribute. Result is 2-D and *is*
  /// in transposed orientation -- pushes to the closure.
  ExpandDimsSwap,
  /// tt.broadcast: swap the result shape (dims swap with operand).
  BroadcastSwap,
  /// tt.trans on a transposed value: emit no new op; the un-transposed
  /// value flows through.
  TransElide,
  /// Downstream tt.dot: swap A and B operands, optionally insert/remove
  /// tt.trans on the un-transposed operand. Recurse on the new result.
  DotFlip,
  /// ttg.convert_layout: rewrite with a transposed src encoding; output
  /// encoding adjusted to match.
  ConvertLayoutAdjust,
  /// ttg.local_alloc / ttg.local_load: insert tt.trans before the op,
  /// keep the shared-mem chain in original orientation.
  SharedMemBoundary,
  /// scf.for iter_arg whose yielded value is in the propagation closure:
  /// retype the iter_arg + initial value (insert tt.trans on init).
  SCFCarryRetype,
  /// Conservative fallback for unhandled ops: insert tt.trans at the
  /// boundary and don't recurse past this op.
  BoundaryInsert,
  /// Unsupported op kind. Bail the whole plan.
  Reject,
};

/// One rule entry in the registry. Operates on a single `(op, transposed
/// operand index)` pair.
struct TransposeRule {
  llvm::StringRef name;
  TransposeRuleKind kind;
  /// Classifier: does this rule apply to `op` when the transposed value
  /// flows in at operand `opIdx`?
  bool (*match)(Operation *op, unsigned opIdx);
  /// Factory (nullable for BoundaryInsert/Reject/SCFCarryRetype which are
  /// handled by the engine itself). Builds the new op given the already-
  /// transposed operand values. Returns the result Value (logically
  /// transposed) on success, or null on failure.
  Value (*transform)(OpBuilder &builder, Operation *op,
                     llvm::ArrayRef<Value> transposedOperands);
};

/// Result of the plan phase. Empty `actions` + nonempty `rejectedAt`
/// indicates a rejected plan.
struct TransposePlan {
  /// Root dots (annotation triggers). Multiple roots per FuncOp allowed.
  llvm::SmallVector<Operation *> roots;
  /// Plan op -> matched rule. Ordered for deterministic commit.
  llvm::SmallVector<Operation *> ops;
  llvm::DenseMap<Operation *, const TransposeRule *> ruleFor;
  /// Operand-side boundary sites: (consumer, opIdx) needing a tt.trans
  /// inserted before the consumer consumes the transposed value.
  llvm::SmallVector<std::pair<Operation *, unsigned>> boundaryOps;
  /// On Reject, the offending op + a human-readable reason.
  Operation *rejectedAt = nullptr;
  llvm::StringRef rejectReason;

  bool empty() const { return ops.empty() && boundaryOps.empty(); }
  bool rejected() const { return rejectedAt != nullptr; }
};

/// Cost-model output. v1 stub: feasible flag only.
struct TransposeScore {
  double value = 0.0;
  bool feasible = false;
};

/// Public API.

/// All rules registered for the propagation engine. Pointers are stable
/// (process-lifetime registry).
llvm::ArrayRef<TransposeRule> getDefaultTransposeRules();

/// Phase 1: classify ops reachable from each annotated root via forward
/// DFS. Returns a plan; on Reject, the plan's `rejected()` is true.
TransposePlan planTransposePropagation(Operation *root);

/// Phase 2: cost-model the plan. v1 stub feasibility.
TransposeScore scoreTransposePlan(const TransposePlan &plan);

/// Phase 3: apply the plan. Mutates IR. Caller should only invoke when
/// `score.feasible == true`.
void commitTransposePlan(const TransposePlan &plan);

/// Dry-run safety net: clones `funcOp` into a temporary module, re-runs
/// plan + commit on the clone, calls the MLIR verifier on the temporary
/// module, then discards. Returns true iff the post-commit IR verified
/// cleanly. Use this before invoking `commitTransposePlan` on the real
/// IR -- if it returns false, the real funcOp should be left untouched.
bool dryRunCommit(Operation *funcOp);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRANSPOSEPROPAGATE_H_
