#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONUPLIFTWHILETOFOR
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Rewrites a countable `scf.while` into the equivalent `scf.for`. The matching
// / rewrite logic is MLIR's `scf::upliftWhileToForLoop`, which only succeeds
// when the loop is provably a counting loop (single `cmpi slt/sgt` in the
// before-region against a dominating bound, and an `addi` induction-var update
// with a loop-invariant step). Loops it cannot prove countable -- e.g. the
// dynamic scheduler's atomic advance or the CLC hardware condition -- are left
// untouched.
//
// Unlike the stock `populateUpliftWhileToForPatterns` driver, this pass calls
// the utility directly so it can capture the resulting `scf.for` and copy the
// AutoWS / pipelining loop annotations (`tt.num_stages`, `tt.warp_specialize`,
// `tt.data_partition_factor`, `llvm.loop_annotation`, ...) from the original
// `scf.while` onto it. Those annotations are attached to the while by the
// frontend (`tl.condition`); transferring them lets a scheduler-driven
// persistent while loop receive the same warp-specialization / pipelining
// treatment as a hand-written `scf.for` persistent loop.
class UpliftWhileToForPass
    : public impl::TritonUpliftWhileToForBase<UpliftWhileToForPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<scf::WhileOp> loops;
    getOperation()->walk(
        [&](scf::WhileOp whileOp) { loops.push_back(whileOp); });

    for (scf::WhileOp whileOp : loops) {
      // Capture the discardable attributes (the `tt.*` / `ttg.*` / `llvm.*`
      // loop annotations) before the op is replaced -- inherent attributes such
      // as `operandSegmentSizes` are not included and must not be transferred.
      DictionaryAttr attrs = whileOp->getDiscardableAttrDictionary();
      rewriter.setInsertionPoint(whileOp);
      FailureOr<scf::ForOp> forOp =
          scf::upliftWhileToForLoop(rewriter, whileOp);
      if (failed(forOp))
        continue;
      for (NamedAttribute attr : attrs)
        (*forOp)->setAttr(attr.getName(), attr.getValue());
    }
  }
};

} // namespace

} // namespace mlir::triton
