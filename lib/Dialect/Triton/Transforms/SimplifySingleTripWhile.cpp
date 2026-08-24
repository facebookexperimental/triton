#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONSIMPLIFYSINGLETRIPWHILE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-simplify-single-trip-while"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Returns true if `v` is a constant `i1` equal to `expected`. Note: a 1-bit
// APInt of value 1 sign-extends to -1, so compare the boolean value rather than
// getConstantIntValue()'s sign-extended integer.
static bool isConstantBool(Value v, bool expected) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)) && val.getBitWidth() == 1)
    return val.getBoolValue() == expected;
  return false;
}

// A single-trip `scf.while` matches the "constant-flip" pattern when:
//  - the before-region has no side effects (it is evaluated by the rewrite but
//    its ops are otherwise dropped), and
//  - the `scf.condition` value is directly a before-block argument at index
//  `c`,
//    whose while init folds to constant `true` (so the loop is entered), while
//    the after-region yield operand at index `c` folds to constant `false` (so
//    the loop is not re-entered).
// The condition need not be a pass-through of all args -- canonicalization
// drops unused loop-carried values, so the real IR often forwards no args / has
// no results.
static bool isSingleTripConstantFlip(scf::WhileOp whileOp) {
  Block &before = whileOp.getBefore().front();
  auto condOp = cast<scf::ConditionOp>(before.getTerminator());

  // The before-region is evaluated (cloned) but never kept, so it must be free
  // of side effects.
  for (Operation &op : before.without_terminator())
    if (!isMemoryEffectFree(&op))
      return false;

  // The condition value must be directly a before-block argument.
  auto condArg = dyn_cast<BlockArgument>(condOp.getCondition());
  if (!condArg || condArg.getOwner() != &before)
    return false;
  unsigned c = condArg.getArgNumber();

  // init[c] == true (enter once) and yield[c] == false (do not re-enter). The
  // yield forwards the next loop-carried values, indexed like the before args.
  // Well-formed IR keeps these arities in sync, but bounds-check `c` so
  // malformed/partially-formed IR makes us decline rather than crash.
  auto yieldOp = cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
  if (c >= whileOp.getInits().size() || c >= yieldOp.getResults().size())
    return false;
  return isConstantBool(whileOp.getInits()[c], /*expected=*/true) &&
         isConstantBool(yieldOp.getResults()[c], /*expected=*/false);
}

// Forward AutoWS to for loops exactly one loop-nesting level inside the
// scheduler while. Intervening non-loop control flow, such as scf.if, does not
// add a nesting level. Existing inner-loop settings take precedence.
static bool forwardAutoWSToInnerLoops(scf::WhileOp whileOp) {
  if (!whileOp->hasAttr(kWarpSpecializeAttrName))
    return true;

  SmallVector<scf::ForOp> innerLoops;
  whileOp.getAfter().walk([&](scf::ForOp forOp) {
    LoopLikeOpInterface parentLoop =
        forOp->getParentOfType<LoopLikeOpInterface>();
    if (parentLoop && parentLoop.getOperation() == whileOp.getOperation())
      innerLoops.push_back(forOp);
  });
  if (innerLoops.empty())
    return false;

  SmallVector<NamedAttribute> attrs = filterAutoWSLoopAttrs(
      whileOp, AutoWSLoopAttrPropagation::ForwardToInnerLoop);
  for (scf::ForOp forOp : innerLoops) {
    for (NamedAttribute attr : attrs) {
      if (!forOp->hasAttr(attr.getName()))
        forOp->setAttr(attr.getName(), attr.getValue());
    }
  }
  return true;
}

// Clone the (pure) before-region ops under `argMap` (before-arg -> concrete
// value) and return the resulting `scf.condition` forwarded args.
static SmallVector<Value> evalBeforeForwardedArgs(OpBuilder &builder,
                                                  scf::WhileOp whileOp,
                                                  IRMapping argMap) {
  Block &before = whileOp.getBefore().front();
  for (Operation &op : before.without_terminator())
    builder.clone(op, argMap);
  auto condOp = cast<scf::ConditionOp>(before.getTerminator());
  SmallVector<Value> fwd;
  for (Value a : condOp.getArgs())
    fwd.push_back(argMap.lookupOrDefault(a));
  return fwd;
}

// Inline the after-region body once and replace the loop with its results.
static void rewriteSingleTripWhile(scf::WhileOp whileOp) {
  OpBuilder builder(whileOp);
  Block &before = whileOp.getBefore().front();
  Block &after = whileOp.getAfter().front();
  auto yieldOp = cast<scf::YieldOp>(after.getTerminator());

  // First before-region evaluation, with the loop inits substituted for the
  // before-block args. Its forwarded args are what feed the after-block on the
  // single entry.
  IRMapping initMap;
  for (auto [arg, init] : llvm::zip(before.getArguments(), whileOp.getInits()))
    initMap.map(arg, init);
  SmallVector<Value> forwarded =
      evalBeforeForwardedArgs(builder, whileOp, initMap);

  // Inline the after-region body once.
  IRMapping bodyMap;
  for (auto [arg, v] : llvm::zip(after.getArguments(), forwarded))
    bodyMap.map(arg, v);
  for (Operation &op : after.without_terminator())
    builder.clone(op, bodyMap);

  // The while results are the forwarded args of the *second* before-region
  // evaluation, run on the loop-carried values yielded by the body (whose
  // condition folds to false, so the loop exits after one iteration). Skip this
  // when the loop has no results (the common canonicalized case).
  SmallVector<Value> results;
  if (whileOp.getNumResults() > 0) {
    IRMapping nextMap;
    for (auto [arg, y] : llvm::zip(before.getArguments(), yieldOp.getResults()))
      nextMap.map(arg, bodyMap.lookupOrDefault(y));
    results = evalBeforeForwardedArgs(builder, whileOp, nextMap);
  }

  whileOp.replaceAllUsesWith(results);
  whileOp.erase();
}

} // namespace

class SimplifySingleTripWhilePass
    : public impl::TritonSimplifySingleTripWhileBase<
          SimplifySingleTripWhilePass> {
public:
  void runOnOperation() override {
    SmallVector<scf::WhileOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>([&](scf::WhileOp whileOp) {
      if (isSingleTripConstantFlip(whileOp))
        loops.push_back(whileOp);
    });

    for (scf::WhileOp whileOp : loops) {
      if (!forwardAutoWSToInnerLoops(whileOp)) {
        LDBG("Keeping annotated single-trip while without a first-level inner "
             "loop: "
             << whileOp);
        continue;
      }
      LDBG("Simplifying single-trip while: " << whileOp);
      rewriteSingleTripWhile(whileOp);
    }
  }
};

} // namespace mlir::triton
