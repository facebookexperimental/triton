#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"

using namespace mlir;

namespace mlir {
namespace triton {

void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, bool)>
        processPeeledOp) {
  SmallVector<Operation *> loopBodyOps;
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();
  Type type = forOp.getStep().getType();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value newUpperBound = rewriter.create<arith::SubIOp>(loc, upperBound, step);

  rewriter.setInsertionPointAfter(forOp);
  Value lastIV = getLastInductionValue(rewriter, forOp);

  auto cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             lowerBound, upperBound);

  // Create an if op to execute the peeled iteration
  IRMapping map;
  map.map(forOp.getRegionIterArgs(), forOp.getResults());
  map.map(forOp.getInductionVar(), lastIV);
  auto ifOp = rewriter.create<scf::IfOp>(loc, forOp.getResultTypes(), cond,
                                         /*hasElse=*/true);
  ifOp.getThenRegion().front().erase();
  forOp.getBodyRegion().cloneInto(&ifOp.getThenRegion(), map);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  rewriter.create<scf::YieldOp>(loc, forOp.getResults());

  forOp->replaceUsesWithIf(ifOp, [&](OpOperand &operand) {
    return !ifOp->isAncestor(operand.getOwner());
  });

  forOp.getUpperBoundMutable().assign(newUpperBound);

  if (processPeeledOp) {
    for (auto &op :
         llvm::make_early_inc_range(forOp.getBody()->without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/false);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
    for (auto &op : llvm::make_early_inc_range(
             ifOp.getThenRegion().front().without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/true);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
  }
}

void peelLoopPrologue(scf::ForOp forOp) {
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();

  Value lowerBound = forOp.getLowerBound();
  Value step = forOp.getStep();

  // Clone the loop body before the loop as the prologue.
  // Map induction var -> lowerBound, region iter args -> init args.
  rewriter.setInsertionPoint(forOp);
  IRMapping map;
  map.map(forOp.getInductionVar(), lowerBound);
  for (auto [regionArg, initArg] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
    map.map(regionArg, initArg);
  }

  // Clone all ops except the yield, and capture yielded values as
  // the new init args for the remaining loop.
  SmallVector<Value> prologueYieldedValues;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
      for (Value v : yieldOp.getOperands()) {
        prologueYieldedValues.push_back(map.lookupOrDefault(v));
      }
    } else {
      rewriter.clone(op, map);
    }
  }

  // Adjust the loop in place: lb -> lb + step, init args -> prologue outputs.
  // Note: This assumes the loop always executes at least once. If the loop
  // could have zero iterations, the prologue would execute unconditionally
  // which would be incorrect.
  Value newLowerBound = rewriter.create<arith::AddIOp>(loc, lowerBound, step);
  forOp.getLowerBoundMutable().assign(newLowerBound);
  for (auto [initArg, prologueVal] :
       llvm::zip(forOp.getInitArgsMutable(), prologueYieldedValues)) {
    initArg.set(prologueVal);
  }
}

} // namespace triton
} // namespace mlir
