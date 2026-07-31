#include "triton/Dialect/TritonGPU/Transforms/PartitionLoopPeeling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {
namespace {

static arith::CmpIOp getFirstIterationPredicate(scf::ForOp forOp) {
  arith::CmpIOp candidate;
  forOp.getBody()->walk([&](arith::CmpIOp cmp) {
    if (candidate || cmp.getPredicate() != arith::CmpIPredicate::slt ||
        cmp.getLhs() != forOp.getInductionVar() ||
        cmp->getBlock() != forOp.getBody())
      return;

    auto add = cmp.getRhs().getDefiningOp<arith::AddIOp>();
    if (!add)
      return;
    bool isFirstIterationBoundary = (add.getLhs() == forOp.getLowerBound() &&
                                     add.getRhs() == forOp.getStep()) ||
                                    (add.getRhs() == forOp.getLowerBound() &&
                                     add.getLhs() == forOp.getStep());
    if (!isFirstIterationBoundary)
      return;

    bool controlsIf = llvm::any_of(cmp->getUsers(), [&](Operation *user) {
      auto ifOp = dyn_cast<scf::IfOp>(user);
      return ifOp && ifOp.getCondition() == cmp.getResult();
    });
    if (controlsIf)
      candidate = cmp;
  });
  return candidate;
}

static SmallVector<Value>
cloneIteration(IRRewriter &rewriter, scf::ForOp source, Block *destination,
               Value inductionValue, ValueRange iterArgs,
               arith::CmpIOp predicate, bool predicateValue) {
  IRMapping mapping;
  mapping.map(source.getInductionVar(), inductionValue);
  mapping.map(source.getRegionIterArgs(), iterArgs);

  rewriter.setInsertionPointToStart(destination);
  auto foldedPredicate = arith::ConstantIntOp::create(
      rewriter, predicate.getLoc(), predicateValue, 1);
  // The remainder is still consumed by PipelineExpander. Preserve the
  // predicate's serialized stage/cluster on its constant replacement so every
  // operation in that loop remains scheduled.
  foldedPredicate->setAttrs(predicate->getAttrs());
  mapping.map(predicate.getResult(), foldedPredicate);

  for (Operation &op : source.getBody()->without_terminator()) {
    if (&op == predicate.getOperation())
      continue;
    rewriter.clone(op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(source.getBody()->getTerminator());
  SmallVector<Value> yielded;
  yielded.reserve(oldYield.getNumOperands());
  for (Value value : oldYield.getOperands())
    yielded.push_back(mapping.lookupOrDefault(value));
  return yielded;
}

static void peelFirstIteration(scf::ForOp forOp, arith::CmpIOp predicate) {
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();

  Value hasFirstIteration =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getLowerBound(), forOp.getUpperBound());
  auto peeled = scf::IfOp::create(rewriter, loc, forOp.getResultTypes(),
                                  hasFirstIteration, /*withElseRegion=*/true);

  Block *thenBlock = peeled.thenBlock();
  SmallVector<Value> firstResults =
      cloneIteration(rewriter, forOp, thenBlock, forOp.getLowerBound(),
                     forOp.getInitArgs(), predicate, /*predicateValue=*/true);

  rewriter.setInsertionPointToEnd(thenBlock);
  Value remainderLowerBound = arith::AddIOp::create(
      rewriter, loc, forOp.getLowerBound(), forOp.getStep());
  auto remainder =
      scf::ForOp::create(rewriter, loc, remainderLowerBound,
                         forOp.getUpperBound(), forOp.getStep(), firstResults);
  remainder->setAttrs(forOp->getAttrs());
  SmallVector<Value> remainderResults = cloneIteration(
      rewriter, forOp, remainder.getBody(), remainder.getInductionVar(),
      remainder.getRegionIterArgs(), predicate, /*predicateValue=*/false);
  rewriter.setInsertionPointToEnd(remainder.getBody());
  auto remainderYield = scf::YieldOp::create(rewriter, loc, remainderResults);
  remainderYield->setAttrs(forOp.getBody()->getTerminator()->getAttrs());

  rewriter.setInsertionPointAfter(remainder);
  scf::YieldOp::create(rewriter, loc, remainder.getResults());

  rewriter.setInsertionPointToStart(peeled.elseBlock());
  scf::YieldOp::create(rewriter, loc, forOp.getInitArgs());

  rewriter.replaceOp(forOp, peeled.getResults());
}

} // namespace

void peelPartitionLoops(ModuleOp moduleOp) {
  SmallVector<std::pair<scf::ForOp, arith::CmpIOp>> candidates;
  moduleOp.walk([&](WarpSpecializeOp wsOp) {
    for (Region *partition : wsOp.getPartitionRegions()) {
      partition->walk([&](scf::ForOp forOp) {
        if (forOp->getParentOfType<WarpSpecializeOp>() != wsOp)
          return;
        if (auto predicate = getFirstIterationPredicate(forOp))
          candidates.emplace_back(forOp, predicate);
      });
    }
  });

  for (auto [forOp, predicate] : llvm::reverse(candidates))
    peelFirstIteration(forOp, predicate);
}

} // namespace mlir::triton::gpu
