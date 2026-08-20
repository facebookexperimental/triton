#include "triton/Dialect/TritonGPU/Transforms/PartitionLoopPeeling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {
namespace {

// Match `iv < lb + step`, the guard the frontend emits for "first iteration
// only" work, and return it when it controls an scf.if in the loop body.
//
// The match is intentionally narrow. The comparison must live directly in the
// loop body (not in a nested region) and must feed an scf.if condition, so any
// other control flow -- nested loops, an scf.if on an unrelated predicate, a
// while loop -- simply fails to match and the loop is left alone. Peeling only
// ever runs on a shape it fully understands; there is no partial rewrite to
// unwind. When several comparisons match, only the first in walk order is
// peeled: the transform is a first-iteration split, so peeling more than one
// guard would need nested prologues, and the HSTU masked prologue this targets
// has exactly one.
static arith::CmpIOp getFirstIterationPredicate(scf::ForOp forOp) {
  arith::CmpIOp candidate;
  forOp.getBody()->walk([&](arith::CmpIOp cmp) {
    if (cmp.getPredicate() != arith::CmpIPredicate::slt ||
        cmp.getLhs() != forOp.getInductionVar() ||
        cmp->getBlock() != forOp.getBody())
      return WalkResult::advance();

    auto add = cmp.getRhs().getDefiningOp<arith::AddIOp>();
    if (!add)
      return WalkResult::advance();
    bool isFirstIterationBoundary = (add.getLhs() == forOp.getLowerBound() &&
                                     add.getRhs() == forOp.getStep()) ||
                                    (add.getRhs() == forOp.getLowerBound() &&
                                     add.getLhs() == forOp.getStep());
    if (!isFirstIterationBoundary)
      return WalkResult::advance();

    bool controlsIf = llvm::any_of(cmp->getUsers(), [&](Operation *user) {
      auto ifOp = dyn_cast<scf::IfOp>(user);
      return ifOp && ifOp.getCondition() == cmp.getResult();
    });
    if (!controlsIf)
      return WalkResult::advance();

    candidate = cmp;
    return WalkResult::interrupt();
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

// The scaffolding below is created inside a physical warp-specialize region,
// where every operation carries the partition task id. Mirror the source loop's
// so the new ops are not the only unannotated ones in that region.
static void copyTaskId(Operation *source, Operation *destination) {
  if (Attribute attr = source->getAttr(kAsyncTaskIdAttrName))
    destination->setAttr(kAsyncTaskIdAttrName, attr);
}

// Copy only the discardable attributes, leaving whatever inherent state the
// destination op was constructed with intact -- setAttrs would overwrite the
// whole dictionary, which is safe for today's scf.for but not for a loop op
// that carries inherent attributes.
static void copyDiscardableAttrs(Operation *source, Operation *destination) {
  for (NamedAttribute attr : source->getDiscardableAttrs())
    destination->setDiscardableAttr(attr.getName(), attr.getValue());
}

static void peelFirstIteration(scf::ForOp forOp, arith::CmpIOp predicate) {
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();

  auto hasFirstIterationOp =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getLowerBound(), forOp.getUpperBound());
  copyTaskId(forOp, hasFirstIterationOp);
  Value hasFirstIteration = hasFirstIterationOp.getResult();
  auto peeled = scf::IfOp::create(rewriter, loc, forOp.getResultTypes(),
                                  hasFirstIteration, /*withElseRegion=*/true);
  copyTaskId(forOp, peeled);
  // scf.IfOp auto-inserts a yield in each region when the op has no results
  // (a loop without iter args). Drop those so the explicit yields below are the
  // only terminators, instead of appending ops after a terminator.
  for (Block *block : {peeled.thenBlock(), peeled.elseBlock()})
    if (block->mightHaveTerminator())
      rewriter.eraseOp(block->getTerminator());

  Block *thenBlock = peeled.thenBlock();
  SmallVector<Value> firstResults =
      cloneIteration(rewriter, forOp, thenBlock, forOp.getLowerBound(),
                     forOp.getInitArgs(), predicate, /*predicateValue=*/true);

  rewriter.setInsertionPointToEnd(thenBlock);
  auto remainderLowerBoundOp = arith::AddIOp::create(
      rewriter, loc, forOp.getLowerBound(), forOp.getStep());
  copyTaskId(forOp, remainderLowerBoundOp);
  Value remainderLowerBound = remainderLowerBoundOp.getResult();
  auto remainder =
      scf::ForOp::create(rewriter, loc, remainderLowerBound,
                         forOp.getUpperBound(), forOp.getStep(), firstResults);
  // Only the remainder loop continues to expandLoops, so it inherits the
  // source loop's schedule metadata. The peeled prologue is a straight-line
  // clone whose ops keep their own per-op stage/cluster attributes; giving it
  // loop-level schedule attributes would present a second schedulable loop to
  // the pipeliner.
  copyDiscardableAttrs(forOp, remainder);
  SmallVector<Value> remainderResults = cloneIteration(
      rewriter, forOp, remainder.getBody(), remainder.getInductionVar(),
      remainder.getRegionIterArgs(), predicate, /*predicateValue=*/false);
  rewriter.setInsertionPointToEnd(remainder.getBody());
  auto remainderYield = scf::YieldOp::create(rewriter, loc, remainderResults);
  copyDiscardableAttrs(forOp.getBody()->getTerminator(), remainderYield);

  rewriter.setInsertionPointAfter(remainder);
  auto thenYield = scf::YieldOp::create(rewriter, loc, remainder.getResults());
  copyTaskId(forOp, thenYield);

  rewriter.setInsertionPointToStart(peeled.elseBlock());
  auto elseYield = scf::YieldOp::create(rewriter, loc, forOp.getInitArgs());
  copyTaskId(forOp, elseYield);

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

  // Peel in walk (post) order, i.e. innermost first. Peeling replaces a loop
  // with an scf.if and erases the original, so an outer loop must be peeled
  // after the inner ones it contains -- the other way round the outer clone
  // erases the inner loop and leaves the remaining entries dangling.
  for (auto [forOp, predicate] : candidates)
    peelFirstIteration(forOp, predicate);
}

} // namespace mlir::triton::gpu
