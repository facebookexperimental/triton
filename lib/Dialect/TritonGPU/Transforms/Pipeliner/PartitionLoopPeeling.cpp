#include "triton/Dialect/TritonGPU/Transforms/PartitionLoopPeeling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {
namespace {

namespace tt = mlir::triton;

static constexpr StringLiteral kPeelIterationsAttrName =
    "ttg.loop_peel_iterations";
static constexpr int64_t kMaxPeeledIterations = 4;

// A `splat(base) + make_range(0, extent)` offset vector, the shape both sides
// of the causal mask have. Pass-local: nothing outside this file reasons about
// the mask operands.
struct OffsetRange {
  Value base;
  int64_t extent;
};

struct PeelCandidate {
  arith::CmpIOp predicate;
  int64_t iterations;
};

static std::optional<PeelCandidate> getPeelCandidate(scf::ForOp forOp);

static Value stripBroadcastAndExpandDims(Value value) {
  while (true) {
    if (auto broadcast = value.getDefiningOp<tt::BroadcastOp>()) {
      value = broadcast.getSrc();
      continue;
    }
    if (auto expand = value.getDefiningOp<tt::ExpandDimsOp>()) {
      value = expand.getSrc();
      continue;
    }
    return value;
  }
}

static std::optional<OffsetRange> matchOffsetRange(Value value) {
  value = stripBroadcastAndExpandDims(value);
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;

  auto match = [](Value splatValue,
                  Value rangeValue) -> std::optional<OffsetRange> {
    auto splat = splatValue.getDefiningOp<tt::SplatOp>();
    auto range = rangeValue.getDefiningOp<tt::MakeRangeOp>();
    if (!splat || !range || range.getStartAttr().getInt() != 0 ||
        range.getEndAttr().getInt() <= 0)
      return std::nullopt;
    return OffsetRange{splat.getSrc(), range.getEndAttr().getInt()};
  };

  if (auto result = match(add.getLhs(), add.getRhs()))
    return result;
  return match(add.getRhs(), add.getLhs());
}

static bool isZeroSplat(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto elements = dyn_cast<SplatElementsAttr>(constant.getValue());
  if (!elements)
    return false;
  Attribute splat = elements.getSplatValue<Attribute>();
  if (auto integer = dyn_cast<IntegerAttr>(splat))
    return integer.getValue().isZero();
  if (auto fp = dyn_cast<FloatAttr>(splat))
    return fp.getValue().isZero();
  return false;
}

static bool isSameUnorderedPair(Value lhs0, Value rhs0, Value lhs1,
                                Value rhs1) {
  return (lhs0 == lhs1 && rhs0 == rhs1) || (lhs0 == rhs1 && rhs0 == lhs1);
}

/// Shared tail of the causal-mask match: `m` must be `iv + [0, M)`, `n` must be
/// `lb + [0, N)`, and every use of `mask` must be a select whose false value is
/// all zero. The first ceil(N / step) iterations may be masked; every later
/// iteration is all true.
static std::optional<int64_t> getCausalMaskIterations(scf::ForOp forOp,
                                                      int64_t step, Value m,
                                                      Value n, Value mask) {
  auto mRange = matchOffsetRange(m);
  auto nRange = matchOffsetRange(n);
  if (!mRange || !nRange || mRange->base != forOp.getInductionVar() ||
      nRange->base != forOp.getLowerBound())
    return std::nullopt;

  int64_t iterations = (nRange->extent + step - 1) / step;
  if (iterations <= 0 || iterations > kMaxPeeledIterations)
    return std::nullopt;

  bool hasSelect = false;
  for (Operation *user : mask.getUsers()) {
    auto select = dyn_cast<arith::SelectOp>(user);
    if (!select || select.getCondition() != mask ||
        !isZeroSplat(select.getFalseValue()))
      return std::nullopt;
    hasSelect = true;
  }
  return hasSelect ? std::optional<int64_t>(iterations) : std::nullopt;
}

/// Match the causal HSTU mask, which is `m >= n` written either as
///
///   (m == n) || ((m - n) > 0)     the form the frontend emits, or
///   m >= n                        the same predicate after canonicalization
///
/// where m is based on the loop IV and n is based on the loop lower bound.
/// Both spellings are accepted so the pattern does not depend on whether an
/// earlier pass folded the disjunction.
struct TensorMaskMatch {
  Value mask;
  int64_t iterations;
};

static std::optional<TensorMaskMatch>
matchFirstIterationsTensorMask(scf::ForOp forOp) {
  APInt stepValue;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepValue)))
    return {};
  int64_t step = stepValue.getSExtValue();
  if (step <= 0)
    return {};

  std::optional<TensorMaskMatch> candidate;
  forOp.getBody()->walk([&](arith::OrIOp orOp) {
    if (candidate || orOp->getBlock() != forOp.getBody())
      return;

    auto tryMatch = [&](Value eqValue, Value gtValue) {
      auto eq = eqValue.getDefiningOp<arith::CmpIOp>();
      auto gt = gtValue.getDefiningOp<arith::CmpIOp>();
      if (!eq || !gt || eq.getPredicate() != arith::CmpIPredicate::eq ||
          gt.getPredicate() != arith::CmpIPredicate::sgt ||
          !isZeroSplat(gt.getRhs()))
        return;

      auto sub = gt.getLhs().getDefiningOp<arith::SubIOp>();
      if (!sub || !isSameUnorderedPair(eq.getLhs(), eq.getRhs(), sub.getLhs(),
                                       sub.getRhs()))
        return;

      if (auto iterations = getCausalMaskIterations(
              forOp, step, sub.getLhs(), sub.getRhs(), orOp.getResult()))
        candidate = TensorMaskMatch{orOp.getResult(), *iterations};
    };

    tryMatch(orOp.getLhs(), orOp.getRhs());
    if (!candidate)
      tryMatch(orOp.getRhs(), orOp.getLhs());
  });
  if (candidate.has_value())
    return candidate;

  forOp.getBody()->walk([&](arith::CmpIOp cmp) {
    if (candidate || cmp->getBlock() != forOp.getBody() ||
        cmp.getPredicate() != arith::CmpIPredicate::sge)
      return;
    if (auto iterations = getCausalMaskIterations(
            forOp, step, cmp.getLhs(), cmp.getRhs(), cmp.getResult()))
      candidate = TensorMaskMatch{cmp.getResult(), *iterations};
  });
  return candidate;
}

static void copyScheduleAttrs(Operation *source, Operation *destination) {
  const StringRef scheduleAttrNames[] = {
      kAsyncTaskIdAttrName, tt::kLoopClusterAttrName, tt::kLoopStageAttrName};
  for (StringRef name : scheduleAttrNames)
    if (Attribute attr = source->getAttr(name))
      destination->setAttr(name, attr);
}

/// Turn a tensor causal mask into a scalar masked-prefix branch. The branch
/// result remains the real mask in the first ceil(N / step) iterations and
/// becomes all-true in the remainder. peelIterations folds the scalar branch
/// immediately when it clones each path.
static bool materializeFirstIterationsMaskBranch(scf::ForOp forOp) {
  if (getPeelCandidate(forOp))
    return false;

  auto match = matchFirstIterationsTensorMask(forOp);
  if (!match)
    return false;
  Value mask = match->mask;

  auto maskType = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType || !maskType.getElementType().isInteger(1))
    return false;

  Operation *maskOp = mask.getDefiningOp();
  IRRewriter rewriter(forOp);
  rewriter.setInsertionPointAfter(maskOp);
  Location loc = mask.getLoc();
  Value boundary = forOp.getLowerBound();
  for (int64_t i = 0; i < match->iterations; ++i) {
    auto add = arith::AddIOp::create(rewriter, loc, boundary, forOp.getStep());
    copyScheduleAttrs(maskOp, add);
    boundary = add;
  }
  auto needsMask =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getInductionVar(), boundary);
  needsMask->setAttr(kPeelIterationsAttrName,
                     rewriter.getI64IntegerAttr(match->iterations));
  auto effectiveMask = scf::IfOp::create(rewriter, loc, TypeRange{maskType},
                                         needsMask, /*withElseRegion=*/true);
  effectiveMask->setAttr(kSyntheticMaskBranchAttrName, rewriter.getUnitAttr());
  copyScheduleAttrs(maskOp, needsMask);
  copyScheduleAttrs(maskOp, effectiveMask);

  rewriter.setInsertionPointToStart(effectiveMask.thenBlock());
  auto thenYield = scf::YieldOp::create(rewriter, loc, mask);
  copyScheduleAttrs(maskOp, thenYield);

  rewriter.setInsertionPointToStart(effectiveMask.elseBlock());
  auto trueAttr = SplatElementsAttr::get(maskType, rewriter.getBoolAttr(true));
  auto trueMask = arith::ConstantOp::create(rewriter, loc, maskType, trueAttr);
  copyScheduleAttrs(maskOp, trueMask);
  auto elseYield = scf::YieldOp::create(rewriter, loc, trueMask.getResult());
  copyScheduleAttrs(maskOp, elseYield);

  mask.replaceUsesWithIf(effectiveMask.getResult(0), [&](OpOperand &use) {
    return use.getOwner() != thenYield;
  });
  return true;
}

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
static std::optional<PeelCandidate> getPeelCandidate(scf::ForOp forOp) {
  arith::CmpIOp candidate;
  forOp.getBody()->walk([&](arith::CmpIOp cmp) {
    if (cmp.getPredicate() != arith::CmpIPredicate::slt ||
        cmp.getLhs() != forOp.getInductionVar() ||
        cmp->getBlock() != forOp.getBody())
      return WalkResult::advance();

    if (auto count = cmp->getAttrOfType<IntegerAttr>(kPeelIterationsAttrName)) {
      int64_t iterations = count.getInt();
      if (iterations > 0 && iterations <= kMaxPeeledIterations) {
        candidate = cmp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

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
  if (!candidate)
    return std::nullopt;
  int64_t iterations = 1;
  if (auto count =
          candidate->getAttrOfType<IntegerAttr>(kPeelIterationsAttrName))
    iterations = count.getInt();
  return PeelCandidate{candidate, iterations};
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

    // Branches introduced by materializeFirstIterationMaskBranch are created
    // after loop scheduling. Inline their selected side while cloning so an
    // unscheduled, constant-conditioned scf.if never reaches PipelineExpander.
    if (auto ifOp = dyn_cast<scf::IfOp>(op);
        ifOp && ifOp->hasAttr(kSyntheticMaskBranchAttrName)) {
      Value condition = mapping.lookupOrDefault(ifOp.getCondition());
      APInt constant;
      if (matchPattern(condition, m_ConstantInt(&constant)) &&
          constant.getBitWidth() == 1) {
        Block *selected =
            constant.isOne() ? ifOp.thenBlock() : ifOp.elseBlock();
        if (selected) {
          for (Operation &nested : selected->without_terminator())
            rewriter.clone(nested, mapping);
          auto yield = cast<scf::YieldOp>(selected->getTerminator());
          for (auto [result, value] :
               llvm::zip(ifOp.getResults(), yield.getOperands()))
            mapping.map(result, mapping.lookupOrDefault(value));
        }
        continue;
      }
    }
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

static void eraseDefaultYield(IRRewriter &rewriter, Block *block) {
  if (block->mightHaveTerminator())
    if (auto yield = dyn_cast<scf::YieldOp>(block->getTerminator()))
      rewriter.eraseOp(yield);
}

static void peelIterations(scf::ForOp forOp, arith::CmpIOp predicate,
                           int64_t iterations) {
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
    eraseDefaultYield(rewriter, block);

  Block *thenBlock = peeled.thenBlock();
  SmallVector<Value> firstResults =
      cloneIteration(rewriter, forOp, thenBlock, forOp.getLowerBound(),
                     forOp.getInitArgs(), predicate, /*predicateValue=*/true);

  Value nextInduction = forOp.getLowerBound();
  SmallVector<Value> prefixResults = firstResults;
  for (int64_t i = 1; i < iterations; ++i) {
    rewriter.setInsertionPointToEnd(thenBlock);
    auto nextInductionOp =
        arith::AddIOp::create(rewriter, loc, nextInduction, forOp.getStep());
    copyTaskId(forOp, nextInductionOp);
    nextInduction = nextInductionOp;
    auto hasIterationOp =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                              nextInduction, forOp.getUpperBound());
    copyTaskId(forOp, hasIterationOp);
    auto optionalIteration =
        scf::IfOp::create(rewriter, loc, forOp.getResultTypes(), hasIterationOp,
                          /*withElseRegion=*/true);
    copyTaskId(forOp, optionalIteration);
    for (Block *block :
         {optionalIteration.thenBlock(), optionalIteration.elseBlock()})
      eraseDefaultYield(rewriter, block);

    SmallVector<Value> iterationResults = cloneIteration(
        rewriter, forOp, optionalIteration.thenBlock(), nextInduction,
        prefixResults, predicate, /*predicateValue=*/true);
    rewriter.setInsertionPointToEnd(optionalIteration.thenBlock());
    auto iterationYield = scf::YieldOp::create(rewriter, loc, iterationResults);
    copyTaskId(forOp, iterationYield);

    rewriter.setInsertionPointToStart(optionalIteration.elseBlock());
    auto skippedYield = scf::YieldOp::create(rewriter, loc, prefixResults);
    copyTaskId(forOp, skippedYield);
    prefixResults.assign(optionalIteration.getResults().begin(),
                         optionalIteration.getResults().end());
  }

  rewriter.setInsertionPointToEnd(thenBlock);
  auto remainderLowerBoundOp =
      arith::AddIOp::create(rewriter, loc, nextInduction, forOp.getStep());
  copyTaskId(forOp, remainderLowerBoundOp);
  Value remainderLowerBound = remainderLowerBoundOp.getResult();
  auto remainder =
      scf::ForOp::create(rewriter, loc, remainderLowerBound,
                         forOp.getUpperBound(), forOp.getStep(), prefixResults);
  // Only the remainder loop continues to expandLoops, so it inherits the
  // source loop's schedule metadata. The peeled prologue is a straight-line
  // clone whose ops keep their own per-op stage/cluster attributes; giving it
  // loop-level schedule attributes would present a second schedulable loop to
  // the pipeliner.
  copyDiscardableAttrs(forOp, remainder);
  scf::YieldOp defaultRemainderYield;
  if (!remainder.getBody()->empty())
    defaultRemainderYield =
        dyn_cast<scf::YieldOp>(remainder.getBody()->getTerminator());
  SmallVector<Value> remainderResults = cloneIteration(
      rewriter, forOp, remainder.getBody(), remainder.getInductionVar(),
      remainder.getRegionIterArgs(), predicate, /*predicateValue=*/false);
  // scf.for creates an empty scf.yield for loops without iter args. Replace it
  // instead of appending a second terminator after it. Loops with iter args
  // start with an empty block, so there is no default terminator to erase.
  if (defaultRemainderYield)
    rewriter.eraseOp(defaultRemainderYield);
  rewriter.setInsertionPointToEnd(remainder.getBody());
  auto remainderYield = scf::YieldOp::create(rewriter, loc, remainderResults);
  copyDiscardableAttrs(forOp.getBody()->getTerminator(), remainderYield);

  rewriter.setInsertionPointAfter(remainder);
  auto thenYield = scf::YieldOp::create(rewriter, loc, remainder.getResults());
  copyTaskId(forOp, thenYield);

  Block *elseBlock = peeled.elseBlock();
  rewriter.setInsertionPointToStart(elseBlock);
  auto elseYield = scf::YieldOp::create(rewriter, loc, forOp.getInitArgs());
  copyTaskId(forOp, elseYield);

  rewriter.replaceOp(forOp, peeled.getResults());
}

} // namespace

void peelPartitionLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> partitionLoops;
  moduleOp.walk([&](WarpSpecializeOp wsOp) {
    for (Region *partition : wsOp.getPartitionRegions()) {
      partition->walk([&](scf::ForOp forOp) {
        if (forOp->getParentOfType<WarpSpecializeOp>() == wsOp)
          partitionLoops.push_back(forOp);
      });
    }
  });
  for (scf::ForOp forOp : partitionLoops)
    materializeFirstIterationsMaskBranch(forOp);

  SmallVector<std::pair<scf::ForOp, PeelCandidate>> candidates;
  moduleOp.walk([&](WarpSpecializeOp wsOp) {
    for (Region *partition : wsOp.getPartitionRegions()) {
      partition->walk([&](scf::ForOp forOp) {
        if (forOp->getParentOfType<WarpSpecializeOp>() != wsOp)
          return;
        if (auto candidate = getPeelCandidate(forOp))
          candidates.emplace_back(forOp, *candidate);
      });
    }
  });

  // Peel in walk (post) order, i.e. innermost first. Peeling replaces a loop
  // with an scf.if and erases the original, so an outer loop must be peeled
  // after the inner ones it contains -- the other way round the outer clone
  // erases the inner loop and leaves the remaining entries dangling.
  for (auto [forOp, candidate] : candidates)
    peelIterations(forOp, candidate.predicate, candidate.iterations);
}

} // namespace mlir::triton::gpu
