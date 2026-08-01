#include "triton/Dialect/TritonGPU/Transforms/PartitionLoopPeeling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {
namespace {

namespace tt = mlir::triton;
constexpr StringLiteral kSyntheticMaskBranchAttr =
    "ttg.loop_peeling.synthetic_mask";

struct OffsetRange {
  Value base;
  int64_t extent;
};

static arith::CmpIOp getFirstIterationPredicate(scf::ForOp forOp);

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

/// Match the base causal HSTU mask
///
///   (m == n) || ((m - n) > 0)
///
/// where m is based on the loop IV and n is based on the loop lower bound.
/// A positive loop step at least as large as the n range proves the mask is all
/// true after the first iteration.
static Value matchFirstIterationTensorMask(scf::ForOp forOp) {
  APInt stepValue;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepValue)))
    return {};
  int64_t step = stepValue.getSExtValue();
  if (step <= 0)
    return {};

  Value candidate;
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

      auto m = matchOffsetRange(sub.getLhs());
      auto n = matchOffsetRange(sub.getRhs());
      if (!m || !n || m->base != forOp.getInductionVar() ||
          n->base != forOp.getLowerBound() || step < n->extent)
        return;

      SmallVector<arith::SelectOp> selects;
      for (Operation *user : orOp.getResult().getUsers()) {
        auto select = dyn_cast<arith::SelectOp>(user);
        if (!select || select.getCondition() != orOp.getResult() ||
            !isZeroSplat(select.getFalseValue()))
          return;
        selects.push_back(select);
      }
      if (!selects.empty())
        candidate = orOp.getResult();
    };

    tryMatch(orOp.getLhs(), orOp.getRhs());
    if (!candidate)
      tryMatch(orOp.getRhs(), orOp.getLhs());
  });
  return candidate;
}

static void copyScheduleAttrs(Operation *source, Operation *destination) {
  for (StringRef name : {"async_task_id", "loop.cluster", "loop.stage"})
    if (Attribute attr = source->getAttr(name))
      destination->setAttr(name, attr);
}

/// Turn a tensor causal mask into a scalar first-iteration branch. The branch
/// result remains the real mask in the first iteration and becomes all-true in
/// the remainder. peelFirstIteration folds the scalar branch immediately when
/// it clones each path.
static bool materializeFirstIterationMaskBranch(scf::ForOp forOp) {
  if (getFirstIterationPredicate(forOp))
    return false;

  Value mask = matchFirstIterationTensorMask(forOp);
  if (!mask)
    return false;

  auto maskType = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType || !maskType.getElementType().isInteger(1))
    return false;

  Operation *maskOp = mask.getDefiningOp();
  IRRewriter rewriter(forOp);
  rewriter.setInsertionPointAfter(maskOp);
  Location loc = mask.getLoc();
  auto boundary = arith::AddIOp::create(rewriter, loc, forOp.getLowerBound(),
                                        forOp.getStep());
  auto needsMask =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getInductionVar(), boundary);
  auto effectiveMask = scf::IfOp::create(rewriter, loc, TypeRange{maskType},
                                         needsMask, /*withElseRegion=*/true);
  effectiveMask->setAttr(kSyntheticMaskBranchAttr, rewriter.getUnitAttr());
  copyScheduleAttrs(maskOp, boundary);
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

    // Branches introduced by materializeFirstIterationMaskBranch are created
    // after loop scheduling. Inline their selected side while cloning so an
    // unscheduled, constant-conditioned scf.if never reaches PipelineExpander.
    if (auto ifOp = dyn_cast<scf::IfOp>(op);
        ifOp && ifOp->hasAttr(kSyntheticMaskBranchAttr)) {
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

static void peelFirstIteration(scf::ForOp forOp, arith::CmpIOp predicate) {
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();

  Value hasFirstIteration =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getLowerBound(), forOp.getUpperBound());
  auto peeled = scf::IfOp::create(rewriter, loc, forOp.getResultTypes(),
                                  hasFirstIteration, /*withElseRegion=*/true);

  Block *thenBlock = peeled.thenBlock();
  if (!thenBlock->empty())
    if (auto yield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator()))
      rewriter.eraseOp(yield);
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
  remainderYield->setAttrs(forOp.getBody()->getTerminator()->getAttrs());

  rewriter.setInsertionPointAfter(remainder);
  scf::YieldOp::create(rewriter, loc, remainder.getResults());

  Block *elseBlock = peeled.elseBlock();
  if (!elseBlock->empty())
    if (auto yield = dyn_cast<scf::YieldOp>(elseBlock->getTerminator()))
      rewriter.eraseOp(yield);
  rewriter.setInsertionPointToStart(elseBlock);
  scf::YieldOp::create(rewriter, loc, forOp.getInitArgs());

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
    materializeFirstIterationMaskBranch(forOp);

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
