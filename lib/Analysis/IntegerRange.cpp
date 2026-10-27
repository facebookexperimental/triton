#include "triton/Analysis/IntegerRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <array>
#include <limits>

using namespace mlir;
using namespace mlir::triton;

namespace {

static arith::CmpIPredicate signedPredicate(arith::CmpIPredicate predicate) {
  using P = arith::CmpIPredicate;
  switch (predicate) {
  case P::ult:
    return P::slt;
  case P::ule:
    return P::sle;
  case P::ugt:
    return P::sgt;
  case P::uge:
    return P::sge;
  default:
    return predicate;
  }
}

static bool refineOrderedEndpoints(arith::CmpIPredicate predicate,
                                   bool isUnsigned, APInt &lo, APInt &hi,
                                   const APInt &boundLo, const APInt &boundHi) {
  auto less = [&](const APInt &a, const APInt &b) {
    return isUnsigned ? a.ult(b) : a.slt(b);
  };
  APInt lower = boundLo, upper = boundHi;
  using P = arith::CmpIPredicate;
  switch (signedPredicate(predicate)) {
  case P::slt:
    if (isUnsigned ? upper.isZero() : upper.isMinSignedValue())
      return false;
    --upper;
    [[fallthrough]];
  case P::sle:
    if (less(upper, hi))
      hi = upper;
    break;
  case P::sgt:
    if (isUnsigned ? lower.isMaxValue() : lower.isMaxSignedValue())
      return false;
    ++lower;
    [[fallthrough]];
  case P::sge:
    if (less(lo, lower))
      lo = lower;
    break;
  default:
    llvm_unreachable("expected ordered integer comparison");
  }
  return true;
}

static ConstantIntRanges refineRange(arith::CmpIPredicate predicate,
                                     const ConstantIntRanges &value,
                                     const ConstantIntRanges &bound) {
  using P = arith::CmpIPredicate;
  bool isUnsigned = signedPredicate(predicate) != predicate;
  APInt lo = isUnsigned ? value.umin() : value.smin();
  APInt hi = isUnsigned ? value.umax() : value.smax();
  APInt boundLo = isUnsigned ? bound.umin() : bound.smin();
  APInt boundHi = isUnsigned ? bound.umax() : bound.smax();
  if (predicate == P::eq) {
    lo = llvm::APIntOps::smax(lo, boundLo);
    hi = llvm::APIntOps::smin(hi, boundHi);
  } else if (predicate == P::ne) {
    if (boundLo == boundHi && lo != hi) {
      if (lo == boundLo)
        ++lo;
      if (hi == boundHi)
        --hi;
    }
  } else if (!refineOrderedEndpoints(predicate, isUnsigned, lo, hi, boundLo,
                                     boundHi))
    return value;
  // Contradictions poison this result; they do not make its block unreachable.
  if (isUnsigned ? hi.ult(lo) : hi.slt(lo))
    return value;
  ConstantIntRanges refined = isUnsigned
                                  ? ConstantIntRanges::fromUnsigned(lo, hi)
                                  : ConstantIntRanges::fromSigned(lo, hi);
  return value.intersection(refined);
}

static ConstantIntRanges inferSignedDivRem(Operation *op,
                                           const ConstantIntRanges &lhs,
                                           const ConstantIntRanges &rhs) {
  unsigned width = lhs.smin().getBitWidth();
  APInt zero(width, 0);
  std::optional<ConstantIntRanges> result;
  // Zero divisors are undefined; analyze each nonzero signed interval.
  for (bool negative : {false, true}) {
    if (negative ? !rhs.smin().isNegative() : !rhs.smax().isStrictlyPositive())
      continue;
    APInt lo = negative ? rhs.smin()
                        : llvm::APIntOps::smax(rhs.smin(), APInt(width, 1));
    APInt hi = negative
                   ? llvm::APIntOps::smin(rhs.smax(), APInt::getAllOnes(width))
                   : rhs.smax();
    if (lo.sgt(hi))
      continue;
    std::array<ConstantIntRanges, 2> operands{
        lhs, ConstantIntRanges::fromSigned(lo, hi)};
    ConstantIntRanges range = isa<arith::DivSIOp>(op)
                                  ? intrange::inferDivS(operands)
                                  : intrange::inferRemS(operands);
    result = result ? result->rangeUnion(range) : range;
  }
  ConstantIntRanges range = result.value_or(ConstantIntRanges::maxRange(width));
  if (isa<arith::RemSIOp>(op))
    // Remainder has the dividend's sign and cannot exceed its magnitude.
    range = range.intersection(
        ConstantIntRanges::fromSigned(llvm::APIntOps::smin(lhs.smin(), zero),
                                      llvm::APIntOps::smax(lhs.smax(), zero)));
  return range;
}

// Tensor ranges bound all elements, independent of their lane layout.
static bool inferTritonRanges(Operation *op,
                              ArrayRef<IntegerValueRange> operands,
                              SetIntLatticeFn setRange,
                              ArrayRef<int64_t> maxProgramIds) {
  if (auto refine = dyn_cast<RefineOp>(op)) {
    if (!operands[0].isUninitialized() && !operands[1].isUninitialized())
      setRange(refine.getResult(),
               refineRange(refine.getPredicate(), operands[0].getValue(),
                           operands[1].getValue()));
    return true;
  }
  if (isa<arith::DivSIOp, arith::RemSIOp>(op)) {
    if (operands[0].isUninitialized() || operands[1].isUninitialized())
      return false;
    setRange(op->getResult(0), inferSignedDivRem(op, operands[0].getValue(),
                                                 operands[1].getValue()));
    return true;
  }
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    setRange(range.getResult(),
             ConstantIntRanges::fromSigned(APInt(32, range.getStart()),
                                           APInt(32, range.getEnd() - 1)));
    return true;
  }
  if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
    setRange(op->getResult(0),
             ConstantIntRanges::fromSigned(
                 APInt(32, 0), APInt(32, maxProgramIds[pid.getAxisAsInt()])));
    return true;
  }
  if (isa<SplatOp, ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp, SplitOp,
          GatherOp, triton::gpu::ConvertLayoutOp, triton::gpu::RequireLayoutOp,
          triton::gpu::ReleaseLayoutOp>(op)) {
    for (Value result : op->getResults())
      setRange(result, operands.front());
    return true;
  }
  if (isa<JoinOp>(op)) {
    setRange(op->getResult(0),
             IntegerValueRange::join(operands[0], operands[1]));
    return true;
  }
  return false;
}

class IntegerRangeAnalysis final : public dataflow::IntegerRangeAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerRangeAnalysis)
  IntegerRangeAnalysis(DataFlowSolver &solver,
                       std::array<int64_t, 3> maxProgramIds)
      : dataflow::IntegerRangeAnalysis(solver), maxProgramIds(maxProgramIds) {}

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    SmallVector<IntegerValueRange> ranges;
    for (const auto *operand : operands)
      ranges.push_back(operand->getValue());
    auto setRange = [&](Value value, const IntegerValueRange &range) {
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        return;
      auto *lattice = results[result.getResultNumber()];
      if (!range.isUninitialized())
        propagateIfChanged(lattice, lattice->join(range));
    };
    if (inferTritonRanges(op, ranges, setRange, maxProgramIds))
      return success();
    if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
      inferrable.inferResultRangesFromOptional(ranges, setRange);
      return success();
    }
    return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                          results);
  }

private:
  std::array<int64_t, 3> maxProgramIds;
};

} // namespace

std::unique_ptr<DataFlowSolver>
mlir::triton::createIntegerRangeSolver(Operation *root) {
  std::array<int64_t, 3> maxProgramIds;
  maxProgramIds.fill(std::numeric_limits<int32_t>::max());
  ModuleOp mod = dyn_cast<ModuleOp>(root);
  if (!mod)
    mod = root->getParentOfType<ModuleOp>();
  if (mod)
    for (Dialect *dialect : root->getContext()->getLoadedDialects())
      if (const auto *bounds =
              dialect->getRegisteredInterface<DialectLaunchBoundsInterface>())
        for (unsigned axis : llvm::seq(3u))
          if (std::optional<int64_t> bound = bounds->getMaxProgramId(mod, axis))
            maxProgramIds[axis] = std::min(maxProgramIds[axis], *bound);

  auto solver = createDataFlowSolver();
  solver->load<IntegerRangeAnalysis>(maxProgramIds);
  return solver;
}

LogicalResult
mlir::triton::collectNonNegativeDividends(Operation *root,
                                          DenseSet<Operation *> &operations) {
  SmallVector<Operation *> queries;
  root->walk([&](Operation *op) {
    if (isa<arith::DivSIOp, arith::RemSIOp>(op) &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      queries.push_back(op);
  });
  if (queries.empty())
    return success();
  auto solver = createIntegerRangeSolver(root);
  if (failed(solver->initializeAndRun(root)))
    return failure();
  for (Operation *op : queries) {
    const auto *range = solver->lookupState<dataflow::IntegerValueRangeLattice>(
        op->getOperand(0));
    if (range && !range->getValue().isUninitialized() &&
        range->getValue().getValue().smin().isNonNegative())
      operations.insert(op);
  }
  return success();
}
