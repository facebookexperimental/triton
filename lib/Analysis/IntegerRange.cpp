#include "triton/Analysis/IntegerRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;

namespace {

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

// TODO: Upstream signed div/rem ranges to LLVM's arith interfaces.
class IntegerRangeAnalysis final : public dataflow::IntegerRangeAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerRangeAnalysis)
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
      return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                            results);
    const IntegerValueRange &lhs = operands[0]->getValue();
    const IntegerValueRange &rhs = operands[1]->getValue();
    if (lhs.isUninitialized() || rhs.isUninitialized())
      return success();
    ConstantIntRanges range =
        inferSignedDivRem(op, lhs.getValue(), rhs.getValue());
    propagateIfChanged(results.front(), results.front()->join(range));
    return success();
  }
};

} // namespace

std::unique_ptr<DataFlowSolver> mlir::triton::createIntegerRangeSolver() {
  auto solver = createDataFlowSolver();
  solver->load<IntegerRangeAnalysis>();
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
  auto solver = createIntegerRangeSolver();
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
