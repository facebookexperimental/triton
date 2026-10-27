#include "triton/Analysis/IntegerRange.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
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
  IntegerRangeAnalysis(DataFlowSolver &solver, Operation *root)
      : dataflow::IntegerRangeAnalysis(solver), dominance(root) {
    root->walk([&](LLVM::AssumeOp assume) {
      SmallVector<Value> pending{assume.getCond()};
      DenseSet<Value> visited;
      while (!pending.empty()) {
        Value condition = pending.pop_back_val();
        if (!visited.insert(condition).second)
          continue;
        if (auto conjunction = condition.getDefiningOp<arith::AndIOp>())
          llvm::append_range(pending, conjunction->getOperands());
        else if (auto cmp = condition.getDefiningOp<arith::CmpIOp>())
          for (Value value : cmp->getOperands())
            if (auto range = getBoundFromCmpOp(cmp, value))
              assumptions[value].push_back({assume, *range});
      }
    });
  }

  // Apply assumptions at operand uses; never change the defining value's range.
  IntegerValueRange atUse(Value value, Operation *use,
                          const IntegerValueRange &inferred) {
    if (inferred.isUninitialized())
      return inferred;
    ConstantIntRanges range = inferred.getValue();
    auto it = assumptions.find(value);
    if (it != assumptions.end())
      for (const auto &fact : it->second)
        if (dominance.properlyDominates(fact.first, use))
          range = range.intersection(fact.second);
    return IntegerValueRange(range);
  }

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    SmallVector<std::unique_ptr<dataflow::IntegerValueRangeLattice>> storage;
    SmallVector<const dataflow::IntegerValueRangeLattice *> inputs;
    for (auto [value, lattice] : llvm::zip(op->getOperands(), operands)) {
      if (!assumptions.contains(value)) {
        inputs.push_back(lattice);
        continue;
      }
      auto local = std::make_unique<dataflow::IntegerValueRangeLattice>(value);
      (void)local->join(atUse(value, op, lattice->getValue()));
      inputs.push_back(local.get());
      storage.push_back(std::move(local));
    }
    if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
      return dataflow::IntegerRangeAnalysis::visitOperation(op, inputs,
                                                            results);
    const IntegerValueRange &lhs = inputs[0]->getValue();
    const IntegerValueRange &rhs = inputs[1]->getValue();
    if (lhs.isUninitialized() || rhs.isUninitialized())
      return success();
    ConstantIntRanges range =
        inferSignedDivRem(op, lhs.getValue(), rhs.getValue());
    propagateIfChanged(results.front(), results.front()->join(range));
    return success();
  }

private:
  DenseMap<Value, SmallVector<std::pair<Operation *, ConstantIntRanges>>>
      assumptions;
  DominanceInfo dominance;
};

} // namespace

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
  auto solver = createDataFlowSolver();
  auto *analysis = solver->load<IntegerRangeAnalysis>(root);
  if (failed(solver->initializeAndRun(root)))
    return failure();
  for (Operation *op : queries) {
    const auto *range = solver->lookupState<dataflow::IntegerValueRangeLattice>(
        op->getOperand(0));
    if (!range)
      continue;
    IntegerValueRange atUse =
        analysis->atUse(op->getOperand(0), op, range->getValue());
    if (!atUse.isUninitialized() && atUse.getValue().smin().isNonNegative())
      operations.insert(op);
  }
  return success();
}
