#include "triton/Analysis/IntegerRange.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <array>
#include <limits>

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

// Tensor ranges bound all elements, independent of their lane layout.
static bool inferTritonRanges(Operation *op,
                              ArrayRef<IntegerValueRange> operands,
                              SetIntLatticeFn setRange,
                              ArrayRef<int64_t> maxProgramIds) {
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
      auto *lattice = results[cast<OpResult>(value).getResultNumber()];
      propagateIfChanged(lattice, lattice->join(range));
    };
    if (inferTritonRanges(op, ranges, setRange, maxProgramIds))
      return success();
    return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                          results);
  }

private:
  std::array<int64_t, 3> maxProgramIds;
};

// Use-local refinement must not narrow the global SSA lattice.
class RangeAtUse {
public:
  RangeAtUse(DataFlowSolver &solver, DominanceInfo &dominance,
             const DenseMap<Value, SmallVector<LLVM::AssumeOp>> &assumptions,
             Operation *use, ArrayRef<int64_t> maxProgramIds)
      : solver(solver), dominance(dominance), assumptions(assumptions),
        use(use), maxProgramIds(maxProgramIds) {}

  IntegerValueRange get(Value value) {
    SmallVector<std::pair<Value, bool>> pending{{value, false}};
    while (!pending.empty()) {
      auto [current, operandsReady] = pending.pop_back_val();
      if (cache.contains(current))
        continue;
      if (Operation *definition = getDefinition(current);
          !operandsReady && definition) {
        pending.emplace_back(current, true);
        for (Value operand : definition->getOperands())
          if (!cache.contains(operand))
            pending.emplace_back(operand, false);
        continue;
      }
      cache.try_emplace(current, infer(current));
    }
    return cache.lookup(value);
  }

private:
  static Operation *getDefinition(Value value) {
    Operation *definition = value.getDefiningOp();
    if (definition && definition->getNumRegions() == 0 &&
        ConstantIntRanges::getStorageBitwidth(value.getType()) != 0)
      return definition;
    return nullptr;
  }

  IntegerValueRange infer(Value value) {
    IntegerValueRange range = IntegerValueRange::getMaxRange(value);
    if (const auto *lattice =
            solver.lookupState<dataflow::IntegerValueRangeLattice>(value))
      if (!lattice->getValue().isUninitialized())
        range = lattice->getValue();

    // Region results and block arguments use the solver's fixed point.
    if (Operation *definition = getDefinition(value)) {
      SmallVector<IntegerValueRange> operands;
      for (Value operand : definition->getOperands())
        operands.push_back(cache.lookup(operand));
      auto setRange = [&](Value result, const IntegerValueRange &inferred) {
        if (result == value && !inferred.isUninitialized())
          range = inferred;
      };
      if (!inferTritonRanges(definition, operands, setRange, maxProgramIds))
        if (auto interface = dyn_cast<InferIntRangeInterface>(definition))
          interface.inferResultRangesFromOptional(operands, setRange);
    }

    auto it = assumptions.find(value);
    if (it != assumptions.end())
      for (LLVM::AssumeOp assume : it->second) {
        if (!dominance.properlyDominates(assume.getOperation(), use))
          continue;
        auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>();
        if (auto bound = getBoundFromCmpOp(cmp, value))
          range = IntegerValueRange(range.getValue().intersection(*bound));
      }
    return range;
  }

  DenseMap<Value, IntegerValueRange> cache;
  DataFlowSolver &solver;
  DominanceInfo &dominance;
  const DenseMap<Value, SmallVector<LLVM::AssumeOp>> &assumptions;
  Operation *use;
  ArrayRef<int64_t> maxProgramIds;
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
  if (failed(solver->initializeAndRun(root)))
    return failure();

  DenseMap<Value, SmallVector<LLVM::AssumeOp>> assumptions;
  root->walk([&](LLVM::AssumeOp assume) {
    if (auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>())
      for (Value operand : cmp->getOperands())
        assumptions[operand].push_back(assume);
  });
  DominanceInfo dominance(root);
  // Q uses, V values, E operand edges, A assumptions: O(Q*(V+E+A)) query
  // time, O(V+E+A) space. Dataflow fixed-point cost is separate.
  // Each use has its own cache because assumptions can differ within a block.
  for (Operation *op : queries) {
    RangeAtUse query(*solver, dominance, assumptions, op, maxProgramIds);
    IntegerValueRange range = query.get(op->getOperand(0));
    if (!range.isUninitialized() && range.getValue().smin().isNonNegative())
      operations.insert(op);
  }
  return success();
}
