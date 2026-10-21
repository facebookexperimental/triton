#include "triton/Analysis/IntegerRange.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Tensor ranges bound all elements, independent of their lane layout.
static bool inferTritonRanges(Operation *op,
                              ArrayRef<IntegerValueRange> operands,
                              SetIntLatticeFn setRange) {
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    setRange(range.getResult(),
             ConstantIntRanges::fromSigned(APInt(32, range.getStart()),
                                           APInt(32, range.getEnd() - 1)));
    return true;
  }
  if (isa<GetProgramIdOp>(op)) {
    setRange(op->getResult(0), ConstantIntRanges::fromSigned(
                                   APInt(32, 0), APInt::getSignedMaxValue(32)));
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
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;

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
    if (inferTritonRanges(op, ranges, setRange))
      return success();
    return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                          results);
  }
};

// Use-local refinement must not narrow the global SSA lattice.
class RangeAtUse {
public:
  RangeAtUse(DataFlowSolver &solver, DominanceInfo &dominance,
             const DenseMap<Value, SmallVector<LLVM::AssumeOp>> &assumptions,
             Operation *use)
      : solver(solver), dominance(dominance), assumptions(assumptions),
        use(use) {}

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
      if (!inferTritonRanges(definition, operands, setRange))
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
  solver->load<IntegerRangeAnalysis>();
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
    RangeAtUse query(*solver, dominance, assumptions, op);
    IntegerValueRange range = query.get(op->getOperand(0));
    if (!range.isUninitialized() && range.getValue().smin().isNonNegative())
      operations.insert(op);
  }
  return success();
}
