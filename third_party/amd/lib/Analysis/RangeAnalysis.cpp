#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace tt = mlir::triton;

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;
constexpr uint64_t kDefaultMaxPrograms = 1ULL << 31; // 2147483648

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

void inferResultRangesPID(Operation *op, uint64_t max,
                          SetIntRangeFn setResultRange) {
  assert(op->getNumResults() == 1 && "expected op to have one result");
  auto result = op->getResult(0);
  assert(llvm::isa<IntegerType>(result.getType()) &&
         "expected result type to be int");
  IntegerType resTy = llvm::cast<IntegerType>(result.getType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(resTy);
  setResultRange(result, ConstantIntRanges::range(
                             /*min*/ {/*numBits*/ bitWidth, /*val*/ 0,
                                      /*isSigned*/ resTy.isSigned()},
                             /*max*/
                             {/*numBits*/ bitWidth, /*val*/ max,
                              /*isSigned*/ resTy.isSigned()},
                             /*isSigned*/ resTy.isSigned()));
}

void inferResultRangesMaxNonNegSigned(Operation *op,
                                      SetIntRangeFn setResultRange) {
  for (auto result : op->getResults()) {
    auto bitWidth =
        mlir::ConstantIntRanges::getStorageBitwidth(result.getType());
    setResultRange(result, ConstantIntRanges::fromSigned(
                               APInt::getZero(bitWidth),
                               APInt::getSignedMaxValue(bitWidth)));
  }
}

// TODO: Move signed div/rem range inference into LLVM arith interfaces.
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

} // namespace

namespace mlir::triton::AMD {

std::optional<int64_t>
TritonIntegerRangeAnalysis::maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  std::optional<Value> iv = loop.getSingleInductionVar();
  if (!iv)
    return {};

  unsigned int width = ConstantIntRanges::getStorageBitwidth(iv->getType());

  auto getLoopRangeInfo = [&](std::optional<OpFoldResult> loopBound,
                              Block *block,
                              std::optional<bool> getUpper = std::nullopt,
                              std::optional<APInt> defaultVal = std::nullopt) {
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = llvm::dyn_cast_if_present<Value>(*loopBound)) {
        const dataflow::IntegerValueRangeLattice *lattice =
            getLatticeElementFor(getProgramPointBefore(block), value);
        if (lattice != nullptr && !lattice->getValue().isUninitialized()) {
          IntegerValueRange range =
              getRangeAt(value, loop, lattice->getValue());
          return getUpper.value_or(false) ? range.getValue().smax()
                                          : range.getValue().smin();
        }
      }
    }
    if (defaultVal)
      return *defaultVal;
    return getUpper.value_or(false) ? APInt::getSignedMaxValue(width)
                                    : APInt::getSignedMinValue(width);
  };

  Block *block = iv->getParentBlock();
  APInt min = getLoopRangeInfo(lowerBound, block,
                               /*getUpper=*/false);
  APInt max = getLoopRangeInfo(upperBound, block,
                               /*getUpper=*/true);
  // We can assume step is 1 if no range information as that gives us the upper
  // bound of the number of iterations.
  APInt stepValDefault = {width, 1, /*isSigned=*/true};
  APInt stepVal =
      getLoopRangeInfo(step, block, /*getUpper=*/{}, stepValDefault);

  if (stepVal.isNegative())
    std::swap(min, max);
  // This is necessary to catch a case like this:
  //  # range = [0 1024]
  //  K = ....
  //  # range = [1, 64]
  //  k = ...
  //  # range = [0, 16] -> stepVal = range.smin() = 0
  //  step = ceildiv(K, k)
  if (stepVal.isZero())
    stepVal = stepValDefault;
  if (max.slt(min))
    return {};
  // Widest bounds and smallest step give the largest number of iterations the
  // loop can run. Over-estimating is safe because extra backedges only widen
  // loop carried values; under-estimating is not. Loops which may not run at
  // all are handled by always propagating the init args.
  int64_t tripCount = llvm::divideCeilSigned(
      max.getSExtValue() - min.getSExtValue(), stepVal.getSExtValue());
  // A negative count means the subtraction above overflowed, i.e. we do not
  // have a usable bound on the number of iterations.
  if (tripCount < 0)
    return {};
  return tripCount;
}

bool isEmptyInitializedRange(ConstantIntRanges rv) {
  if (!rv.umin().getBitWidth() || !rv.umax().getBitWidth() ||
      !rv.smin().getBitWidth() || !rv.smax().getBitWidth())
    return true;
  return false;
}

std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values) {
  SmallVector<std::optional<ConstantIntRanges>> ranges;
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange ||
        maybeInferredRange->getValue().isUninitialized()) {
      ranges.push_back(std::nullopt);
      continue;
    }
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    if (isEmptyInitializedRange(inferredRange)) {
      ranges.push_back(std::nullopt);
      continue;
    }
    ranges.push_back(inferredRange);
  }
  return ranges;
}

std::optional<bool> evaluateCmpI(const DataFlowSolver &solver,
                                 arith::CmpIOp cmpOp) {
  const auto *lattice =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(cmpOp.getResult());
  if (lattice && !lattice->getValue().isUninitialized()) {
    if (auto value = lattice->getValue().getValue().getConstantValue())
      return !value->isZero();
  }
  return std::nullopt;
}

IntegerValueRange TritonIntegerRangeAnalysis::getRangeAt(
    Value value, Operation *use, const IntegerValueRange &inferred) const {
  if (inferred.isUninitialized())
    return inferred;
  ConstantIntRanges range = inferred.getValue();
  // assume y > 5; if cond { assume z < 3; x = y + z; }
  // Both bounds apply to x; z's bound does not apply in the sibling branch.
  auto it = assumptions.find(value);
  if (it != assumptions.end())
    for (const IntegerAssumption &assumption : it->second)
      if (domInfo->properlyDominates(assumption.owner, use))
        range = range.intersection(assumption.range);
  // Intersection keeps signed and unsigned bounds separate; reconcile them.
  // For 0 <= x <= 1024, signed bounds can be [0, 1024] while umax is UINT_MAX.
  range = range.intersection(
      ConstantIntRanges::fromSigned(range.smin(), range.smax()));
  range = range.intersection(
      ConstantIntRanges::fromUnsigned(range.umin(), range.umax()));
  return IntegerValueRange(range);
}

int64_t
TritonIntegerRangeAnalysis::getTotalLoopTripCount(LoopLikeOpInterface loop) {
  SmallVector<LoopLikeOpInterface> loops{loop};
  getEnclosingLoops(*loop, loops);
  return std::accumulate(loops.begin(), loops.end(), (int64_t)1,
                         [this](int64_t accum, LoopLikeOpInterface loop) {
                           return accum * maybeGetTripCount(loop).value_or(
                                              kDefaultMaxTripCount + 1);
                         });
}

int64_t
TritonIntegerRangeAnalysis::getLoopSimulationSteps(LoopLikeOpInterface loop) {
  int64_t steps = getTotalLoopTripCount(loop);
  // A lattice of this loop changes once per iteration, but also once for every
  // step a nested loop needs to converge. This keeps the simulation steps
  // bounded for deeply nested loops.
  loop->walk([&](LoopLikeOpInterface nested) {
    if (nested == loop)
      return;
    int64_t nestedSteps = getTotalLoopTripCount(nested);
    if (nestedSteps <= kDefaultMaxTripCount)
      steps = std::max(steps, nestedSteps);
  });
  return steps;
}

void TritonIntegerRangeAnalysis::setToEntryState(
    dataflow::IntegerValueRangeLattice *lattice) {
  auto anchor = lattice->getAnchor();
  if (!llvm::isa<IndexType>(getElementTypeOrSelf(anchor)) &&
      !llvm::isa<IntegerType>(getElementTypeOrSelf(anchor)))
    return;

  IntegerValueRange range = IntegerValueRange::getMaxRange(anchor);
  auto changed = lattice->join(range);
  LLVM_DEBUG({
    if (changed == ChangeResult::Change) {
      DBGS() << "Set range of ";
      anchor.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << " to " << range << "\n";
    }
  });
  propagateIfChanged(lattice, changed);
}

void TritonIntegerRangeAnalysis::defaultTransferFunc(
    Operation *op, Value resultVal,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> srcLattices,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices,
    const IntegerValueRange &incomingRange) {

  auto result = dyn_cast<OpResult>(resultVal);
  if (!result)
    return;
  assert(llvm::is_contained(op->getResults(), result));

  dataflow::IntegerValueRangeLattice *lattice =
      resultsLattices[result.getResultNumber()];
  // Join takes the range union. Monotone updates are required for convergence.
  ChangeResult changed = lattice->join(incomingRange);
  LLVM_DEBUG({
    OpPrintingFlags flags;
    flags.skipRegions(true);
    DBGS() << ((changed == ChangeResult::Change) ? ">Inferred range for: "
                                                 : ">Remain unchanged: ");
    resultVal.printAsOperand(llvm::dbgs(), flags);
    llvm::dbgs() << ", resulting state:" << lattice->getValue()
                 << ", in value-range: " << incomingRange << "\n";
  });

  propagateIfChanged(lattice, changed);
}

LogicalResult TritonIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  SmallVector<const dataflow::IntegerValueRangeLattice *> inputs;
  SmallVector<std::unique_ptr<dataflow::IntegerValueRangeLattice>> storage;
  for (auto [value, lattice] : llvm::zip(op->getOperands(), operands)) {
    if (!assumptions.contains(value)) {
      inputs.push_back(lattice);
      continue;
    }
    // Local assumptions must not narrow the defining value for sibling uses.
    // Intersection with fixed assumption bounds preserves monotonicity.
    auto local = std::make_unique<dataflow::IntegerValueRangeLattice>(value);
    (void)local->join(getRangeAt(value, op, lattice->getValue()));
    inputs.push_back(local.get());
    storage.push_back(std::move(local));
  }
  return visitOperationHelper(op, inputs, resultsLattices);
}

LogicalResult TritonIntegerRangeAnalysis::visitOperationHelper(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  LDBG("Inferring ranges for " << *op);

  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &operands, &resultsLattices,
                       this](Value v, const IntegerValueRange &incomingRange) {
    this->defaultTransferFunc(op, v, operands, resultsLattices, incomingRange);
  };

  // Ops with fixed/constant ranges.
  if (llvm::isa<GetProgramIdOp, HistogramOp, GetNumProgramsOp>(op)) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<GetProgramIdOp>([&](auto getPIDOp) {
          int axis = getPIDOp.getAxisAsInt();
          auto it = pidBounds.find(axis);
          if (it != pidBounds.end())
            inferResultRangesPID(getPIDOp, it->second, joinCallback);
          else
            getPIDOp.inferResultRangesFromOptional({}, joinCallback);
        })
        .Case<GetNumProgramsOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<HistogramOp>([&](HistogramOp histOp) {
          return inferResultRangesMaxNonNegSigned(histOp, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  SmallVector<IntegerValueRange> argIntValueRanges = llvm::map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  if (isa<arith::DivSIOp, arith::RemSIOp>(op)) {
    if (!argIntValueRanges[0].isUninitialized() &&
        !argIntValueRanges[1].isUninitialized())
      joinCallback(op->getResult(0),
                   inferSignedDivRem(op, argIntValueRanges[0].getValue(),
                                     argIntValueRanges[1].getValue()));
    return success();
  }

  if (auto sliceOp = dyn_cast<triton::amdgpu::ExtractSliceOp>(op)) {
    joinCallback(sliceOp->getResult(0), argIntValueRanges[0]);
    return success();
  }

  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    inferrable.inferResultRangesFromOptional(argIntValueRanges, joinCallback);
    return success();
  }

  setAllToEntryStates(resultsLattices);
  return success();
}

void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionSuccessor successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  LLVM_DEBUG({
    DBGS() << "Visit Region Succesors of ";
    OpPrintingFlags flags;
    flags.skipRegions(true);
    branch.print(llvm::dbgs(), flags);
    llvm::dbgs() << "\n";
  });
  SmallVector<dataflow::IntegerValueRangeLattice *> lattices;
  for (auto abstractLat : abstractLattices) {
    lattices.push_back(
        static_cast<dataflow::IntegerValueRangeLattice *>(abstractLat));
  }
  // Initialize the number of simulation steps
  LoopLikeOpInterface loop =
      llvm::dyn_cast<LoopLikeOpInterface>(branch.getOperation());
  if (loop) {
    if (!loopSimulationSteps.contains(loop)) {
      loopSimulationSteps[loop] = 0;
      for (auto argLat : lattices)
        loopVisits[{loop, argLat}] = 0;
    }

    int64_t steps = getLoopSimulationSteps(loop);
    LLVM_DEBUG({
      DBGS() << "Simulation steps for ";
      OpPrintingFlags flags;
      flags.skipRegions(true);
      loop->print(llvm::dbgs(), flags);
      llvm::dbgs() << "\n";
      DBGS() << " --> " << steps << '\n';
    });
    if (steps < 0)
      steps = kDefaultMaxTripCount + 1;
    // We can only widen the simulation steps (conservative).
    loopSimulationSteps[loop] = std::max(loopSimulationSteps[loop], steps);
  }

  // Loop iter_args need init values plus values from simulated iterations.
  // Loop results need one extra backedge to observe the final iteration's yield
  int64_t maxBackedges = 0;
  if (loop)
    maxBackedges = successor.isOperation() ? loopSimulationSteps[loop]
                                           : loopSimulationSteps[loop] - 1;

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  // Note: It does not seems to be quite obvious; this loop could update SCF
  // operations' LHS. e.g. If the given "branch" argument is scf.if, and the
  // scf.if construct looks like following:
  //   x = scf.if cond
  //    m = ... // op_m
  //    yield m
  //   else
  //    n = ... // op_n
  //    yield n
  //
  // This loop tries to update lattice(x) = join(lattice(m), lattice(n),
  // provided lattice(m) and lattice(n) are initialized.
  //
  // Note that the state of lattice(m) and lattice(n) was updated in the
  // "previous" round. In this "round", the scf.if is visitied right now, and
  // it takes this moment to update its LHS.
  //
  // Alternatively, when we visit, say op_m, we notice its result is used by
  // a yieldOp, get the yieldOp's corresponding receiver, in this case x, and
  // update its state accordingly.
  //
  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    // Loop init args are not backedges and should never be blocked or counted.
    bool isBackedge = loop && op != branch.getOperation();
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }
    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    auto valueToLattices = [&](Value v) { return getLatticeElement(v); };
    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        RegionSuccessor parentSuccessor =
            RegionSuccessor(branch.getOperation());
        SmallVector<Value> nonSuccessorInputs =
            branch.getNonSuccessorInputs(parentSuccessor);
        SmallVector<dataflow::IntegerValueRangeLattice *>
            nonSuccessorInputLattices =
                llvm::map_to_vector(nonSuccessorInputs, valueToLattices);
        visitNonControlFlowArguments(branch, parentSuccessor,
                                     nonSuccessorInputs,
                                     nonSuccessorInputLattices);
      } else {
        if (!inputs.empty())
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        Region *region = point->getBlock()->getParent();
        SmallVector<Value> nonSuccessorInputs =
            branch.getNonSuccessorInputs(RegionSuccessor(region));
        SmallVector<dataflow::IntegerValueRangeLattice *>
            nonSuccessorInputLattices =
                llvm::map_to_vector(nonSuccessorInputs, valueToLattices);
        visitNonControlFlowArguments(branch, RegionSuccessor(region),
                                     nonSuccessorInputs,
                                     nonSuccessorInputLattices);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #maxBackedges times, stop propagating.
      if (isBackedge && loopVisits[loopArgLat] >= maxBackedges)
        continue;

      ChangeResult changed;
      if (loop && loopSimulationSteps[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).
        changed = argLat->join(IntegerValueRange::getMaxRange(oper));
      } else {
        // Else, propagate pred operands. Known-trip-count loops are bounded by
        // loopVisits, so join the value directly and avoid LLVM's generic
        // merge-site widening for long-but-finite loop simulations.
        auto *operLat = getLatticeElementFor(point, oper);
        changed =
            loop ? argLat->join(operLat->getValue()) : argLat->join(*operLat);
        LLVM_DEBUG({
          if (changed == ChangeResult::Change) {
            DBGS() << "Operand lattice ";
            oper.printAsOperand(llvm::dbgs(), {});
            llvm::dbgs() << " --> " << operLat->getValue() << "\n";
          }
        });
      }
      propagateIfChanged(argLat, changed);
      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (isBackedge && changed == ChangeResult::Change)
        ++loopVisits[loopArgLat];
    }
  }
}

IntegerAssumptions
TritonIntegerRangeAnalysis::collectAssumptions(Operation *rootOp,
                                               bool filterConstants) {
  IntegerAssumptions assumptions;
  rootOp->walk([&](LLVM::AssumeOp assume) {
    SmallVector<Value> pending{assume.getCond()};
    DenseSet<Value> visited;
    while (!pending.empty()) {
      Value condition = pending.pop_back_val();
      if (!visited.insert(condition).second)
        continue;
      if (auto conjunction = condition.getDefiningOp<arith::AndIOp>()) {
        llvm::append_range(pending, conjunction->getOperands());
        continue;
      }
      auto cmp = condition.getDefiningOp<arith::CmpIOp>();
      if (!cmp)
        continue;
      for (Value value : cmp->getOperands()) {
        if (filterConstants && getConstantIntValue(value))
          continue;
        if (auto range = triton::getBoundFromCmpOp(cmp, value))
          assumptions[value].push_back({*range, assume});
      }
    }
  });
  return assumptions;
}

struct FoldTrueCmpIOp : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  FoldTrueCmpIOp(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {};

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    auto result = evaluateCmpI(*solver, cmpOp);
    if (!result)
      return failure();

    TypedAttr constAttr = *result ? rewriter.getOneAttr(cmpOp.getType())
                                  : rewriter.getZeroAttr(cmpOp.getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(cmpOp, constAttr);
    return success();
  }

  DataFlowSolver *solver;
};

void populateFoldTrueCmpIOpPatterns(RewritePatternSet &patterns,
                                    DataFlowSolver *solver) {
  patterns.add<FoldTrueCmpIOp>(patterns.getContext(), solver);
}

} // namespace mlir::triton::AMD
