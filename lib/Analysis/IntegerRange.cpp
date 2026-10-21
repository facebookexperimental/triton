#include "triton/Analysis/IntegerRange.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/DebugLog.h"
#include <array>
#include <limits>

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "triton-integer-facts"

namespace {

struct IntegerBound {
  Value value;
  arith::CmpIPredicate predicate;
  bool operator==(const IntegerBound &other) const {
    return value == other.value && predicate == other.predicate;
  }
};

struct IntegerFacts {
  SmallVector<IntegerBound, 4> bounds;
  // Modulus zero denotes an exact bit pattern; one denotes no congruence.
  std::optional<std::pair<APInt, APInt>> congruence;

  bool operator==(const IntegerFacts &other) const {
    return bounds == other.bounds && congruence == other.congruence;
  }

  static IntegerFacts unknown(Value value) {
    unsigned width =
        std::max(1u, ConstantIntRanges::getStorageBitwidth(value.getType()));
    return {{},
            std::make_pair(APInt(2 * width + 2, 1), APInt(2 * width + 2, 0))};
  }

  static IntegerFacts join(const IntegerFacts &a, const IntegerFacts &b) {
    if (!a.congruence)
      return b;
    if (!b.congruence)
      return a;
    const auto &[am, ar] = *a.congruence;
    const auto &[bm, br] = *b.congruence;
    APInt delta = ar.uge(br) ? ar - br : br - ar;
    APInt modulus = llvm::APIntOps::GreatestCommonDivisor(
        llvm::APIntOps::GreatestCommonDivisor(am, bm), delta);
    IntegerFacts result{
        {}, std::make_pair(modulus, modulus.isZero() ? ar : ar.urem(modulus))};
    for (IntegerBound bound : a.bounds)
      if (llvm::is_contained(b.bounds, bound))
        result.bounds.push_back(bound);
    return result;
  }

  void print(raw_ostream &os) const {
    if (!congruence) {
      os << "uninitialized";
      return;
    }
    os << "modulus=" << congruence->first << ", residue=" << congruence->second;
  }
};

class IntegerFactsLattice : public dataflow::Lattice<IntegerFacts> {
public:
  using Lattice::Lattice;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerFactsLattice)
};

static Value stripRefinements(Value value) {
  while (auto refine = value.getDefiningOp<RefineOp>())
    value = refine.getValue();
  return value;
}

class IntegerFactsAnalysis final
    : public dataflow::SparseForwardDataFlowAnalysis<IntegerFactsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerFactsAnalysis)

  void setToEntryState(IntegerFactsLattice *lattice) override {
    propagateIfChanged(
        lattice, lattice->join(IntegerFacts::unknown(lattice->getAnchor())));
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor, ValueRange inputs,
      ArrayRef<IntegerFactsLattice *> lattices) override {
    auto loop = dyn_cast<scf::ForOp>(op);
    if (!loop || !isa<IntegerType>(loop.getInductionVar().getType())) {
      SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, inputs, lattices);
      return;
    }
    const auto *lower = getLatticeElementFor(
        getProgramPointBefore(loop.getBody()), loop.getLowerBound());
    const auto *step = getLatticeElementFor(
        getProgramPointBefore(loop.getBody()), loop.getStep());
    if (!lower->getValue().congruence || !step->getValue().congruence)
      return;
    const auto &[lm, lr] = *lower->getValue().congruence;
    const auto &[sm, sr] = *step->getValue().congruence;
    unsigned bits =
        cast<IntegerType>(loop.getInductionVar().getType()).getWidth();
    APInt modulus = llvm::APIntOps::GreatestCommonDivisor(
        llvm::APIntOps::GreatestCommonDivisor(lm, sm),
        llvm::APIntOps::GreatestCommonDivisor(
            sr, APInt::getOneBitSet(lm.getBitWidth(), bits)));
    IntegerFacts facts{{}, std::make_pair(modulus, lr.urem(modulus))};
    auto *iv = getLatticeElement(loop.getInductionVar());
    propagateIfChanged(iv, iv->join(facts));
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<const IntegerFactsLattice *> operands,
                 ArrayRef<IntegerFactsLattice *> results) override {
    if (results.size() != 1) {
      setAllToEntryStates(results);
      return success();
    }
    Value result = op->getResult(0);
    unsigned bits = ConstantIntRanges::getStorageBitwidth(result.getType());
    if (!bits) {
      setAllToEntryStates(results);
      return success();
    }
    for (const auto *operand : operands)
      if (!operand->getValue().congruence)
        return success();
    IntegerFacts facts = IntegerFacts::unknown(result);
    APInt constant;
    if (matchPattern(result, m_ConstantInt(&constant))) {
      facts.congruence = {APInt(2 * bits + 2, 0), constant.zext(2 * bits + 2)};
    } else if (auto refine = dyn_cast<RefineOp>(op)) {
      facts = operands[0]->getValue();
      if (refine.getPredicate() == arith::CmpIPredicate::eq &&
          operands[1]->getValue().congruence->first.isZero())
        facts.congruence = operands[1]->getValue().congruence;
      IntegerBound bound{refine.getBound(), refine.getPredicate()};
      // Keep a finite prefix. Existing relations cannot change order on
      // revisit.
      if (facts.bounds.size() < 8 && !llvm::is_contained(facts.bounds, bound))
        facts.bounds.push_back(bound);
      else if (!llvm::is_contained(facts.bounds, bound))
        LDBG() << "Symbolic bound budget exhausted: " << *op;
    } else if (isa<SplatOp, ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp,
                   triton::gpu::ConvertLayoutOp>(op)) {
      facts.congruence = operands[0]->getValue().congruence;
    } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      const auto &[am, ar] = *operands[0]->getValue().congruence;
      const auto &[bm, br] = *operands[1]->getValue().congruence;
      APInt wrap = APInt::getOneBitSet(2 * bits + 2, bits);
      APInt modulus = llvm::APIntOps::GreatestCommonDivisor(am, bm);
      APInt residue = ar + br;
      if (isa<arith::SubIOp>(op))
        residue = ar + wrap - br;
      if (isa<arith::MulIOp>(op)) {
        modulus = llvm::APIntOps::GreatestCommonDivisor(
            llvm::APIntOps::GreatestCommonDivisor(am * bm, am * br), bm * ar);
        residue = ar * br;
      }
      if (!modulus.isZero())
        modulus = llvm::APIntOps::GreatestCommonDivisor(modulus, wrap);
      facts.congruence = {modulus,
                          residue.urem(modulus.isZero() ? wrap : modulus)};
    }
    if (isa<arith::DivSIOp>(op)) {
      const auto *lhs = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
          getProgramPointAfter(op), op->getOperand(0));
      const auto *rhs = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
          getProgramPointAfter(op), op->getOperand(1));
      if (lhs->getValue().isUninitialized() ||
          rhs->getValue().isUninitialized())
        return success();
      if (lhs->getValue().getValue().smin().isNonNegative() &&
          rhs->getValue().getValue().smin().isStrictlyPositive()) {
        for (IntegerBound bound : operands[0]->getValue().bounds) {
          if (bound.predicate != arith::CmpIPredicate::slt)
            continue;
          auto product =
              stripRefinements(bound.value).getDefiningOp<arith::MulIOp>();
          if (!product)
            continue;
          Value divisor = stripRefinements(op->getOperand(1));
          Value factor;
          if (stripRefinements(product.getLhs()) == divisor)
            factor = product.getRhs();
          else if (stripRefinements(product.getRhs()) == divisor)
            factor = product.getLhs();
          if (!factor)
            continue;
          const auto *a = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
              getProgramPointAfter(op), product.getLhs());
          const auto *b = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
              getProgramPointAfter(op), product.getRhs());
          if (a->getValue().isUninitialized() ||
              b->getValue().isUninitialized())
            return success();
          bool noWrap = product.hasNoSignedWrap();
          if (!noWrap) {
            const ConstantIntRanges &ar = a->getValue().getValue();
            const ConstantIntRanges &br = b->getValue().getValue();
            noWrap = true;
            for (const APInt &av : {ar.smin(), ar.smax()})
              for (const APInt &bv : {br.smin(), br.smax()}) {
                bool overflow;
                (void)av.smul_ov(bv, overflow);
                noWrap &= !overflow;
              }
          }
          if (noWrap)
            facts.bounds.push_back({factor, arith::CmpIPredicate::slt});
        }
      }
    }
    propagateIfChanged(results[0], results[0]->join(facts));
    return success();
  }
};

static ConstantIntRanges refineRange(arith::CmpIPredicate predicate,
                                     const ConstantIntRanges &value,
                                     const ConstantIntRanges &bound) {
  using P = arith::CmpIPredicate;
  bool isUnsigned = predicate == P::ult || predicate == P::ule ||
                    predicate == P::ugt || predicate == P::uge;
  APInt lo = isUnsigned ? value.umin() : value.smin();
  APInt hi = isUnsigned ? value.umax() : value.smax();
  APInt boundLo = isUnsigned ? bound.umin() : bound.smin();
  APInt boundHi = isUnsigned ? bound.umax() : bound.smax();
  auto less = [&](const APInt &a, const APInt &b) {
    return isUnsigned ? a.ult(b) : a.slt(b);
  };
  auto setLower = [&](const APInt &v) {
    if (less(lo, v))
      lo = v;
  };
  auto setUpper = [&](const APInt &v) {
    if (less(v, hi))
      hi = v;
  };
  switch (predicate) {
  case P::eq:
    setLower(boundLo);
    setUpper(boundHi);
    break;
  case P::slt:
  case P::ult:
    if (isUnsigned ? boundHi.isZero() : boundHi.isMinSignedValue())
      return value;
    setUpper(boundHi - 1);
    break;
  case P::sle:
  case P::ule:
    setUpper(boundHi);
    break;
  case P::sgt:
  case P::ugt:
    if (isUnsigned ? boundLo.isMaxValue() : boundLo.isMaxSignedValue())
      return value;
    setLower(boundLo + 1);
    break;
  case P::sge:
  case P::uge:
    setLower(boundLo);
    break;
  case P::ne:
    if (boundLo == boundHi && lo != hi) {
      if (lo == boundLo)
        ++lo;
      if (hi == boundHi)
        --hi;
    }
    break;
  }
  // Contradictions poison this result; they do not make its block unreachable.
  if (less(hi, lo))
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
      auto *lattice = results[cast<OpResult>(value).getResultNumber()];
      const auto *facts =
          getOrCreateFor<IntegerFactsLattice>(getProgramPointAfter(op), value);
      if (!facts->getValue().congruence || range.isUninitialized())
        return;
      ConstantIntRanges refined = range.getValue();
      const auto &[modulus, residue] = *facts->getValue().congruence;
      if (modulus.isZero()) {
        APInt exact = residue.trunc(refined.umin().getBitWidth());
        if (exact.uge(refined.umin()) && exact.ule(refined.umax()) &&
            exact.sge(refined.smin()) && exact.sle(refined.smax()))
          refined = ConstantIntRanges::constant(exact);
      } else if (!modulus.isOne()) {
        APInt lo = refined.umin().zext(modulus.getBitWidth());
        APInt hi = refined.umax().zext(modulus.getBitWidth());
        lo += (residue + modulus - lo.urem(modulus)).urem(modulus);
        APInt delta = (hi + modulus - residue).urem(modulus);
        if (hi.uge(delta) && lo.ule(hi - delta))
          refined = refined.intersection(ConstantIntRanges::fromUnsigned(
              lo.trunc(refined.umin().getBitWidth()),
              (hi - delta).trunc(refined.umin().getBitWidth())));
      }
      for (IntegerBound bound : facts->getValue().bounds) {
        const auto *limit = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
            getProgramPointAfter(op), bound.value);
        if (limit->getValue().isUninitialized())
          return;
        refined =
            refineRange(bound.predicate, refined, limit->getValue().getValue());
      }
      propagateIfChanged(lattice, lattice->join(refined));
    };
    if (inferTritonRanges(op, ranges, setRange, maxProgramIds))
      return success();
    return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                          results);
  }

private:
  std::array<int64_t, 3> maxProgramIds;
};

} // namespace

IntegerFactAnalysis::IntegerFactAnalysis() = default;
IntegerFactAnalysis::~IntegerFactAnalysis() = default;

LogicalResult IntegerFactAnalysis::initialize(Operation *root) {
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

  solver = createDataFlowSolver();
  solver->load<IntegerFactsAnalysis>();
  solver->load<IntegerRangeAnalysis>(maxProgramIds);
  return solver->initializeAndRun(root);
}

IntegerValueRange IntegerFactAnalysis::getRange(Value value) const {
  assert(solver && "initialize integer facts before querying");
  const auto *state =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(value);
  return state ? state->getValue() : IntegerValueRange::getMaxRange(value);
}

IntegerProof IntegerFactAnalysis::prove(arith::CmpIPredicate predicate,
                                        Value lhs, Value rhs) const {
  assert(solver && "initialize integer facts before querying");
  if (const auto *facts = solver->lookupState<IntegerFactsLattice>(lhs))
    for (IntegerBound bound : facts->getValue().bounds)
      if (stripRefinements(bound.value) == stripRefinements(rhs)) {
        if (bound.predicate == predicate)
          return IntegerProof::Proven;
        if (bound.predicate == arith::invertPredicate(predicate))
          return IntegerProof::Disproven;
      }
  IntegerValueRange a = getRange(lhs);
  IntegerValueRange b = getRange(rhs);
  if (a.isUninitialized() || b.isUninitialized())
    return IntegerProof::Unknown;
  using P = arith::CmpIPredicate;
  intrange::CmpPredicate pred;
  switch (predicate) {
  case P::eq:
    pred = intrange::CmpPredicate::eq;
    break;
  case P::ne:
    pred = intrange::CmpPredicate::ne;
    break;
  case P::slt:
    pred = intrange::CmpPredicate::slt;
    break;
  case P::sle:
    pred = intrange::CmpPredicate::sle;
    break;
  case P::sgt:
    pred = intrange::CmpPredicate::sgt;
    break;
  case P::sge:
    pred = intrange::CmpPredicate::sge;
    break;
  case P::ult:
    pred = intrange::CmpPredicate::ult;
    break;
  case P::ule:
    pred = intrange::CmpPredicate::ule;
    break;
  case P::ugt:
    pred = intrange::CmpPredicate::ugt;
    break;
  case P::uge:
    pred = intrange::CmpPredicate::uge;
    break;
  }
  std::optional<bool> result =
      intrange::evaluatePred(pred, a.getValue(), b.getValue());
  if (!result)
    return IntegerProof::Unknown;
  return *result ? IntegerProof::Proven : IntegerProof::Disproven;
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
  IntegerFactAnalysis facts;
  if (failed(facts.initialize(root)))
    return failure();
  for (Operation *op : queries) {
    IntegerValueRange range = facts.getRange(op->getOperand(0));
    if (!range.isUninitialized() && range.getValue().smin().isNonNegative())
      operations.insert(op);
  }
  return success();
}
