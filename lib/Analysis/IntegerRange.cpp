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
#include "llvm/Support/MathExtras.h"
#include <array>
#include <limits>
#include <map>
#include <tuple>

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
  enum class Sign { Unknown, Nonnegative, Positive };
  SmallVector<IntegerBound, 4> bounds;
  // Modulus zero denotes an exact bit pattern; one denotes no congruence.
  std::optional<std::pair<APInt, APInt>> congruence;
  Sign sign = Sign::Unknown;

  bool operator==(const IntegerFacts &other) const {
    return bounds == other.bounds && congruence == other.congruence &&
           sign == other.sign;
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
    result.sign = std::min(a.sign, b.sign);
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

// 64 SSA nodes; 16 relations/terms; degree four; normalization depth 12.
// K relations and T terms: pairwise proof costs O(K^2*T*log(T)).
// At most eight bounds and eight positive scale factors per transfer.
class IntegerRelations {
  struct Symbol {
    unsigned value;
    unsigned divisor;
    bool operator<(const Symbol &other) const {
      return std::tie(value, divisor) < std::tie(other.value, other.divisor);
    }
    bool operator==(const Symbol &other) const {
      return value == other.value && divisor == other.divisor;
    }
  };
  using Monomial = SmallVector<Symbol, 2>;
  using Polynomial = std::map<Monomial, int64_t>;
  struct Constraint {
    Polynomial expression;
    bool strict;
  };

public:
  IntegerRelations(ValueRange roots,
                   function_ref<IntegerValueRange(Value)> getRange) {
    SmallVector<Value> pending(roots);
    DenseSet<Value> visited;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      if (values.size() == 64) {
        LDBG() << "Integer relation SSA budget exhausted";
        valid = false;
        return;
      }
      values.push_back(value);
      if (ConstantIntRanges::getStorageBitwidth(value.getType()) > 64) {
        valid = false;
        return;
      }
      IntegerValueRange range = getRange(value);
      if (!range.isUninitialized()) {
        Value identity = stripRefinements(value);
        auto it = ranges.find(identity);
        if (it == ranges.end())
          ranges.try_emplace(identity, range.getValue());
        else
          it->second = it->second.intersection(range.getValue());
      }
      Operation *op = value.getDefiningOp();
      // These operations require all operands to be non-poison. Select does
      // not.
      if (op &&
          isa<RefineOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::DivSIOp, arith::RemSIOp, arith::MinSIOp, arith::MaxSIOp>(
              op))
        llvm::append_range(pending, op->getOperands());
    }
    for (Value value : values) {
      if (auto refine = value.getDefiningOp<RefineOp>()) {
        using P = arith::CmpIPredicate;
        P pred = refine.getPredicate();
        if ((pred == P::slt || pred == P::sle) &&
            !matchPattern(refine.getBound(), m_Constant()))
          addConstraint(
              subtract(normalize(refine.getBound()), normalize(value)),
              pred == P::slt);
        if ((pred == P::sgt || pred == P::sge) &&
            !matchPattern(refine.getBound(), m_Constant()))
          addConstraint(
              subtract(normalize(value), normalize(refine.getBound())),
              pred == P::sgt);
      }
      Operation *op = value.getDefiningOp();
      if (auto minimum = dyn_cast_or_null<arith::MinSIOp>(op)) {
        addConstraint(subtract(normalize(minimum.getLhs()), normalize(value)),
                      false);
        addConstraint(subtract(normalize(minimum.getRhs()), normalize(value)),
                      false);
      }
      if (!op || !isa<arith::DivSIOp, arith::RemSIOp>(op) ||
          !nonnegativeRange(op->getOperand(0)) ||
          !positiveRange(op->getOperand(1)))
        continue;
      auto dividend = normalize(op->getOperand(0));
      auto divisor = normalize(op->getOperand(1));
      auto quotient = quotientSymbol(op->getOperand(0), op->getOperand(1));
      auto remainder = subtract(dividend, multiply(quotient, divisor));
      addConstraint(remainder, false);
      addConstraint(subtract(divisor, remainder), true);
    }
  }

  bool nonnegative(Value value) {
    if (auto minimum = stripRefinements(value).getDefiningOp<arith::MinSIOp>())
      return prove(normalize(minimum.getLhs()), false) &&
             prove(normalize(minimum.getRhs()), false);
    return prove(normalize(value), false);
  }

  bool positive(Value value) {
    if (!valid)
      return false;
    if (auto minimum = stripRefinements(value).getDefiningOp<arith::MinSIOp>())
      return prove(normalize(minimum.getLhs()), true) &&
             prove(normalize(minimum.getRhs()), true);
    return prove(normalize(value), true);
  }

  bool productUpperBound(Value value, Value divisor, Value bound) {
    if (!valid)
      return false;
    auto upper = normalize(bound);
    auto input = normalize(value);
    if (auto minimum =
            stripRefinements(divisor).getDefiningOp<arith::MinSIOp>())
      return prove(
                 subtract(multiply(normalize(minimum.getLhs()), upper), input),
                 true) &&
             prove(
                 subtract(multiply(normalize(minimum.getRhs()), upper), input),
                 true);
    return prove(subtract(multiply(normalize(divisor), upper), input), true);
  }

  ArrayRef<Value> candidates() const { return values; }
  bool hasRelations() const { return valid && !constraints.empty(); }
  bool upperBound(Value value, Value bound) {
    return prove(subtract(normalize(bound), normalize(value)), true);
  }

private:
  static std::optional<Polynomial> add(std::optional<Polynomial> a,
                                       const std::optional<Polynomial> &b,
                                       int64_t sign = 1) {
    if (!a || !b)
      return std::nullopt;
    for (const auto &[term, coefficient] : *b) {
      int64_t scaled, sum;
      if (llvm::MulOverflow(coefficient, sign, scaled) ||
          llvm::AddOverflow((*a)[term], scaled, sum))
        return std::nullopt;
      if (sum)
        (*a)[term] = sum;
      else
        a->erase(term);
      if (a->size() > 16)
        return std::nullopt;
    }
    return a;
  }

  static std::optional<Polynomial>
  subtract(std::optional<Polynomial> a, const std::optional<Polynomial> &b) {
    return add(std::move(a), b, -1);
  }

  static std::optional<Polynomial>
  multiply(const std::optional<Polynomial> &a,
           const std::optional<Polynomial> &b) {
    if (!a || !b)
      return std::nullopt;
    std::optional<Polynomial> result = Polynomial{};
    for (const auto &[at, ac] : *a)
      for (const auto &[bt, bc] : *b) {
        if (at.size() + bt.size() > 4)
          return std::nullopt;
        int64_t coefficient;
        if (llvm::MulOverflow(ac, bc, coefficient))
          return std::nullopt;
        Monomial term(at);
        llvm::append_range(term, bt);
        llvm::sort(term);
        result = add(std::move(result), Polynomial{{term, coefficient}});
        if (!result)
          return std::nullopt;
      }
    return result;
  }

  unsigned symbolID(Value value) {
    value = stripRefinements(value);
    auto [it, inserted] = symbolIDs.try_emplace(value, symbols.size());
    if (inserted)
      symbols.push_back(value);
    return it->second;
  }

  Polynomial quotientSymbol(Value dividend, Value divisor) {
    return {{{{symbolID(dividend), symbolID(divisor)}}, 1}};
  }

  bool positiveRange(Value value) const {
    auto it = ranges.find(stripRefinements(value));
    return it != ranges.end() && it->second.smin().isStrictlyPositive();
  }
  bool nonnegativeRange(Value value) const {
    auto it = ranges.find(stripRefinements(value));
    return it != ranges.end() && it->second.smin().isNonNegative();
  }

  bool noSignedWrap(Operation *op) const {
    if (auto overflow = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op))
      if (overflow.hasNoSignedWrap())
        return true;
    auto a = ranges.find(stripRefinements(op->getOperand(0)));
    auto b = ranges.find(stripRefinements(op->getOperand(1)));
    if (a == ranges.end() || b == ranges.end())
      return false;
    bool noOverflow = true;
    for (const APInt &av : {a->second.smin(), a->second.smax()})
      for (const APInt &bv : {b->second.smin(), b->second.smax()}) {
        bool overflow;
        if (isa<arith::AddIOp>(op))
          (void)av.sadd_ov(bv, overflow);
        else if (isa<arith::SubIOp>(op))
          (void)av.ssub_ov(bv, overflow);
        else
          (void)av.smul_ov(bv, overflow);
        noOverflow &= !overflow;
      }
    if (noOverflow)
      return true;
    if (!isa<arith::MulIOp>(op) || !a->second.smin().isNonNegative() ||
        !b->second.smin().isNonNegative())
      return false;
    // 0 <= a <= limit / b, b > 0 implies 0 <= a*b <= limit.
    for (Value value : values) {
      auto refine = value.getDefiningOp<RefineOp>();
      if (!refine || (refine.getPredicate() != arith::CmpIPredicate::sle &&
                      refine.getPredicate() != arith::CmpIPredicate::slt))
        continue;
      auto quotient =
          stripRefinements(refine.getBound()).getDefiningOp<arith::DivSIOp>();
      if (!quotient || !nonnegativeRange(quotient.getLhs()) ||
          !positiveRange(quotient.getRhs()))
        continue;
      Value bounded = stripRefinements(refine.getValue());
      Value divisor = stripRefinements(quotient.getRhs());
      Value lhs = stripRefinements(op->getOperand(0));
      Value rhs = stripRefinements(op->getOperand(1));
      if ((bounded == lhs && divisor == rhs) ||
          (bounded == rhs && divisor == lhs))
        return true;
    }
    return false;
  }

  std::optional<Polynomial> normalize(Value value, unsigned depth = 0) {
    if (depth == 12)
      return std::nullopt;
    value = stripRefinements(value);
    if (auto constant = getConstantIntValue(value))
      return Polynomial{{{}, *constant}};
    Operation *op = value.getDefiningOp();
    if (op && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op) &&
        noSignedWrap(op)) {
      auto a = normalize(op->getOperand(0), depth + 1);
      auto b = normalize(op->getOperand(1), depth + 1);
      if (isa<arith::AddIOp>(op))
        return add(a, b);
      if (isa<arith::SubIOp>(op))
        return subtract(a, b);
      return multiply(a, b);
    }
    if (op && isa<arith::DivSIOp>(op))
      return quotientSymbol(op->getOperand(0), op->getOperand(1));
    if (op && isa<arith::RemSIOp>(op) && nonnegativeRange(op->getOperand(0)) &&
        positiveRange(op->getOperand(1)))
      return subtract(
          normalize(op->getOperand(0), depth + 1),
          multiply(quotientSymbol(op->getOperand(0), op->getOperand(1)),
                   normalize(op->getOperand(1), depth + 1)));
    return Polynomial{{{{symbolID(value), 0}}, 1}};
  }

  void addConstraint(std::optional<Polynomial> expression, bool strict) {
    if (expression)
      for (Constraint &constraint : constraints)
        if (constraint.expression == *expression) {
          constraint.strict |= strict;
          return;
        }
    if (expression && constraints.size() < 16)
      constraints.push_back({std::move(*expression), strict});
  }

  bool constantNonnegative(const Polynomial &expression, bool strict) const {
    APInt sum(256, 0);
    for (const auto &[term, coefficient] : expression) {
      APInt lo(256, 1), hi(256, 1);
      for (Symbol symbol : term) {
        auto it = ranges.find(symbols[symbol.value]);
        if (it == ranges.end() || it->second.smin().getBitWidth() > 64)
          return false;
        APInt a = it->second.smin().sext(256);
        APInt b = it->second.smax().sext(256);
        if (symbol.divisor) {
          auto divisor = ranges.find(symbols[symbol.divisor]);
          if (a.isNegative() || divisor == ranges.end() ||
              !divisor->second.smin().isStrictlyPositive())
            return false;
          a = a.sdiv(divisor->second.smax().sext(256));
          b = b.sdiv(divisor->second.smin().sext(256));
        }
        std::array<APInt, 4> products{lo * a, lo * b, hi * a, hi * b};
        lo = *llvm::min_element(
            products, [](const APInt &x, const APInt &y) { return x.slt(y); });
        hi = *llvm::max_element(
            products, [](const APInt &x, const APInt &y) { return x.slt(y); });
      }
      bool overflow;
      APInt product = (coefficient < 0 ? hi : lo)
                          .smul_ov(APInt(256, coefficient, true), overflow);
      if (overflow)
        return false;
      sum = sum.sadd_ov(product, overflow);
      if (overflow)
        return false;
    }
    return strict ? sum.isStrictlyPositive() : sum.isNonNegative();
  }

  bool proveDirect(const Polynomial &expression, bool strict) const {
    if (constantNonnegative(expression, strict))
      return true;
    for (const Constraint &constraint : constraints)
      if (constraint.expression == expression && (!strict || constraint.strict))
        return true;
    for (const Constraint &a : constraints) {
      if (!cancelsTerm(expression, a.expression))
        continue;
      auto rest = subtract(expression, a.expression);
      if (!rest)
        continue;
      if (constantNonnegative(*rest, strict && !a.strict))
        return true;
      for (const Constraint &b : constraints) {
        if (!cancelsTerm(*rest, b.expression))
          continue;
        auto tail = subtract(rest, b.expression);
        if (tail &&
            constantNonnegative(*tail, strict && !a.strict && !b.strict))
          return true;
      }
    }
    return false;
  }

  static bool cancelsTerm(const Polynomial &expression,
                          const Polynomial &constraint) {
    for (const auto &[term, coefficient] : constraint) {
      auto it = expression.find(term);
      if (!term.empty() && it != expression.end() && it->second == coefficient)
        return true;
    }
    return false;
  }

  bool prove(const std::optional<Polynomial> &expression, bool strict) {
    if (!valid || !expression)
      return false;
    if (proveDirect(*expression, strict))
      return true;
    // Positive scaling preserves the inequality; never assume a divisor's sign.
    DenseSet<Value> tried;
    for (Value value : values) {
      if (!positiveRange(value) || matchPattern(value, m_Constant()))
        continue;
      if (!tried.insert(stripRefinements(value)).second)
        continue;
      if (tried.size() > 8)
        break;
      auto scaled = multiply(expression, normalize(value));
      if (scaled && proveDirect(*scaled, strict))
        return true;
    }
    return false;
  }

  SmallVector<Value> values;
  SmallVector<Value> symbols{Value{}};
  SmallVector<Constraint> constraints;
  DenseMap<Value, unsigned> symbolIDs;
  DenseMap<Value, ConstantIntRanges> ranges;
  bool valid = true;
};

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
    if (isa<arith::DivSIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
            arith::MinSIOp>(op) &&
        isa<IntegerType>(result.getType())) {
      bool pending = false;
      auto range = [&](Value value) {
        const auto *state = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
            getProgramPointAfter(op), value);
        pending |= state->getValue().isUninitialized();
        return state->getValue();
      };
      IntegerRelations relations(op->getOperands(), range);
      if (pending)
        return success();
      bool division = isa<arith::DivSIOp>(op);
      Value dividend = op->getOperand(0), divisor = op->getOperand(1);
      bool nonnegative =
          division
              ? relations.nonnegative(dividend) && relations.positive(divisor)
              : relations.hasRelations() && relations.nonnegative(result);
      if (nonnegative) {
        DenseSet<Value> tried;
        for (Value bound : relations.candidates()) {
          IntegerValueRange limit = range(bound);
          if (bound.getType() != result.getType() ||
              matchPattern(bound, m_Constant()) || limit.isUninitialized() ||
              !limit.getValue().smin().isStrictlyPositive() ||
              !tried.insert(stripRefinements(bound)).second)
            continue;
          if (tried.size() > 8)
            break;
          bool proven =
              division ? relations.productUpperBound(dividend, divisor, bound)
                       : relations.upperBound(result, bound);
          if (!proven)
            continue;
          if (facts.bounds.size() < 8)
            facts.bounds.push_back({bound, arith::CmpIPredicate::slt});
        }
        facts.sign = !division && relations.positive(result)
                         ? IntegerFacts::Sign::Positive
                         : IntegerFacts::Sign::Nonnegative;
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
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        return;
      auto *lattice = results[result.getResultNumber()];
      const auto *facts =
          getOrCreateFor<IntegerFactsLattice>(getProgramPointAfter(op), value);
      if (!facts->getValue().congruence || range.isUninitialized())
        return;
      ConstantIntRanges refined = range.getValue();
      IntegerFacts::Sign sign = facts->getValue().sign;
      if (sign != IntegerFacts::Sign::Unknown) {
        unsigned width = refined.smin().getBitWidth();
        refined = refined.intersection(ConstantIntRanges::fromSigned(
            APInt(width, sign == IntegerFacts::Sign::Positive),
            APInt::getSignedMaxValue(width)));
      }
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

IntegerFactAnalysis::IntegerFactAnalysis() = default;
IntegerFactAnalysis::~IntegerFactAnalysis() = default;

std::unique_ptr<DataFlowSolver>
mlir::triton::createIntegerFactSolver(Operation *root) {
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
  solver->load<IntegerFactsAnalysis>();
  solver->load<IntegerRangeAnalysis>(maxProgramIds);
  return solver;
}

LogicalResult IntegerFactAnalysis::initialize(Operation *root) {
  solver = createIntegerFactSolver(root);
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
