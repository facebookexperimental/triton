#include "triton/Analysis/IntegerRange.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
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
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "triton-integer-facts"

namespace {

static Value stripRefinements(Value value);

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
    for (IntegerBound bound : a.bounds) {
      if (llvm::is_contained(b.bounds, bound)) {
        if (!llvm::is_contained(result.bounds, bound))
          result.bounds.push_back(bound);
        continue;
      }
      // Distinct refinements agree on bits, but not on poison across edges.
      bound.value = stripRefinements(bound.value);
      if (llvm::any_of(b.bounds,
                       [&](IntegerBound other) {
                         return other.predicate == bound.predicate &&
                                stripRefinements(other.value) == bound.value;
                       }) &&
          !llvm::is_contained(result.bounds, bound))
        result.bounds.push_back(bound);
    }
    result.sign = std::min(a.sign, b.sign);
    return result;
  }

  void print(raw_ostream &os) const {
    if (!congruence) {
      os << "uninitialized";
      return;
    }
    os << "modulus=" << congruence->first << ", residue=" << congruence->second;
    os << ", sign=" << static_cast<unsigned>(sign);
    for (IntegerBound bound : bounds)
      os << ", " << stringifyCmpIPredicate(bound.predicate) << " "
         << bound.value;
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

// 64 SSA nodes/relations; 16 terms; degree four; normalization depth 12.
// K relations, T terms: O(K^2*T^4*log(T)); cap pair subtractions at 256.
// At most eight candidate bounds and eight positive scale factors per transfer.
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
  using Monomial = SmallVector<Symbol, 4>;
  // Sorted, bounded terms avoid a tree allocation for each proof subtraction.
  using Polynomial = SmallVector<std::pair<Monomial, int64_t>, 16>;
  struct Constraint {
    Polynomial expression;
    bool strict;
  };

public:
  IntegerRelations(ValueRange roots,
                   function_ref<IntegerValueRange(Value)> getRange,
                   function_ref<const IntegerFacts &(Value)> getFacts)
      : getFacts(getFacts) {
    collectValues(roots, getRange);
    if (!valid)
      return;
    for (Value value : values) {
      addRefinementConstraint(value);
      addArithmeticConstraints(value);
    }
    for (auto [value, bound] : inheritedBounds)
      addInheritedBound(value, bound);
  }

private:
  void collectValues(ValueRange roots,
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
      for (IntegerBound bound : getFacts(value).bounds) {
        inheritedBounds.emplace_back(value, bound);
        pending.push_back(bound.value);
      }
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
      if (op && isa<RefineOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
                    arith::DivSIOp, arith::RemSIOp, arith::DivUIOp,
                    arith::RemUIOp, arith::MinSIOp, arith::MinUIOp,
                    arith::MaxSIOp, arith::ExtSIOp, arith::ExtUIOp>(op))
        llvm::append_range(pending, op->getOperands());
    }
  }

  void addRefinementConstraint(Value value) {
    if (auto refine = value.getDefiningOp<RefineOp>()) {
      using P = arith::CmpIPredicate;
      P pred = refine.getPredicate();
      if ((pred == P::slt || pred == P::sle) &&
          !matchPattern(refine.getBound(), m_Constant()))
        addConstraint(subtract(normalize(refine.getBound()), normalize(value)),
                      pred == P::slt);
      if ((pred == P::sgt || pred == P::sge) &&
          !matchPattern(refine.getBound(), m_Constant()))
        addConstraint(subtract(normalize(value), normalize(refine.getBound())),
                      pred == P::sgt);
    }
  }

  void addArithmeticConstraints(Value value) {
    Operation *op = value.getDefiningOp();
    if (Operation *minimum = signedMinimum(value)) {
      addConstraint(
          subtract(normalize(minimum->getOperand(0)), normalize(value)), false);
      addConstraint(
          subtract(normalize(minimum->getOperand(1)), normalize(value)), false);
    }
    if (!op ||
        !isa<arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp>(
            op) ||
        !nonnegativeRange(op->getOperand(0)) ||
        !positiveRange(op->getOperand(1)))
      return;
    auto dividend = normalize(op->getOperand(0));
    auto divisor = normalize(op->getOperand(1));
    auto quotient = quotientSymbol(op->getOperand(0), op->getOperand(1));
    auto remainder = subtract(dividend, multiply(quotient, divisor));
    addConstraint(remainder, false);
    addConstraint(subtract(divisor, remainder), true);
  }

  void addInheritedBound(Value value, IntegerBound bound) {
    if (matchPattern(bound.value, m_Constant()))
      return;
    using P = arith::CmpIPredicate;
    P pred = bound.predicate;
    bool unsignedOrder =
        nonnegativeRange(value) && nonnegativeRange(bound.value);
    if (pred == P::slt || pred == P::sle ||
        (unsignedOrder && (pred == P::ult || pred == P::ule)))
      addConstraint(subtract(normalize(bound.value), normalize(value)),
                    pred == P::slt || pred == P::ult);
    if (pred == P::sgt || pred == P::sge ||
        (unsignedOrder && (pred == P::ugt || pred == P::uge)))
      addConstraint(subtract(normalize(value), normalize(bound.value)),
                    pred == P::sgt || pred == P::ugt);
  }

public:
  bool nonnegative(Value value) {
    if (nonnegativeRange(value))
      return true;
    Operation *op = stripRefinements(value).getDefiningOp();
    if (op && isa<arith::AddIOp, arith::MulIOp>(op) && noSignedWrap(op) &&
        nonnegativeRange(op->getOperand(0)) &&
        nonnegativeRange(op->getOperand(1)))
      return true;
    if (Operation *minimum = signedMinimum(value))
      return prove(normalize(minimum->getOperand(0)), false) &&
             prove(normalize(minimum->getOperand(1)), false);
    return prove(normalize(value), false);
  }

  bool positive(Value value) {
    if (!valid)
      return false;
    if (positiveRange(value))
      return true;
    if (Operation *minimum = signedMinimum(value))
      return prove(normalize(minimum->getOperand(0)), true) &&
             prove(normalize(minimum->getOperand(1)), true);
    return prove(normalize(value), true);
  }

  bool productUpperBound(Value value, Value divisor, Value bound) {
    if (!valid)
      return false;
    auto upper = normalize(bound);
    auto input = normalize(value);
    if (Operation *minimum = signedMinimum(divisor))
      return prove(subtract(multiply(normalize(minimum->getOperand(0)), upper),
                            input),
                   true) &&
             prove(subtract(multiply(normalize(minimum->getOperand(1)), upper),
                            input),
                   true);
    return prove(subtract(multiply(normalize(divisor), upper), input), true);
  }

  ArrayRef<Value> candidates() const { return values; }
  bool hasRelations() const { return valid && !constraints.empty(); }
  bool upperBound(Value value, Value bound, bool strict = true) {
    if (auto add = value.getDefiningOp<arith::AddIOp>();
        add && noSignedWrap(add))
      for (unsigned i : llvm::seq(2u)) {
        Value base = add->getOperand(i), offset = add->getOperand(1 - i);
        auto range = ranges.find(stripRefinements(offset));
        if (range == ranges.end())
          continue;
        for (IntegerBound upper : getFacts(base).bounds) {
          using P = arith::CmpIPredicate;
          if (stripRefinements(upper.value) != stripRefinements(bound) ||
              (upper.predicate != P::slt && upper.predicate != P::sle))
            continue;
          int64_t gap = congruenceGap(base, upper);
          APInt maximum = range->second.smax();
          APInt distance(maximum.getBitWidth(), gap);
          if (strict ? maximum.slt(distance) : maximum.sle(distance))
            return true;
        }
      }
    return prove(subtract(normalize(bound), normalize(value)), strict);
  }

private:
  int64_t congruenceGap(Value base, IntegerBound upper) const {
    int64_t gap = upper.predicate == arith::CmpIPredicate::slt;
    const auto &a = getFacts(base).congruence;
    const auto &b = getFacts(upper.value).congruence;
    if (a && b) {
      APInt modulus = llvm::APIntOps::GreatestCommonDivisor(a->first, b->first);
      if (!modulus.isZero() && modulus.getActiveBits() < 64 &&
          (modulus.isPowerOf2() ||
           (nonnegativeRange(base) && nonnegativeRange(upper.value)))) {
        APInt residue =
            (b->second.urem(modulus) + modulus - a->second.urem(modulus))
                .urem(modulus);
        gap = (residue.isZero() && gap ? modulus : residue).getZExtValue();
      }
    }
    return gap;
  }

  Operation *signedMinimum(Value value) const {
    Operation *op = stripRefinements(value).getDefiningOp();
    if (op &&
        (isa<arith::MinSIOp>(op) ||
         (isa<arith::MinUIOp>(op) && nonnegativeRange(op->getOperand(0)) &&
          nonnegativeRange(op->getOperand(1)))))
      return op;
    return nullptr;
  }

  static std::optional<Polynomial> add(std::optional<Polynomial> a,
                                       const std::optional<Polynomial> &b,
                                       int64_t sign = 1) {
    if (!a || !b)
      return std::nullopt;
    Polynomial result;
    auto lhs = a->begin();
    auto rhs = b->begin();
    while (lhs != a->end() || rhs != b->end()) {
      if (rhs == b->end() || (lhs != a->end() && lhs->first < rhs->first)) {
        result.push_back(*lhs++);
      } else {
        int64_t scaled, sum;
        if (llvm::MulOverflow(rhs->second, sign, scaled))
          return std::nullopt;
        if (lhs != a->end() && lhs->first == rhs->first) {
          if (llvm::AddOverflow(lhs->second, scaled, sum))
            return std::nullopt;
          ++lhs;
        } else {
          sum = scaled;
        }
        if (sum)
          result.emplace_back(rhs->first, sum);
        ++rhs;
      }
      if (result.size() > 16)
        return std::nullopt;
    }
    return result;
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
    if (rangeArithmeticFits(op, a->second, b->second))
      return true;
    if (!isa<arith::MulIOp>(op) || !a->second.smin().isNonNegative() ||
        !b->second.smin().isNonNegative())
      return false;
    return hasProductBound(op);
  }

  static bool rangeArithmeticFits(Operation *op, const ConstantIntRanges &a,
                                  const ConstantIntRanges &b) {
    bool noOverflow = true;
    for (const APInt &av : {a.smin(), a.smax()})
      for (const APInt &bv : {b.smin(), b.smax()}) {
        bool overflow;
        if (isa<arith::AddIOp>(op))
          (void)av.sadd_ov(bv, overflow);
        else if (isa<arith::SubIOp>(op))
          (void)av.ssub_ov(bv, overflow);
        else
          (void)av.smul_ov(bv, overflow);
        noOverflow &= !overflow;
      }
    return noOverflow;
  }

  bool hasProductBound(Operation *op) const {
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
    if (depth)
      return normalizeImpl(value, depth);
    value = stripRefinements(value);
    auto it = normalized.find(value);
    if (it != normalized.end())
      return it->second;
    auto expression = normalizeImpl(value, 0);
    normalized.try_emplace(value, expression);
    return expression;
  }

  std::optional<Polynomial> normalizeImpl(Value value, unsigned depth) {
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
    if (op && isa<arith::ExtSIOp>(op))
      return normalize(op->getOperand(0), depth + 1);
    if (op && isa<arith::ExtUIOp>(op) && nonnegativeRange(op->getOperand(0)))
      return normalize(op->getOperand(0), depth + 1);
    return normalizeDivRem(value, depth);
  }

  std::optional<Polynomial> normalizeDivRem(Value value, unsigned depth) {
    Operation *op = value.getDefiningOp();
    if (op &&
        (isa<arith::DivSIOp>(op) ||
         (isa<arith::DivUIOp>(op) && nonnegativeRange(op->getOperand(0)) &&
          positiveRange(op->getOperand(1)))))
      return quotientSymbol(op->getOperand(0), op->getOperand(1));
    if (op && isa<arith::RemSIOp, arith::RemUIOp>(op) &&
        nonnegativeRange(op->getOperand(0)) && positiveRange(op->getOperand(1)))
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
    if (expression && constraints.size() < 64)
      constraints.push_back({std::move(*expression), strict});
    else if (expression)
      LDBG() << "Integer relation constraint budget exhausted";
  }

  std::optional<std::pair<APInt, APInt>> termRange(const Monomial &term) const {
    APInt lo(256, 1), hi(256, 1);
    for (Symbol symbol : term) {
      auto it = ranges.find(symbols[symbol.value]);
      if (it == ranges.end() || it->second.smin().getBitWidth() > 64)
        return std::nullopt;
      APInt a = it->second.smin().sext(256);
      APInt b = it->second.smax().sext(256);
      if (symbol.divisor) {
        auto divisor = ranges.find(symbols[symbol.divisor]);
        if (a.isNegative() || divisor == ranges.end() ||
            !divisor->second.smin().isStrictlyPositive())
          return std::nullopt;
        a = a.sdiv(divisor->second.smax().sext(256));
        b = b.sdiv(divisor->second.smin().sext(256));
      }
      std::array<APInt, 4> products{lo * a, lo * b, hi * a, hi * b};
      lo = *llvm::min_element(
          products, [](const APInt &x, const APInt &y) { return x.slt(y); });
      hi = *llvm::max_element(
          products, [](const APInt &x, const APInt &y) { return x.slt(y); });
    }
    return std::make_pair(std::move(lo), std::move(hi));
  }

  bool constantNonnegative(const Polynomial &expression, bool strict) {
    APInt sum(256, 0);
    for (const auto &[term, coefficient] : expression) {
      auto [it, inserted] = termRanges.try_emplace(term);
      if (inserted)
        it->second = termRange(term);
      if (!it->second)
        return false;
      const auto &[lo, hi] = *it->second;
      bool overflow = false;
      APInt product =
          coefficient == 1
              ? lo
              : (coefficient < 0 ? hi : lo)
                    .smul_ov(APInt(256, coefficient, true), overflow);
      if (overflow)
        return false;
      sum = sum.sadd_ov(product, overflow);
      if (overflow)
        return false;
    }
    return strict ? sum.isStrictlyPositive() : sum.isNonNegative();
  }

  bool knownNonnegative(const Polynomial &expression, bool strict) {
    for (const Constraint &constraint : constraints)
      if (constraint.expression == expression && (!strict || constraint.strict))
        return true;
    return constantNonnegative(expression, strict);
  }

  bool proveDirect(const Polynomial &expression, bool strict) {
    if (knownNonnegative(expression, strict))
      return true;
    unsigned budget = 256;
    for (const Constraint &a : constraints) {
      for (int64_t scale : cancellationScales(expression, a.expression)) {
        auto rest = subtractConstraint(expression, a, scale);
        if (!rest)
          continue;
        if (knownNonnegative(*rest, strict))
          return true;
        for (const Constraint &b : constraints) {
          for (int64_t scale : cancellationScales(*rest, b.expression)) {
            if (!budget--) {
              LDBG() << "Integer relation combination budget exhausted";
              return false;
            }
            auto tail = subtractConstraint(*rest, b, scale);
            if (tail && knownNonnegative(*tail, strict))
              return true;
          }
        }
      }
    }
    return false;
  }

  static std::optional<Polynomial>
  subtractConstraint(const Polynomial &expression, const Constraint &constraint,
                     int64_t scale) {
    auto rest = add(expression, constraint.expression, -scale);
    // Strict integer bounds have at least one unit of slack.
    return constraint.strict ? add(rest, Polynomial{{{}, scale}}) : rest;
  }

  static SmallVector<int64_t, 2>
  cancellationScales(const Polynomial &expression,
                     const Polynomial &constraint) {
    SmallVector<int64_t, 2> scales;
    for (const auto &[term, coefficient] : constraint) {
      auto it = llvm::lower_bound(expression, term,
                                  [](const auto &entry, const Monomial &key) {
                                    return entry.first < key;
                                  });
      if (term.empty() || it == expression.end() || it->first != term ||
          !coefficient ||
          (coefficient == -1 &&
           it->second == std::numeric_limits<int64_t>::min()))
        continue;
      if (it->second % coefficient == 0 && it->second / coefficient > 0)
        if (!llvm::is_contained(scales, it->second / coefficient))
          scales.push_back(it->second / coefficient);
    }
    return scales;
  }

  bool prove(const std::optional<Polynomial> &expression, bool strict) {
    if (!valid || !expression)
      return false;
    if (proveDirect(*expression, strict))
      return true;
    // Positive scaling preserves the inequality; never assume a divisor's sign.
    DenseSet<Value> tried;
    for (Value value : values) {
      if (!positiveRange(value))
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
  SmallVector<std::pair<Value, IntegerBound>> inheritedBounds;
  SmallVector<Value> symbols{Value{}};
  SmallVector<Constraint, 0> constraints;
  DenseMap<Value, unsigned> symbolIDs;
  DenseMap<Value, ConstantIntRanges> ranges;
  DenseMap<Value, std::optional<Polynomial>> normalized;
  // Ranges are immutable within one local proof; never cache across transfers.
  std::map<Monomial, std::optional<std::pair<APInt, APInt>>> termRanges;
  function_ref<const IntegerFacts &(Value)> getFacts;
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

  std::optional<IntegerFacts>
  remapFacts(Value value, ProgramPoint *point,
             const DenseMap<Value, Value> &mapping) {
    const IntegerFacts &source = getLatticeElementFor(point, value)->getValue();
    if (!source.congruence)
      return std::nullopt;
    IntegerFacts result = source;
    result.bounds.clear();
    for (IntegerBound bound : source.bounds)
      if (Value mapped = mapping.lookup(stripRefinements(bound.value))) {
        if (signedPredicate(bound.predicate) != bound.predicate) {
          const auto *a =
              getOrCreateFor<dataflow::IntegerValueRangeLattice>(point, value);
          const auto *b = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
              point, bound.value);
          if (a->getValue().isUninitialized() ||
              b->getValue().isUninitialized())
            return std::nullopt;
          if (a->getValue().getValue().smin().isNonNegative() &&
              b->getValue().getValue().smin().isNonNegative())
            bound.predicate = signedPredicate(bound.predicate);
        }
        IntegerBound translated{mapped, bound.predicate};
        if (!llvm::is_contained(result.bounds, translated))
          result.bounds.push_back(translated);
      }
    return result;
  }

  void visitCallableOperation(
      CallableOpInterface callable,
      ArrayRef<dataflow::AbstractSparseLattice *> arguments) override {
    Block &entry = callable.getCallableRegion()->front();
    ProgramPoint *point = getProgramPointBefore(&entry);
    const auto *calls = getOrCreateFor<dataflow::PredecessorState>(
        point, getProgramPointAfter(callable));
    if (!calls->allPredecessorsKnown() ||
        !getSolverConfig().isInterprocedural()) {
      AbstractSparseForwardDataFlowAnalysis::visitCallableOperation(callable,
                                                                    arguments);
      return;
    }
    for (Operation *caller : calls->getKnownPredecessors()) {
      auto call = cast<CallOpInterface>(caller);
      DenseMap<Value, Value> mapping;
      for (auto [operand, argument] :
           llvm::zip_equal(call.getArgOperands(), entry.getArguments()))
        mapping.try_emplace(stripRefinements(operand), argument);
      for (auto [operand, target] :
           llvm::zip_equal(call.getArgOperands(), arguments)) {
        auto *lattice = static_cast<IntegerFactsLattice *>(target);
        if (auto facts = remapFacts(operand, point, mapping))
          propagateIfChanged(lattice, lattice->join(*facts));
      }
    }
  }

  LogicalResult visitCallOperation(
      CallOpInterface call,
      ArrayRef<const dataflow::AbstractSparseLattice *> operands,
      ArrayRef<dataflow::AbstractSparseLattice *> results) override {
    auto callable =
        dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    if (!callable || !callable.getCallableRegion() ||
        !getSolverConfig().isInterprocedural())
      return AbstractSparseForwardDataFlowAnalysis::visitCallOperation(
          call, operands, results);
    ProgramPoint *point = getProgramPointAfter(call);
    const auto *returns =
        getOrCreateFor<dataflow::PredecessorState>(point, point);
    if (!returns->allPredecessorsKnown())
      return AbstractSparseForwardDataFlowAnalysis::visitCallOperation(
          call, operands, results);
    DenseMap<Value, Value> mapping;
    for (auto [argument, operand] :
         llvm::zip_equal(callable.getCallableRegion()->front().getArguments(),
                         call.getArgOperands()))
      mapping.try_emplace(argument, operand);
    for (Operation *terminator : returns->getKnownPredecessors())
      for (auto [operand, target] :
           llvm::zip_equal(terminator->getOperands(), results)) {
        auto *lattice = static_cast<IntegerFactsLattice *>(target);
        if (auto facts = remapFacts(operand, point, mapping))
          propagateIfChanged(lattice, lattice->join(*facts));
      }
    return success();
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
      if (inferRefinement(refine, operands, facts) == FactStatus::Pending)
        return success();
    } else if (isa<SplatOp, ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp,
                   triton::gpu::ConvertLayoutOp>(op)) {
      facts.congruence = operands[0]->getValue().congruence;
    } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      if (inferArithmetic(op, operands, bits, facts) == FactStatus::Pending)
        return success();
    }
    if (inferRelations(op, facts) == FactStatus::Pending ||
        inferTransitiveBounds(op, facts) == FactStatus::Pending)
      return success();
    LDBG_OS([&](raw_ostream &os) {
      os << *op << ": ";
      facts.print(os);
    });
    propagateIfChanged(results[0], results[0]->join(facts));
    return success();
  }

private:
  enum class FactStatus { Pending, Ready };

  FactStatus inferRefinement(RefineOp refine,
                             ArrayRef<const IntegerFactsLattice *> operands,
                             IntegerFacts &facts) {
    Operation *op = refine;
    facts = operands[0]->getValue();
    if (refine.getPredicate() == arith::CmpIPredicate::eq &&
        operands[1]->getValue().congruence->first.isZero())
      facts.congruence = operands[1]->getValue().congruence;
    IntegerBound bound{refine.getBound(), refine.getPredicate()};
    if (signedPredicate(bound.predicate) != bound.predicate) {
      const auto *valueRange =
          getOrCreateFor<dataflow::IntegerValueRangeLattice>(
              getProgramPointAfter(op), refine.getValue());
      const auto *boundRange =
          getOrCreateFor<dataflow::IntegerValueRangeLattice>(
              getProgramPointAfter(op), refine.getBound());
      if (valueRange->getValue().isUninitialized() ||
          boundRange->getValue().isUninitialized())
        return FactStatus::Pending;
      if (valueRange->getValue().getValue().smin().isNonNegative() &&
          boundRange->getValue().getValue().smin().isNonNegative())
        bound.predicate = signedPredicate(bound.predicate);
    }
    // Keep a finite prefix. Existing relations cannot change order on
    // revisit.
    if (facts.bounds.size() < 8 && !llvm::is_contained(facts.bounds, bound))
      facts.bounds.push_back(bound);
    else if (!llvm::is_contained(facts.bounds, bound))
      LDBG() << "Symbolic bound budget exhausted: " << *op;
    return FactStatus::Ready;
  }

  FactStatus inferArithmetic(Operation *op,
                             ArrayRef<const IntegerFactsLattice *> operands,
                             unsigned bits, IntegerFacts &facts) {
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
    if (!modulus.isZero() && !modulus.isPowerOf2()) {
      std::array<const ConstantIntRanges *, 2> ranges;
      for (unsigned i : llvm::seq(2u)) {
        const auto *state = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
            getProgramPointAfter(op), op->getOperand(i));
        if (state->getValue().isUninitialized())
          return FactStatus::Pending;
        ranges[i] = &state->getValue().getValue();
      }
      bool wraps;
      if (isa<arith::AddIOp>(op))
        (void)ranges[0]->umax().uadd_ov(ranges[1]->umax(), wraps);
      else if (isa<arith::MulIOp>(op))
        (void)ranges[0]->umax().umul_ov(ranges[1]->umax(), wraps);
      else
        (void)ranges[0]->umin().usub_ov(ranges[1]->umax(), wraps);
      if (wraps)
        modulus = llvm::APIntOps::GreatestCommonDivisor(modulus, wrap);
      else if (isa<arith::SubIOp>(op))
        residue = ar + modulus - br.urem(modulus);
    } else if (!modulus.isZero())
      modulus = llvm::APIntOps::GreatestCommonDivisor(modulus, wrap);
    facts.congruence = {modulus,
                        residue.urem(modulus.isZero() ? wrap : modulus)};
    return FactStatus::Ready;
  }

  FactStatus inferRelations(Operation *op, IntegerFacts &facts) {
    Value result = op->getResult(0);
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::AddIOp, arith::SubIOp,
            arith::MulIOp, arith::MinSIOp, arith::MinUIOp>(op) &&
        isa<IntegerType>(result.getType())) {
      bool pending = false;
      auto range = [&](Value value) {
        const auto *state = getOrCreateFor<dataflow::IntegerValueRangeLattice>(
            getProgramPointAfter(op), value);
        pending |= state->getValue().isUninitialized();
        return state->getValue();
      };
      auto getFacts = [&](Value value) -> const IntegerFacts & {
        const auto *state =
            getLatticeElementFor(getProgramPointAfter(op), value);
        pending |= !state->getValue().congruence;
        return state->getValue();
      };
      IntegerRelations relations(op->getOperands(), range, getFacts);
      if (pending)
        return FactStatus::Pending;
      bool division = isa<arith::DivSIOp, arith::DivUIOp>(op);
      Value dividend = op->getOperand(0), divisor = op->getOperand(1);
      bool nonnegative =
          division
              ? relations.nonnegative(dividend) && relations.positive(divisor)
              : relations.hasRelations() && relations.nonnegative(result);
      if (nonnegative) {
        inferCandidateBounds(op, relations, range, facts);
        facts.sign = !division && relations.positive(result)
                         ? IntegerFacts::Sign::Positive
                         : IntegerFacts::Sign::Nonnegative;
      }
    }
    return FactStatus::Ready;
  }

  void inferCandidateBounds(Operation *op, IntegerRelations &relations,
                            function_ref<IntegerValueRange(Value)> range,
                            IntegerFacts &facts) {
    Value result = op->getResult(0);
    bool division = isa<arith::DivSIOp, arith::DivUIOp>(op);
    Value dividend = op->getOperand(0), divisor = op->getOperand(1);
    DenseSet<Value> tried;
    SmallVector<Value> candidates(relations.candidates());
    std::stable_partition(candidates.begin(), candidates.end(),
                          [&](Value value) {
                            IntegerValueRange limit = range(value);
                            return !limit.isUninitialized() &&
                                   limit.getValue().smin().isStrictlyPositive();
                          });
    for (Value bound : candidates) {
      IntegerValueRange limit = range(bound);
      if (bound.getType() != result.getType() ||
          matchPattern(bound, m_Constant()) || limit.isUninitialized() ||
          !limit.getValue().smin().isNonNegative() ||
          !tried.insert(stripRefinements(bound)).second)
        continue;
      if (tried.size() > 8)
        break;
      bool proven = division
                        ? relations.productUpperBound(dividend, divisor, bound)
                        : relations.upperBound(result, bound);
      bool inclusive = !proven && relations.upperBound(result, bound, false);
      if (!proven && !inclusive)
        continue;
      if (facts.bounds.size() < 8)
        facts.bounds.push_back({bound, proven ? arith::CmpIPredicate::slt
                                              : arith::CmpIPredicate::sle});
    }
  }

  FactStatus inferTransitiveBounds(Operation *op, IntegerFacts &facts) {
    for (unsigned i = 0; i < facts.bounds.size() && facts.bounds.size() < 8;
         ++i) {
      IntegerBound bound = facts.bounds[i];
      using P = arith::CmpIPredicate;
      if (bound.predicate != P::slt && bound.predicate != P::sle)
        continue;
      const auto *limit =
          getLatticeElementFor(getProgramPointAfter(op), bound.value);
      if (!limit->getValue().congruence)
        return FactStatus::Pending;
      for (IntegerBound upper : limit->getValue().bounds) {
        if (upper.predicate != P::slt && upper.predicate != P::sle)
          continue;
        IntegerBound transitive{upper.value, bound.predicate == P::slt ||
                                                     upper.predicate == P::slt
                                                 ? P::slt
                                                 : P::sle};
        if (facts.bounds.size() < 8 &&
            !llvm::is_contained(facts.bounds, transitive))
          facts.bounds.push_back(transitive);
      }
    }
    return FactStatus::Ready;
  }
};

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

static ConstantIntRanges refineCongruence(ConstantIntRanges refined,
                                          const IntegerFacts &facts) {
  const auto &[modulus, residue] = *facts.congruence;
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
  return refined;
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
      refined = refineCongruence(refined, facts->getValue());
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

static intrange::CmpPredicate rangePredicate(arith::CmpIPredicate predicate) {
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
  return pred;
}

static bool hasNonnegativeRange(const IntegerValueRange &range) {
  return !range.isUninitialized() && range.getValue().smin().isNonNegative();
}

IntegerProof IntegerFactAnalysis::prove(arith::CmpIPredicate predicate,
                                        Value lhs, Value rhs) const {
  assert(solver && "initialize integer facts before querying");
  IntegerValueRange a = getRange(lhs);
  IntegerValueRange b = getRange(rhs);
  bool nonnegative = hasNonnegativeRange(a) && hasNonnegativeRange(b);
  if (nonnegative)
    predicate = signedPredicate(predicate);
  if (const auto *facts = solver->lookupState<IntegerFactsLattice>(lhs))
    for (IntegerBound bound : facts->getValue().bounds)
      if (stripRefinements(bound.value) == stripRefinements(rhs)) {
        if (nonnegative)
          bound.predicate = signedPredicate(bound.predicate);
        if (bound.predicate == predicate)
          return IntegerProof::Proven;
        if (bound.predicate == arith::invertPredicate(predicate))
          return IntegerProof::Disproven;
      }
  if (a.isUninitialized() || b.isUninitialized())
    return IntegerProof::Unknown;
  intrange::CmpPredicate pred = rangePredicate(predicate);
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
