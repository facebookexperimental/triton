#ifndef TRITON_STOCHASTIC_STORE_OWNERSHIP_H
#define TRITON_STOCHASTIC_STORE_OWNERSHIP_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
namespace {

struct AffineIndex {
  int64_t constant = 0;
  DenseMap<Value, int64_t> terms = DenseMap<Value, int64_t>();
};

bool add(AffineIndex &dst, const AffineIndex &src, int64_t scale = 1) {
  int64_t value;
  if (llvm::MulOverflow(src.constant, scale, value) ||
      llvm::AddOverflow(dst.constant, value, dst.constant))
    return false;
  for (auto [symbol, coefficient] : src.terms) {
    if (llvm::MulOverflow(coefficient, scale, value) ||
        llvm::AddOverflow(dst.terms[symbol], value, dst.terms[symbol]))
      return false;
    if (dst.terms[symbol] == 0)
      dst.terms.erase(symbol);
  }
  return true;
}

struct StoreDimension {
  int64_t stride, extent;
};
struct StoreFootprint {
  int64_t origin = 0, span = 1;
  SmallVector<StoreDimension> dimensions;
};

class StoreOwnership {
  FuncOp function;
  int64_t program;
  DenseMap<Value, std::pair<int64_t, int64_t>> domains;
  DenseSet<Operation *> loops;
  bool inactiveProgram = false;

  std::optional<int64_t> uniform(Value value) {
    auto index = expression(value);
    if (!index || !index->terms.empty())
      return std::nullopt;
    return index->constant;
  }

  bool fits(Value value, const AffineIndex &index) {
    auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    if (!type)
      return false;
    int64_t lower = index.constant, upper = index.constant;
    for (auto [symbol, coefficient] : index.terms) {
      auto domain = domains.find(symbol);
      if (domain == domains.end())
        return false;
      auto [lo, hi] = domain->second;
      if (coefficient < 0)
        std::swap(lo, hi);
      int64_t a, b;
      if (llvm::MulOverflow(coefficient, lo, a) ||
          llvm::MulOverflow(coefficient, hi, b) ||
          llvm::AddOverflow(lower, a, lower) ||
          llvm::AddOverflow(upper, b, upper))
        return false;
    }
    return type.getWidth() >= 64 || (llvm::isIntN(type.getWidth(), lower) &&
                                     llvm::isIntN(type.getWidth(), upper));
  }

public:
  StoreOwnership(FuncOp function, int64_t program)
      : function(function), program(program) {}

  bool inactive() const { return inactiveProgram; }

  std::optional<AffineIndex> expression(Value value) {
    APInt integer;
    if (matchPattern(value, m_ConstantInt(&integer)))
      return AffineIndex{integer.getSExtValue(), DenseMap<Value, int64_t>()};
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner()->getParentOp() == function.getOperation()) {
        auto attr = function.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                           "tt.sr_value");
        if (attr)
          return AffineIndex{attr.getInt(), DenseMap<Value, int64_t>()};
        return std::nullopt;
      }
      auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!loop || arg != loop.getInductionVar())
        return std::nullopt;
      if (!domains.contains(value)) {
        auto lower = uniform(loop.getLowerBound());
        auto upper = uniform(loop.getUpperBound());
        auto step = uniform(loop.getStep());
        if (!lower || !upper || !step || *step <= 0)
          return std::nullopt;
        if (*upper <= *lower) {
          inactiveProgram = true;
          return AffineIndex{*lower, DenseMap<Value, int64_t>()};
        }
        int64_t distance;
        if (llvm::SubOverflow(*upper, *lower, distance))
          return std::nullopt;
        domains[value] = {*lower, *lower + ((distance - 1) / *step) * *step};
        loops.insert(loop);
      }
      if (domains[value].first == domains[value].second)
        return AffineIndex{domains[value].first, DenseMap<Value, int64_t>()};
      return AffineIndex{0, {{value, 1}}};
    }
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
      if (pid.getAxis() != ProgramIDDim::X)
        return std::nullopt;
      return AffineIndex{program, DenseMap<Value, int64_t>()};
    }
    if (auto range = dyn_cast<MakeRangeOp>(op)) {
      domains.try_emplace(value, std::pair<int64_t, int64_t>(
                                     range.getStart(), range.getEnd() - 1));
      return AffineIndex{0, {{value, 1}}};
    }
    if (isa<arith::ExtUIOp>(op)) {
      auto input = expression(op->getOperand(0));
      if (!input || input->constant < 0)
        return std::nullopt;
      for (auto [symbol, coefficient] : input->terms)
        if (coefficient < 0 || domains.lookup(symbol).first < 0)
          return std::nullopt;
      return input;
    }
    if (isa<SplatOp, BroadcastOp, ExpandDimsOp, arith::ExtSIOp>(op))
      return expression(op->getOperand(0));
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      auto a = expression(op->getOperand(0));
      auto b = expression(op->getOperand(1));
      if (!a || !b)
        return std::nullopt;
      AffineIndex result;
      if (isa<arith::MulIOp>(op)) {
        if (a->terms.empty()) {
          if (!add(result, *b, a->constant))
            return std::nullopt;
        } else if (b->terms.empty()) {
          if (!add(result, *a, b->constant))
            return std::nullopt;
        } else {
          return std::nullopt;
        }
      } else {
        result = *a;
        if (!add(result, *b, isa<arith::SubIOp>(op) ? -1 : 1))
          return std::nullopt;
      }
      return fits(value, result) ? std::optional(result) : std::nullopt;
    }
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::MinSIOp, arith::MinUIOp>(
            op)) {
      auto a = uniform(op->getOperand(0)), b = uniform(op->getOperand(1));
      if (!a || !b || *a < 0 || *b <= 0)
        return std::nullopt;
      return AffineIndex{
          isa<arith::DivSIOp, arith::DivUIOp>(op) ? *a / *b : std::min(*a, *b),
          DenseMap<Value, int64_t>()};
    }
    return std::nullopt;
  }

  void applyMask(Value mask) {
    if (!mask)
      return;
    auto *op = mask.getDefiningOp();
    if (!op)
      return;
    if (isa<BroadcastOp, ExpandDimsOp, SplatOp>(op)) {
      applyMask(op->getOperand(0));
      return;
    }
    if (isa<arith::AndIOp>(op)) {
      applyMask(op->getOperand(0));
      applyMask(op->getOperand(1));
      return;
    }
    auto compare = dyn_cast<arith::CmpIOp>(op);
    if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt)
      return;
    auto lhs = expression(compare.getLhs());
    auto rhs = uniform(compare.getRhs());
    if (!lhs || !rhs)
      return;
    for (auto [symbol, coefficient] : lhs->terms) {
      if (!symbol.getDefiningOp<MakeRangeOp>() || coefficient != 1)
        continue;
      int64_t maximum;
      if (llvm::SubOverflow(*rhs, int64_t(1), maximum) ||
          llvm::SubOverflow(maximum, lhs->constant, maximum))
        continue;
      bool overflow = false;
      for (auto [other, scale] : lhs->terms) {
        if (other == symbol)
          continue;
        auto [lower, upper] = domains.lookup(other);
        int64_t contribution;
        overflow |=
            llvm::MulOverflow(scale, scale < 0 ? upper : lower, contribution);
        if (!overflow)
          overflow |= llvm::SubOverflow(maximum, contribution, maximum);
      }
      if (overflow)
        continue;
      auto &domain = domains[symbol];
      domain.second = std::min(domain.second, maximum);
      if (domain.second < domain.first)
        inactiveProgram = true;
    }
  }

  Value base(Value pointer) {
    if (auto *op = pointer.getDefiningOp())
      if (isa<SplatOp, BroadcastOp, ExpandDimsOp>(op))
        return base(op->getOperand(0));
    if (auto op = pointer.getDefiningOp<AddPtrOp>())
      return base(op.getPtr());
    auto arg = dyn_cast<BlockArgument>(pointer);
    return arg && arg.getOwner()->getParentOp() == function.getOperation()
               ? pointer
               : Value();
  }

  std::optional<AffineIndex> offset(Value pointer) {
    if (auto *op = pointer.getDefiningOp())
      if (isa<SplatOp, BroadcastOp, ExpandDimsOp>(op))
        return offset(op->getOperand(0));
    if (auto op = pointer.getDefiningOp<AddPtrOp>()) {
      auto a = offset(op.getPtr()), b = expression(op.getOffset());
      if (a && b && add(*a, *b))
        return a;
      return std::nullopt;
    }
    return base(pointer) ? std::optional(AffineIndex{}) : std::nullopt;
  }

  std::optional<StoreFootprint> footprint(const AffineIndex &index) {
    StoreFootprint result;
    result.origin = index.constant;
    for (auto [symbol, coefficient] : index.terms) {
      if (coefficient <= 0)
        return std::nullopt;
      auto [lower, upper] = domains.lookup(symbol);
      int64_t step = 1;
      if (auto arg = dyn_cast<BlockArgument>(symbol)) {
        auto loop = cast<scf::ForOp>(arg.getOwner()->getParentOp());
        auto value = uniform(loop.getStep());
        if (!value)
          return std::nullopt;
        step = *value;
      }
      int64_t origin, stride;
      if (llvm::MulOverflow(coefficient, lower, origin) ||
          llvm::AddOverflow(result.origin, origin, result.origin) ||
          llvm::MulOverflow(coefficient, step, stride))
        return std::nullopt;
      result.dimensions.push_back({stride, (upper - lower) / step + 1});
    }
    llvm::sort(result.dimensions,
               [](auto a, auto b) { return a.stride < b.stride; });
    for (auto dimension : result.dimensions) {
      int64_t growth;
      if (dimension.stride < result.span ||
          llvm::MulOverflow(dimension.stride, dimension.extent - 1, growth) ||
          llvm::AddOverflow(result.span, growth, result.span))
        return std::nullopt;
    }
    int64_t end;
    if (llvm::AddOverflow(result.origin, result.span, end))
      return std::nullopt;
    return result;
  }

  bool regionsValid(SrStoreOp store) {
    for (Operation *parent = store->getParentOp();
         parent != function.getOperation(); parent = parent->getParentOp())
      if (!loops.contains(parent))
        return false;
    return true;
  }
};

bool footprintsDisjoint(const StoreFootprint &a, const StoreFootprint &b) {
  if (a.origin + a.span <= b.origin || b.origin + b.span <= a.origin)
    return true;
  for (auto dimension : a.dimensions) {
    int64_t period = dimension.stride;
    auto residue = [period](const StoreFootprint &footprint)
        -> std::optional<std::pair<int64_t, int64_t>> {
      int64_t span = 1;
      for (auto dimension : footprint.dimensions) {
        if (dimension.stride < period)
          span += dimension.stride * (dimension.extent - 1);
        else if (dimension.stride % period != 0)
          return std::nullopt;
      }
      int64_t begin = footprint.origin % period;
      if (begin < 0)
        begin += period;
      if (span > period - begin)
        return std::nullopt;
      return std::pair(begin, begin + span);
    };
    auto x = residue(a), y = residue(b);
    if (x && y && (x->second <= y->first || y->second <= x->first))
      return true;
  }
  return false;
}

bool allFootprintsDisjoint(SmallVector<StoreFootprint> footprints) {
  llvm::sort(footprints,
             [](const auto &a, const auto &b) { return a.origin < b.origin; });
  bool intervals = true;
  for (size_t i = 1; i < footprints.size(); ++i)
    intervals &= footprints[i - 1].origin + footprints[i - 1].span <=
                 footprints[i].origin;
  if (intervals)
    return true;
  for (size_t i = 0; i < footprints.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (!footprintsDisjoint(footprints[i], footprints[j]))
        return false;
  return true;
}

bool proveLaunchOwnership(SrStoreOp store) {
  auto function = store->getParentOfType<FuncOp>();
  if (!isKernel(function))
    return false;
  auto grid = function->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "tt.sr_grid");
  constexpr int64_t maxCheckedPrograms = 65536;
  if (!grid || grid.getInt() <= 0 || grid.getInt() > maxCheckedPrograms)
    return false;
  SmallVector<StoreFootprint> physical, logical;
  for (int64_t program = 0; program < grid.getInt(); ++program) {
    StoreOwnership proof(function, program);
    auto offset = proof.offset(store.getPtr());
    auto index = proof.expression(store.getLogicalIndex());
    proof.applyMask(store.getMask());
    if (proof.inactive())
      continue;
    if (!offset || !index || offset->terms.size() != index->terms.size())
      return false;
    for (auto [symbol, coefficient] : offset->terms)
      if (!index->terms.contains(symbol))
        return false;
    int64_t lanes = 1;
    for (auto [symbol, coefficient] : offset->terms)
      if (auto range = symbol.getDefiningOp<MakeRangeOp>())
        if (llvm::MulOverflow(lanes, int64_t(range.getEnd() - range.getStart()),
                              lanes))
          return false;
    if (auto tensor = dyn_cast<RankedTensorType>(store.getPtr().getType()))
      if (lanes != tensor.getNumElements())
        return false;
    auto p = proof.footprint(*offset), l = proof.footprint(*index);
    if (!p || !l || !proof.regionsValid(store))
      return false;
    physical.push_back(*p);
    logical.push_back(*l);
  }
  if (!allFootprintsDisjoint(physical) || !allFootprintsDisjoint(logical))
    return false;
  bool conflict = false;
  StoreOwnership proof(function, 0);
  auto base = proof.base(store.getPtr());
  function.walk([&](Operation *other) {
    if (other == store.getOperation())
      return;
    if (isa<CallOp>(other))
      conflict = true;
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(other)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          continue;
        Value pointer;
        if (auto ordinary = dyn_cast<StoreOp>(other))
          pointer = ordinary.getPtr();
        else if (auto annotated = dyn_cast<SrStoreOp>(other))
          pointer = annotated.getPtr();
        conflict |=
            !pointer || !proof.base(pointer) || proof.base(pointer) == base;
      }
    }
  });
  return base && !conflict;
}

bool structuralIndex(Value value, DenseSet<Value> &varying,
                     bool uniformLoads = false) {
  APInt integer;
  if (matchPattern(value, m_ConstantInt(&integer)))
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (isa<FuncOp>(argument.getOwner()->getParentOp()))
      return isa<IntegerType>(value.getType());
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument != loop.getInductionVar())
      return false;
    varying.insert(value);
    return true;
  }
  auto *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<MakeRangeOp, GetProgramIdOp>(op)) {
    varying.insert(value);
    return true;
  }
  if (auto load = dyn_cast<LoadOp>(op))
    return uniformLoads && isa<IntegerType>(load.getType()) &&
           isa<PointerType>(load.getPtr().getType());
  if (!isa<SplatOp, BroadcastOp, ExpandDimsOp, arith::ExtSIOp, arith::ExtUIOp,
           arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
           arith::DivUIOp, arith::MinSIOp, arith::MinUIOp>(op))
    return false;
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return structuralIndex(operand, varying, uniformLoads);
  });
}

Value declaredPointerBase(Value pointer, DenseSet<Value> &varying) {
  auto *op = pointer.getDefiningOp();
  if (op && isa<SplatOp, BroadcastOp, ExpandDimsOp, BitcastOp>(op))
    return declaredPointerBase(op->getOperand(0), varying);
  if (auto addptr = pointer.getDefiningOp<AddPtrOp>()) {
    if (!structuralIndex(addptr.getOffset(), varying, true))
      return {};
    return declaredPointerBase(addptr.getPtr(), varying);
  }
  auto argument = dyn_cast<BlockArgument>(pointer);
  return argument && isa<FuncOp>(argument.getOwner()->getParentOp()) ? pointer
                                                                     : Value();
}

bool proveDeclaredOwnership(SrStoreOp store) {
  DenseSet<Value> physical, logical;
  auto base = declaredPointerBase(store.getPtr(), physical);
  if (!base || !structuralIndex(store.getLogicalIndex(), logical) ||
      physical != logical)
    return false;
  for (Operation *parent = store->getParentOp(); !isa<FuncOp>(parent);
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    APInt step;
    DenseSet<Value> bounds;
    if (!loop || !physical.contains(loop.getInductionVar()) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        step.getSExtValue() <= 0 ||
        !structuralIndex(loop.getLowerBound(), bounds, true) ||
        !structuralIndex(loop.getUpperBound(), bounds, true))
      return false;
  }
  bool invalid = false;
  store->getParentOfType<FuncOp>().walk([&](Operation *other) {
    if (other == store.getOperation())
      return;
    if (isa<CallOp>(other))
      invalid = true;
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(other)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          continue;
        Value pointer;
        if (auto ordinary = dyn_cast<StoreOp>(other))
          pointer = ordinary.getPtr();
        else if (auto annotated = dyn_cast<SrStoreOp>(other))
          pointer = annotated.getPtr();
        DenseSet<Value> dimensions;
        invalid |= !pointer || pointer == store.getPtr() ||
                   !declaredPointerBase(pointer, dimensions);
      }
    }
  });
  return !invalid;
}

} // namespace
} // namespace mlir::triton
#endif
