#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONMATERIALIZESTOCHASTICROUNDING
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
namespace {

struct AffineIndex {
  int64_t constant = 0;
  DenseMap<Value, int64_t> terms;
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

std::optional<AffineIndex> affine(Value value) {
  APInt integer;
  if (matchPattern(value, m_ConstantInt(&integer)))
    return AffineIndex{integer.getSExtValue(), DenseMap<Value, int64_t>()};
  auto *op = value.getDefiningOp();
  if (op &&
      isa<SplatOp, BroadcastOp, ExpandDimsOp, arith::ExtSIOp, arith::ExtUIOp>(
          op))
    return affine(op->getOperand(0));
  if (op && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
    auto a = affine(op->getOperand(0)), b = affine(op->getOperand(1));
    if (!a || !b)
      return std::nullopt;
    if ((!a->terms.empty() || !b->terms.empty()) &&
        !getElementTypeOrSelf(value.getType()).isInteger(64))
      return std::nullopt;
    if (isa<arith::MulIOp>(op)) {
      if (a->terms.empty()) {
        AffineIndex result;
        return add(result, *b, a->constant) ? std::optional(result)
                                            : std::nullopt;
      }
      if (b->terms.empty()) {
        AffineIndex result;
        return add(result, *a, b->constant) ? std::optional(result)
                                            : std::nullopt;
      }
      return std::nullopt;
    }
    if (add(*a, *b, isa<arith::SubIOp>(op) ? -1 : 1))
      return a;
    return std::nullopt;
  }
  if ((op && isa<MakeRangeOp, GetProgramIdOp>(op)) || isa<BlockArgument>(value))
    return AffineIndex{0, {{value, 1}}};
  return std::nullopt;
}

bool isVarying(Value symbol) {
  if (symbol.getDefiningOp<MakeRangeOp>() ||
      symbol.getDefiningOp<GetProgramIdOp>())
    return true;
  auto arg = dyn_cast<BlockArgument>(symbol);
  return arg && isa<scf::ForOp>(arg.getOwner()->getParentOp());
}

Value pointerBase(Value ptr, AffineIndex &offset) {
  if (auto splat = ptr.getDefiningOp<SplatOp>())
    return pointerBase(splat.getSrc(), offset);
  if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) {
    auto index = affine(addptr.getOffset());
    if (!index || !add(offset, *index))
      return {};
    return pointerBase(addptr.getPtr(), offset);
  }
  auto arg = dyn_cast<BlockArgument>(ptr);
  if (!arg || !isa<FuncOp>(arg.getOwner()->getParentOp()))
    return {};
  return ptr;
}

bool proveOwnership(SrStoreOp store) {
  AffineIndex offset;
  Value base = pointerBase(store.getPtr(), offset);
  auto logical = affine(store.getLogicalIndex());
  if (!base || !logical)
    return false;
  for (auto [symbol, coefficient] : offset.terms)
    if (isVarying(symbol) && logical->terms.lookup(symbol) != coefficient)
      return false;
  for (auto [symbol, coefficient] : logical->terms)
    if (isVarying(symbol) && offset.terms.lookup(symbol) != coefficient)
      return false;

  SmallVector<std::pair<int64_t, int64_t>> dimensions;
  int64_t programStride = 0;
  unsigned programAxes = 0;
  DenseSet<Operation *> loops;
  for (auto [symbol, coefficient] : offset.terms) {
    if (!isVarying(symbol))
      continue;
    if (coefficient <= 0)
      return false;
    if (auto range = symbol.getDefiningOp<MakeRangeOp>()) {
      dimensions.emplace_back(coefficient, range.getEnd() - range.getStart());
    } else if (auto program = symbol.getDefiningOp<GetProgramIdOp>()) {
      if (program.getAxis() != ProgramIDDim::X)
        return false;
      programStride = coefficient;
      ++programAxes;
    } else {
      auto arg = cast<BlockArgument>(symbol);
      auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      APInt lower, upper, step;
      if (!loop || arg != loop.getInductionVar() ||
          !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
          !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
          !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
          step.getSExtValue() <= 0 || upper.sle(lower))
        return false;
      int64_t stride, distance;
      if (llvm::SubOverflow(upper.getSExtValue(), lower.getSExtValue(),
                            distance) ||
          llvm::MulOverflow(coefficient, step.getSExtValue(), stride))
        return false;
      dimensions.emplace_back(
          stride, llvm::divideCeilSigned(distance, step.getSExtValue()));
      loops.insert(loop);
    }
  }
  for (Operation *parent = store->getParentOp(); !isa<FuncOp>(parent);
       parent = parent->getParentOp())
    if (!loops.contains(parent))
      return false;
  if (programAxes != 1)
    return false;
  llvm::sort(dimensions);
  int64_t span = 1;
  for (auto [stride, extent] : dimensions) {
    int64_t growth;
    if (stride < span || extent <= 0 ||
        llvm::MulOverflow(stride, extent - 1, growth) ||
        llvm::AddOverflow(span, growth, span))
      return false;
  }
  if (programAxes && programStride < span)
    return false;
  if (auto tensor = dyn_cast<RankedTensorType>(store.getPtr().getType())) {
    int64_t lanes = 1;
    for (auto [stride, extent] : dimensions)
      if (llvm::MulOverflow(lanes, extent, lanes))
        return false;
    if (lanes < tensor.getNumElements())
      return false;
  }
  bool conflictingStore = false;
  store->getParentOfType<FuncOp>().walk([&](Operation *other) {
    if (other == store.getOperation())
      return;
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(other)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances)
        conflictingStore |= isa<MemoryEffects::Write>(effect.getEffect());
    }
  });
  return !conflictingStore;
}

bool containsLoop(Operation *op) {
  return op->walk([](LoopLikeOpInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

// Inserts after the logical index and after the last loop that precedes the
// store in its block.
void setEntropyInsertionPoint(OpBuilder &builder, SrStoreOp store) {
  auto *index = store.getLogicalIndex().getDefiningOp();
  if (index && index->getBlock() != store->getBlock())
    index = nullptr;
  for (Operation *op = store->getPrevNode(); op != index;
       op = op->getPrevNode()) {
    if (containsLoop(op)) {
      builder.setInsertionPointAfter(op);
      return;
    }
  }
  if (index)
    builder.setInsertionPointAfter(index);
  else
    builder.setInsertionPointToStart(store->getBlock());
}

class EntropyBuilder {
  OpBuilder &b;
  Location loc;
  Type lane32, lane64;

public:
  EntropyBuilder(OpBuilder &builder, SrStoreOp store)
      : b(builder), loc(store.getLoc()),
        lane64(store.getLogicalIndex().getType()) {
    lane32 = b.getI32Type();
    if (auto tensor = dyn_cast<RankedTensorType>(lane64))
      lane32 = tensor.clone(b.getI32Type());
  }
  Value constant(Type type, uint64_t value) {
    auto scalar = cast<IntegerType>(getElementTypeOrSelf(type));
    auto attr = b.getIntegerAttr(scalar, APInt(scalar.getWidth(), value));
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return arith::ConstantOp::create(b, loc,
                                       DenseElementsAttr::get(tensor, attr));
    return arith::ConstantOp::create(b, loc, attr);
  }
  Value splat(Value value) {
    if (isa<RankedTensorType>(lane32))
      return SplatOp::create(b, loc, lane32, value);
    return value;
  }
  Value low(Value value, Type type) {
    return arith::TruncIOp::create(b, loc, type, value);
  }
  Value high(Value value, Type type) {
    auto shifted =
        arith::ShRUIOp::create(b, loc, value, constant(value.getType(), 32));
    return low(shifted, type);
  }
  Value xorWith(Value a, Value c) {
    return arith::XOrIOp::create(b, loc, a, c);
  }
  Value mix(Value value) {
    for (uint64_t multiplier : {0xff51afd7ed558ccdULL, 0xc4ceb9fe1a85ec53ULL}) {
      value = xorWith(value, arith::ShRUIOp::create(
                                 b, loc, value, constant(value.getType(), 33)));
      value = arith::MulIOp::create(b, loc, value,
                                    constant(value.getType(), multiplier));
    }
    return xorWith(value, arith::ShRUIOp::create(
                              b, loc, value, constant(value.getType(), 33)));
  }
  Value hash(Value a, Value c, uint64_t domain) {
    return mix(xorWith(mix(xorWith(a, constant(a.getType(), domain))), c));
  }
  Value generate(SrStoreOp store, FuncOp function, int rounds) {
    Value dynamicSeed = LoadOp::create(b, loc, function.getArgument(0));
    Value seed =
        hash(function.getArgument(1), dynamicSeed, 0x747269746f6e5301ULL);
    Value invocation = hash(function.getArgument(2),
                            constant(b.getI64Type(), 0), 0x747269746f6e5302ULL);
    Value c0 = low(store.getLogicalIndex(), lane32);
    Value c1 = high(store.getLogicalIndex(), lane32);
    Value c2 = splat(low(invocation, b.getI32Type()));
    Value c3 = splat(high(invocation, b.getI32Type()));
    Value k0 = low(seed, b.getI32Type()), k1 = high(seed, b.getI32Type());
    Value m0 = constant(lane32, 0xd2511f53), m1 = constant(lane32, 0xcd9e8d57);
    for (int round = 0; round < rounds; ++round) {
      Value lo0 = arith::MulIOp::create(b, loc, c0, m0);
      Value hi0 = MulhiUIOp::create(b, loc, c0, m0);
      Value lo1 = arith::MulIOp::create(b, loc, c2, m1);
      Value hi1 = MulhiUIOp::create(b, loc, c2, m1);
      Value mixed0 = xorWith(hi1, c1);
      Value next0 = xorWith(mixed0, splat(k0));
      Value mixed2 = xorWith(hi0, c3);
      Value next2 = xorWith(mixed2, splat(k1));
      c0 = next0;
      c1 = lo1;
      c2 = next2;
      c3 = lo0;
      if (round + 1 < rounds) {
        k0 = arith::AddIOp::create(b, loc, k0,
                                   constant(b.getI32Type(), 0x9e3779b9));
        k1 = arith::AddIOp::create(b, loc, k1,
                                   constant(b.getI32Type(), 0xbb67ae85));
      }
    }
    return c0;
  }
};

struct TritonMaterializeStochasticRounding
    : impl::TritonMaterializeStochasticRoundingBase<
          TritonMaterializeStochasticRounding> {
  using TritonMaterializeStochasticRoundingBase::
      TritonMaterializeStochasticRoundingBase;

  void runOnOperation() override {
    SmallVector<SrStoreOp> stores;
    getOperation().walk([&](SrStoreOp op) { stores.push_back(op); });
    if (mode.empty() && stores.empty())
      return;
    if ((mode != "plain" && mode != "enabled") || stores.size() != 1 ||
        stores[0].getSite() != site) {
      getOperation().emitError(
          "requires exactly one declared SR site and plain or enabled mode");
      signalPassFailure();
      return;
    }
    auto store = stores[0];
    if (mode == "enabled" && !proveOwnership(store)) {
      store.emitOpError(
          "cannot prove injective matching pointer and logical indices");
      signalPassFailure();
      return;
    }
    OpBuilder builder(store);
    auto valueType = getPointeeType(store.getPtr().getType());
    Value converted;
    if (mode == "plain") {
      converted = FpToFpOp::create(
          builder, store.getLoc(), valueType, store.getValue(),
          RoundingModeAttr::get(&getContext(), RoundingMode::RTNE));
    } else {
      auto function = store->getParentOfType<FuncOp>();
      auto signature = function.getFunctionType();
      auto seedPointer = signature.getNumInputs() >= 3
                             ? dyn_cast<PointerType>(signature.getInput(0))
                             : PointerType();
      if ((computeCapability != 100 && computeCapability != 103) ||
          ptxVersion < 87 || (rounds != 7 && rounds != 10) || !seedPointer ||
          !seedPointer.getPointeeType().isInteger(64) ||
          seedPointer.getAddressSpace() != PtrAddrSpace::Global ||
          !signature.getInput(1).isInteger(64) ||
          !signature.getInput(2).isInteger(64)) {
        store.emitOpError(
            "enabled ABI v1 requires sm100/sm103, PTX 8.7, 7 or 10 rounds and "
            "three seed arguments (ptr<i64>, i64, i64)");
        signalPassFailure();
        return;
      }
      OpBuilder entropyBuilder(&getContext());
      setEntropyInsertionPoint(entropyBuilder, store);
      EntropyBuilder entropy(entropyBuilder, store);
      auto random = entropy.generate(store, function, rounds);
      auto conversion = FpToFpOp::create(
          builder, store.getLoc(), valueType, store.getValue(), random,
          RoundingModeAttr::get(&getContext(), RoundingMode::RS));
      conversion->setAttr("sr_ieee", builder.getUnitAttr());
      converted = conversion;
    }
    Attribute cachePolicy;
    if (store.getCache() != CacheModifier::NONE ||
        store.getEvict() != EvictionPolicy::NORMAL)
      cachePolicy = CachePolicyAttr::get(&getContext(), store.getCache(),
                                         store.getEvict());
    StoreOp::create(builder, store.getLoc(), store.getPtr(), converted,
                    store.getMask(), cachePolicy);
    store.erase();
  }
};
} // namespace
} // namespace mlir::triton
