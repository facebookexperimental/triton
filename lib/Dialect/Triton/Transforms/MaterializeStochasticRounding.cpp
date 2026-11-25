#include "StochasticStoreOwnership.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONMATERIALIZESTOCHASTICROUNDING
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
namespace {

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
    for (auto [symbol, coefficient] : offset.terms)
      if (auto range = symbol.getDefiningOp<MakeRangeOp>())
        if (llvm::MulOverflow(lanes, int64_t(range.getEnd() - range.getStart()),
                              lanes))
          return false;
    if (lanes != tensor.getNumElements())
      return false;
  }
  bool conflictingStore = false;
  store->getParentOfType<FuncOp>().walk([&](Operation *other) {
    if (other == store.getOperation())
      return;
    conflictingStore |= isa<CallOp>(other);
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(other)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances)
        if (isa<MemoryEffects::Write>(effect.getEffect())) {
          AffineIndex otherOffset;
          Value pointer;
          if (auto ordinary = dyn_cast<StoreOp>(other))
            pointer = ordinary.getPtr();
          else if (auto annotated = dyn_cast<SrStoreOp>(other))
            pointer = annotated.getPtr();
          auto otherBase =
              pointer ? pointerBase(pointer, otherOffset) : Value();
          conflictingStore |= !otherBase || otherBase == base;
        }
    }
  });
  return !conflictingStore;
}

bool containsLoop(Operation *op) {
  return op->walk([](LoopLikeOpInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

// Inserts before the first entry-block op that holds an SR store and after the
// last loop that precedes it.
void setSeedInsertionPoint(OpBuilder &builder, FuncOp function) {
  Block &entry = function.getBody().front();
  Operation *anchor = entry.getTerminator();
  function.walk([&](SrStoreOp store) {
    Operation *ancestor = entry.findAncestorOpInBlock(*store.getOperation());
    if (ancestor && ancestor->isBeforeInBlock(anchor))
      anchor = ancestor;
  });
  for (Operation *op = anchor->getPrevNode(); op; op = op->getPrevNode()) {
    if (containsLoop(op)) {
      builder.setInsertionPointAfter(op);
      return;
    }
  }
  builder.setInsertionPointToStart(&entry);
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
  Value softwareRound(SrStoreOp store, Value random) {
    Value bits = BitcastOp::create(b, loc, lane32, store.getValue());
    Value magnitude =
        arith::AndIOp::create(b, loc, bits, constant(lane32, 0x7fffffff));
    Value special =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, magnitude,
                              constant(lane32, 0x7f800000));
    Value nan = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                      magnitude, constant(lane32, 0x7f800000));
    Value entropy =
        arith::AndIOp::create(b, loc, random, constant(lane32, 0xffff));
    Value rounded = arith::AddIOp::create(b, loc, bits, entropy);
    rounded = arith::ShRUIOp::create(b, loc, rounded, constant(lane32, 16));
    if (store->hasAttr("sr_satfinite")) {
      Value magnitude =
          arith::AndIOp::create(b, loc, rounded, constant(lane32, 0x7fff));
      Value overflow =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, magnitude,
                                constant(lane32, 0x7f80));
      Value maximum = arith::OrIOp::create(
          b, loc,
          arith::AndIOp::create(b, loc, rounded, constant(lane32, 0x8000)),
          constant(lane32, 0x7f7f));
      rounded = arith::SelectOp::create(b, loc, overflow, maximum, rounded);
    }
    Value original = arith::ShRUIOp::create(b, loc, bits, constant(lane32, 16));
    original = arith::SelectOp::create(b, loc, nan, constant(lane32, 0x7fff),
                                       original);
    rounded = arith::SelectOp::create(b, loc, special, original, rounded);
    Type lane16 = b.getI16Type();
    if (auto tensor = dyn_cast<RankedTensorType>(lane32))
      lane16 = tensor.clone(b.getI16Type());
    rounded = arith::TruncIOp::create(b, loc, lane16, rounded);
    return BitcastOp::create(b, loc, getPointeeType(store.getPtr().getType()),
                             rounded);
  }

  Value generate(SrStoreOp store, FuncOp function, int rounds,
                 Value dynamicSeed, bool packed) {
    OpBuilder entryBuilder(function.getContext());
    entryBuilder.setInsertionPointAfter(dynamicSeed.getDefiningOp());
    EntropyBuilder uniformEntropy(entryBuilder, store);
    Value seed = uniformEntropy.hash(function.getArgument(1), dynamicSeed,
                                     0x747269746f6e5301ULL);
    Value invocation = uniformEntropy.hash(
        function.getArgument(2),
        uniformEntropy.constant(b.getI64Type(), store.getSiteId()),
        0x747269746f6e5302ULL);
    Value index = store.getLogicalIndex();
    if (packed)
      index = arith::ShRUIOp::create(b, loc, index, constant(lane64, 3));
    Value c0 = low(index, lane32);
    Value c1 = high(index, lane32);
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
    if (!packed)
      return c0;
    Value slice = arith::AndIOp::create(
        b, loc, low(store.getLogicalIndex(), lane32), constant(lane32, 7));
    auto bitSet = [&](uint32_t bit) {
      return arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::ne,
          arith::AndIOp::create(b, loc, slice, constant(lane32, bit)),
          constant(lane32, 0));
    };
    Value first = arith::SelectOp::create(b, loc, bitSet(2), c1, c0);
    Value second = arith::SelectOp::create(b, loc, bitSet(2), c3, c2);
    Value word = arith::SelectOp::create(b, loc, bitSet(4), second, first);
    Value shift = arith::MulIOp::create(
        b, loc, arith::AndIOp::create(b, loc, slice, constant(lane32, 1)),
        constant(lane32, 16));
    word = arith::ShRUIOp::create(b, loc, word, shift);
    return arith::AndIOp::create(b, loc, word, constant(lane32, 0xffff));
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
    llvm::StringMap<int32_t> declarations;
    SmallVector<StringRef> entries;
    StringRef(site).split(entries, ',');
    bool invalid = (mode != "plain" && mode != "enabled") ||
                   (ownership != "checked" && ownership != "declared") ||
                   (entropyMode != "element" && entropyMode != "packed8");
    for (auto entry : entries) {
      auto [name, number] = entry.split('=');
      int32_t id = 0;
      invalid |= name.empty() ||
                 (!number.empty() && number.getAsInteger(10, id)) || id < 0;
      invalid |= !declarations.try_emplace(name, id).second;
    }
    llvm::StringMap<unsigned> counts;
    for (auto store : stores) {
      auto declaration = declarations.find(store.getSite());
      if (declaration == declarations.end()) {
        invalid = true;
        continue;
      }
      store.setSiteId(declaration->second);
      ++counts[store.getSite()];
    }
    for (auto &entry : declarations)
      invalid |= counts.lookup(entry.getKey()) != 1;
    if (invalid) {
      getOperation().emitError("requires exactly one declared SR site instance "
                               "per name and plain or enabled mode");
      signalPassFailure();
      return;
    }
    DenseMap<Operation *, Value> dynamicSeeds;
    for (auto store : stores) {
      bool eligible =
          getElementTypeOrSelf(store.getValue()).isF32() &&
          getElementTypeOrSelf(getPointeeType(store.getPtr().getType()))
              .isBF16();
      if (mode == "enabled" && eligible && !proveOwnership(store) &&
          !proveLaunchOwnership(store) &&
          !(ownership == "declared" && proveDeclaredOwnership(store))) {
        store.emitOpError(
            "cannot prove injective matching pointer and logical indices");
        signalPassFailure();
        return;
      }
      if (mode == "enabled") {
        auto *next = store->getNextNode();
        while (next && !next->hasTrait<OpTrait::IsTerminator>() &&
               isMemoryEffectFree(next)) {
          store->moveAfter(next);
          next = store->getNextNode();
        }
      }
      OpBuilder builder(store);
      auto valueType = getPointeeType(store.getPtr().getType());
      Value converted;
      if (mode == "plain" || !eligible) {
        if (valueType == store.getValue().getType())
          converted = store.getValue();
        else
          converted = FpToFpOp::create(
              builder, store.getLoc(), valueType, store.getValue(),
              RoundingModeAttr::get(&getContext(), RoundingMode::RTNE));
      } else {
        auto function = store->getParentOfType<FuncOp>();
        auto signature = function.getFunctionType();
        auto seedPointer = signature.getNumInputs() >= 3
                               ? dyn_cast<PointerType>(signature.getInput(0))
                               : PointerType();
        if ((rounds != 7 && rounds != 10) || !seedPointer ||
            !seedPointer.getPointeeType().isInteger(64) ||
            seedPointer.getAddressSpace() != PtrAddrSpace::Global ||
            !signature.getInput(1).isInteger(64) ||
            !signature.getInput(2).isInteger(64)) {
          store.emitOpError("enabled ABI v1 requires 7 or 10 rounds and three "
                            "seed arguments (ptr<i64>, i64, i64)");
          signalPassFailure();
          return;
        }
        for (unsigned arg = 3; arg < function.getNumArguments(); ++arg) {
          auto constraint =
              function.getArgAttrOfType<IntegerAttr>(arg, "tt.sr_value");
          auto type =
              dyn_cast<IntegerType>(function.getArgument(arg).getType());
          if (!constraint || !type)
            continue;
          OpBuilder entryBuilder(function.getContext());
          entryBuilder.setInsertionPointToStart(&function.getBody().front());
          auto constant = arith::ConstantOp::create(
              entryBuilder, store.getLoc(), type,
              entryBuilder.getIntegerAttr(type, constraint.getInt()));
          function.getArgument(arg).replaceAllUsesWith(constant);
        }
        OpBuilder entropyBuilder(&getContext());
        entropyBuilder.setInsertionPoint(store);
        EntropyBuilder entropy(entropyBuilder, store);
        auto &dynamicSeed = dynamicSeeds[function.getOperation()];
        if (!dynamicSeed) {
          OpBuilder seedBuilder(function.getContext());
          setSeedInsertionPoint(seedBuilder, function);
          dynamicSeed = LoadOp::create(seedBuilder, store.getLoc(),
                                       function.getArgument(0));
        }
        auto random = entropy.generate(store, function, rounds, dynamicSeed,
                                       entropyMode == "packed8");
        if ((computeCapability == 100 || computeCapability == 103) &&
            ptxVersion >= 87) {
          auto conversion = FpToFpOp::create(
              builder, store.getLoc(), valueType, store.getValue(), random,
              RoundingModeAttr::get(&getContext(), RoundingMode::RS));
          conversion->setAttr("sr_ieee", builder.getUnitAttr());
          converted = conversion;
          if (store->hasAttr("sr_satfinite")) {
            conversion->setAttr("sr_satfinite", builder.getUnitAttr());
            Type integerType = builder.getI32Type();
            if (auto tensor =
                    dyn_cast<RankedTensorType>(store.getValue().getType()))
              integerType = tensor.clone(builder.getI32Type());
            Value bits = BitcastOp::create(builder, store.getLoc(), integerType,
                                           store.getValue());
            Value exponent = arith::AndIOp::create(
                builder, store.getLoc(), bits,
                entropy.constant(integerType, 0x7f800000));
            Value special = arith::CmpIOp::create(
                builder, store.getLoc(), arith::CmpIPredicate::eq, exponent,
                entropy.constant(integerType, 0x7f800000));
            Value original = FpToFpOp::create(
                builder, store.getLoc(), valueType, store.getValue(),
                RoundingModeAttr::get(&getContext(), RoundingMode::RTNE));
            converted = arith::SelectOp::create(builder, store.getLoc(),
                                                special, original, converted);
          }
        } else {
          converted = entropy.softwareRound(store, random);
        }
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
  }
};
} // namespace
} // namespace mlir::triton
