#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONNPOTPAD
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool isNpot(ArrayRef<int64_t> shape) {
  return llvm::any_of(shape, [](int64_t d) { return !llvm::isPowerOf2_64(d); });
}

bool touchesNpot(Type type) {
  if (auto ptr = dyn_cast<PointerType>(type))
    return touchesNpot(ptr.getPointeeType());
  if (auto desc = dyn_cast<TensorDescInterface>(type))
    return isNpot(desc.getShape());
  // A memdesc's first dim counts buffers: memdesc_index peels it, and a TLX
  // barrier array is [N]. Only the rest is the tile.
  if (auto memDesc = dyn_cast<gpu::MemDescType>(type))
    return isNpot(memDesc.getShape().drop_front());
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasRank() && isNpot(shaped.getShape());
}

RankedTensorType padded(RankedTensorType type) {
  return type.clone(llvm::map_to_vector(type.getShape(), [](int64_t d) {
    return static_cast<int64_t>(llvm::PowerOf2Ceil(d));
  }));
}

// Split src and dst into groups of dims with equal element counts. Only a
// group's leading non-unit dim may pad, so its live elements stay a row-major
// prefix on both sides.
bool padsOnlyGroupLeads(ArrayRef<int64_t> src, ArrayRef<int64_t> dst) {
  auto leadOnly = [](ArrayRef<int64_t> group) {
    auto rest = group.drop_while([](int64_t d) { return d == 1; });
    return rest.empty() || !isNpot(rest.drop_front());
  };
  size_t i = 0, j = 0;
  while (i < src.size() || j < dst.size()) {
    size_t i0 = i, j0 = j;
    int64_t srcElems = 1, dstElems = 1;
    do {
      if (srcElems <= dstElems && i < src.size())
        srcElems *= src[i++];
      else if (j < dst.size())
        dstElems *= dst[j++];
      else
        return false;
    } while (srcElems != dstElems);
    if (!leadOnly(src.slice(i0, i - i0)) || !leadOnly(dst.slice(j0, j - j0)))
      return false;
  }
  return true;
}

bool combines(ValueRange operands, Value lhs, Value rhs) {
  return operands.size() == 2 && ((operands[0] == lhs && operands[1] == rhs) ||
                                  (operands[0] == rhs && operands[1] == lhs));
}

// The exact identity of a single-op combiner, else null. getNeutralElement
// gives +0.0 for addf, and for float max/min an inf or NaN that some types
// lack; for those the largest finite value is exact.
TypedAttr identityOf(Operation *combiner) {
  Type type = combiner->getResult(0).getType();
  Builder b(type.getContext());
  if (isa<arith::AddFOp>(combiner))
    return b.getFloatAttr(type, -0.0);
  bool isNum = isa<arith::MaxNumFOp, arith::MinNumFOp>(combiner);
  if (isNum || isa<arith::MaximumFOp, arith::MinimumFOp>(combiner)) {
    const llvm::fltSemantics &sem = cast<FloatType>(type).getFloatSemantics();
    bool isMax = isa<arith::MaximumFOp, arith::MaxNumFOp>(combiner);
    if (isNum ? !APFloat::semanticsHasNaN(sem) : !APFloat::semanticsHasInf(sem))
      return b.getFloatAttr(type, APFloat::getLargest(sem, /*Negative=*/isMax));
  }
  return arith::getNeutralElement(combiner).value_or(TypedAttr());
}

// True if each result is a select, max, min, and or or of its two inputs, so
// combine(x, x) == x and the last live lane can stand in for the pad lanes.
bool isIdempotent(Region &combine) {
  Block &block = combine.front();
  unsigned n = block.getNumArguments() / 2;
  for (auto [k, result] :
       llvm::enumerate(block.getTerminator()->getOperands())) {
    Operation *def = result.getDefiningOp();
    if (!def)
      return false;
    std::optional<ValueRange> inputs =
        llvm::TypeSwitch<Operation *, std::optional<ValueRange>>(def)
            .Case([](arith::SelectOp op) {
              return std::optional<ValueRange>(op->getOperands().drop_front());
            })
            .Case<arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
                  arith::MinimumFOp, arith::MaxSIOp, arith::MinSIOp,
                  arith::MaxUIOp, arith::MinUIOp, arith::AndIOp, arith::OrIOp>(
                [](Operation *op) {
                  return std::optional<ValueRange>(op->getOperands());
                })
            .Default(std::optional<ValueRange>());
    if (!inputs ||
        !combines(*inputs, block.getArgument(k), block.getArgument(n + k)))
      return false;
  }
  return true;
}

class Padder {
public:
  LogicalResult admit(Operation *op);
  void apply(ModuleOp mod);

private:
  Value live(Operation *user, RankedTensorType orig);
  template <typename OpT> void andLive(OpT op, RankedTensorType orig);
  void fill(OpOperand &operand, RankedTensorType orig, TypedAttr value);
  void padLanes(Operation *op, RankedTensorType orig, unsigned axis,
                ArrayRef<TypedAttr> identities);

  SmallVector<std::function<void()>> fixups;
  DenseMap<std::pair<Region *, Type>, Value> liveCache;
};

// One i1 mask per (function, original shape): make_range(0, P) < N along
// each padded dim, broadcast to the padded shape and and-ed.
Value Padder::live(Operation *user, RankedTensorType orig) {
  Region *region = user->getParentRegion();
  while (!region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
    region = region->getParentOp()->getParentRegion();
  auto b = OpBuilder::atBlockBegin(&region->front());
  Value &mask = liveCache[{region, orig.clone(b.getI1Type())}];
  if (mask)
    return mask;
  Location loc = region->getParentOp()->getLoc();
  auto maskTy = padded(orig).clone(b.getI1Type());
  for (int d = 0, rank = orig.getRank(); d < rank; ++d) {
    int64_t n = orig.getDimSize(d), p = maskTy.getDimSize(d);
    if (n == p)
      continue;
    auto rangeTy = RankedTensorType::get({p}, b.getI32Type());
    Value range = MakeRangeOp::create(b, loc, rangeTy, 0, p);
    Value bound = arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(rangeTy, b.getI32IntegerAttr(n)));
    Value m =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, range, bound);
    for (int k = 0; k < rank; ++k)
      if (k != d)
        m = ExpandDimsOp::create(b, loc, m, k);
    if (m.getType() != maskTy)
      m = BroadcastOp::create(b, loc, maskTy, m);
    mask = mask ? arith::AndIOp::create(b, loc, mask, m).getResult() : m;
  }
  return mask;
}

// Pad lanes never touch memory.
template <typename OpT> void Padder::andLive(OpT op, RankedTensorType orig) {
  Value mask = live(op, orig);
  if (Value old = op.getMask()) {
    OpBuilder b(op);
    mask = arith::AndIOp::create(b, op.getLoc(), old, mask);
  }
  op.getMaskMutable().assign(mask);
}

// andLive sends a load's pad lanes (also seen through a transpose) to `other`,
// so the fill folds into the load when `other` is absent or already the fill,
// as in a hand-padded load, and the load keeps feeding the dot directly.
bool foldFill(Value v, TypedAttr value) {
  while (auto trans = v.getDefiningOp<TransOp>())
    v = trans.getSrc();
  auto load = v.getDefiningOp<LoadOp>();
  if (!load)
    return false;
  if (!load.getOther()) {
    OpBuilder b(load);
    load.getOtherMutable().assign(arith::ConstantOp::create(
        b, load.getLoc(),
        DenseElementsAttr::get(cast<RankedTensorType>(load.getType()), value)));
    return true;
  }
  DenseElementsAttr other;
  return matchPattern(load.getOther(), m_Constant(&other)) && other.isSplat() &&
         other.getSplatValue<Attribute>() == value;
}

// Pad lanes of `operand` take `value`.
void Padder::fill(OpOperand &operand, RankedTensorType orig, TypedAttr value) {
  Operation *op = operand.getOwner();
  OpBuilder b(op);
  auto type = cast<RankedTensorType>(operand.get().getType());
  Value pad = arith::ConstantOp::create(b, op->getLoc(),
                                        DenseElementsAttr::get(type, value));
  operand.set(arith::SelectOp::create(b, op->getLoc(), live(op, orig),
                                      operand.get(), pad));
}

// Pad lanes of each source take its identity, or else (no identities) a copy
// of the last live lane along `axis`.
void Padder::padLanes(Operation *op, RankedTensorType orig, unsigned axis,
                      ArrayRef<TypedAttr> identities) {
  for (OpOperand &src : op->getOpOperands()) {
    if (!identities.empty()) {
      fill(src, orig, identities[src.getOperandNumber()]);
      continue;
    }
    OpBuilder b(op);
    auto type = cast<RankedTensorType>(src.get().getType());
    Value last = arith::ConstantOp::create(
        b, op->getLoc(),
        DenseElementsAttr::get(type.clone(b.getI32Type()),
                               b.getI32IntegerAttr(orig.getDimSize(axis) - 1)));
    Value copy = GatherOp::create(b, op->getLoc(), src.get(), last, axis);
    src.set(arith::SelectOp::create(b, op->getLoc(), live(op, orig), src.get(),
                                    copy));
  }
}

// A result that combines only its own pair of block args (k and n+k) is
// independent of the others, so each source pads with its own identity. Empty
// unless every result has one.
SmallVector<TypedAttr> identitiesOf(Region &combine) {
  Block &block = combine.front();
  unsigned n = block.getNumArguments() / 2;
  SmallVector<TypedAttr> identities;
  for (auto [k, result] :
       llvm::enumerate(block.getTerminator()->getOperands())) {
    Operation *def = result.getDefiningOp();
    if (!def || !combines(def->getOperands(), block.getArgument(k),
                          block.getArgument(n + k)))
      return {};
    identities.push_back(identityOf(def));
    if (!identities.back())
      return {};
  }
  return identities;
}

// Padding moves an element unless each NPOT dim leads its group, so each live
// lane of the padded result gathers the source lane holding its row-major
// element of the original shapes. Pad lanes gather lane 0.
void relayout(ReshapeOp reshape, RankedTensorType origSrc,
              RankedTensorType origDst) {
  OpBuilder b(reshape);
  Location loc = reshape.getLoc();
  RankedTensorType src = reshape.getSrc().getType(), dst = reshape.getType();
  int64_t srcElems = src.getNumElements();
  auto indexTy = RankedTensorType::get({dst.getNumElements()}, b.getI32Type());
  auto splat = [&](int64_t v) -> Value {
    return arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(indexTy, b.getI32IntegerAttr(v)));
  };
  // Digit d of a row-major index whose dims after d hold `stride` elements.
  auto digit = [&](Value x, int64_t d, int64_t stride, int64_t size) -> Value {
    if (stride != 1)
      x = arith::DivUIOp::create(b, loc, x, splat(stride));
    return d == 0 ? x
                  : arith::RemUIOp::create(b, loc, x, splat(size)).getResult();
  };
  auto addScaled = [&](Value sum, Value x, int64_t stride) -> Value {
    if (stride != 1)
      x = arith::MulIOp::create(b, loc, x, splat(stride));
    return sum ? arith::AddIOp::create(b, loc, sum, x).getResult() : x;
  };
  Value lane = MakeRangeOp::create(b, loc, indexTy, 0, indexTy.getDimSize(0));
  Value elem, live;
  for (int64_t d = dst.getRank() - 1, laneStride = 1, stride = 1; d >= 0; --d) {
    Value c = digit(lane, d, laneStride, dst.getDimSize(d));
    if (origDst.getDimSize(d) != dst.getDimSize(d)) {
      Value in = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, c,
                                       splat(origDst.getDimSize(d)));
      live = live ? arith::AndIOp::create(b, loc, live, in).getResult() : in;
    }
    elem = addScaled(elem, c, stride);
    laneStride *= dst.getDimSize(d);
    stride *= origDst.getDimSize(d);
  }
  Value index;
  for (int64_t d = src.getRank() - 1, laneStride = 1, stride = 1; d >= 0; --d) {
    index = addScaled(index, digit(elem, d, stride, origSrc.getDimSize(d)),
                      laneStride);
    laneStride *= src.getDimSize(d);
    stride *= origSrc.getDimSize(d);
  }
  index = arith::SelectOp::create(b, loc, live, index, splat(0));
  Value flat = reshape.getSrc();
  if (src.getRank() != 1)
    flat = ReshapeOp::create(b, loc, ArrayRef(srcElems), flat);
  Value result = GatherOp::create(b, loc, flat, index, 0u);
  if (dst.getRank() != 1)
    result = ReshapeOp::create(b, loc, dst.getShape(), result);
  reshape.replaceAllUsesWith(result);
  reshape.erase();
}

LogicalResult Padder::admit(Operation *op) {
  if (auto func = dyn_cast<FunctionOpInterface>(op)) {
    if (llvm::any_of(func.getArgumentTypes(), touchesNpot) ||
        llvm::any_of(func.getResultTypes(), touchesNpot))
      return op->emitOpError(
          "with an NPOT tensor in its signature is unsupported");
    return success();
  }
  SmallVector<Type> types(op->getOperandTypes());
  llvm::append_range(types, op->getResultTypes());
  for (Region &region : op->getRegions())
    llvm::append_range(types, region.getArgumentTypes());
  if (!llvm::any_of(types, touchesNpot))
    return success();
  if (llvm::any_of(types, [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        return touchesNpot(type) && !(tensor && !tensor.getEncoding());
      }))
    return op->emitOpError("on an NPOT pointer-to-tensor, descriptor, memory "
                           "descriptor or encoded tensor is unsupported");

  // Lanewise ops and control flow are only retyped.
  if (op->hasTrait<OpTrait::Elementwise>()) {
    if (!isMemoryEffectFree(op))
      return op->emitOpError(
          "with side effects on an NPOT tensor is unsupported");
    return success();
  }
  if (isa<SplatOp, BroadcastOp, ExpandDimsOp, TransOp, JoinOp, SplitOp,
          RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(op))
    return success();
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    int64_t end =
        range.getStartAttr().getInt() + padded(range.getType()).getDimSize(0);
    if (end > std::numeric_limits<int32_t>::max())
      return op->emitOpError(
          "with an NPOT extent whose padded end overflows i32 is unsupported");
    fixups.push_back([range, end = static_cast<uint32_t>(end)]() mutable {
      range.setEnd(end);
    });
    return success();
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
    if (!splat)
      return op->emitOpError("with a non-splat NPOT value is unsupported");
    auto type = padded(cast<RankedTensorType>(cst.getType()));
    fixups.push_back([cst, splat, type]() mutable {
      cst.setValueAttr(splat.resizeSplat(type));
    });
    return success();
  }
  if (auto load = dyn_cast<LoadOp>(op)) {
    auto type = cast<RankedTensorType>(load.getPtr().getType());
    fixups.push_back([this, load, type] { andLive(load, type); });
    return success();
  }
  if (auto store = dyn_cast<StoreOp>(op)) {
    auto type = cast<RankedTensorType>(store.getPtr().getType());
    fixups.push_back([this, store, type] { andLive(store, type); });
    return success();
  }
  // Any element order is legal under allow_reorder, so it pads as ordered.
  if (auto reshape = dyn_cast<ReshapeOp>(op)) {
    RankedTensorType src = reshape.getSrc().getType(), dst = reshape.getType();
    if (!padsOnlyGroupLeads(src.getShape(), dst.getShape()))
      fixups.push_back([reshape, src, dst] { relayout(reshape, src, dst); });
    else if (reshape.getAllowReorder())
      fixups.push_back([reshape]() mutable { reshape.setAllowReorder(false); });
    return success();
  }
  // A forward scan needs no pad value, since pad lanes come last.
  auto reduce = dyn_cast<ReduceOp>(op);
  auto scan = dyn_cast<ScanOp>(op);
  if (reduce || scan) {
    auto orig = cast<RankedTensorType>(op->getOperand(0).getType());
    unsigned axis = reduce ? reduce.getAxis() : scan.getAxis();
    if (llvm::isPowerOf2_64(orig.getDimSize(axis)) ||
        (scan && !scan.getReverse()))
      return success();
    SmallVector<TypedAttr> identities = identitiesOf(op->getRegion(0));
    if (identities.empty() && !isIdempotent(op->getRegion(0)))
      return op->emitOpError(
          "over an NPOT axis with a combiner that has no pad value is "
          "unsupported");
    fixups.push_back([this, op, orig, axis, identities] {
      padLanes(op, orig, axis, identities);
    });
    return success();
  }
  if (auto hist = dyn_cast<HistogramOp>(op)) {
    auto type = cast<RankedTensorType>(hist.getSrc().getType());
    if (isNpot(type.getShape()))
      fixups.push_back([this, hist, type] { andLive(hist, type); });
    return success();
  }
  // Pad lanes gather index 0, never past the source.
  if (auto gather = dyn_cast<GatherOp>(op)) {
    RankedTensorType type = gather.getIndices().getType();
    if (isNpot(type.getShape()))
      fixups.push_back([this, gather, type]() mutable {
        fill(gather.getIndicesMutable(), type,
             Builder(gather.getContext()).getZeroAttr(type.getElementType()));
      });
    return success();
  }
  if (auto rmw = dyn_cast<AtomicRMWOp>(op)) {
    auto type = cast<RankedTensorType>(rmw.getPtr().getType());
    fixups.push_back([this, rmw, type] { andLive(rmw, type); });
    return success();
  }
  // A padded K is zero-filled on both operands and a scale pad is 1. A padded
  // M or N only adds rows and columns that the store masks.
  if (isa<DotOp, DotScaledOp>(op)) {
    auto a = cast<RankedTensorType>(op->getOperand(0).getType());
    if (!isNpot(a.getShape().take_back()))
      return success();
    SmallVector<std::pair<OpOperand *, bool>> operands = {
        {&op->getOpOperand(0), false}, {&op->getOpOperand(1), false}};
    if (auto scaled = dyn_cast<DotScaledOp>(op))
      for (MutableOperandRange scale :
           {scaled.getAScaleMutable(), scaled.getBScaleMutable()})
        if (!scale.empty())
          operands.push_back({&scale[0], true});
    Builder b(op->getContext());
    for (auto [operand, isScale] : operands) {
      auto orig = cast<RankedTensorType>(operand->get().getType());
      Type elem = orig.getElementType();
      TypedAttr value = b.getZeroAttr(elem);
      // Integer scale 127 is 1.0 as E8M0 (bias 127), a finite 1.875 as UE5M3.
      if (isScale && isa<FloatType>(elem))
        value = b.getFloatAttr(elem, 1.0);
      else if (isScale)
        value = b.getIntegerAttr(elem, 127);
      fixups.push_back([this, use = operand, orig, value] {
        if (!foldFill(use->get(), value))
          fill(*use, orig, value);
      });
    }
    return success();
  }
  return op->emitOpError("on an NPOT tensor is unsupported");
}

void Padder::apply(ModuleOp mod) {
  auto retype = [](Value value) {
    if (auto type = dyn_cast<RankedTensorType>(value.getType());
        type && isNpot(type.getShape()))
      value.setType(padded(type));
  };
  mod.walk([&](Operation *op) {
    llvm::for_each(op->getResults(), retype);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        llvm::for_each(block.getArguments(), retype);
  });
  for (auto &fixup : fixups)
    fixup();
}

} // namespace

struct NpotPadPass : impl::TritonNpotPadBase<NpotPadPass> {
  // Admits every op before changing any, so an error leaves the IR unchanged.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Padder padder;
    if (mod.walk([&](Operation *op) { return WalkResult(padder.admit(op)); })
            .wasInterrupted())
      return signalPassFailure();
    padder.apply(mod);
  }
};

} // namespace mlir::triton
