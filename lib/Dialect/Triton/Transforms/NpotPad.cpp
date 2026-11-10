#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
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

SmallVector<Value> npotValues(Operation *op) {
  SmallVector<Value> values(op->getOperands());
  llvm::append_range(values, op->getResults());
  for (Region &region : op->getRegions())
    for (Block &block : region)
      llvm::append_range(values, block.getArguments());
  llvm::erase_if(values, [](Value v) { return !touchesNpot(v.getType()); });
  return values;
}

// Padding needs a mask to keep a pad lane's side effect from landing. An atomic
// without a predicate can't be masked, nor can a TMA store's box, so selection
// keeps its connected component exact where it can.
static bool hasPaddedLowering(Operation *op) {
  return !(isa<AtomicOpInterface>(op) && !isa<PredicatedOpInterface>(op)) &&
         !isa<DescriptorStoreLikeOpInterface>(op);
}

// Groups the NPOT tensors connected through their ops (a connected component
// of the def-use graph). A connected component keeps its shape iff its ops
// lower a mixed-radix layout without an unsupported reason and, unless one has
// no padded lowering, each access keeps its padded vector and its mixed-radix
// layouts take a smaller total in-dim size in sum than its padded Blocked
// layouts.
class ExactComponents {
public:
  ExactComponents(ModuleOp mod, int numWarps, int threadsPerWarp, int numCTAs,
                  StringRef target, int nonKDim, int numStages,
                  bool useAsyncCopy);
  bool contains(Value v) const {
    return !paddedComponents.contains(components.getLeaderValue(v));
  }
  bool contains(Operation *op) const {
    SmallVector<Value> values = npotValues(op);
    return !values.empty() && contains(values.front());
  }

private:
  llvm::EquivalenceClasses<Value> components;
  DenseSet<Value> paddedComponents;
};

ExactComponents::ExactComponents(ModuleOp mod, int numWarps, int threadsPerWarp,
                                 int numCTAs, StringRef target, int nonKDim,
                                 int numStages, bool useAsyncCopy) {
  mod.walk([&](Operation *op) {
    SmallVector<Value> values = npotValues(op);
    for (Value v : values)
      components.unionSets(values.front(), v);
    // A value forwarded along control flow and the region argument, parent
    // result or block argument it becomes take one layout.
    auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op);
    if (auto parent = terminator
                          ? dyn_cast<RegionBranchOpInterface>(op->getParentOp())
                          : nullptr) {
      RegionBranchSuccessorMapping mapping;
      parent.getSuccessorOperandInputMapping(mapping,
                                             RegionBranchPoint(terminator));
      for (auto &[operand, inputs] : mapping)
        for (Value input : inputs)
          if (touchesNpot(input.getType()))
            components.unionSets(operand->get(), input);
    }
    if (auto branch = dyn_cast<BranchOpInterface>(op))
      for (OpOperand &operand : op->getOpOperands())
        if (std::optional<BlockArgument> arg =
                branch.getSuccessorBlockArgument(operand.getOperandNumber());
            arg && touchesNpot(arg->getType()))
          components.unionSets(operand.get(), *arg);
  });
  MLIRContext *ctx = mod.getContext();
  // A value reaching expand_dims or a growing broadcast takes its
  // consumer's layout, so tiles alone decide the connected component:
  // consumer-determined values are excluded from both total in-dim size sums.
  auto isConsumerDetermined = [&](Value v) {
    SmallVector<Value> worklist{v};
    DenseSet<Value> seen{v};
    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      auto curShaped = cast<RankedTensorType>(cur.getType());
      for (Operation *user : cur.getUsers()) {
        if (isa<ExpandDimsOp>(user))
          return true;
        if (auto bcast = dyn_cast<BroadcastOp>(user)) {
          if (auto resTy = dyn_cast<RankedTensorType>(bcast.getType());
              resTy && resTy.getNumElements() > curShaped.getNumElements())
            return true;
        }
        if (user->getNumResults() != 1)
          continue;
        Value res = user->getResult(0);
        auto resTy = dyn_cast<RankedTensorType>(res.getType());
        if (resTy && resTy.getShape() == curShaped.getShape() &&
            seen.insert(res).second)
          worklist.push_back(res);
      }
    }
    return false;
  };
  DenseSet<Value> consumerDetermined;
  mod.walk([&](Operation *op) {
    for (Value v : npotValues(op))
      if (isa<RankedTensorType>(v.getType()) && isConsumerDetermined(v))
        consumerDetermined.insert(v);
  });
  // Adds v's total in-dim size (registers x lanes x warps) under its
  // mixed-radix and Blocked layouts to sum; false if v cannot take a
  // mixed-radix layout.
  auto addTotalInDimSizes = [&](Value v, std::pair<int64_t, int64_t> &sum) {
    if (auto desc = dyn_cast<TensorDescType>(v.getType()))
      return !desc.getSharedLayout();
    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (!type || type.getEncoding())
      return false;
    FailureOr<gpu::LinearEncodingAttr> exact = gpu::getMixedRadixEncoding(
        ctx, type.getShape(), numWarps, threadsPerWarp, numCTAs);
    if (failed(exact))
      return false;
    if (consumerDetermined.contains(v))
      return true;
    ArrayRef<int64_t> pow2 = padded(type).getShape();
    auto pad = gpu::getDefaultBlockedEncoding(ctx, pow2, numWarps,
                                              threadsPerWarp, numCTAs);
    sum.first += exact->getLinearLayout().getTotalInDimSize();
    sum.second += pad.toLinearLayout(pow2).getTotalInDimSize();
    return true;
  };
  // Vector first, per the spec: a load or store must keep the vector AxisInfo
  // proves for its padded pointer along the dim Coalesce vectorizes.
  std::optional<ModuleAxisInfoAnalysis> axisInfo;
  auto losesVector = [&](Operation *op) -> std::optional<std::string> {
    Value ptr =
        llvm::TypeSwitch<Operation *, Value>(op)
            .Case<LoadOp, StoreOp>([](auto access) { return access.getPtr(); })
            .Default(Value());
    auto type = ptr ? dyn_cast<RankedTensorType>(ptr.getType()) : nullptr;
    if (!type || !isNpot(type.getShape()))
      return std::nullopt;
    if (!axisInfo)
      axisInfo.emplace(mod);
    const AxisInfo &info = *axisInfo->getAxisInfo(ptr);
    auto order = getOrderFromContiguity(info.getContiguity());
    int64_t threads = int64_t(numWarps) * threadsPerWarp;
    unsigned pow2Vector = getAxisInfoVector(info, type, padded(type).getShape(),
                                            order[0], threads);
    unsigned vector =
        getAxisInfoVector(info, type, type.getShape(), order[0], threads);
    unsigned exactVector = gpu::getMixedRadixVector(
        ctx, type.getShape(), vector, order, numWarps, threadsPerWarp);
    if (exactVector >= pow2Vector)
      return std::nullopt;
    return llvm::formatv("NPOT {0} pads: its exact vector is {1} elements, {2} "
                         "on the pow2 tile",
                         op->getName(), exactVector, pow2Vector)
        .str();
  };
  // The pipeliner copies a dot's operands to LDS in the order and vector
  // Coalesce gives their loads: the last dim fastest and 128 bits unless
  // AxisInfo proves otherwise. Only loads in a multi-stage loop are copied.
  auto slowDot = [&](Operation *op) -> std::optional<std::string> {
    auto dot = dyn_cast<DotOpInterface>(op);
    if (!dot)
      return std::nullopt;
    SmallVector<NpotDotOperandAccess, 2> operands;
    for (Value operand : {dot.getA(), dot.getB()}) {
      auto type = cast<RankedTensorType>(operand.getType());
      NpotDotOperandAccess &access = operands.emplace_back();
      access.order =
          llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(type.getRank())));
      access.vector = std::max(128u / type.getElementTypeBitWidth(), 1u);
      access.asyncStages = 0;
      auto load = operand.getDefiningOp<LoadOp>();
      if (!load)
        continue;
      if (auto forOp = load->getParentOfType<scf::ForOp>()) {
        IntegerAttr stages =
            TritonDialect::getLoaded(forOp)->getNumStagesAttrHelper().getAttr(
                forOp);
        int64_t loopStages = stages ? stages.getInt() : numStages;
        if (useAsyncCopy && loopStages > 1)
          access.asyncStages = loopStages;
      }
      if (!axisInfo)
        axisInfo.emplace(mod);
      const AxisInfo &info = *axisInfo->getAxisInfo(load.getPtr());
      access.order =
          llvm::to_vector(getOrderFromContiguity(info.getContiguity()));
      access.vector = getAxisInfoVector(info, load.getPtr().getType(),
                                        type.getShape(), access.order[0],
                                        int64_t(numWarps) * threadsPerWarp);
    }
    return getNpotDotSlowReason(op, operands, numWarps, threadsPerWarp, nonKDim,
                                target);
  };
  // A mixed-radix op's NPOT dim may be on any operand, e.g. a dot's B.
  auto rejects = [&](Operation *op) -> std::optional<std::string> {
    auto mixedRadix = dyn_cast<MixedRadixLayoutOpInterface>(op);
    RankedTensorType type;
    for (Type t : op->getOperandTypes()) {
      if (auto desc = dyn_cast<TensorDescType>(t))
        t = desc.getBlockType();
      type = mixedRadix || isa<DotOpInterface>(op)
                 ? dyn_cast<RankedTensorType>(t)
                 : nullptr;
      if (type && isNpot(type.getShape()))
        break;
      type = nullptr;
    }
    FailureOr<gpu::LinearEncodingAttr> exact = failure();
    if (type && !type.getEncoding())
      exact = gpu::getMixedRadixEncoding(ctx, type.getShape(), numWarps,
                                         threadsPerWarp, numCTAs);
    if (failed(exact))
      return std::nullopt;
    if (isa<DescriptorOpInterface>(op))
      return getNpotDescriptorUnsupportedReason(op, exact->getLinearLayout(),
                                                target);
    if (!isa<DotOpInterface>(op))
      return mixedRadix.getMixedRadixUnsupportedReason(
          exact->getLinearLayout());
    return getNpotDotUnsupportedReason(op, exact->getLinearLayout(), target,
                                       numWarps, nonKDim);
  };
  // A dot or a descriptor remarks its target reason; else a padded connected
  // component with a dot remarks the first reason it pads, once.
  using Reasons = llvm::MapVector<Value, std::pair<Operation *, std::string>>;
  Reasons padReasons, slowReasons;
  auto pad = [&](Value leader, Operation *op, std::string reason) {
    paddedComponents.insert(leader);
    padReasons.try_emplace(leader, op, std::move(reason));
  };
  DenseMap<Value, std::pair<int64_t, int64_t>> totalInDimSizes;
  DenseMap<Value, Operation *> firstOps;
  DenseSet<Value> seen, dots, remarked, unpaddable;
  mod.walk([&](Operation *op) {
    SmallVector<Value> values = npotValues(op);
    if (values.empty())
      return;
    Value leader = components.getLeaderValue(values.front());
    firstOps.try_emplace(leader, op);
    if (isa<DotOpInterface>(op))
      dots.insert(leader);
    if (!gpu::isMixedRadixLayoutOp(op->getName())) {
      pad(leader, op,
          llvm::formatv("NPOT mixed-radix layout is not yet supported on {0}",
                        op->getName()));
    } else if (std::optional<std::string> reason = rejects(op)) {
      if (isa<DotOpInterface, DescriptorOpInterface>(op)) {
        op->emitRemark(*reason);
        remarked.insert(leader);
      }
      pad(leader, op, *reason);
    } else if (std::optional<std::string> reason = losesVector(op)) {
      slowReasons.try_emplace(leader, op, *reason);
    } else if (std::optional<std::string> reason = slowDot(op)) {
      slowReasons.try_emplace(leader, op, *reason);
    }
    if (!hasPaddedLowering(op))
      unpaddable.insert(leader);
    for (Value v : values)
      if (seen.insert(v).second &&
          !addTotalInDimSizes(v, totalInDimSizes[leader]))
        pad(leader, op,
            llvm::formatv(
                "NPOT mixed-radix layout is not yet supported for {0}",
                v.getType()));
  });
  // Summed over the connected component: a thin index value may take a larger
  // total in-dim size than its padded layout while the tile it addresses takes
  // less.
  for (auto &[leader, sum] : totalInDimSizes)
    if (sum.first >= sum.second)
      slowReasons.try_emplace(
          leader, firstOps.lookup(leader),
          llvm::formatv("NPOT tensors pad: their mixed-radix layouts take {0} "
                        "registers x lanes x warps, their pow2 layouts {1}",
                        sum.first, sum.second));
  for (auto &[leader, reason] : slowReasons)
    if (!unpaddable.contains(leader))
      pad(leader, reason.first, reason.second);
  for (auto &[leader, reason] : padReasons)
    if (dots.contains(leader) && !remarked.contains(leader))
      reason.first->emitRemark(reason.second);
}

class Padder {
public:
  LogicalResult accept(Operation *op);
  void apply(ModuleOp mod, const ExactComponents &exact);

private:
  Value live(Operation *user, RankedTensorType orig);
  template <typename OpT> void andLive(OpT op, RankedTensorType orig);
  void fill(OpOperand &operand, RankedTensorType orig, TypedAttr value);
  void padLanes(Operation *op, RankedTensorType orig, unsigned axis,
                ArrayRef<TypedAttr> identities);
  void storeThroughPointers(DescriptorStoreOp store, RankedTensorType orig);

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

// A pad lane, or a coordinate outside the descriptor's shape (which TMA would
// clip), stores nothing.
void Padder::storeThroughPointers(DescriptorStoreOp store,
                                  RankedTensorType orig) {
  auto desc = store.getDesc().getDefiningOp<MakeTensorDescOp>();
  OpBuilder b(store);
  Location loc = store.getLoc();
  RankedTensorType type = store.getSrc().getType();
  int rank = type.getRank();
  auto grow = [&](Value row, int d) -> Value {
    for (int k = 0; k < rank; ++k)
      if (k != d)
        row = ExpandDimsOp::create(b, loc, row, k);
    auto to = cast<RankedTensorType>(row.getType()).clone(type.getShape());
    return row.getType() == to ? row : BroadcastOp::create(b, loc, to, row);
  };
  Value ptr = SplatOp::create(b, loc, type.clone(desc.getBase().getType()),
                              desc.getBase());
  Value mask = live(store, orig);
  for (auto [d, index] : llvm::enumerate(store.getIndices())) {
    auto rowTy = RankedTensorType::get({type.getDimSize(d)}, b.getI32Type());
    auto row64Ty = rowTy.clone(b.getI64Type());
    Value c = arith::AddIOp::create(
        b, loc, MakeRangeOp::create(b, loc, rowTy, 0, type.getDimSize(d)),
        SplatOp::create(b, loc, rowTy, index));
    Value zero = arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(rowTy, b.getI32IntegerAttr(0)));
    Value in = arith::AndIOp::create(
        b, loc,
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, c, zero),
        arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::slt, c,
            SplatOp::create(b, loc, rowTy, desc.getShape()[d])));
    in = grow(in, d);
    mask = mask ? arith::AndIOp::create(b, loc, mask, in).getResult() : in;
    Value offset = arith::MulIOp::create(
        b, loc, arith::ExtSIOp::create(b, loc, row64Ty, c),
        SplatOp::create(b, loc, row64Ty, desc.getStrides()[d]));
    ptr = AddPtrOp::create(b, loc, ptr.getType(), ptr, grow(offset, d));
  }
  StoreOp::create(b, loc, ptr, store.getSrc(), mask);
  store.erase();
  if (desc->use_empty())
    desc.erase();
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

LogicalResult Padder::accept(Operation *op) {
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
        auto desc = dyn_cast<TensorDescType>(type);
        auto tensor = desc && !desc.getSharedLayout()
                          ? desc.getBlockType()
                          : dyn_cast<RankedTensorType>(type);
        return touchesNpot(type) && !(tensor && !tensor.getEncoding());
      }))
    return op->emitOpError("on an NPOT pointer-to-tensor, descriptor, memory "
                           "descriptor or encoded tensor is unsupported");
  // A padded TMA box would write its pad lanes in bounds, so the store moves to
  // pointers.
  if (auto store = dyn_cast<DescriptorStoreOp>(op);
      store && store.getReduceKind() == DescriptorReduceKind::NONE &&
      store.getDesc().getDefiningOp<MakeTensorDescOp>()) {
    RankedTensorType orig = store.getSrc().getType();
    fixups.push_back(
        [this, store, orig] { storeThroughPointers(store, orig); });
    return success();
  }
  if (!hasPaddedLowering(op))
    return op->emitOpError("on an NPOT tensor is unsupported");

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
  // A padded box reads past the tile only into pad lanes, and TMA fills the
  // part out of bounds.
  if (isa<MakeTensorDescOp, DescriptorLoadOp, DescriptorGatherOp>(op))
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

void Padder::apply(ModuleOp mod, const ExactComponents &exact) {
  auto retype = [&](Value value) {
    if (!touchesNpot(value.getType()) || exact.contains(value))
      return;
    if (auto type = dyn_cast<RankedTensorType>(value.getType()))
      value.setType(padded(type));
    else if (auto desc = dyn_cast<TensorDescType>(value.getType()))
      value.setType(TensorDescType::get(padded(desc.getBlockType()).getShape(),
                                        desc.getElementType(),
                                        desc.getSharedLayout()));
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
  using impl::TritonNpotPadBase<NpotPadPass>::TritonNpotPadBase;

  // Checks every op before changing any, so an error leaves the IR unchanged.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ExactComponents exact(mod, numWarps, threadsPerWarp, numCTAs, target,
                          matrixInstructionSize, numStages, useAsyncCopy);
    Padder padder;
    if (mod.walk([&](Operation *op) {
             return WalkResult(exact.contains(op) ? success()
                                                  : padder.accept(op));
           })
            .wasInterrupted())
      return signalPassFailure();
    padder.apply(mod, exact);
  }
};

} // namespace mlir::triton
