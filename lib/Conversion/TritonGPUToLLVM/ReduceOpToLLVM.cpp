#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    // Multi-CTA reduction pass generates tt.reduce on 1-element tensors
    // loaded from DSM buffers. These are within-CTA (each CTA has its own
    // buffer copy), but the encoding may not reflect this if cluster_dims > 1.
    // Only allow these specific 1-element cases through.
    if (!helper.isReduceWithinCTA()) {
      auto srcTy = cast<RankedTensorType>(op.getOperands()[0].getType());
      if (srcTy.getShape()[op.getAxis()] != 1) {
        return op.emitError(
            "cross-CTA reduce on tensor with reduction axis size > 1 is not "
            "supported; only 1-element tensors from multi-CTA DSM exchange "
            "are allowed");
      }
      LDBG("Cross-CTA reduce on 1-element tensor (multi-CTA DSM exchange), "
           "proceeding with within-CTA lowering");
    }
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    // Identity-fill phantom slots before the reduce; maskers below no-op for
    // pow2 (see the "NPOT reduction wrapping prediction" section for why).
    if (failed(maskWrappedRegisters(helper, op, srcValues, rewriter)))
      return failure();

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Identity-fill phantom lane/warp accumulators before the cross-thread
    // butterfly + inter-warp accumulation.
    if (failed(maskWrappedLanesAndWarps(helper, op, accs, rewriter)))
      return failure();

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    if (failed(accumulatePartialReductions(helper, smemBases, rewriter)))
      return failure();

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  bool isInnerTree(triton::ReduceOp op) const {
    auto attr = op.getReductionOrderingAttr();
    return attr && attr.getValue() == "inner_tree";
  }

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<unsigned> getRegGroupKey(ArrayRef<unsigned> offset, unsigned axis,
                                       unsigned groupSpan) const {
    SmallVector<unsigned> key(offset.begin(), offset.end());
    unsigned axisOffset = key[axis];
    key[axis] = (axisOffset / groupSpan) * groupSpan;
    return key;
  }

  SmallVector<Value>
  reduceValueSequence(Location loc, triton::ReduceOp op,
                      SmallVector<SmallVector<Value>> values,
                      ConversionPatternRewriter &rewriter) const {
    if (values.empty())
      return {};

    if (isInnerTree(op)) {
      while (values.size() > 1) {
        SmallVector<SmallVector<Value>> next;
        for (unsigned i = 0; i + 1 < values.size(); i += 2) {
          SmallVector<Value> merged = values[i];
          accumulate(loc, rewriter, op.getCombineOp(), merged, values[i + 1]);
          next.push_back(std::move(merged));
        }
        if (values.size() % 2 == 1)
          next.push_back(std::move(values.back()));
        values = std::move(next);
      }
      return std::move(values[0]);
    }

    SmallVector<Value> acc;
    for (auto &cur : values)
      accumulate(loc, rewriter, op.getCombineOp(), acc, cur);
    return acc;
  }

  bool matchesResultOffset(ArrayRef<unsigned> key,
                           ArrayRef<unsigned> resultOffs, unsigned axis) const {
    if (key.size() != resultOffs.size() + 1)
      return false;
    for (unsigned dim = 0; dim < resultOffs.size(); ++dim) {
      unsigned keyDim = dim < axis ? dim : dim + 1;
      if (key[keyDim] != resultOffs[dim])
        return false;
    }
    return true;
  }

  // Get the neutral/identity element for the reduction operation in the combine
  // region. Works around upstream not supporting MaxNumFOp/MinNumFOp.
  std::optional<TypedAttr> getNeutralElement(Operation *op) const {
    if (isa<arith::MaxNumFOp, arith::MinNumFOp>(op)) {
      OpBuilder builder(op->getContext());
      Type resultType = op->getResult(0).getType();
      const llvm::fltSemantics &semantic =
          llvm::cast<FloatType>(resultType).getFloatSemantics();
      if (isa<arith::MaxNumFOp>(op))
        return builder.getFloatAttr(
            resultType, APFloat::getInf(semantic, /*Negative=*/true));
      if (isa<arith::MinNumFOp>(op))
        return builder.getFloatAttr(
            resultType, APFloat::getInf(semantic, /*Negative=*/false));
    }
    return mlir::arith::getNeutralElement(op);
  }

  // Get the single non-terminator op from the combine region, if it exists.
  std::optional<Operation *> getReductionOp(triton::ReduceOp op) const {
    Region &region = op.getCombineOp();
    if (region.getBlocks().size() != 1)
      return std::nullopt;
    Block &block = region.front();
    auto body = block.without_terminator();
    if (std::distance(body.begin(), body.end()) != 1)
      return std::nullopt;
    return &block.front();
  }

  // Classify a compare op as (isMax, isSigned). nullopt for non-directional
  // predicates (EQ/NE/unordered-only). Float compares report isSigned=true
  // (unused for floats). Integer compares report signedness from the predicate.
  static std::optional<std::pair<bool, bool>>
  classifyCmpPredicate(Operation *cmp) {
    if (auto f = dyn_cast<arith::CmpFOp>(cmp)) {
      using P = arith::CmpFPredicate;
      // Ordered predicates only; unordered (true on NaN) are treated
      // conservatively (unrecognized -> reject) so a -inf/INT_MIN phantom can't
      // beat a real NaN.
      switch (f.getPredicate()) {
      case P::OGT:
      case P::OGE:
        return std::pair<bool, bool>{true, true};
      case P::OLT:
      case P::OLE:
        return std::pair<bool, bool>{false, true};
      default:
        return std::nullopt;
      }
    }
    if (auto i = dyn_cast<arith::CmpIOp>(cmp)) {
      using P = arith::CmpIPredicate;
      switch (i.getPredicate()) {
      case P::sgt:
      case P::sge:
        return std::pair<bool, bool>{true, true};
      case P::slt:
      case P::sle:
        return std::pair<bool, bool>{false, true};
      case P::ugt:
      case P::uge:
        return std::pair<bool, bool>{true, false};
      case P::ult:
      case P::ule:
        return std::pair<bool, bool>{false, false};
      default:
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  // Identities for a multi-operand arg-reduce (argmax/argmin): only the value
  // operand gets the losing extreme so a phantom slot never wins the select;
  // the index operand is left unchanged. nullopt for any unrecognized combine.
  std::optional<SmallVector<std::optional<TypedAttr>>>
  getArgReduceIdentities(triton::ReduceOp op) const {
    unsigned n = op.getNumOperands();
    if (n < 2)
      return std::nullopt;
    Region &region = op.getCombineOp();
    if (!region.hasOneBlock())
      return std::nullopt;
    Block &block = region.front();
    if (block.getNumArguments() != 2 * n)
      return std::nullopt;
    // Operand 0 is the reduced value; its two sides are args[0] and args[n].
    Value lhs = block.getArgument(0);
    Value rhs = block.getArgument(n);
    Type valTy = lhs.getType();
    if (!valTy.isIntOrFloat())
      return std::nullopt;

    // Read the direction (max/min) from the canonical "value0 <pred> value1"
    // compare that drives the selects.
    std::optional<bool> isMax;
    bool isSigned = true;
    for (Operation &inner : block.without_terminator()) {
      Value cmpLhs, cmpRhs;
      if (auto f = dyn_cast<arith::CmpFOp>(&inner)) {
        cmpLhs = f.getLhs();
        cmpRhs = f.getRhs();
      } else if (auto i = dyn_cast<arith::CmpIOp>(&inner)) {
        cmpLhs = i.getLhs();
        cmpRhs = i.getRhs();
      } else {
        continue;
      }
      if (cmpLhs != lhs || cmpRhs != rhs)
        continue;
      if (auto classified = classifyCmpPredicate(&inner)) {
        isMax = classified->first;
        isSigned = classified->second;
        break;
      }
    }
    if (!isMax)
      return std::nullopt;

    OpBuilder builder(op.getContext());
    TypedAttr valIdentity;
    if (auto ft = dyn_cast<FloatType>(valTy)) {
      valIdentity = builder.getFloatAttr(
          ft, APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/*isMax));
    } else {
      auto it = cast<IntegerType>(valTy);
      unsigned w = it.getWidth();
      APInt v = *isMax ? (isSigned ? APInt::getSignedMinValue(w)
                                   : APInt::getMinValue(w))
                       : (isSigned ? APInt::getSignedMaxValue(w)
                                   : APInt::getMaxValue(w));
      valIdentity = builder.getIntegerAttr(it, v);
    }
    SmallVector<std::optional<TypedAttr>> identities(n, std::nullopt);
    identities[0] = valIdentity;
    return identities;
  }

  // Select the reduction identity into acc where `outOfRange` is true. Handles
  // single-op reductions (known scalar neutral) and arg-reduce (value operand
  // only). Returns false if no identity could be determined.
  bool predicateAccWithIdentity(ConversionPatternRewriter &rewriter,
                                Location loc, SmallVector<Value> &acc,
                                triton::ReduceOp op, Value outOfRange) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Single-operand reduction with a known scalar identity (sum, max, ...).
    if (auto reductionOp = getReductionOp(op)) {
      if (auto neutralAttr = getNeutralElement(*reductionOp)) {
        for (unsigned i = 0; i < acc.size(); ++i) {
          Value identity = arith::ConstantOp::create(
              rewriter, loc, acc[i].getType(), cast<TypedAttr>(*neutralAttr));
          acc[i] = b.select(outOfRange, identity, acc[i]);
        }
        return true;
      }
    }
    // Multi-operand arg-reduce (argmax/argmin): mask only the value operand.
    if (auto identities = getArgReduceIdentities(op)) {
      for (unsigned i = 0; i < acc.size() && i < identities->size(); ++i) {
        if (!(*identities)[i])
          continue;
        Value identity = arith::ConstantOp::create(
            rewriter, loc, acc[i].getType(), *(*identities)[i]);
        acc[i] = b.select(outOfRange, identity, acc[i]);
      }
      return true;
    }
    return false;
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier(triton::gpu::AddrSpace::Local);
  }

  // === NPOT reduction wrapping prediction ===
  // NPOT axis rounded up to pow2 creates phantom register/lane/warp slots.
  // Reconstruct the raw position as the sum of per-bit bases, flag it >=
  // dimSize, and overwrite with the reduction identity before the reduce.
  // Pow2/MMA axes skip this -> byte-identical.

  // True when the axis is NPOT (rounds up, creating phantom slots) and uses the
  // register/lane/warp basis decomposition. False for pow2 and MMA/dot-operand
  // encodings, so every caller short-circuits (byte-identical to pow2).
  static bool axisNeedsNpotWrapMasking(int64_t dimSize, Attribute enc) {
    return !llvm::isPowerOf2_64(dimSize) &&
           !isa<NvidiaMmaEncodingAttr, DotOperandEncodingAttr>(enc);
  }

  // Per-axis lane/warp bases from a pow2 LinearLayout, with their summed max
  // contributions (used to short-circuit before emitting runtime IR).
  struct AxisLaneWarpBases {
    SmallVector<unsigned> laneAxisBases;
    SmallVector<unsigned> warpAxisBases;
    unsigned maxLaneContrib = 0;
    unsigned maxWarpContrib = 0;
  };

  // Extract lane/warp axis bases from `pow2LL` for output dim `axis` and sum
  // each set into maxLaneContrib/maxWarpContrib. No IR emission.
  AxisLaneWarpBases extractAxisLaneWarpBases(const LinearLayout &pow2LL,
                                             unsigned axis,
                                             MLIRContext *ctx) const {
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");

    AxisLaneWarpBases out;
    const auto &laneBases = pow2LL.getBases().find(kLane)->second;
    for (const auto &basis : laneBases)
      out.laneAxisBases.push_back(basis[axis]);
    const auto &warpBases = pow2LL.getBases().find(kWarp)->second;
    for (const auto &basis : warpBases)
      out.warpAxisBases.push_back(basis[axis]);

    for (auto v : out.laneAxisBases)
      out.maxLaneContrib += v;
    for (auto v : out.warpAxisBases)
      out.maxWarpContrib += v;
    return out;
  }

  // Build the pre-modulo pow2 LinearLayout (reduction axis rounded up) and its
  // lane/warp axis bases. Returns both so callers needing the register bases
  // can reuse the layout.
  std::pair<LinearLayout, AxisLaneWarpBases>
  buildPow2AxisLayout(ArrayRef<int64_t> srcShape, Attribute srcEncoding,
                      unsigned axis) const {
    SmallVector<int64_t> pow2Shape(srcShape.begin(), srcShape.end());
    pow2Shape[axis] = llvm::NextPowerOf2(srcShape[axis]);
    auto pow2LL = triton::gpu::toLinearLayout(pow2Shape, srcEncoding);
    auto bases =
        extractAxisLaneWarpBases(pow2LL, axis, srcEncoding.getContext());
    return {std::move(pow2LL), std::move(bases)};
  }

  // Reconstruct the runtime axis contribution of `id` (a lane or warp id) from
  // its per-bit position bases: acc += bases[i] for every set bit i of `id`.
  // Zero bases are skipped (they emit no IR). Emits the same op sequence as the
  // per-bit loops it replaces, so codegen is unchanged.
  Value buildContrib(ArrayRef<unsigned> bases, Value id, Location loc,
                     ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value acc = b.i32_val(0);
    for (unsigned i = 0; i < bases.size(); ++i) {
      if (bases[i] == 0)
        continue;
      Value bit = b.and_(b.lshr(id, b.i32_val(i)), b.i32_val(1));
      acc = b.add(acc, b.mul(bit, b.i32_val(bases[i])));
    }
    return acc;
  }

  // Build the runtime lane+warp contribution to the axis (i32) from the
  // axis-specific lane/warp bases extracted by extractAxisLaneWarpBases.
  Value buildAxisLaneWarpContrib(const AxisLaneWarpBases &bases, Location loc,
                                 ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    Value laneContrib =
        buildContrib(bases.laneAxisBases, laneId, loc, rewriter);
    Value warpContrib =
        buildContrib(bases.warpAxisBases, warpId, loc, rewriter);
    return b.add(laneContrib, warpContrib);
  }

  // Per-register wrapping predicates: preds[i] is true when register i's raw
  // axis offset can reach dimSize, null when it never wraps. Empty if the axis
  // never wraps at all.
  SmallVector<Value>
  computeRegisterWrappingPreds(ArrayRef<int64_t> srcShape,
                               Attribute srcEncoding, Location loc,
                               unsigned axis, unsigned numTotalRegs,
                               ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (!axisNeedsNpotWrapMasking(dimSize, srcEncoding))
      return {};

    // Pre-modulo pow2 layout: positions >= dimSize are the wrapped ones.
    auto pow2Layout = buildPow2AxisLayout(srcShape, srcEncoding, axis);
    const LinearLayout &pow2LL = pow2Layout.first;
    const AxisLaneWarpBases &bases = pow2Layout.second;
    auto *ctx = srcEncoding.getContext();
    auto kRegister = StringAttr::get(ctx, "register");

    unsigned numRegs = pow2LL.getInDimSize(kRegister);

    // Extract register bases for the reduction axis from the pow2 layout.
    const auto &regBases = pow2LL.getBases().find(kRegister)->second;
    unsigned regBasisCount = regBases.size();

#ifndef NDEBUG
    // Precondition for the ADD reconstruction below: the ADD of per-bit axis
    // contributions equals the true GF(2)/XOR axis position only when the
    // pow2-rounded layout's per-axis register bases are distinct powers of two.
    // Verify each non-zero axis-component base is a power of two and that no
    // two are equal (pairwise distinct), so ADD == XOR.
    {
      llvm::DenseSet<unsigned> seenAxisBases;
      for (unsigned bit = 0; bit < regBasisCount; ++bit) {
        unsigned axisBase = regBases[bit][axis];
        if (axisBase == 0)
          continue;
        assert(llvm::isPowerOf2_64(axisBase) &&
               "NPOT reduce: axis register base must be a power of two so the "
               "ADD reconstruction equals the XOR axis position");
        assert(seenAxisBases.insert(axisBase).second &&
               "NPOT reduce: axis register bases must be pairwise distinct so "
               "the ADD reconstruction equals the XOR axis position");
      }
    }
#endif

    // Compute raw register contribution to the axis for each register index. In
    // the pow2-rounded layout the per-axis register bases are distinct powers
    // of two (asserted above), so the ADD reconstruction below equals the XOR
    // axis position: bases combined via XOR equal addition because they are
    // linearly independent over GF(2).
    SmallVector<unsigned> rawRegOffsets(numRegs, 0);
    for (unsigned r = 0; r < numRegs; ++r) {
      for (unsigned bit = 0; bit < regBasisCount; ++bit) {
        if (r & (1u << bit))
          rawRegOffsets[r] += regBases[bit][axis];
      }
    }

    // Check if wrapping can occur at all.
    unsigned maxRegOffset =
        *std::max_element(rawRegOffsets.begin(), rawRegOffsets.end());
    unsigned maxRawTotal =
        maxRegOffset + bases.maxLaneContrib + bases.maxWarpContrib;
    if (maxRawTotal < static_cast<unsigned>(dimSize))
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value threadBase = buildAxisLaneWarpContrib(bases, loc, rewriter);

    // Compute per-register predicates. Group registers by their axis-only
    // contribution to avoid redundant runtime comparisons.
    llvm::DenseMap<unsigned, Value> axisPredCache;
    unsigned maxThreadBase = bases.maxLaneContrib + bases.maxWarpContrib;

    auto getAxisPred = [&](unsigned regOff) -> Value {
      auto it = axisPredCache.find(regOff);
      if (it != axisPredCache.end())
        return it->second;
      Value pred;
      if (regOff + maxThreadBase < static_cast<unsigned>(dimSize)) {
        pred = {};
      } else if (regOff >= static_cast<unsigned>(dimSize)) {
        pred = b.true_val();
      } else {
        Value rawTotal = b.add(threadBase, b.i32_val(regOff));
        pred = b.icmp_uge(rawTotal, b.i32_val(dimSize));
      }
      axisPredCache[regOff] = pred;
      return pred;
    };

    // Build per-register predicates. For multi-dim tensors, each total register
    // index maps to an axis sub-index via the register bases.
    SmallVector<Value> preds(numTotalRegs);
    for (unsigned i = 0; i < numTotalRegs; ++i) {
      unsigned regAxisOff = 0;
      for (unsigned bit = 0; bit < regBasisCount; ++bit) {
        if (i & (1u << bit))
          regAxisOff += regBases[bit][axis];
      }
      preds[i] = getAxisPred(regAxisOff);
    }
    return preds;
  }

  // Predicate that is true for lanes/warps holding wrapped/duplicate data.
  // Null when no wrapping occurs (pow2 dim or lane+warp coverage <= dimSize).
  Value computeModularWrappingPred(ArrayRef<int64_t> srcShape,
                                   Attribute srcEncoding, Location loc,
                                   unsigned axis,
                                   ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (!axisNeedsNpotWrapMasking(dimSize, srcEncoding))
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Build pow2 layout to get raw (pre-modulo) offsets.
    auto bases = buildPow2AxisLayout(srcShape, srcEncoding, axis).second;

    // Check if total lane+warp coverage can exceed dimSize.
    unsigned totalCoverage = bases.maxLaneContrib + bases.maxWarpContrib;
    if (totalCoverage < static_cast<unsigned>(dimSize))
      return {};

    Value rawOffset = buildAxisLaneWarpContrib(bases, loc, rewriter);
    return b.icmp_uge(rawOffset, b.i32_val(dimSize));
  }

  // Predicate that is true for phantom warps (raw axis offset out of range),
  // which must not write to smem (they would clobber a real warp's slot). Null
  // when no warp can wrap on its own (pow2 dim or warp coverage <= dimSize).
  Value computeWarpWrappingPred(ArrayRef<int64_t> srcShape,
                                Attribute srcEncoding, Location loc,
                                unsigned axis, Value warpId,
                                ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (!axisNeedsNpotWrapMasking(dimSize, srcEncoding))
      return {};

    auto bases = buildPow2AxisLayout(srcShape, srcEncoding, axis).second;

    // No warp can exceed dimSize on its own -> no phantom warps.
    if (bases.maxWarpContrib < static_cast<unsigned>(dimSize))
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Runtime warp contribution to the axis from the warp axis bases.
    Value warpContrib =
        buildContrib(bases.warpAxisBases, warpId, loc, rewriter);
    return b.icmp_uge(warpContrib, b.i32_val(dimSize));
  }

  // Identity-fill wrapped register slots before within-thread reduction folds
  // them in (empty preds = no-op). Returns failure() when wrapping is present
  // but the combine identity is unknown -- reject rather than miscompute.
  LogicalResult
  maskWrappedRegisters(ReduceOpHelper &helper, triton::ReduceOp op,
                       SmallVector<SmallVector<Value>> &srcValues,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> regPreds = computeRegisterWrappingPreds(
        helper.getSrcShape(), helper.getSrcLayout(), loc, axis,
        srcValues.size(), rewriter);
    if (regPreds.empty())
      return success();

    // regPreds is sized to srcValues.size() (== 2^#regBases) by construction;
    // the NPOT fold reduces basis values, not the basis count.
    for (unsigned i = 0; i < srcValues.size(); ++i) {
      // A null predicate means this register never wraps (purely in-range).
      if (!regPreds[i])
        continue;
      if (!predicateAccWithIdentity(rewriter, loc, srcValues[i], op,
                                    regPreds[i]))
        return op.emitError(
            "NPOT reduction over an axis with wrapped register slots requires "
            "a known reduction identity, but the combine op's identity could "
            "not be determined");
    }
    return success();
  }

  // Identity-fill phantom lane/warp accumulators before the within-warp shuffle
  // butterfly (one mask covers both intra-warp lanes and inter-warp warps).
  // Null pred = no-op; returns failure() when wrapping is present but the
  // combine identity is unknown.
  LogicalResult maskWrappedLanesAndWarps(
      ReduceOpHelper &helper, triton::ReduceOp op,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    Value lanePred = computeModularWrappingPred(
        helper.getSrcShape(), helper.getSrcLayout(), loc, axis, rewriter);
    if (!lanePred)
      return success();

    for (auto &it : accs) {
      if (!predicateAccWithIdentity(rewriter, loc, it.second, op, lanePred))
        return op.emitError(
            "NPOT reduction over an axis with wrapped lane/warp slots requires "
            "a known reduction identity, but the combine op's identity could "
            "not be determined");
    }
    return success();
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (int i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }

    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);
    unsigned axis = op.getAxis();
    unsigned contigPerThread = helper.getContigPerThreadOnReductionAxis();

    SmallVector<int> iterOrder;
    for (const auto &[_, i] : uniqueOffsets)
      iterOrder.push_back(i);

    std::map<SmallVector<unsigned>, SmallVector<int>> regGroups;
    for (int idx : iterOrder) {
      regGroups[getRegGroupKey(offsets[idx], axis, contigPerThread)].push_back(
          idx);
    }

    if (!isInnerTree(op)) {
      for (int idx : iterOrder) {
        SmallVector<unsigned> key = offsets[idx];
        key[axis] = 0;
        bool isFirst = accs.find(key) == accs.end();
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), accs[key],
                   srcValues[idx]);
        if (isFirst)
          indices[key] = srcIndices[idx];
      }
      return;
    }

    for (auto &[key, group] : regGroups) {
      llvm::sort(group, [&](int lhs, int rhs) {
        return offsets[lhs][axis] < offsets[rhs][axis];
      });

      SmallVector<SmallVector<Value>> groupValues;
      groupValues.reserve(group.size());
      for (int idx : group)
        groupValues.push_back(srcValues[idx]);

      accs[key] = reduceValueSequence(op.getLoc(), op, std::move(groupValues),
                                      rewriter);
      indices[key] = srcIndices[group.front()];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;

    if (isInnerTree(op)) {
      // INNER_TREE: count-up shuffle order (1, 2, 4, ...) to build the
      // reduction tree from adjacent lanes first. This ensures bitwise-
      // identical results regardless of num_warps, because the tree
      // structure is determined by lane proximity, not by the total
      // number of active lanes.
      for (unsigned N = 1; N <= numLaneToReduce / 2; N <<= 1) {
        SmallVector<Value> shfl(acc.size());
        for (unsigned i = 0; i < acc.size(); ++i) {
          shfl[i] =
              targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
      }
    } else {
      for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
        SmallVector<Value> shfl(acc.size());
        for (unsigned i = 0; i < acc.size(); ++i) {
          shfl[i] =
              targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
      }
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    // The shuffle butterfly in warpReduce halves the lane span each step and
    // therefore requires a power-of-two lane count. For an NPOT reduction axis,
    // getIntraWarpSizeWithUniqueData() can be NPOT (e.g. 3), which would make
    // the butterfly skip lanes (3/2 -> a single xor-1 step, leaving lane 2
    // unreduced). Round up to the next power of two so every real lane is
    // visited; the phantom lanes in [realSize, pow2) were already identity-
    // filled by maskWrappedLanesAndWarps, so they contribute nothing. For pow2
    // axes this is a no-op (sizeIntraWarps is already pow2).
    unsigned sizeIntraWarps =
        llvm::PowerOf2Ceil(helper.getIntraWarpSizeWithUniqueData());
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    if (auto firstResultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto resultLayout = cast<SliceEncodingAttr>(firstResultTy.getEncoding());
      unsigned resultElems = getTotalElemsPerThread(firstResultTy);
      SmallVector<SmallVector<unsigned>> resultOffsets =
          emitOffsetForLayout(resultLayout, firstResultTy);
      SmallVector<SmallVector<Value>> resultVals(
          op.getNumOperands(), SmallVector<Value>(resultElems));

      for (unsigned j = 0; j < resultElems; ++j) {
        SmallVector<SmallVector<Value>> groupVals;
        for (const auto &[key, vals] : accs) {
          if (matchesResultOffset(key, resultOffsets[j], axis))
            groupVals.push_back(vals);
        }
        SmallVector<Value> reduced =
            reduceValueSequence(loc, op, std::move(groupVals), rewriter);
        for (unsigned i = 0; i < op.getNumOperands(); ++i)
          resultVals[i][j] = reduced[i];
      }

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto resultTy = cast<RankedTensorType>(op.getResult()[i].getType());
        results[i] = packLLElements(loc, getTypeConverter(), resultVals[i],
                                    rewriter, resultTy);
      }
    } else {
      SmallVector<SmallVector<Value>> groupVals;
      groupVals.reserve(accs.size());
      for (const auto &[_, vals] : accs)
        groupVals.push_back(vals);
      SmallVector<Value> reduced =
          reduceValueSequence(loc, op, std::move(groupVals), rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        results[i] = reduced[i];
    }
    rewriter.replaceOp(op, results);
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout =
        mlir::cast<DistributedEncodingTrait>(helper.getSrcLayout());
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto srcShape = helper.getSrcShape();
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, srcLayout, srcShape, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, srcLayout, srcShape, kWarp, warpId);

    Value laneIdAxis = multiDimLaneId[axis];
    Value laneZero = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value write =
        b.and_(b.and_(isRepresentativeLane, isRepresentativeWarp), laneZero);

    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    // NPOT warp collapse: when the reduction axis is NPOT, more warps may map
    // onto the axis than there are unique warp slots (e.g. 4 warps over a dim
    // of 96 -> getInterWarpSizeWithUniqueData() == 3). The extra warps are
    // modular duplicates: their axis index (multiDimWarpId[axis] below) wraps
    // (3 % 3 == 0), so they would write to the *same* smem slot as a real warp.
    // Those duplicate warps hold identity (filled by maskWrappedLanesAndWarps),
    // so the duplicate store races the real store and can clobber it with
    // identity. Suppress writes from the phantom warps, identified from the raw
    // (pre-collapse) warp axis offset >= dimSize. For pow2 axes there are no
    // phantom warps so this guard is never added and codegen is unchanged.
    if (Value warpPhantom = computeWarpWrappingPred(
            srcShape, helper.getSrcLayout(), loc, axis, warpId, rewriter))
      write = b.and_(write, b.icmp_eq(warpPhantom, b.i1_val(false)));

    Value warpIdAxis = multiDimWarpId[axis];

    unsigned numRegGroups = helper.getNumRegGroupsOnAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();

    std::map<unsigned, unsigned> axisOffsetToGroupIdx;
    if (numRegGroups > 1) {
      SmallVector<unsigned> axisOffsets;
      axisOffsets.reserve(accs.size());
      for (const auto &[key, _] : accs)
        axisOffsets.push_back(key[axis]);
      llvm::sort(axisOffsets);

      for (unsigned axisVal : axisOffsets) {
        if (axisOffsetToGroupIdx.find(axisVal) == axisOffsetToGroupIdx.end())
          axisOffsetToGroupIdx[axisVal] = axisOffsetToGroupIdx.size();
      }
      assert(axisOffsetToGroupIdx.size() == numRegGroups &&
             "unexpected number of register groups");
    }

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      if (numRegGroups > 1) {
        unsigned regGroupIdx = axisOffsetToGroupIdx[key[axis]];
        Value groupOffset =
            b.add(b.i32_val(regGroupIdx * sizeInterWarps), warpIdAxis);
        writeIdx[axis] = groupOffset;
      } else {
        writeIdx[axis] = warpIdAxis;
      }
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], write);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  LogicalResult
  accumulatePartialReductions(ReduceOpHelper &helper,
                              SmallVector<Value> &smemBases,
                              ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned realInterWarps = helper.getInterWarpSizeWithUniqueData();
    // The inter-warp shuffle butterfly (warpReduce below) halves the lane span
    // each step and therefore requires a power-of-two count. For an NPOT
    // reduction axis the per-warp axis count collapses to an NPOT value (e.g.
    // 4 warps over a dim of 96 -> 3 unique warp slots), which would make the
    // butterfly skip warp partials. Round up so every real partial is visited;
    // the phantom slots in [realInterWarps, pow2) are identity-filled below so
    // they contribute nothing. For pow2 axes this is a no-op.
    unsigned sizeInterWarps = llvm::PowerOf2Ceil(realInterWarps);
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = numLanes * numWarps;

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(numLanes);
    Value laneId = b.urem(threadId, warpSize);
    Value zero = b.i32_val(0);

    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = b.icmp_slt(threadId, b.i32_val(elems));

    // An inter-warp smem slot is phantom when the warp that owns it carries a
    // raw axis offset >= dimSize (it wrapped): its store was suppressed (see
    // computeWarpWrappingPred), so the load returns undef and the slot must be
    // identity-filled before the butterfly folds it in. Counting unique warp
    // bases (realInterWarps) is not enough -- e.g. 8 warps over a dim of 192
    // reports 8 unique slots, yet warps 6,7 (raw 192,224) are phantom -- so
    // detect phantoms from the warp axis bases directly. Only the single-
    // register-group case maps a slot index contiguously onto the warp axis.
    // For pow2 axes no warp wraps, so no phantom is found and the fill is
    // skipped (byte-identical).
    unsigned axis = op.getAxis();
    ArrayRef<int64_t> srcShape = helper.getSrcShape();
    Attribute srcEncoding = helper.getSrcLayout();
    int64_t dimSize = srcShape[axis];
    SmallVector<unsigned> warpAxisBases;
    bool interWarpNeedsIdentity = false;
    if (axisNeedsNpotWrapMasking(dimSize, srcEncoding) &&
        helper.getNumRegGroupsOnAxis() == 1) {
      warpAxisBases =
          buildPow2AxisLayout(srcShape, srcEncoding, axis).second.warpAxisBases;
      for (unsigned slot = 0; slot < sizeInterWarps && !interWarpNeedsIdentity;
           ++slot) {
        unsigned raw = 0;
        for (unsigned i = 0; i < warpAxisBases.size(); ++i)
          if (slot & (1u << i))
            raw += warpAxisBases[i];
        if (raw >= static_cast<unsigned>(dimSize))
          interWarpNeedsIdentity = true;
      }
    }

    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                       threadIsNeeded);
      }
      if (interWarpNeedsIdentity) {
        // The slot's owning warp is its position within the pow2 inter-warp
        // group; reconstruct that warp's raw axis offset from the warp bases
        // (mirroring computeWarpWrappingPred) and flag it phantom when the
        // offset reaches dimSize.
        Value slotInGroup = b.urem(readOffset, b.i32_val(sizeInterWarps));
        Value warpContrib =
            buildContrib(warpAxisBases, slotInGroup, loc, rewriter);
        Value isPhantom =
            b.icmp_uge(warpContrib, b.i32_val(static_cast<int64_t>(dimSize)));
        if (!predicateAccWithIdentity(rewriter, loc, acc, op, isPhantom))
          return op.emitError("NPOT inter-warp reduction with phantom warp "
                              "slots requires a known reduction identity");
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                 threadIsNeeded);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = b.urem(laneId, b.i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          b.icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = b.and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = b.add(readOffset, b.i32_val(numThreads));
      }
    }
    return success();
  }

  SmallVector<Value> loadAndReduceRegGroups(
      Location loc, triton::ReduceOp op, ArrayRef<Value> smemBases,
      SmallVector<Value> readIdx, ArrayRef<unsigned> smemShape,
      ArrayRef<unsigned> smemOrder, unsigned axis, unsigned numRegGroups,
      unsigned sizeInterWarps, ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<SmallVector<Value>> groupVals;
    groupVals.reserve(numRegGroups);
    for (unsigned g = 0; g < numRegGroups; ++g) {
      SmallVector<Value> vals;
      vals.reserve(op.getNumOperands());
      SmallVector<Value> groupReadIdx = readIdx;
      groupReadIdx[axis] = b.i32_val(g * sizeInterWarps);
      Value offset =
          linearize(rewriter, loc, groupReadIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value ptr = b.gep(smemBases[i].getType(), elemTy, smemBases[i], offset);
        vals.push_back(b.load(elemTy, ptr));
      }
      groupVals.push_back(std::move(vals));
    }
    return reduceValueSequence(loc, op, std::move(groupVals), rewriter);
  }

  SmallVector<Value>
  loadAndReduceScalarRegGroups(Location loc, triton::ReduceOp op,
                               ArrayRef<Value> smemBases, unsigned numRegGroups,
                               unsigned sizeInterWarps,
                               ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<SmallVector<Value>> groupVals;
    groupVals.reserve(numRegGroups);
    for (unsigned g = 0; g < numRegGroups; ++g) {
      SmallVector<Value> vals;
      vals.reserve(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value ptr = b.gep(smemBases[i].getType(), elemTy, smemBases[i],
                          b.i32_val(g * sizeInterWarps));
        vals.push_back(b.load(elemTy, ptr));
      }
      groupVals.push_back(std::move(vals));
    }
    return reduceValueSequence(loc, op, std::move(groupVals), rewriter);
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();

    unsigned numRegGroups = helper.getNumRegGroupsOnAxis();
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();

    SmallVector<Value> results(op.getNumOperands());
    if (numRegGroups > 1) {
      if (auto firstResultTy =
              dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
        auto resultLayout =
            cast<SliceEncodingAttr>(firstResultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(firstResultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, firstResultTy, true);
        auto resultShape = firstResultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<SmallVector<Value>> resultVals(
            op.getNumOperands(), SmallVector<Value>(resultElems));
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + axis, b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < axis ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }

          SmallVector<Value> vals = loadAndReduceRegGroups(
              loc, op, smemBases, readIdx, smemShape, smemOrder, axis,
              numRegGroups, sizeInterWarps, rewriter);
          for (unsigned i = 0; i < op.getNumOperands(); ++i)
            resultVals[i][j] = vals[i];
        }

        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto resultTy = cast<RankedTensorType>(op.getResult()[i].getType());
          results[i] = packLLElements(loc, getTypeConverter(), resultVals[i],
                                      rewriter, resultTy);
        }
      } else {
        SmallVector<Value> vals = loadAndReduceScalarRegGroups(
            loc, op, smemBases, numRegGroups, sizeInterWarps, rewriter);
        for (unsigned i = 0; i < op.getNumOperands(); ++i)
          results[i] = vals[i];
      }
      rewriter.replaceOp(op, results);
      return;
    }

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }

          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = b.load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = b.load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
