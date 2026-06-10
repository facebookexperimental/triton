#include "ReduceScanCommon.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
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

    // NPOT correctness: toLinearLayout rounds an NPOT reduction axis up to the
    // next power of two, creating phantom register/lane/warp slots whose data
    // is wrapped (duplicated / cross-row) modulo the real axis size. If those
    // slots are folded into the reduction they corrupt the result.
    // Identity-fill them before they enter the reduction. The two maskers below
    // short-circuit for pow2 axes (returning empty preds), so pow2 codegen is
    // byte-identical.
    if (failed(maskWrappedRegisters(helper, op, srcValues, rewriter)))
      return failure();

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Identity-fill accumulators held by phantom (wrapped) lanes/warps before
    // the cross-thread butterfly + inter-warp shared-memory accumulation.
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
    accumulatePartialReductions(helper, smemBases, rewriter);

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

  // For NPOT reductions, replace acc with identity elements for out-of-range
  // lanes so the pow2 butterfly produces correct results.
  // Returns true if identity elements were successfully applied, false if the
  // identity could not be determined.
  bool predicateAccWithIdentity(ConversionPatternRewriter &rewriter,
                                Location loc, SmallVector<Value> &acc,
                                triton::ReduceOp op, Value outOfRange) const {
    auto reductionOp = getReductionOp(op);
    if (!reductionOp)
      return false;
    auto neutralAttr = getNeutralElement(*reductionOp);
    if (!neutralAttr)
      return false;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned i = 0; i < acc.size(); ++i) {
      Value identity = arith::ConstantOp::create(
          rewriter, loc, acc[i].getType(), cast<TypedAttr>(*neutralAttr));
      acc[i] = b.select(outOfRange, identity, acc[i]);
    }
    return true;
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

  // Per-axis lane/warp data extracted from a pow2 LinearLayout. Used by both
  // NPOT wrapping-prediction helpers below as their first step: they extract
  // the lane/warp axis bases and compute the max contributions identically,
  // then (if wrapping can occur) build the runtime lane+warp contribution.
  // Each helper layers its own wrapping-predicate construction on top.
  struct AxisLaneWarpBases {
    SmallVector<unsigned> laneAxisBases;
    SmallVector<unsigned> warpAxisBases;
    unsigned maxLaneContrib = 0;
    unsigned maxWarpContrib = 0;
  };

  // Extract lane/warp axis bases from `pow2LL` for output dim `axis` and
  // sum each set to compute the max possible contribution. Cheap (no IR
  // emission) so callers can use the result to short-circuit before building
  // runtime contributions.
  AxisLaneWarpBases extractAxisLaneWarpBases(const LinearLayout &pow2LL,
                                             unsigned axis,
                                             MLIRContext *ctx) const {
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    unsigned axisDimIdx = axis;

    AxisLaneWarpBases out;
    const auto &laneBases = pow2LL.getBases().find(kLane)->second;
    for (const auto &basis : laneBases)
      out.laneAxisBases.push_back(basis[axisDimIdx]);
    const auto &warpBases = pow2LL.getBases().find(kWarp)->second;
    for (const auto &basis : warpBases)
      out.warpAxisBases.push_back(basis[axisDimIdx]);

    for (auto v : out.laneAxisBases)
      out.maxLaneContrib += v;
    for (auto v : out.warpAxisBases)
      out.maxWarpContrib += v;
    return out;
  }

  // Build the runtime lane+warp contribution to the axis (i32) from the
  // axis-specific lane/warp bases extracted by extractAxisLaneWarpBases.
  Value buildAxisLaneWarpContrib(const AxisLaneWarpBases &bases, Location loc,
                                 ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    // Runtime lane contribution (sum of lane bases where the corresponding
    // bit is set in laneId).
    Value laneContrib = b.i32_val(0);
    for (unsigned i = 0; i < bases.laneAxisBases.size(); ++i) {
      if (bases.laneAxisBases[i] == 0)
        continue;
      Value bit = b.and_(b.lshr(laneId, b.i32_val(i)), b.i32_val(1));
      laneContrib =
          b.add(laneContrib, b.mul(bit, b.i32_val(bases.laneAxisBases[i])));
    }

    // Runtime warp contribution.
    Value warpContrib = b.i32_val(0);
    for (unsigned i = 0; i < bases.warpAxisBases.size(); ++i) {
      if (bases.warpAxisBases[i] == 0)
        continue;
      Value bit = b.and_(b.lshr(warpId, b.i32_val(i)), b.i32_val(1));
      warpContrib =
          b.add(warpContrib, b.mul(bit, b.i32_val(bases.warpAxisBases[i])));
    }
    return b.add(laneContrib, warpContrib);
  }

  // Compute per-register wrapping predicates for NPOT reductions.
  // For NPOT dims, register extension (ensureLayoutNotSmallerThan) can cause
  // elements to wrap mod dimSize, duplicating across threads. Returns a vector
  // of predicates (one per register) that are true when the element wraps.
  // Returns empty if no wrapping occurs.
  //
  // Uses LinearLayout with pow2 shape (pre-modulo) to compute raw offsets
  // for any distributed encoding (blocked, MMA, dot operand, etc.).
  SmallVector<Value>
  computeRegisterWrappingPreds(ArrayRef<int64_t> srcShape,
                               Attribute srcEncoding, Location loc,
                               unsigned axis, unsigned numTotalRegs,
                               ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (llvm::isPowerOf2_64(dimSize))
      return {};

    // Build the pow2 shape by rounding the NPOT reduction axis up to next
    // pow2. The resulting layout is pre-modulo: each (register, lane, warp)
    // maps to a unique position in [0, pow2Shape). Positions >= dimSize are
    // the "wrapped" ones that need to be masked with identity.
    SmallVector<int64_t> pow2Shape(srcShape.begin(), srcShape.end());
    pow2Shape[axis] = llvm::NextPowerOf2(dimSize);

    auto pow2LL = triton::gpu::toLinearLayout(pow2Shape, srcEncoding);
    auto *ctx = srcEncoding.getContext();
    auto kRegister = StringAttr::get(ctx, "register");

    unsigned numRegs = pow2LL.getInDimSize(kRegister);
    unsigned axisDimIdx = axis;

    // Extract register bases for the reduction axis from the pow2 layout.
    const auto &regBases = pow2LL.getBases().find(kRegister)->second;
    unsigned regBasisCount = regBases.size();

    // Compute raw register contribution to the axis for each register index.
    // In the pow2 layout (all dims pow2), bases are combined via XOR which
    // equals addition because the bases are linearly independent over GF(2).
    SmallVector<unsigned> rawRegOffsets(numRegs, 0);
    for (unsigned r = 0; r < numRegs; ++r) {
      for (unsigned b = 0; b < regBasisCount; ++b) {
        if (r & (1u << b))
          rawRegOffsets[r] += regBases[b][axisDimIdx];
      }
    }

    auto bases = extractAxisLaneWarpBases(pow2LL, axis, ctx);

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

    // Build per-register predicates. For multi-dim tensors, each total
    // register index maps to an axis sub-index via the register bases.
    SmallVector<Value> preds(numTotalRegs);
    for (unsigned i = 0; i < numTotalRegs; ++i) {
      unsigned regAxisOff = 0;
      for (unsigned bit = 0; bit < regBasisCount; ++bit) {
        if (i & (1u << bit))
          regAxisOff += regBases[bit][axisDimIdx];
      }
      preds[i] = getAxisPred(regAxisOff);
    }
    return preds;
  }

  // For NPOT reduction dims, compute a predicate that is true for lanes
  // holding wrapped/duplicate data due to modular arithmetic. Returns a null
  // Value when no wrapping occurs (pow2 dims or coverage <= dimSize).
  //
  // Uses LinearLayout with pow2 shape (pre-modulo) to compute raw lane+warp
  // offsets for any distributed encoding.
  Value computeModularWrappingPred(ArrayRef<int64_t> srcShape,
                                   Attribute srcEncoding, Location loc,
                                   unsigned axis, unsigned sizeIntraWarps,
                                   unsigned interleave,
                                   ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (llvm::isPowerOf2_64(dimSize))
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Build pow2 layout to get raw (pre-modulo) offsets.
    SmallVector<int64_t> pow2Shape(srcShape.begin(), srcShape.end());
    pow2Shape[axis] = llvm::NextPowerOf2(dimSize);

    auto pow2LL = triton::gpu::toLinearLayout(pow2Shape, srcEncoding);
    auto *ctx = srcEncoding.getContext();

    auto bases = extractAxisLaneWarpBases(pow2LL, axis, ctx);

    // Check if total lane+warp coverage can exceed dimSize.
    unsigned totalCoverage = bases.maxLaneContrib + bases.maxWarpContrib;
    if (totalCoverage < static_cast<unsigned>(dimSize))
      return {};

    Value rawOffset = buildAxisLaneWarpContrib(bases, loc, rewriter);
    return b.icmp_uge(rawOffset, b.i32_val(dimSize));
  }

  // For an NPOT reduction dim, return a predicate that is true for warps whose
  // raw (pre-modular-collapse) axis offset is out of range, i.e. modular
  // duplicates of an earlier warp. These phantom warps must not write to shared
  // memory (they would collide with / clobber a real warp's slot). Returns a
  // null Value when no warp can wrap (pow2 dim or warp coverage <= dimSize), so
  // pow2 codegen is unchanged.
  Value computeWarpWrappingPred(ArrayRef<int64_t> srcShape,
                                Attribute srcEncoding, Location loc,
                                unsigned axis, Value warpId,
                                ConversionPatternRewriter &rewriter) const {
    int64_t dimSize = srcShape[axis];
    if (llvm::isPowerOf2_64(dimSize))
      return {};

    SmallVector<int64_t> pow2Shape(srcShape.begin(), srcShape.end());
    pow2Shape[axis] = llvm::NextPowerOf2(dimSize);
    auto pow2LL = triton::gpu::toLinearLayout(pow2Shape, srcEncoding);
    auto *ctx = srcEncoding.getContext();
    auto bases = extractAxisLaneWarpBases(pow2LL, axis, ctx);

    // No warp can exceed dimSize on its own -> no phantom warps.
    if (bases.maxWarpContrib < static_cast<unsigned>(dimSize))
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Runtime warp contribution to the axis from the warp axis bases.
    Value warpContrib = b.i32_val(0);
    for (unsigned i = 0; i < bases.warpAxisBases.size(); ++i) {
      if (bases.warpAxisBases[i] == 0)
        continue;
      Value bit = b.and_(b.lshr(warpId, b.i32_val(i)), b.i32_val(1));
      warpContrib =
          b.add(warpContrib, b.mul(bit, b.i32_val(bases.warpAxisBases[i])));
    }
    return b.icmp_uge(warpContrib, b.i32_val(dimSize));
  }

  // Identity-fill the register slots that hold wrapped data for an NPOT
  // reduction axis, BEFORE within-thread reduction folds them in.
  //
  // No-op (and byte-identical codegen) for pow2 axes or when no register slot
  // can wrap: computeRegisterWrappingPreds returns an empty vector and we
  // return immediately without emitting any IR.
  //
  // Returns failure() only when wrapping IS present but the reduction identity
  // cannot be determined (unknown combine op) -- miscomputing silently would be
  // worse, so we reject with a clear error instead.
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

    assert(regPreds.size() == srcValues.size() &&
           "register wrapping predicate count must match registers per thread");
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

  // Identity-fill the accumulators held by phantom (wrapped) lanes/warps for an
  // NPOT reduction axis, BEFORE the within-warp shuffle butterfly. Because the
  // masked value also persists into the shared-memory inter-warp accumulation,
  // a single mask here covers both wrapped lanes (intra-warp butterfly) and
  // wrapped warps (inter-warp accumulation).
  //
  // No-op (and byte-identical codegen) for pow2 axes or when lane+warp coverage
  // cannot exceed the axis size: computeModularWrappingPred returns a null
  // Value and we return immediately without emitting any IR.
  //
  // Returns failure() only when wrapping IS present but the reduction identity
  // cannot be determined.
  LogicalResult maskWrappedLanesAndWarps(
      ReduceOpHelper &helper, triton::ReduceOp op,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned interleave = helper.getThreadOffsetOnReductionAxis();
    Value lanePred = computeModularWrappingPred(
        helper.getSrcShape(), helper.getSrcLayout(), loc, axis, sizeIntraWarps,
        interleave, rewriter);
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
  void accumulatePartialReductions(ReduceOpHelper &helper,
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

    // For an NPOT axis we rounded sizeInterWarps up to a power of two. Each
    // pow2 group then contains phantom slots in [realInterWarps,
    // sizeInterWarps) that no warp wrote, so the shared-memory load returns
    // undef there. Those slots must be identity-filled before the butterfly
    // folds them in. We only need this in the single-register-group case
    // (numRegGroups == 1), which is where the NPOT inter-warp collapse occurs
    // and where the pow2 group maps contiguously onto the smem axis (slot ==
    // warp axis index). For pow2 axes sizeInterWarps == realInterWarps and this
    // predicate is never true, so the identity fill is skipped and codegen is
    // unchanged.
    bool interWarpNeedsIdentity =
        sizeInterWarps != realInterWarps && helper.getNumRegGroupsOnAxis() == 1;

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
        // Slot is phantom when its position within the pow2 inter-warp group is
        // >= the real warp count. With numRegGroups == 1 the smem axis is the
        // inter-warp dimension, so the slot index is readOffset.
        Value slotInGroup = b.urem(readOffset, b.i32_val(sizeInterWarps));
        Value isPhantom = b.icmp_uge(slotInGroup, b.i32_val(realInterWarps));
        if (!predicateAccWithIdentity(rewriter, loc, acc, op, isPhantom))
          return; // identity unknown: leave as-is (load already predicated).
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
