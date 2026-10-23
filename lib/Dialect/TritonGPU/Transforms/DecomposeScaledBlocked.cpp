#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::gpu {

SmallVector<int, 2> DecomposeScaledBlocked::getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

// NV-only classification, mirroring getScaleKind's native-path rule in
// MMAv5.cpp: int8 scales with a block factor of 16 are UE5M3. Callers must
// additionally gate on allowUE5M3Scales: AMD treats int8 scales as E8M0 at
// any block size.
static bool isUE5M3ScaleTensor(DotScaledOp dotOp, Value scale) {
  return scale &&
         cast<RankedTensorType>(scale.getType())
             .getElementType()
             .isInteger(8) &&
         dotOp.deduceScaleFactor() == 16;
}

LogicalResult
DecomposeScaledBlocked::matchAndRewrite(DotScaledOp scaledDotOp,
                                        PatternRewriter &rewriter) const {
  // This fallback constructs dot operands with the result encoding as their
  // parent. Only a blocked parent is supported here; native MMA layouts and
  // deferred/custom layouts must retain their own lowering contract.
  if (!isa_and_nonnull<BlockedEncodingAttr>(
          scaledDotOp.getResult().getType().getEncoding()))
    return rewriter.notifyMatchFailure(scaledDotOp,
                                       "expected blocked result encoding");

  // fp6 upcast requires K-packing (4 values per 3 bytes along K).
  auto isFp6 = [](ScaleDotElemType t) {
    return t == ScaleDotElemType::E2M3 || t == ScaleDotElemType::E3M2;
  };
  if ((isFp6(scaledDotOp.getAElemType()) && !scaledDotOp.getLhsKPack()) ||
      (isFp6(scaledDotOp.getBElemType()) && !scaledDotOp.getRhsKPack())) {
    return rewriter.notifyMatchFailure(scaledDotOp,
                                       "NYI: fp6 without K-packing");
  }

  // Types
  auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                    scaledDotOp.getBElemType(), rewriter);

  // UE5M3 scales reach 114688 and overflow f16 (max 65504) at decode time,
  // even when scale * value would fit. Name-reject FP16-compute UE5M3 dots
  // instead of producing silent inf. This is NV-only (allowUE5M3Scales):
  // AMD classifies int8 scales as E8M0 and is unaffected.
  if (allowUE5M3Scales && computeType == rewriter.getF16Type() &&
      (isUE5M3ScaleTensor(scaledDotOp, scaledDotOp.getAScale()) ||
       isUE5M3ScaleTensor(scaledDotOp, scaledDotOp.getBScale())))
    return scaledDotOp.emitError(
        "cannot decompose dot_scaled with UE5M3 (int8 block-16) scales and "
        "f16 compute type: scales up to 114688 overflow f16");

  auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
  scaledA = cvtDotOperand(rewriter, scaledDotOp, 0, scaledA);
  auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
  scaledB = cvtDotOperand(rewriter, scaledDotOp, 1, scaledB);
  auto newDot = DotOp::create(rewriter, scaledDotOp.getLoc(), scaledA, scaledB,
                              scaledDotOp.getC());

  rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp,
                                               scaledDotOp.getType(), newDot);
  return success();
}

FloatType
DecomposeScaledBlocked::getComputeType(ScaleDotElemType aType,
                                       ScaleDotElemType bType,
                                       PatternRewriter &rewriter) const {
  if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
    return rewriter.getF16Type();
  return rewriter.getBF16Type();
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::decodeUE5M3To16(PatternRewriter &rewriter,
                                        TypedValue<RankedTensorType> scale,
                                        FloatType computeType) const {
  // Decode a UE5M3 (unsigned E5M3, NVFP4 block-16) scale byte to a 16-bit
  // float: exponent bits [7:3] with bias 15, mantissa bits [2:0]. Matches
  // the ue5m3_to_float32 reference in test_nvfp4_ue5m3_matmul: exp == 0 is
  // subnormal (mant * 2^-17), 0xFF (exp == 31, mant == 7) is NaN, otherwise
  // normal ((1 + mant/8) * 2^(exp-15)), including 0xF8-0xFE as finite
  // (1+m/8)*2^16. 0xFF additionally matches maskNan's byte comparison, so
  // both agree on the NaN byte.
  // Compute type is always bf16 here: UE5M3 scales reach 114688 and overflow
  // f16, so f16-compute UE5M3 dots are name-rejected in matchAndRewrite.
  assert(computeType == rewriter.getBF16Type() &&
         "UE5M3 decode requires bf16 compute type");
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  auto f32Type = rewriter.getF32Type();
  auto i32Type = rewriter.getI32Type();

  auto splatI32 = [&](int32_t v) {
    auto cst = arith::ConstantIntOp::create(rewriter, loc, v, 32);
    return SplatOp::create(rewriter, loc, scaleTy.clone(i32Type), cst);
  };

  // exp = (byte >> 3) & 0x1F, mant = byte & 0x7.
  auto x = arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(i32Type), scale);
  auto shr3 = arith::ShRUIOp::create(rewriter, loc, x, splatI32(3));
  auto exp = arith::AndIOp::create(rewriter, loc, shr3, splatI32(0x1F));
  auto mant = arith::AndIOp::create(rewriter, loc, x, splatI32(0x7));

  // Subnormal (exp == 0): mant * 2^-17, exact in f32 (power-of-two scaling
  // far from overflow/underflow).
  auto mantF32 =
      arith::UIToFPOp::create(rewriter, loc, scaleTy.clone(f32Type), mant);
  auto twoNeg17 =
      SplatOp::create(rewriter, loc, scaleTy.clone(f32Type),
                      arith::ConstantOp::create(
                          rewriter, loc, rewriter.getF32FloatAttr(0x1p-17)));
  auto sub = arith::MulFOp::create(rewriter, loc, mantF32, twoNeg17);

  // Normal: f32 bits ((exp + 112) << 23) | (mant << 20). The f32 exponent
  // field is exp - 15 + 127 = exp + 112. Exact by construction.
  auto f32Exp = arith::AddIOp::create(rewriter, loc, exp, splatI32(112));
  auto expBits = arith::ShLIOp::create(rewriter, loc, f32Exp, splatI32(23));
  auto mantBits = arith::ShLIOp::create(rewriter, loc, mant, splatI32(20));
  auto normalBits = arith::OrIOp::create(rewriter, loc, expBits, mantBits);
  auto normal =
      BitcastOp::create(rewriter, loc, scaleTy.clone(f32Type), normalBits);

  // 0xFF only: quiet NaN. 0xF8-0xFE take the normal path above, which
  // yields (1+m/8)*2^16 for exp == 31.
  auto nan = BitcastOp::create(rewriter, loc, scaleTy.clone(f32Type),
                               splatI32(static_cast<int32_t>(0x7FC00000)));

  auto isSub = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     exp, splatI32(0));
  auto isFF = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, x,
                                    splatI32(0xFF));
  auto notSub = arith::SelectOp::create(rewriter, loc, isFF, nan, normal);
  auto scaleF32 = arith::SelectOp::create(rewriter, loc, isSub, sub, notSub);

  // Every UE5M3 value is exactly representable in bf16 (3-bit mantissa,
  // exponent range [-17, 16]), so the truncation is exact.
  return cast<TypedValue<RankedTensorType>>(
      arith::TruncFOp::create(rewriter, loc, scaleTy.clone(computeType),
                              scaleF32)
          .getResult());
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleTo16(PatternRewriter &rewriter,
                                  TypedValue<RankedTensorType> scale,
                                  FloatType computeType) const {
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  assert(computeType == rewriter.getBF16Type() ||
         computeType == rewriter.getF16Type());

  if (isa<FloatType>(scaleTy.getElementType())) {
    auto scaleType = scaleTy.clone(computeType);
    return cast<TypedValue<RankedTensorType>>(
        FpToFpOp::create(rewriter, loc, scaleType, scale).getResult());
  }

  // Choose an fp type that can fit the scale value.
  FloatType largeFpType = computeType == rewriter.getF16Type()
                              ? rewriter.getF32Type()
                              : computeType;
  int intWidth = largeFpType.getIntOrFloatBitWidth();
  auto intType = rewriter.getIntegerType(intWidth);

  auto zexted =
      arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intType), scale);
  // getFpMantissaWidth() returns the number of bits in the mantissa plus the
  // sign bit!
  int shiftValue = largeFpType.getFPMantissaWidth() - 1;
  auto shiftConst =
      arith::ConstantIntOp::create(rewriter, loc, shiftValue, intWidth);
  auto shift =
      SplatOp::create(rewriter, loc, scaleTy.clone(intType), shiftConst);
  auto shlRes = arith::ShLIOp::create(rewriter, loc, zexted, shift);
  Value scaleFP =
      BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shlRes);
  if (largeFpType != computeType) {
    scaleFP = arith::TruncFOp::create(rewriter, loc, scaleTy.clone(computeType),
                                      scaleFP);
  }
  return cast<TypedValue<RankedTensorType>>(scaleFP);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::broadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> scale, int dim, Attribute dstEncoding) const {
  auto loc = scale.getLoc();
  auto scaleTy = scale.getType();
  int32_t scaleFactor = scaledDotOp.deduceScaleFactor();

  // We want to broadcast the scales along dim. To do this:
  //   1. Introduce a size 1 dimension right after dim
  //   2. Broadcast the new dim to the scale factor
  //   3. Reshape it to get the result shape.

  // Compute the shape after each step.
  auto expandedShape = to_vector(scaleTy.getShape());
  expandedShape.insert(expandedShape.begin() + dim + 1, 1);
  auto broadcastShape = expandedShape;
  broadcastShape[dim + 1] = scaleFactor;
  auto resultShape = to_vector(scaleTy.getShape());
  resultShape[dim] *= scaleFactor;

  // It is more efficient to perform a layout conversion before broadcasting,
  // since there are fewer elements. Infer the source encoding that will produce
  // dstEncoding after the broadcast and reshape.
  auto interface = cast<DialectInferLayoutInterface>(&dstEncoding.getDialect());
  Attribute broadcastEncoding;
  auto result = interface->inferReshapeOpEncoding(
      resultShape, dstEncoding, broadcastShape, broadcastEncoding,
      /*allowReorder=*/false, loc);
  assert(succeeded(result));
  Attribute srcEncoding;
  result = interface->inferReshapeOpEncoding(expandedShape, broadcastEncoding,
                                             scaleTy.getShape(), srcEncoding,
                                             /*allowReorder=*/false, loc);
  assert(succeeded(result));

  auto srcType = scaleTy.cloneWithEncoding(srcEncoding);

  // Convert the scales to the desired type.
  scale = ConvertLayoutOp::create(rewriter, loc, srcType, scale);

  // Introduce the new dimension.
  auto expandType = RankedTensorType::get(
      expandedShape, scaleTy.getElementType(), broadcastEncoding);
  // We know this layout avoids having any layout conversions after we expand
  // the scales, so mark the layout as efficient. Otherwise, forward layout
  // propagation may try to sink the convert layout.
  auto expandScale = ReshapeOp::create(
      rewriter, loc, expandType, scale,
      /*allow_reorder=*/nullptr, /*efficient_layout=*/rewriter.getUnitAttr());
  // Broadcast the dimension to the microscaling factor.
  auto broadcastType = RankedTensorType::get(
      broadcastShape, scaleTy.getElementType(), broadcastEncoding);
  auto broadcastScale =
      BroadcastOp::create(rewriter, loc, broadcastType, expandScale);
  auto resultType =
      RankedTensorType::get(resultShape, scaleTy.getElementType(), dstEncoding);
  // Reshape to fold in the broadcasted dimension and get the final result.
  return ReshapeOp::create(rewriter, loc, resultType, broadcastScale);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::maskNan(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> mxfp, TypedValue<RankedTensorType> scale,
    int dim) const {
  // Skip NaN checks if fastMath
  if (scaledDotOp.getFastMath())
    return mxfp;

  // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
  auto loc = scale.getLoc();

  // Scale is NaN
  auto scaleTy = scale.getType();
  TypedValue<RankedTensorType> scaleIsNan;
  if (isa<FloatType>(scaleTy.getElementType())) {
    auto computeType = cast<FloatType>(mxfp.getType().getElementType());
    auto scaleFp = scaleTo16(rewriter, scale, computeType);
    scaleIsNan = cast<TypedValue<RankedTensorType>>(
        arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNO, scaleFp,
                              scaleFp)
            .getResult());
  } else {
    auto constFF = arith::ConstantOp::create(
        rewriter, loc, scaleTy,
        DenseElementsAttr::get(scaleTy,
                               APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
    scaleIsNan = cast<TypedValue<RankedTensorType>>(
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                              constFF)
            .getResult());
  }
  auto cond = broadcastScale(rewriter, scaledDotOp, scaleIsNan, dim,
                             mxfp.getType().getEncoding());

  // Create NaN
  auto mxfpTy = mxfp.getType();
  auto nan = APFloat::getNaN(
      cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
  auto constNan = arith::ConstantOp::create(
      rewriter, loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

  auto result = arith::SelectOp::create(rewriter, loc, cond, constNan, mxfp);
  return cast<TypedValue<RankedTensorType>>(result.getResult());
}

// Convert packed fp6 codes (i32 values 0..63) to computeType floats.
// Matches the interpreter: denormals and zero via (m/2^mbits) * 2^(1-bias),
// normals via (1 + m/2^mbits) * 2^(e-bias), times the sign. Every intermediate
// value is exactly representable, so computing directly in f16/bf16 is exact.
static TypedValue<RankedTensorType>
fp6BitsToFloat(PatternRewriter &rewriter, TypedValue<RankedTensorType> codes,
               int ebits, int mbits, int bias, FloatType computeType) {
  auto loc = codes.getLoc();
  auto i32Ty = codes.getType();
  auto fTy = i32Ty.clone(computeType);
  auto i32 = rewriter.getI32Type();
  auto i32Splat = [&](int64_t val) {
    return arith::ConstantOp::create(
               rewriter, loc, i32Ty,
               SplatElementsAttr::get(i32Ty, rewriter.getIntegerAttr(i32, val)))
        .getResult();
  };
  auto fSplat = [&](double val) {
    return arith::ConstantOp::create(
               rewriter, loc, fTy,
               SplatElementsAttr::get(fTy,
                                      rewriter.getFloatAttr(computeType, val)))
        .getResult();
  };
  // 2^k for the small k used here, without <cmath>.
  auto pow2 = [](int k) {
    return k >= 0 ? double(1 << k) : 1.0 / double(1 << -k);
  };

  Value c = codes;
  // Bit 5 is the sign for both e2m3 and e3m2.
  Value s = arith::AndIOp::create(
      rewriter, loc,
      arith::ShRUIOp::create(rewriter, loc, c, i32Splat(ebits + mbits)),
      i32Splat(1));
  Value e = arith::AndIOp::create(
      rewriter, loc, arith::ShRUIOp::create(rewriter, loc, c, i32Splat(mbits)),
      i32Splat((1 << ebits) - 1));
  Value m = arith::AndIOp::create(rewriter, loc, c, i32Splat((1 << mbits) - 1));

  Value isDenorm = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, e, i32Splat(0));
  Value mF = arith::UIToFPOp::create(rewriter, loc, fTy, m);
  // Multiply by the reciprocal: some backends have no f16/bf16 divide, and the
  // product is exact for these dyadic constants.
  Value mantFrac = arith::MulFOp::create(rewriter, loc, mF,
                                         fSplat(1.0 / double(1 << mbits)));

  // 2^(e-bias) via f32 bit construction: exact, and avoids a select chain.
  // e == 0 takes the denormal path, so its (valid) power of two is unused.
  Value expBits = arith::ShLIOp::create(
      rewriter, loc,
      arith::AddIOp::create(
          rewriter, loc,
          arith::SubIOp::create(rewriter, loc, e, i32Splat(bias)),
          i32Splat(127)),
      i32Splat(23));
  Value pow2F32 = BitcastOp::create(rewriter, loc,
                                    i32Ty.clone(rewriter.getF32Type()), expBits)
                      .getResult();
  // Exact powers of two, so the required rounding mode is immaterial.
  Value pow2E = FpToFpOp::create(rewriter, loc, fTy, pow2F32,
                                 RoundingModeAttr::get(rewriter.getContext(),
                                                       RoundingMode::RTNE))
                    .getResult();
  Value normal = arith::MulFOp::create(
      rewriter, loc,
      arith::AddFOp::create(rewriter, loc, fSplat(1.0), mantFrac), pow2E);
  Value denorm =
      arith::MulFOp::create(rewriter, loc, mantFrac, fSplat(pow2(1 - bias)));
  Value mag = arith::SelectOp::create(rewriter, loc, isDenorm, denorm, normal);
  Value isNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                      s, i32Splat(0));
  Value negMag = arith::NegFOp::create(rewriter, loc, mag);
  return cast<TypedValue<RankedTensorType>>(
      arith::SelectOp::create(rewriter, loc, isNeg, negMag, mag).getResult());
}

// Warp-local gather layout, mirroring OptimizeThreadLocality's
// setOptimizedGatherLayout. Gathers created here run after that pass, so the
// recipe is applied directly; the shared-memory fallback miscompiles.
static BlockedEncodingAttr
computeWarpLocalGatherLayout(MLIRContext *ctx, RankedTensorType srcType,
                             ArrayRef<int64_t> idxShape, unsigned axis,
                             unsigned numThreadsPerWarp, unsigned numWarps) {
  unsigned rank = srcType.getRank();
  assert(rank > 1);
  SmallVector<unsigned> threadsPerWarp(rank);
  SmallVector<unsigned> warpsPerCTA(rank);
  SmallVector<unsigned> order;
  order.push_back(axis);
  unsigned maxThreadsInAxis =
      std::min<unsigned>(srcType.getDimSize(axis), numThreadsPerWarp);
  threadsPerWarp[axis] = maxThreadsInAxis;
  unsigned threadsToAlloc = numThreadsPerWarp / maxThreadsInAxis;
  for (unsigned dim : getThreadOrder(srcType)) {
    if (dim == axis)
      continue;
    order.push_back(dim);
    unsigned nextThreadAlloc =
        std::min<unsigned>(srcType.getDimSize(dim), threadsToAlloc);
    threadsPerWarp[dim] = nextThreadAlloc;
    threadsToAlloc /= nextThreadAlloc;
  }
  warpsPerCTA[axis] = 1;
  unsigned warpsToAlloc = numWarps;
  for (unsigned dim : getWarpOrder(srcType)) {
    if (dim == axis)
      continue;
    unsigned warpsCanFit = srcType.getDimSize(dim) / threadsPerWarp[dim];
    assert(warpsCanFit != 0);
    unsigned nextWarpAlloc = std::min<unsigned>(warpsCanFit, warpsToAlloc);
    warpsPerCTA[dim] = nextWarpAlloc;
    warpsToAlloc /= nextWarpAlloc;
  }
  SmallVector<unsigned> sizePerThread(rank, 1);
  sizePerThread[axis] =
      std::max<int64_t>(srcType.getDimSize(axis), idxShape[axis]) /
      threadsPerWarp[axis];
  threadsPerWarp[axis] *= threadsToAlloc;
  warpsPerCTA[axis] *= warpsToAlloc;
  auto baseLayout = cast<LayoutEncodingTrait>(srcType.getEncoding());
  auto cgaLayout = getCGALayout(baseLayout);
  return BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp,
                                  warpsPerCTA, order, cgaLayout);
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::upcastFp6(PatternRewriter &rewriter,
                                  DotScaledOp scaledDotOp, int opIdx,
                                  FloatType computeType) const {
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto elemType =
      opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType();
  int ebits, mbits, bias;
  if (elemType == ScaleDotElemType::E2M3) {
    ebits = 2;
    mbits = 3;
    bias = 1;
  } else {
    assert(elemType == ScaleDotElemType::E3M2);
    ebits = 3;
    mbits = 2;
    bias = 3;
  }

  auto loc = v.getLoc();
  auto vTy = v.getType();
  int rank = vTy.getRank();
  int kDim = opIdx == 0 ? rank - 1 : rank - 2;
  // The K dim is padded to the unpacked width; the leading 3/4 hold dense
  // triples and the trailing quarter is ignored padding.
  // TODO: once NPOT gather/layout codegen is robust, accept tight packed-K
  // (packedK % 3 == 0, unpacked K = packedK * 4 / 3) here and in
  // `tl.dot_scaled` shape deduction instead of requiring padded POT input.
  assert(vTy.getShape()[kDim] % 4 == 0);

  SmallVector<int64_t> outShape(to_vector(vTy.getShape()));

  // 0) Gather the two bytes feeding each output code directly at output
  // positions. Output K-coordinate j = 4t+i decodes packed triple t as
  //   v = ((lo >> s_lo) & m_lo) | ((hi << s_hi) & m_hi)
  // with per-output constants.
  auto *ctx = rewriter.getContext();
  auto mod = scaledDotOp->getParentOfType<ModuleOp>();
  int numWarps = lookupNumWarps(scaledDotOp);
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  auto idxEnc = computeWarpLocalGatherLayout(ctx, vTy, outShape, kDim,
                                             threadsPerWarp, numWarps);
  auto i32 = rewriter.getI32Type();
  auto idxTy = RankedTensorType::get(outShape, i32, idxEnc);

  // Per-output K-coordinate j = cumsum(ones) - 1 along K. Triple
  // (b0, b1, b2) holds v0 = b0[5:0], v1 = b0[7:6]|b1[3:0]<<2,
  // v2 = b1[7:4]|b2[1:0]<<4, v3 = b2[7:2]; the two bytes and the
  // shift/mask tables below are computed arithmetically from j.
  auto i32Splat = [&](int64_t val) {
    return arith::ConstantOp::create(
               rewriter, loc, idxTy,
               SplatElementsAttr::get(idxTy, rewriter.getIntegerAttr(i32, val)))
        .getResult();
  };
  auto scan = ScanOp::create(rewriter, loc, ValueRange{i32Splat(1)}, kDim,
                             /*reverse=*/false, StringAttr());
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Region &combineOp = scan.getCombineOp();
    Block *body = rewriter.createBlock(&combineOp);
    Value acc = body->addArgument(i32, loc);
    Value cur = body->addArgument(i32, loc);
    rewriter.setInsertionPointToEnd(body);
    Value sum = arith::AddIOp::create(rewriter, loc, acc, cur);
    OperationState retState(loc, ScanReturnOp::getOperationName());
    retState.addOperands(sum);
    rewriter.create(retState);
  }
  Value j =
      arith::SubIOp::create(rewriter, loc, scan->getResult(0), i32Splat(1))
          .getResult();
  Value t = arith::DivUIOp::create(rewriter, loc, j, i32Splat(4)).getResult();
  Value i = arith::RemUIOp::create(rewriter, loc, j, i32Splat(4)).getResult();
  Value threeT =
      arith::MulIOp::create(rewriter, loc, t, i32Splat(3)).getResult();
  Value iGe2 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                                     i, i32Splat(2))
                   .getResult();
  Value loOff =
      arith::SelectOp::create(
          rewriter, loc, iGe2,
          arith::SubIOp::create(rewriter, loc, i, i32Splat(1)), i32Splat(0))
          .getResult();
  Value loIdx = arith::AddIOp::create(rewriter, loc, threeT, loOff).getResult();
  Value hiOff =
      arith::MinUIOp::create(rewriter, loc, i, i32Splat(2)).getResult();
  Value hiIdx = arith::AddIOp::create(rewriter, loc, threeT, hiOff).getResult();
  auto sel4 = [&](int v0, int v1, int v2, int v3) {
    Value is1 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      i, i32Splat(1))
                    .getResult();
    Value is2 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      i, i32Splat(2))
                    .getResult();
    Value is3 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      i, i32Splat(3))
                    .getResult();
    Value s23 =
        arith::SelectOp::create(rewriter, loc, is3, i32Splat(v3), i32Splat(v0))
            .getResult();
    Value s12 = arith::SelectOp::create(rewriter, loc, is2, i32Splat(v2), s23)
                    .getResult();
    return arith::SelectOp::create(rewriter, loc, is1, i32Splat(v1), s12)
        .getResult();
  };
  Value sLo = sel4(0, 6, 4, 2);
  Value mLo = sel4(63, 3, 15, 63);
  Value sHi = sel4(0, 2, 4, 0);
  Value mHi = sel4(0, 60, 48, 0);
  Value cvtSrc =
      ConvertLayoutOp::create(rewriter, loc, vTy.cloneWithEncoding(idxEnc), v)
          .getResult();
  auto gatherLo = GatherOp::create(rewriter, loc, cvtSrc, loIdx, kDim);
  gatherLo.setEfficientLayout(true);
  auto gatherHi = GatherOp::create(rewriter, loc, cvtSrc, hiIdx, kDim);
  gatherHi.setEfficientLayout(true);
  assert(GatherLoweringHelper(gatherLo).isWarpLocal());
  assert(GatherLoweringHelper(gatherHi).isWarpLocal());
  Value lo = gatherLo.getResult();
  Value hi = gatherHi.getResult();
  auto laneTy = cast<RankedTensorType>(lo.getType()).clone(i32);

  // 1) Decode each output code from its two bytes. K stays in place.
  Value lo32 = arith::ExtUIOp::create(rewriter, loc, laneTy, lo);
  Value hi32 = arith::ExtUIOp::create(rewriter, loc, laneTy, hi);
  auto codes = cast<TypedValue<RankedTensorType>>(
      arith::OrIOp::create(
          rewriter, loc,
          arith::AndIOp::create(
              rewriter, loc, arith::ShRUIOp::create(rewriter, loc, lo32, sLo),
              mLo),
          arith::AndIOp::create(rewriter, loc,
                                arith::ShLIOp::create(rewriter, loc, hi32, sHi),
                                mHi))
          .getResult());
  return fp6BitsToFloat(rewriter, codes, ebits, mbits, bias, computeType);
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::scaleArg(PatternRewriter &rewriter,
                                 DotScaledOp scaledDotOp, int opIdx,
                                 FloatType computeType) const {
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
  auto elemType =
      opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType();
  auto isFp4 = ScaleDotElemType::E2M1 == elemType;
  auto isFp6 =
      elemType == ScaleDotElemType::E2M3 || elemType == ScaleDotElemType::E3M2;

  auto loc = v.getLoc();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // 0) Upcast value to computeType (fp16/bf16)
  if (isFp6) {
    v = upcastFp6(rewriter, scaledDotOp, opIdx, computeType);
  } else if (isFp4) {
    bool kPack =
        opIdx == 0 ? scaledDotOp.getLhsKPack() : scaledDotOp.getRhsKPack();
    int packedDim = kPack ? kDim : (opIdx == 0 ? rank - 2 : rank - 1);
    v = Fp4ToFpOp::create(rewriter, loc, v, computeType, packedDim);
  } else {
    auto vType16 = v.getType().clone(computeType);
    v = cast<TypedValue<RankedTensorType>>(
        FpToFpOp::create(rewriter, loc, vType16, v).getResult());
  }
  if (!scale)
    return v;

  // 1) Cast scale to fp16/bf16, broadcast it and convert its layout
  auto reshapeScale = extendAndBroadcastScale(rewriter, scaledDotOp, scale,
                                              computeType, v.getType(), opIdx);

  // 2) Multiply
  auto mxfp = cast<TypedValue<RankedTensorType>>(
      arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

  // 3) If the scale is NaN, return NaN, else return the scaled value.
  return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
}

TypedValue<RankedTensorType> DecomposeScaledBlocked::extendAndBroadcastScale(
    PatternRewriter &rewriter, DotScaledOp scaledDotOp,
    TypedValue<RankedTensorType> &scale, FloatType computeType,
    RankedTensorType dstType, int opIdx) const {
  auto loc = scale.getLoc();
  auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
  auto rank = v.getType().getRank();
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

  // For some weird reason, we take the scale with shape as if it were coming
  // from the lhs even when it's the rhs. In a normal world, we should accept
  // this parameter transposed, as we do with the mxfp.
  //
  // Notice: this is an inplace change.
  if (opIdx == 1) {
    auto order = getTransposeOrder(rank);
    scale = TransOp::create(rewriter, loc, scale, order);
  }

  // 1) Cast scale to compute type (fp16/bf16). On NV, int8 block-16
  // scales are UE5M3, not E8M0, and need their own decode.
  auto scale16 = allowUE5M3Scales && isUE5M3ScaleTensor(scaledDotOp, scale)
                     ? decodeUE5M3To16(rewriter, scale, computeType)
                     : scaleTo16(rewriter, scale, computeType);

  // 2) Broadcast scale to the same shape as v and convert the layout
  return broadcastScale(rewriter, scaledDotOp, scale16, kDim,
                        dstType.getEncoding());
}

TypedValue<RankedTensorType>
DecomposeScaledBlocked::cvtDotOperand(PatternRewriter &rewriter,
                                      DotScaledOp scaledDotOp, int opIdx,
                                      TypedValue<RankedTensorType> v) const {
  auto *ctx = rewriter.getContext();
  auto retEnc = scaledDotOp.getType().getEncoding();
  auto vType = v.getType();
  auto encoding =
      DotOperandEncodingAttr::get(ctx, opIdx, retEnc, vType.getElementType());
  auto retTy = vType.cloneWithEncoding(encoding);
  return ConvertLayoutOp::create(rewriter, v.getLoc(), retTy, v);
}

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit,
                                            bool allowUE5M3Scales) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit,
                                       allowUE5M3Scales);
}

} // namespace mlir::triton::gpu
