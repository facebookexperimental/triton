#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;

namespace {

/// OperandValueKey structure represents compile time part
/// of spatial coordinates of a value in a tensor.
///
/// Every Value spatial coordinates(i.e. [batch;nonK;k]) in tensor can be
/// defined as:
///
/// batch = (bRepIdx * CTABSize + bIdx) + (laneBCoord + warpBCoord)
/// nonK = (nonKRepIdx * CTANKSize + nonKIdx) + (laneNonKCoord + warpNonKCoord)
/// k = kIdx
///
/// Where:
/// CTABSize, CTANKSize: constants;
/// laneBCoord, warpBCoord, laneNonKCoord, warpNonKCoord: runtime components;
/// bRepIdx, nonKRepIdx, bIdx, nonKIdx, kIdx: compile time components.
struct OperandValueKey {
  unsigned bRepIdx, nonKRepIdx;
  unsigned bIdx, nonKIdx, kIdx;

  bool operator==(const OperandValueKey &other) const {
    return (bRepIdx == other.bRepIdx && nonKRepIdx == other.nonKRepIdx &&
            bIdx == other.bIdx && nonKIdx == other.nonKIdx &&
            kIdx == other.kIdx);
  }
};

} // namespace

template <> struct std::hash<OperandValueKey> {
  std::size_t operator()(const OperandValueKey &k) const {
    return llvm::hash_combine(k.bRepIdx, k.nonKRepIdx, k.bIdx, k.nonKIdx,
                              k.kIdx);
  }
};

namespace {

using ValueTableFMA = std::unordered_map<OperandValueKey, Value>;

ValueTableFMA getValueTableFromStructFMA(
    Value val, ArrayRef<unsigned> perRepShape, ArrayRef<unsigned> repetitions,
    unsigned kDim, unsigned nonKDim, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<unsigned> inRepOrder, ArrayRef<unsigned> repOrder) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(perRepShape.size() == 3);
  auto numElemsRep = product(perRepShape);
  assert(elems.size() == numElemsRep * product(repetitions));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  // Check if any dimension is NPOT. For NPOT K, the modular layout stores
  // elements in a different order than delinearize assumes. Use the
  // LinearLayout to compute correct coordinates in that case.
  bool hasNpotDim = llvm::any_of(
      perRepShape, [](unsigned s) { return s > 0 && !llvm::isPowerOf2_32(s); });

  if (hasNpotDim) {
    // For NPOT K, the register ordering follows the LinearLayout's modular
    // arithmetic (ADD + mod), NOT the simple row-major order that delinearize
    // assumes. Register index i maps to coordinates by accumulating basis
    // values: coord[d] = SUM(basis[d] for each active bit) mod dimSize.
    //
    // The dot_op encoding rounds NPOT dims to pow2 for register allocation.
    // So the actual register space is pow2-rounded, and we need to use the
    // pow2 size for bit indexing, then apply modular reduction to get the
    // actual NPOT coordinate.

    // Round perRepShape to pow2 for register bit counting.
    SmallVector<unsigned> pow2PerRepShape(3);
    for (int i = 0; i < 3; i++) {
      unsigned s = perRepShape[i];
      pow2PerRepShape[i] = (s <= 1)                 ? s
                           : llvm::isPowerOf2_32(s) ? s
                                                    : llvm::NextPowerOf2(s);
    }
    unsigned pow2NumElemsRep = product(pow2PerRepShape);

    // Count register bits per dimension (in inRepOrder).
    SmallVector<unsigned> bitsPerDim(3);
    for (int i = 0; i < 3; i++) {
      unsigned s = pow2PerRepShape[i];
      bitsPerDim[i] = (s <= 1) ? 0 : llvm::Log2_32(s);
    }

    for (unsigned idx = 0; idx < elems.size(); ++idx) {
      auto inRepLinearIdx = idx % pow2NumElemsRep;
      auto repLinearIdx = idx / pow2NumElemsRep;

      // Compute in-rep coordinates from register bits using basis
      // accumulation. Bits are assigned to dims in inRepOrder
      // (fastest-changing dim first).
      SmallVector<unsigned> inRepCoords(3, 0);
      unsigned bitOffset = 0;
      for (auto d : inRepOrder) {
        unsigned nBits = bitsPerDim[d];
        unsigned dimSize = perRepShape[d];
        if (nBits == 0)
          continue;
        unsigned coord = 0;
        for (unsigned bit = 0; bit < nBits; ++bit) {
          if (inRepLinearIdx & (1u << (bitOffset + bit))) {
            unsigned basisVal = (1u << bit);
            coord += basisVal;
          }
        }
        // Modular reduction: ADD+mod for NPOT, AND for pow2.
        if (llvm::isPowerOf2_32(dimSize))
          inRepCoords[d] = coord & (dimSize - 1);
        else
          inRepCoords[d] = coord % dimSize;
        bitOffset += nBits;
      }

      // Rep coordinates: all dims in repetitions are pow2.
      auto repSpatialIdx =
          mlir::LLVM::delinearize(repLinearIdx, repetitions, repOrder);

      OperandValueKey key{repSpatialIdx[0], repSpatialIdx[nonKDim],
                          inRepCoords[0], inRepCoords[nonKDim],
                          inRepCoords[kDim]};
      res[key] = elems[idx];
    }
    return res;
  }

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    auto inRepLinearIdx = idx % numElemsRep;
    auto repLinearIdx = idx / numElemsRep;
    auto inRepSpatialIdx =
        mlir::LLVM::delinearize(inRepLinearIdx, perRepShape, inRepOrder);
    auto repSpatialIdx =
        mlir::LLVM::delinearize(repLinearIdx, repetitions, repOrder);
    OperandValueKey key{repSpatialIdx[0], repSpatialIdx[nonKDim],
                        inRepSpatialIdx[0], inRepSpatialIdx[nonKDim],
                        inRepSpatialIdx[kDim]};
    res[key] = elems[idx];
  }
  return res;
}

} // namespace

namespace mlir::triton::gpu {

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  // TODO process A and B operand separately
  auto inRepOrder = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto repOrder = expandMatrixOrderWithBatch(dLayout.getRepOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getContigPerThread(dTensorTy);
  auto numElemsPerThread = product(sizePerThread);
  SmallVector<unsigned> shapePerCTATile;
  for (auto [reg, thread, warp] :
       llvm::zip(sizePerThread, dLayout.getThreadsPerWarp(),
                 dLayout.getWarpsPerCTA())) {
    shapePerCTATile.push_back(reg * thread * warp);
  }
  shapePerCTATile = expandMatrixShapeWithBatch(ArrayRef(shapePerCTATile));
  sizePerThread = expandMatrixShapeWithBatch(ArrayRef(sizePerThread));

  unsigned K = aShapePerCTA[2];
  // For NPOT K, the dot_op LinearLayout rounds K up to pow2 in the register
  // dim (fmaDotToLinearLayout uses pow2ThreadSize). The LLVM struct stores
  // pow2-many K elements per thread, including degenerate wrap-around
  // duplicates. Use pow2 K in the perRepShape so getValueTableFromStructFMA
  // iterates over all struct elements correctly. The dot loop below still
  // uses actual K for the summation.
  unsigned Kpow2 = llvm::isPowerOf2_32(K) ? K : (unsigned)llvm::NextPowerOf2(K);

  // For NPOT non-K dims (M/N/batch), the operand registers are materialized
  // from the dot_op LinearLayout which rounds the shape up to pow2 in the
  // register dim (see LinearLayoutConversions.cpp NextPow2 of shapePerCTA).
  // The number of CTA-tile repetitions covering that pow2-rounded register
  // space is NextPow2(shapePerCTA)/CTATile, which can exceed
  // ceil(shapePerCTA/CTATile) for NPOT shapes (e.g. M=80/96 -> pow2 128). The
  // operand value tables (and the accumulator indexing) must use this padded
  // count so that getValueTableFromStructFMA iterates over all struct elements
  // correctly (matching the Kpow2 handling for the K dim) and the dot loop
  // below writes every register slot of the pow2-rounded C struct. See the
  // dot-loop comment for why computing the padding reps is correct.
  unsigned repetitionsPadded[3];
  for (int i = 0; i < 3; ++i) {
    int64_t shape = dShapePerCTA[i];
    int64_t pow2Shape = llvm::isPowerOf2_64(shape)
                            ? shape
                            : (int64_t)llvm::NextPowerOf2((uint64_t)shape);
    repetitionsPadded[i] =
        ceil(pow2Shape, static_cast<int64_t>(shapePerCTATile[i]));
  }

  auto has = getValueTableFromStructFMA(
      llA, {sizePerThread[0], sizePerThread[1], Kpow2},
      {repetitionsPadded[0], repetitionsPadded[1], 1},
      /*kDim*/ 2, /*nonKDim*/ 1, rewriter, loc, inRepOrder, repOrder);
  auto hbs = getValueTableFromStructFMA(
      llB, {sizePerThread[0], Kpow2, sizePerThread[2]},
      {repetitionsPadded[0], 1, repetitionsPadded[2]},
      /*kDim*/ 1, /*nonKDim*/ 2, rewriter, loc, inRepOrder, repOrder);

  SmallVector<Value> acc = cc;

  // Iterate over the pow2-padded repetition count. The accumulator (operand C)
  // is materialized from the dot_op/blocked LinearLayout, which rounds each
  // NPOT non-K dim up to pow2 in the register space. The padding reps cover the
  // dead rows [shapePerCTA, NextPow2(shapePerCTA)); under the layout's modular
  // (% origSize) reduction these wrap onto valid logical rows. By computing the
  // FMA for every padded rep using the (wrapped) operand value tables, each
  // padding rep produces the SAME result as the canonical rep that owns the
  // wrapped row, so the redundant store is harmless rather than clobbering the
  // valid rows with stale C values (the m80 wrong-results bug). linearAccumIdx
  // must be linearized with the padded rep count to match the C struct layout.
  for (unsigned bRep = 0; bRep < repetitionsPadded[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitionsPadded[1]; ++mRep)
      for (unsigned nRep = 0; nRep < repetitionsPadded[2]; ++nRep)
        for (unsigned b = 0; b < sizePerThread[0]; ++b)
          for (unsigned m = 0; m < sizePerThread[1]; ++m)
            for (unsigned n = 0; n < sizePerThread[2]; ++n) {
              SmallVector<unsigned> multiDimAccumIdx = {b, m, n};
              unsigned linearInRepIdx =
                  LLVM::linearize(multiDimAccumIdx, sizePerThread, inRepOrder);
              SmallVector<unsigned> multiDimRepIdx = {bRep, mRep, nRep};
              unsigned linearRepIdx =
                  LLVM::linearize(multiDimRepIdx, repetitionsPadded, repOrder);
              unsigned linearAccumIdx =
                  linearInRepIdx + linearRepIdx * numElemsPerThread;

              SmallVector<Value> aOpVector;
              SmallVector<Value> bOpVector;

              for (unsigned k = 0; k < K; ++k) {
                aOpVector.push_back(has.at({bRep, mRep, b, m, k}));
                bOpVector.push_back(hbs.at({bRep, nRep, b, n, k}));
              }

              acc[linearAccumIdx] = multiplier.multiplyVectors(
                  aOpVector, bOpVector, acc[linearAccumIdx]);
            }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}

} // namespace mlir::triton::gpu
