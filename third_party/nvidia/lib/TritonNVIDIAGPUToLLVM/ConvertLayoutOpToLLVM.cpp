#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::LLVM::NVIDIA::lowerLdStMatrix;

struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const NVIDIA::TargetInfo &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(
      LLVMTypeConverter &typeConverter, const NVIDIA::TargetInfo &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto *ctx = op.getContext();

    // Modular layouts use the generic conversion path, except odd tiles.
    if (hasNpotShape(srcTy) || hasNpotShape(dstTy)) {
      auto tiles =
          nvidia_gpu::getMixedRadixTileTransfer(srcTy, dstTy, targetInfo);
      if (!tiles)
        return failure();
      return transferMixedRadixTiles(op, adaptor, *tiles, rewriter);
    }

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);
    srcLayout = srcLayout.removeZeroBasesAlongDim(str_attr("register"));
    dstLayout = dstLayout.removeZeroBasesAlongDim(str_attr("register"));

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    if (cvtNeedsSharedMemory(op)) {
      auto loc = op.getLoc();

      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      auto inVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, dstLayout,
                                       inVals, llvmElemTy, smemBase, op);

      Value result = packUniqueTensorElements(loc, getTypeConverter(), outVals,
                                              rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  // `inVals` holds `numTiles` transfers of `srcLayout` back to back.
  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase,
      Operation *sourceOp, int32_t numTiles = 1,
      int32_t tilesPerRound = 1) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, llvmElemTyPtr,
          smemBase, sourceOp, numTiles, tilesPerRound);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase,
          sourceOp, numTiles, tilesPerRound);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v, LLVM::IntegerOverflowFlags::nuw);
      }
      return outVals;
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto kBlock = str_attr("block");
    bool crossCTA =
        !dstLayout.invertAndCompose(srcLayout).isTrivialOver({kBlock});
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth, crossCTA);
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = invertAndComposeLocal(smem, dstLayout, {kBlock});

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    SmallVector<Value> permutedInVals;
    int32_t valsPerTile = inVals.size() / numTiles;
    for (int32_t tile = 0; tile < numTiles; ++tile)
      llvm::append_range(permutedInVals, permStore.apply(inVals.slice(
                                             tile * valsPerTile, valsPerTile)));
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    auto nBlock = storeCvt.getInDimSize(kBlock);
    storeCvt = storeCvt.reshapeOuts(
        {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
    loadCvt = loadCvt.reshapeOuts(
        {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

    // We never do cross-CTA writes by construction. We may do cross-CTA reads,
    // but in that case we lower to ld.shared/st.shared. A write is in-CTA when
    // the block input either acts as the identity (each CTA writes its own
    // smem) or is dropped entirely (broadcast-only CGA, block maps to zero).
    // Only a block that targets a *different* CTA would be a cross-CTA write.
    assert(storeCvt.isTrivialOver({kBlock}) ||
           storeCvt.sublayoutIsZero({kBlock},
                                    to_vector(storeCvt.getOutDimNames())));
    assert(loadCvt.isTrivialOver({kBlock}) ||
           loadCvt.sublayoutIsZero({kBlock},
                                   to_vector(loadCvt.getOutDimNames())) ||
           idxDst == 0);

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps * numTiles);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    bool isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
    bool isBlockSync = mlir::isCvtDimSync(srcLayout, dstLayout, kBlock);
    auto emitBarrier = [&]() {
      if (isWarpSync) {
        targetInfo.warpSync(loc, rewriter);
      } else if (isBlockSync) {
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      } else {
        targetInfo.clusterBarrier(loc, rewriter, sourceOp);
      }
    };
    auto dropBlock = [&](const LinearLayout &cvt) {
      SmallVector<StringAttr> inDims = {kReg, kLane, kWarp};
      SmallVector<StringAttr> outDims = {kOffset};
      return cvt.sublayout(inDims, outDims);
    };
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    // Each rep of a round is one store/bar/load of that rep of the round's
    // tiles, at disjoint offsets.
    int32_t tileElems = storeCvt.getTotalOutDimSize() / nBlock;
    auto tileBase = [&](int32_t tile) {
      tile %= tilesPerRound;
      return tile == 0 ? smemBase
                       : b.gep(smemBase.getType(), llvmElemTy, smemBase,
                               b.i32_val(tile * tileElems));
    };
    SmallVector<SmallVector<Value>> outValsByTile(numTiles);
    int32_t numRounds = llvm::divideCeil(numTiles, tilesPerRound);
    for (int i = 0; i < nReps * numRounds; ++i) {
      if (i > 0)
        emitBarrier();
      int32_t rep = i % nReps, first = i / nReps * tilesPerRound;
      auto roundTiles =
          llvm::seq<int32_t>(first, std::min(numTiles, first + tilesPerRound));
      for (int32_t tile : roundTiles) {
        auto tileInVals =
            to_vector(ArrayRef(permutedInVals)
                          .slice((tile * nReps + rep) * tileSize, tileSize));
        Value base = tileBase(tile);
        // Store
        // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans
        if (idxSrc == 0) {
          lowerLdSt(loc, ctx, storeCvt, tileInVals, llvmElemTy, base,
                    /*paddingShifts=*/{}, affineOffset, maskSpanAffineOffset,
                    /*affineBlockOffset=*/Value(), /*maskSpanAffineBlock=*/0,
                    laneId, warpId, rewriter, targetInfo,
                    /*maybeMaxVecElems=*/{},
                    makeSharedStoreEmitter(targetInfo, b.true_val()));
        } else {
          assert(idxSrc == 1 || idxSrc == 2);
          bool transpose = idxSrc == 2;
          auto storeCvtNoBlock = dropBlock(storeCvt);
          auto result = lowerLdStMatrix(
              loc, storeCvtNoBlock, transpose, tileInVals, base, affineOffset,
              maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
          assert(succeeded(result));
        }
      }
      emitBarrier();
      for (int32_t tile : roundTiles) {
        Value base = tileBase(tile);
        // Load
        SmallVector<Value> tileOutVals;
        // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans
        if (idxDst == 0) {
          tileOutVals = lowerLdSt(
              loc, ctx, loadCvt, {}, llvmElemTy, base, /*paddingShifts=*/{},
              affineOffset, maskSpanAffineOffset,
              /*affineBlockOffset=*/Value(),
              /*maskSpanAffineBlock=*/0, laneId, warpId, rewriter, targetInfo,
              /*maybeMaxVecElems=*/{}, makeSharedLoadEmitter(targetInfo));
        } else {
          assert(idxDst == 1 || idxDst == 2);
          bool transpose = idxDst == 2;
          auto loadCvtNoBlock = dropBlock(loadCvt);
          auto result = lowerLdStMatrix(
              loc, loadCvtNoBlock, transpose, tileOutVals, base, affineOffset,
              maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
          assert(succeeded(result));
        }
        llvm::append_range(outValsByTile[tile], tileOutVals);
      }
    }

    // Undo the permLoad used to divideRight
    for (auto &vals : outValsByTile)
      llvm::append_range(outVals, permLoad.inverse().apply(vals));
    return outVals;
  }

  // Moves each odd tile of a mixed-radix conversion as one pow2 transfer.
  LogicalResult
  transferMixedRadixTiles(ConvertLayoutOp op, OpAdaptor adaptor,
                          const nvidia_gpu::MixedRadixTileTransfer &transfer,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto kReg = str_attr("register");
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto src = toLinearLayout(srcTy).removeZeroBasesAlongDim(kReg);
    auto dst = toLinearLayout(dstTy).removeZeroBasesAlongDim(kReg);
    auto inVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    const MixedRadixSharedTiling &tiles = transfer.tiles;
    SmallVector<Value> tileVals;
    for (int32_t tile = 0; tile < transfer.numTiles; ++tile)
      for (int32_t reg : getTileRegisters(src, tiles.sourceDigits, tile))
        tileVals.push_back(inVals[reg]);
    auto outValsByTile = transferWithinBlockSwizzling(
        loc, rewriter, tiles.sourceTile, tiles.destinationTile, tileVals,
        getTypeConverter()->convertType(srcTy.getElementType()),
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op), op,
        transfer.numTiles, transfer.tilesPerRound);
    SmallVector<Value> outVals(dst.getInDimSize(kReg));
    auto next = outValsByTile.begin();
    for (int32_t tile = 0; tile < transfer.numTiles; ++tile)
      for (int32_t reg : getTileRegisters(dst, tiles.destinationDigits, tile))
        outVals[reg] = *next++;
    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    outVals, rewriter, dstTy));
    return success();
  }

  // The registers of `layout` whose `digits` hold the values of `tile`,
  // digits[0] fastest, in register order.
  static SmallVector<int32_t> getTileRegisters(const LinearLayout &layout,
                                               ArrayRef<int32_t> digits,
                                               int32_t tile) {
    auto kReg = StringAttr::get(layout.getInDimNames().begin()->getContext(),
                                "register");
    ArrayRef<int32_t> radices = layout.getInputRadices(kReg);
    SmallVector<int32_t> registers;
    for (int32_t reg = 0; reg < layout.getInDimSize(kReg); ++reg) {
      SmallVector<int32_t> values;
      for (int32_t rest = reg, digit = 0; digit < int32_t(radices.size());
           rest /= radices[digit++])
        values.push_back(rest % radices[digit]);
      int32_t rest = tile;
      bool inTile = true;
      for (int32_t digit : digits) {
        inTile &= values[digit] == rest % radices[digit];
        rest /= radices[digit];
      }
      if (inTile)
        registers.push_back(reg);
    }
    return registers;
  }

  LogicalResult
  transferWithinBlockSwizzling(ConvertLayoutOp op, Value src,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, src, rewriter);
    auto outVals = transferWithinBlockSwizzling(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase, op);

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpSwizzlingConversion a higher benefit as it
  // matches optimized ldmatrix/stmatrix cases
  patterns.add<ConvertLayoutOpSwizzlingConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
