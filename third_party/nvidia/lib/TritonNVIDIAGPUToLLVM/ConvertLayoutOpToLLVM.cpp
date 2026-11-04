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

    // Modular layouts use the generic conversion path, except odd slabs.
    if (hasNpotShape(srcTy) || hasNpotShape(dstTy)) {
      auto slabs = nvidia_gpu::getProductSlabTransfer(srcTy, dstTy, targetInfo);
      if (!slabs)
        return failure();
      return transferProductSlabs(op, adaptor, *slabs, rewriter);
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

  // `inVals` holds `numSlabs` transfers of `srcLayout` back to back.
  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase,
      Operation *sourceOp, int32_t numSlabs = 1) const {
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
          smemBase, sourceOp, numSlabs);
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
          sourceOp, numSlabs);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
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
    int32_t slabSize = inVals.size() / numSlabs;
    for (int32_t slab = 0; slab < numSlabs; ++slab)
      llvm::append_range(permutedInVals, permStore.apply(inVals.slice(
                                             slab * slabSize, slabSize)));
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

    assert(permutedInVals.size() == tileSize * nReps * numSlabs);
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
    // Each rep is one phase that moves that rep of every slab, at disjoint
    // offsets.
    int32_t tileElems = storeCvt.getTotalOutDimSize() / nBlock;
    auto slabBase = [&](int32_t slab) {
      return slab == 0 ? smemBase
                       : b.gep(smemBase.getType(), llvmElemTy, smemBase,
                               b.i32_val(slab * tileElems));
    };
    SmallVector<SmallVector<Value>> slabOutVals(numSlabs);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        emitBarrier();
      for (int32_t slab = 0; slab < numSlabs; ++slab) {
        auto tileInVals =
            to_vector(ArrayRef(permutedInVals)
                          .slice((slab * nReps + i) * tileSize, tileSize));
        Value base = slabBase(slab);
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
      for (int32_t slab = 0; slab < numSlabs; ++slab) {
        Value base = slabBase(slab);
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
        llvm::append_range(slabOutVals[slab], tileOutVals);
      }
    }

    // Undo the permLoad used to divideRight
    for (auto &vals : slabOutVals)
      llvm::append_range(outVals, permLoad.inverse().apply(vals));
    return outVals;
  }

  // Moves each odd slab of a Product conversion as one pow2 transfer.
  LogicalResult
  transferProductSlabs(ConvertLayoutOp op, OpAdaptor adaptor,
                       const nvidia_gpu::ProductSlabTransfer &transfer,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto kReg = str_attr("register");
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto src = toLinearLayout(srcTy).removeZeroBasesAlongDim(kReg);
    auto dst = toLinearLayout(dstTy).removeZeroBasesAlongDim(kReg);
    auto inVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    const ProductSharedTiling &slabs = transfer.slabs;
    SmallVector<Value> slabVals;
    for (int32_t slab = 0; slab < transfer.numSlabs; ++slab)
      for (int32_t reg : getSlabRegisters(src, slabs.sourceDigits, slab))
        slabVals.push_back(inVals[reg]);
    auto slabOutVals = transferWithinBlockSwizzling(
        loc, rewriter, slabs.sourceTile, slabs.destinationTile, slabVals,
        getTypeConverter()->convertType(srcTy.getElementType()),
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op), op,
        transfer.numSlabs);
    SmallVector<Value> outVals(dst.getInDimSize(kReg));
    auto next = slabOutVals.begin();
    for (int32_t slab = 0; slab < transfer.numSlabs; ++slab)
      for (int32_t reg : getSlabRegisters(dst, slabs.destinationDigits, slab))
        outVals[reg] = *next++;
    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    outVals, rewriter, dstTy));
    return success();
  }

  // The registers of `layout` whose `digits` hold slab `slab`'s values,
  // digits[0] fastest, in register order.
  static SmallVector<int32_t> getSlabRegisters(const LinearLayout &layout,
                                               ArrayRef<int32_t> digits,
                                               int32_t slab) {
    auto kReg = StringAttr::get(layout.getInDimNames().begin()->getContext(),
                                "register");
    ArrayRef<int32_t> radices = layout.getInputRadices(kReg);
    SmallVector<int32_t> registers;
    for (int32_t reg = 0; reg < layout.getInDimSize(kReg); ++reg) {
      SmallVector<int32_t> values;
      for (int32_t rest = reg, digit = 0; digit < int32_t(radices.size());
           rest /= radices[digit++])
        values.push_back(rest % radices[digit]);
      int32_t rest = slab;
      bool inSlab = true;
      for (int32_t digit : digits) {
        inSlab &= values[digit] == rest % radices[digit];
        rest /= radices[digit];
      }
      if (inSlab)
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
