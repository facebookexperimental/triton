#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonCPUOpToLLVM.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <cassert>

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const CPU::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // CPUs can have a sufficiently large vector size. Nvidia GPUs have 128.
    assert(triton::gpu::isCPUMode());
    return std::min<unsigned>(65536 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const CPU::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const CPU::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned vecOrig = vec;
    if (llMask) {
      LLVM_DEBUG(DBGS() << "vec = " << vec
                        << " mask_alignment = " << getMaskAlignment(mask));
      // Note: PTX's predicate register is 1 bit, even for a vectorized load.
      // So, the constancy of the mask limits the vectorization size in PTX.
      // But, AVX512 has wider mask registers. We don't need to limit the
      // vectorization size like GPU. (Hopefully, I'm correct.)
      if (!triton::gpu::isCPUMode())
        vec = std::min<size_t>(vec, getMaskAlignment(mask));
      LLVM_DEBUG(llvm::dbgs() << " vec = " << vec << '\n');
    }

    if (vec == 1 && numElems > 1) {
      int maskValue = !llMask ? -1 : getMaskAlignment(mask);
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " numElems = " << numElems << " mask is " << maskValue
                       << "\n";
    }
    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits << " "
                              << op.getType());
    SmallVector<Value> loadedVals;
    assert(triton::gpu::isCPUMode());
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: if vec == 1, no need to use vector type?
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value vecLoadVal;
      if (mask) {
        // TODO: We should optimize back-to-back unpacking and packing.
        Value preds =
            packLLVectorRange(loc, maskElems, vecStart, vec, rewriter);

        mlir::Attribute zeroAttr = rewriter.getZeroAttr(valueElemTy);
        auto denseValue =
            DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
        Value zeroVal =
            rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseValue);

        Value falseVal = zeroVal;
        // If we need to mask the loaded value with other elements
        if (!otherElems.empty())
          falseVal =
              packLLVectorRange(loc, otherElems, vecStart, vec, rewriter);

        vecLoadVal = rewriter.create<LLVM::MaskedLoadOp>(
            loc, vecTy, ptrElems[vecStart], preds, falseVal, vec);
      } else {
        vecLoadVal =
            rewriter.create<LLVM::LoadOp>(loc, vecTy, ptrElems[vecStart]);
      }
      auto unpacked = unpackLLVector(loc, vecLoadVal, rewriter);
      loadedVals.append(unpacked);
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const CPU::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    unsigned vecOrig = vec;
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      if (!triton::gpu::isCPUMode()) {
        unsigned maskAlign = getMaskAlignment(mask);
        vec = std::min(vec, maskAlign);
      }
    }

    if (vec == 1 && elemsPerThread > 1) {
      int mask = !llMask ? -1 : getMaskAlignment(op.getMask());
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " elemsPerThread = " << elemsPerThread << " mask is "
                       << mask << "\n";
    }

    assert(triton::gpu::isCPUMode());
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      Value llWord =
          packLLVectorRange(loc, valueElems, vecStart, vec, rewriter);
      auto address = ptrElems[vecStart];
      if (llMask) {
        auto maskWord =
            packLLVectorRange(loc, maskElems, vecStart, vec, rewriter);
        rewriter.create<LLVM::MaskedStoreOp>(loc, llWord, address, maskWord,
                                             vec);
      } else {
        rewriter.create<LLVM::StoreOp>(loc, llWord, address);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Copied from AMD's.
LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

// Copied from TritonCPU fork.
LLVM::AtomicBinOp getAtomicBinOp(RMWOp op, Type type) {
  switch (op) {
  case RMWOp::AND:
    return LLVM::AtomicBinOp::_and;
  case RMWOp::OR:
    return LLVM::AtomicBinOp::_or;
  case RMWOp::XOR:
    return LLVM::AtomicBinOp::_xor;
  case RMWOp::ADD:
    return LLVM::AtomicBinOp::add;
  case RMWOp::FADD:
    return LLVM::AtomicBinOp::fadd;
  case RMWOp::MAX:
    return type.isIntOrIndex() ? LLVM::AtomicBinOp::max
                               : LLVM::AtomicBinOp::fmax;
  case RMWOp::MIN:
    return type.isIntOrIndex() ? LLVM::AtomicBinOp::min
                               : LLVM::AtomicBinOp::fmin;
  case RMWOp::UMAX:
    return LLVM::AtomicBinOp::umax;
  case RMWOp::UMIN:
    return LLVM::AtomicBinOp::umin;
  case RMWOp::XCHG:
    return LLVM::AtomicBinOp::xchg;
  default:
    llvm_unreachable("Unexpected atomic op");
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const CPU::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    auto vecOrig = vec;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    if (vec == 1 && elemsPerThread > 1)
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " elemsPerThread = " << elemsPerThread << "\n";

    assert(triton::gpu::isCPUMode());
    SmallVector<Value> resultVals(elemsPerThread);
    auto successOrdering = getMemoryOrdering(op.getSem());
    auto failureOrdering = successOrdering != LLVM::AtomicOrdering::monotonic
                               ? LLVM::AtomicOrdering::acquire
                               : successOrdering;
    // Obviously, x86 cmpxchg doesn't take SIMD registers. No vector types.
    Value oldVal;
    for (size_t i = 0; i < elemsPerThread; i++) {
      auto ptr = ptrElements[i];
      auto cmp = cmpElements[i];
      auto val = valElements[i];
      auto ret = rewriter.create<LLVM::AtomicCmpXchgOp>(
          loc, ptr, cmp, val, successOrdering, failureOrdering);
      oldVal = rewriter.create<LLVM::ExtractValueOp>(loc, ret, 0);
      if (tensorTy)
        resultVals[i] = oldVal;
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    } else {
      rewriter.replaceOp(op, oldVal);
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const CPU::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  bool supportsVectorized(RMWOp opType, Type elementType) const {
    // vectorized atomics are only supported on hopper,
    // and only for specific atomic ops (add, min, max).
    // Note that "packed types" like f16x2 are supported sm60+.
    if (!targetInfo.supportVectorizedAtomics()) {
      return false;
    }

    return opType == RMWOp::FADD &&
           (elementType.isF16() || elementType.isBF16() || elementType.isF32());
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // packed: e.g. packed=2 for f16x2
    // vec: e.g. .v2, .v4, .v8 version of atom instruction.
    unsigned vec, vecOrig;
    int numElems, packed;
    if (tensorTy) {
      vec = getVectorSize(ptr);
      if (llMask) {
        vec = std::min<unsigned>(vec, getMaskAlignment(op.getMask()));
      }
      vecOrig = vec;
      packed = 1;
      auto valTy = cast<RankedTensorType>(val.getType());
      if (!supportsVectorized(atomicRmwAttr, valTy.getElementType())) {
        packed =
            std::min<unsigned>(vecOrig, valTy.getElementType().isF16() ? 2 : 1);
        vec = 1;
      }
      numElems = tensorTy.getNumElements();
    } else {
      // scalar
      vec = 1;
      vecOrig = 1;
      numElems = 1;
      packed = 1;
    }
    assert((packed == 1 || vec == 1) && "packed or vec must be 1");

    if (vec * packed == 1 && numElems > 1)
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " packed = " << packed << " origin vec = " << vecOrig
                       << " numElems = " << numElems;

    assert(triton::gpu::isCPUMode());
    Value ret;
    SmallVector<Value> resultVals(elemsPerThread);
    auto ordering = getMemoryOrdering(op.getSem());
    auto binOp = getAtomicBinOp(atomicRmwAttr, valueTy);
    auto scope = stringifyMemSyncScope(op.getScope()).str();
    // TODO: llvm.atomicrmw takes vectors (really?) for fadd/fsub/fmax/fmin.
    // We could try it to see generated x86/ARM code?
    for (size_t i = 0; i < elemsPerThread; i++) {
      auto ptr = ptrElements[i];
      auto val = valElements[i];
      auto mask = llMask ? maskElements[i] : llMask;
      ret = lowerScalarMaskToCF(loc, binOp, ptr, val, mask, ordering, scope,
                                rewriter);
      // TODO: Not sure we need to do this?
      if (tensorTy)
        resultVals[i] = ret;
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    } else {
      rewriter.replaceOp(op, ret);
    }
    return success();
  }

  // Based on the CPU fork.
  Value lowerScalarMaskToCF(Location loc, LLVM::AtomicBinOp binOp, Value ptr,
                            Value val, Value mask,
                            LLVM::AtomicOrdering ordering, StringRef scope,
                            ConversionPatternRewriter &rewriter) const {
    // Check for constant mask.
    // TODO: Is it necessary? Can compiler optimize it?
    if (auto maskDef = mask.getDefiningOp<LLVM::ConstantOp>()) {
      auto maskVal = cast<IntegerAttr>(maskDef.getValue());
      if (maskVal.getValue().isZero()) {
        return rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getZeroAttr(val.getType()));
      } else {
        return rewriter.create<LLVM::AtomicRMWOp>(loc, binOp, ptr, val,
                                                  ordering, scope);
      }
    }

    // There's no predicate support for x86 atomic ops. Need a branch.
    Block *headerBlock = rewriter.getBlock();
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getZeroAttr(val.getType()));
    Block *condBlock =
        rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(condBlock);
    Value resVal = rewriter.create<LLVM::AtomicRMWOp>(loc, binOp, ptr, val,
                                                      ordering, scope);
    Block *footerBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    Value res = footerBlock->addArgument(resVal.getType(), resVal.getLoc());
    rewriter.setInsertionPointToEnd(headerBlock);
    rewriter.create<cf::CondBranchOp>(loc, mask, condBlock, footerBlock, zero);
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cf::BranchOp>(loc, footerBlock, resVal);
    rewriter.setInsertionPointToStart(footerBlock);
    return res;
  }
};

} // namespace

void mlir::triton::CPU::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                  benefit);
}
