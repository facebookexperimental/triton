#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter,
                   const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto queuing = funcOp->getAttrOfType<BoolAttr>("amdg.enable_wmma_queuing");
    bool enableQueuing = triton::isKernel(funcOp) && !funcOp.isExternal() &&
                         queuing && queuing.getValue();
    if (enableQueuing && targetInfo.getArch() != "gfx1250")
      return funcOp.emitError(
          "enable_wmma_queuing requires an AMD gfx1250 target");

    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    newFuncOp->removeAttr("amdg.enable_wmma_queuing");
    handleArgPtrDatatype(funcOp, newFuncOp);

    if (triton::isKernel(funcOp)) {
      newFuncOp.setLinkage(LLVM::Linkage::External);
      if (enableQueuing) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
        // HW_REG_WAVE_SCHED_MODE is register 26. Write only bit 2, preserving
        // bits 0..1 used by LLVM's expert scheduling mode.
        constexpr unsigned hwreg = 26 | (2 << 6);
        Value reg = LLVM::createConstantI32(funcOp.getLoc(), rewriter, hwreg);
        Value value = LLVM::createConstantI32(funcOp.getLoc(), rewriter, 1);
        LLVM::createLLVMIntrinsicCallOp(rewriter, funcOp.getLoc(),
                                        "llvm.amdgcn.s.setreg", {},
                                        {reg, value});
      }
    } else {
      SmallVector<Attribute> passthroughAttrs = {
          rewriter.getStringAttr("noinline"),
          rewriter.getStringAttr("convergent")};
      newFuncOp.setPassthroughAttr(ArrayAttr::get(ctx, passthroughAttrs));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
      if (Attribute numWarps = funcOp->getAttr(triton::gpu::AttrNumWarpsName))
        newFuncOp->setAttr("ws_num_warps", numWarps);
    }

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, targetInfo, benefit);
}
