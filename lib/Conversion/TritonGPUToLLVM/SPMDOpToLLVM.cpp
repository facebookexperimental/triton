#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = targetInfo.programId(
        rewriter, op->getLoc(), op->getParentOfType<ModuleOp>(), op.getAxis());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct WarpBallotOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::WarpBallotOp> {
  explicit WarpBallotOpConversion(LLVMTypeConverter &typeConverter,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::WarpBallotOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::WarpBallotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicates =
        unpackLLElements(op.getLoc(), adaptor.getPred(), rewriter);
    if (predicates.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "warp_ballot predicate must have one element per lane");

    int warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    Type hardwareMaskType = rewriter.getIntegerType(warpSize);
    Value result = targetInfo.ballot(rewriter, op.getLoc(), hardwareMaskType,
                                     predicates.front());
    if (warpSize < 64) {
      TritonLLVMOpBuilder b(op.getLoc(), rewriter);
      result = b.zext(rewriter.getI64Type(), result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<WarpBallotOpConversion>(typeConverter, targetInfo, benefit);
}
