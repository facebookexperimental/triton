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

struct WarpVoteOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::WarpVoteOp> {
  explicit WarpVoteOpConversion(LLVMTypeConverter &typeConverter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::WarpVoteOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::WarpVoteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicates =
        unpackLLElements(op.getLoc(), adaptor.getPred(), rewriter);
    if (predicates.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "warp_vote predicate must have one element per lane");

    int warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    Type hardwareMaskType = rewriter.getIntegerType(warpSize);
    Value mask = targetInfo.ballot(rewriter, op.getLoc(), hardwareMaskType,
                                   predicates.front());
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value expected = b.int_val(warpSize, op.getKind() == "all" ? -1 : 0);
    Value result = op.getKind() == "all" ? b.icmp_eq(mask, expected)
                                         : b.icmp_ne(mask, expected);
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
  patterns.add<WarpVoteOpConversion>(typeConverter, targetInfo, benefit);
}
