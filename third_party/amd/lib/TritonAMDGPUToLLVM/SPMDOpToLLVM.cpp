#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        ::mlir::gpu::GridDimOp::create(rewriter, loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct Clock64OpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::Clock64Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::Clock64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto readClock = LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                                     "llvm.amdgcn.s.memtime",
                                                     rewriter.getI64Type(), {});
    rewriter.replaceOp(op, readClock.getResult(0));
    return success();
  }
};

struct CondBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::CondBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::CondBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCondBarBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, adaptor.getPred(), trueBlock,
                           afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    ROCDL::SBarrierOp::create(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AssumeUniformOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::AssumeUniformOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::AssumeUniformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value src = adaptor.getSrc();
    Value asInt = b.ptrtoint(i64_ty, src);
    Value uniform =
        ROCDL::ReadfirstlaneOp::create(rewriter, loc, i64_ty, asInt);
    rewriter.replaceOp(op, b.inttoptr(src.getType(), uniform));
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<Clock64OpConversion>(typeConverter, benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
  patterns.add<AssumeUniformOpConversion>(typeConverter, benefit);
}
