#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "Utility.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
// using ::mlir::triton::gpu::SharedEncodingAttr;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace {

Value getBarrierField(triton::TritonLLVMOpBuilder builder,
                      SharedMemoryObject barrierSmemObj, int fieldIndex) {
  OpBuilder b = *builder.builder;
  auto i32ty = b.getIntegerType(32);
  auto baseAddr = barrierSmemObj.getBase();
  return builder.gep(baseAddr.getType(), i32ty, baseAddr,
                     builder.i32_val(fieldIndex));
}

Value getPhaseBaseAddress(TritonLLVMOpBuilder builder,
                          SharedMemoryObject barrierSmemObj) {
  return getBarrierField(builder, barrierSmemObj, 1);
}

Value getCountBaseAddress(TritonLLVMOpBuilder builder,
                          SharedMemoryObject barrierSmemObj) {
  return getBarrierField(builder, barrierSmemObj, 0);
}

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    int count = op.getCount();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto countBaseAddr = getCountBaseAddress(b, barrierSmemObj);
    auto phaseBaseAddr = getPhaseBaseAddress(b, barrierSmemObj);

    // Set countVal to count -1 because we use DS_DEC_RTN which does count -= 1
    // and wraps around when post dec value reaches -1. For example,
    // initializing count to 2 will allow 3 arrives (2->1->0->-1) before the
    // value gets reset to 2
    Value countVal = b.i32_val(count - 1);
    Value phaseVal = b.i32_val(0);
    b.store(countVal, countBaseAddr);
    b.store(phaseVal, phaseBaseAddr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto count = op.getCount();

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Use the AMDGCN barrier arrive intrinsic
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64", i64_ty,
        {barrierSmemObj.getBase(), b.i64_val(count)});

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReadBarrierPhaseOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ReadBarrierPhaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ReadBarrierPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto barrierSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto countBaseAddr = barrierSmemObj.getBase();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto phaseBaseAddr = getPhaseBaseAddress(b, barrierSmemObj);

    auto res = b.load(i32_ty, phaseBaseAddr);

    GCNBuilder phaseReadBuilder;
    MLIRContext *ctx = rewriter.getContext();
    auto &wait_cnt = *phaseReadBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    phaseReadBuilder.launch(rewriter, loc, void_ty(ctx),
                            true /*hasSideEffects*/);
    rewriter.replaceOp(op, res);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     const TargetInfo &targetInfo,
                                     RewritePatternSet &patterns,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     PatternBenefit benefit) {
  patterns.add<InitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<ReadBarrierPhaseOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
