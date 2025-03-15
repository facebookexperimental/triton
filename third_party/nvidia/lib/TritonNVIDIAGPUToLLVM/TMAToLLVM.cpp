#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
constexpr int64_t TMA_SIZE_BYTES = 128;

// Retrieve the local thread ID that is aligned with the warp group boundary.
Value getLocalThreadId(ConversionPatternRewriter &rewriter, Operation *op) {
  auto mod = op->getParentOfType<ModuleOp>();
  Location loc = op->getLoc();
  Value threadId = getThreadId(rewriter, loc);
  if (Attribute attr = mod->getAttr("triton_gpu.num-warp-groups-per-cta")) {
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    threadId = urem(threadId, i32_val(numWarps * warpSize));
  }
  return threadId;
}

void tensormap_cp_fenceproxy(Operation *op, MLIRContext *ctx,
                             ConversionPatternRewriter &rewriter, Value outPtr,
                             Value inPtr) {
  PTXBuilder ptxBuilder;
  Location loc = op->getLoc();

  // prepare asm operands
  auto *outAddrOpr = ptxBuilder.newAddrOperand(outPtr, "l");
  auto *inAddrOpr = ptxBuilder.newAddrOperand(inPtr, "l");
  auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

  // Define the instruction opcode
  auto &cp =
      *ptxBuilder.create<>("tensormap.cp_fenceproxy.global.shared::cta."
                           "tensormap::generic.release.gpu.sync.aligned");

  // Execute collectively on first warp in block
  constexpr int kWarpSize = 32;
  Value threadId = getLocalThreadId(rewriter, op);
  Value pred = icmp_slt(threadId, i32_val(kWarpSize));
  cp(outAddrOpr, inAddrOpr, sizeOpr).predicate(pred);

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
};

void tensormap_replace_generic(Operation *op, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               std::string fieldName, Value descPtr,
                               int32_t newVal) {
  PTXBuilder ptxBuilder;
  Location loc = op->getLoc();
  // prepare asm operands
  auto *descAddrOpr = ptxBuilder.newAddrOperand(descPtr, "l");
  auto newValOpr = ptxBuilder.newConstantOperand(newVal);

  // Define the instruction opcode
  auto &replace = ptxBuilder.create<>("tensormap.replace.tile")
                      ->o(fieldName)
                      .o("shared::cta")
                      .o("b1024")
                      .o("b32");

  Value threadId = getLocalThreadId(rewriter, op);
  Value pred = icmp_eq(threadId, i32_val(0));
  replace(descAddrOpr, newValOpr).predicate(pred);

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
}

void tensormap_replace_generic(Operation *op, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               std::string fieldName, Value descPtr,
                               Value newVal,
                               std::optional<int32_t> ord = std::nullopt) {
  PTXBuilder ptxBuilder;
  Location loc = op->getLoc();

  auto newValTy = newVal.getType();
  int width = 0;

  // prepare asm operands
  auto *descAddrOpr = ptxBuilder.newAddrOperand(descPtr, "l");
  PTXInstr::Operand *ordOpr =
      ord ? ptxBuilder.newConstantOperand(*ord) : nullptr;
  PTXInstr::Operand *newValOpr = nullptr;
  if (mlir::isa<IntegerType>(newValTy)) {
    width = mlir::cast<IntegerType>(newValTy).getWidth();
  } else {
    assert(mlir::isa<mlir::LLVM::LLVMPointerType>(newValTy));
    width = 64;
  }
  const char *constraint = width == 64 ? "l" : "r";
  newValOpr = ptxBuilder.newOperand(newVal, constraint);

  // Define the instruction opcode
  auto &replace = ptxBuilder.create<>("tensormap.replace.tile")
                      ->o(fieldName)
                      .o("shared::cta")
                      .o("b1024")
                      .o("b32", width == 32)
                      .o("b64", width == 64);

  Value threadId = getLocalThreadId(rewriter, op);
  Value pred = icmp_eq(threadId, i32_val(0));

  if (ord) {
    replace(descAddrOpr, ordOpr, newValOpr).predicate(pred);
  } else {
    replace(descAddrOpr, newValOpr).predicate(pred);
  }

  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
}

void tensormap_replace_global_address(Operation *op, MLIRContext *ctx,
                                      ConversionPatternRewriter &rewriter,
                                      Value descPtr, Value newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "global_address", descPtr,
                            newVal);
}

void tensormap_replace_rank(Operation *op, MLIRContext *ctx,
                            ConversionPatternRewriter &rewriter, Value descPtr,
                            int32_t newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "rank", descPtr, newVal);
}

void tensormap_replace_box_dim(Operation *op, MLIRContext *ctx,
                               ConversionPatternRewriter &rewriter,
                               Value descPtr, int32_t ord, Value newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "box_dim", descPtr, newVal, ord);
}

void tensormap_replace_global_dim(Operation *op, MLIRContext *ctx,
                                  ConversionPatternRewriter &rewriter,
                                  Value descPtr, int32_t ord, Value newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "global_dim", descPtr, newVal,
                            ord);
}

void tensormap_replace_global_stride(Operation *op, MLIRContext *ctx,
                                     ConversionPatternRewriter &rewriter,
                                     Value descPtr, int32_t ord, Value newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "global_stride", descPtr, newVal,
                            ord);
}

void tensormap_replace_element_stride(Operation *op, MLIRContext *ctx,
                                      ConversionPatternRewriter &rewriter,
                                      Value descPtr, int32_t ord,
                                      Value newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "element_stride", descPtr,
                            newVal, ord);
}

void tensormap_replace_elemtype(Operation *op, MLIRContext *ctx,
                                ConversionPatternRewriter &rewriter,
                                Value descPtr, int32_t newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "elemtype", descPtr, newVal);
}

void tensormap_replace_interleave_layout(Operation *op, MLIRContext *ctx,
                                         ConversionPatternRewriter &rewriter,
                                         Value descPtr, int32_t newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "interleave_layout", descPtr,
                            newVal);
}

void tensormap_replace_swizzle_mode(Operation *op, MLIRContext *ctx,
                                    ConversionPatternRewriter &rewriter,
                                    Value descPtr, int32_t newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "swizzle_mode", descPtr, newVal);
}

void tensormap_replace_fill_mode(Operation *op, MLIRContext *ctx,
                                 ConversionPatternRewriter &rewriter,
                                 Value descPtr, int32_t newVal) {
  tensormap_replace_generic(op, ctx, rewriter, "fill_mode", descPtr, newVal);
}

struct ExperimentalTensormapFenceproxyAcquireOpConversion
    : public ConvertOpToLLVMPattern<
          triton::ExperimentalTensormapFenceproxyAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ExperimentalTensormapFenceproxyAcquireOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    PTXBuilder ptxBuilder;

    // prepare asm operands
    auto *descAddrOpr = ptxBuilder.newAddrOperand(adaptor.getDescPtr(), "l");
    auto *sizeOpr = ptxBuilder.newConstantOperand(TMA_SIZE_BYTES);

    // Define the instruction opcode
    int kWarpSize = 32;
    Value threadId = getLocalThreadId(rewriter, op);
    Value pred = icmp_slt(threadId, i32_val(kWarpSize));
    auto &fence =
        *ptxBuilder.create<>("fence.proxy.tensormap::generic.acquire.gpu");
    fence(descAddrOpr, sizeOpr).predicate(pred);

    ptxBuilder.launch(rewriter, loc, getVoidType());

    // We run the fence on a single warp, then use a barrier to synchronize the
    // rest. This ends up being faster than running the fence on each warp.
    // TODO: Ideally we only emit one barrier after all fences are issued
    insertBarrier(rewriter, op);

    rewriter.eraseOp(op);
    return success();
  }
};

void zero_fill_tma(Operation *op, MLIRContext *ctx,
                   ConversionPatternRewriter &rewriter,
                   const NVIDIA::TargetInfo &targetInfo, Value descPtr) {
  Location loc = op->getLoc();
  // Write out zeros
  constexpr int kWarpSize = 32;
  Value threadId = getLocalThreadId(rewriter, op);
  Value pred = icmp_slt(threadId, i32_val(kWarpSize));

  auto fillVal = i32_val(0);
  auto writeAddr = gep(descPtr.getType(), fillVal.getType(), descPtr, threadId);
  targetInfo.storeShared(rewriter, loc, writeAddr, fillVal, pred);

  // Sync warp
  PTXBuilder ptxBuilder;
  auto &bar = *ptxBuilder.create<>("bar.warp.sync");
  auto *maskOpr = ptxBuilder.newConstantOperand(0xffffffff);
  bar(maskOpr).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(ctx));
}

struct ExperimentalTensormapCreateOpConversion
    : public ConvertOpToLLVMPattern<ExperimentalTensormapCreateOp> {
  const NVIDIA::TargetInfo &targetInfo;

  ExperimentalTensormapCreateOpConversion(LLVMTypeConverter &converter,
                                          const NVIDIA::TargetInfo &targetInfo,
                                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ExperimentalTensormapCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = getContext();

    auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

    zero_fill_tma(op, ctx, rewriter, targetInfo, smemBase);
    tensormap_replace_global_address(op, ctx, rewriter, smemBase,
                                     adaptor.getGlobalAddress());
    tensormap_replace_rank(op, ctx, rewriter, smemBase, op.getRank() - 1);
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_box_dim(op, ctx, rewriter, smemBase, i,
                                op.getBoxDim()[i]);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_global_dim(op, ctx, rewriter, smemBase, i,
                                   op.getGlobalDim()[i]);
    }
    for (int i = 0; i + 1 < op.getRank(); ++i) {
      tensormap_replace_global_stride(op, ctx, rewriter, smemBase, i,
                                      op.getGlobalStride()[i]);
    }
    for (int i = 0; i < op.getRank(); ++i) {
      tensormap_replace_element_stride(op, ctx, rewriter, smemBase, i,
                                       op.getElementStride()[i]);
    }
    tensormap_replace_elemtype(op, ctx, rewriter, smemBase, op.getElemType());
    tensormap_replace_interleave_layout(op, ctx, rewriter, smemBase,
                                        op.getInterleaveLayout());
    tensormap_replace_swizzle_mode(op, ctx, rewriter, smemBase,
                                   op.getSwizzleMode());
    tensormap_replace_fill_mode(op, ctx, rewriter, smemBase, op.getFillMode());
    tensormap_cp_fenceproxy(op, ctx, rewriter, adaptor.getDescPtr(), smemBase);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTMAToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ExperimentalTensormapCreateOpConversion>(typeConverter,
                                                        targetInfo, benefit);
  patterns.add<ExperimentalTensormapFenceproxyAcquireOpConversion>(
      typeConverter, benefit);
}
