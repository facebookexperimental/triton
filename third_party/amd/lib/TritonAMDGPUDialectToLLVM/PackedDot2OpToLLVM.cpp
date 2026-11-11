#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct PackedDot2OpPattern : ConvertOpToLLVMPattern<amdgpu::PackedDot2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::PackedDot2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> lhs =
        unpackUniqueTensorElements(loc, adaptor.getLhs(), rewriter);
    SmallVector<Value> rhs =
        unpackUniqueTensorElements(loc, adaptor.getRhs(), rewriter);
    SmallVector<Value> acc =
        unpackUniqueTensorElements(loc, adaptor.getAcc(), rewriter);
    assert(lhs.size() == rhs.size() && lhs.size() == acc.size());

    bool isBF16 = op.getInputType().isBF16();
    assert((isBF16 || op.getInputType().isF16()) &&
           "packed dot2 input type must be FP16 or BF16");
    Type packedInput = vec_ty(isBF16 ? bf16_ty : f16_ty, 2);
    StringRef intrinsic =
        isBF16 ? "llvm.amdgcn.fdot2.f32.bf16" : "llvm.amdgcn.fdot2";

    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> results;
    results.reserve(lhs.size());
    for (auto [lhsValue, rhsValue, accValue] : llvm::zip_equal(lhs, rhs, acc)) {
      Value dot =
          LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, f32_ty,
                                          {b.bitcast(lhsValue, packedInput),
                                           b.bitcast(rhsValue, packedInput),
                                           accValue, b.false_val()})
              .getResult(0);
      results.push_back(dot);
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populatePackedDot2OpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<PackedDot2OpPattern>(typeConverter, benefit);
}
