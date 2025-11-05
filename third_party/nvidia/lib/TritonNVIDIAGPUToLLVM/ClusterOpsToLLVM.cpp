/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ClusterArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto unitAttr = UnitAttr::get(ctx);
    if (op.getRelaxed()) {
      rewriter.replaceOpWithNewOp<NVVM::ClusterArriveRelaxedOp>(op, unitAttr);
    } else {
      rewriter.replaceOpWithNewOp<NVVM::ClusterArriveOp>(op, unitAttr);
    }
    return success();
  }
};

struct ClusterWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    rewriter.replaceOpWithNewOp<NVVM::ClusterWaitOp>(op, UnitAttr::get(ctx));
    return success();
  }
};

// lower MapToRemoteBufferOp
struct MapToRemoteBufferOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::MapToRemoteBufferOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::MapToRemoteBufferOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::MapToRemoteBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(op.getSrc().getType().getElementType()),
        rewriter);
    auto srcSmemPtr = srcSmemObj.getBase();

    auto ptrTy = cast<LLVM::LLVMPointerType>(srcSmemPtr.getType());
    assert(ptrTy.getAddressSpace() == 3 &&
           "Invalid src llvm addr space for MapToRemoteBufferOp");

    // The result pointer is referring to a memory buffer living in a CTA
    // cluster, so it has a different memory space. NVVM::MapaOp verifies its
    // src and result ptr type, so we need to construct the result ptr type
    // from typeConverter output here
    LLVM::LLVMStructType convertedRetTy =
        cast<LLVM::LLVMStructType>(typeConverter->convertType(op.getType()));
    Type convertedPtrTy = convertedRetTy.getBody()[0];

    // map an SMEM ptr in mem space 3 to a ptr in mem space 7
    auto remotePtr = rewriter.create<NVVM::MapaOp>(
        loc, convertedPtrTy, srcSmemPtr, adaptor.getCtaRank());

    // everything stays the same except base ptr comparing to srcSmemObj
    auto dstSmemObj = SharedMemoryObject(
        remotePtr, srcSmemObj.getBaseElemType(), srcSmemObj.getOffsets());
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateClusterOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ClusterArriveOpConversion>(typeConverter, benefit);
  patterns.add<ClusterWaitOpConversion>(typeConverter, benefit);
  patterns.add<MapToRemoteBufferOpConversion>(typeConverter, benefit);
  return;
}
