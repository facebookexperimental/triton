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

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
static void createClusterArrive(OpBuilder &b, Location loc, bool relaxed) {
  auto unitAttr = UnitAttr::get(b.getContext());
  if (relaxed)
    NVVM::ClusterArriveRelaxedOp::create(b, loc, unitAttr);
  else
    NVVM::ClusterArriveOp::create(b, loc, unitAttr);
}

static void createClusterWait(OpBuilder &b, Location loc) {
  NVVM::ClusterWaitOp::create(b, loc, UnitAttr::get(b.getContext()));
}

template <typename EmitFn>
LogicalResult lowerClusterSyncForAllWarps(Operation *op,
                                          ConversionPatternRewriter &rewriter,
                                          EmitFn emit) {
  auto loc = op->getLoc();
  auto mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return rewriter.notifyMatchFailure(op, "expected parent module");

  // Emit the cluster sync directly (no warp_specialize wrapping) when either:
  //  (a) it is already inside a warp_specialize region (e.g. the user placed
  //      cluster_barrier() inside each async_task so every warp group
  //      participates). The warp split already happened, so every warp of the
  //      current region executes it; wrapping again would form an illegal
  //      nested ttg.warp_specialize. OR
  //  (b) the kernel carries the cluster-sync-kernel-init marker -- the
  //      entry-block cluster sync inserted by maybeInsertClusterSync. For that
  //      sync, ConvertWarpSpecializeToLLVM already emits the non-default warps'
  //      "@!isDefault barrier.cluster.arrive" at WS entry. Wrapping here too
  //      would double-count worker participation: the extra worker arrive
  //      imbalances the cluster barrier (deadlock) and adds a spurious worker
  //      wait.
  if (op->getParentOfType<triton::gpu::WarpSpecializeOp>() ||
      mlir::triton::tlx::hasClusterSyncKernelInit(op)) {
    rewriter.setInsertionPoint(op);
    emit(rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  auto defaultNumWarps = triton::gpu::maybeLookupNumWarps(op);
  if (!defaultNumWarps)
    return rewriter.notifyMatchFailure(op, "missing contextual num-warps");
  int totalNumWarps = *defaultNumWarps;
  if (auto totalNumWarpsAttr =
          mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    totalNumWarps = totalNumWarpsAttr.getInt();
  int workerNumWarps = totalNumWarps - *defaultNumWarps;
  if (workerNumWarps < 0)
    return rewriter.notifyMatchFailure(op, "invalid total/default num-warps");

  rewriter.setInsertionPoint(op);
  if (workerNumWarps == 0) {
    emit(rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<int32_t> partitionNumWarps;
  for (int remainingWarps = workerNumWarps; remainingWarps > 0;) {
    int32_t partitionWarps =
        llvm::bit_floor(static_cast<uint32_t>(remainingWarps));
    partitionNumWarps.push_back(partitionWarps);
    remainingWarps -= partitionWarps;
  }

  auto wsOp = triton::gpu::WarpSpecializeOp::create(rewriter, loc, TypeRange{},
                                                    partitionNumWarps);
  SmallVector<int32_t> startIds;
  int startId = *defaultNumWarps;
  for (int32_t partitionWarps : partitionNumWarps) {
    startIds.push_back(startId);
    startId += partitionWarps;
  }
  wsOp.setWarpGroupStartIds(startIds);

  Block *defaultBlock = rewriter.createBlock(&wsOp.getDefaultRegion());
  rewriter.setInsertionPointToEnd(defaultBlock);
  emit(rewriter);
  triton::gpu::WarpYieldOp::create(rewriter, loc, TypeRange(), ValueRange());

  Block *partitionHolder = rewriter.createBlock(&wsOp.getPartitionOpHolder());
  rewriter.setInsertionPointToStart(partitionHolder);
  auto partitions = triton::gpu::WarpSpecializePartitionsOp::create(
      rewriter, loc, ValueRange(),
      /*numPartitionRegions=*/partitionNumWarps.size());
  for (Region &partitionRegion : partitions.getPartitionRegions()) {
    Block *partitionBlock = rewriter.createBlock(&partitionRegion);
    rewriter.setInsertionPointToEnd(partitionBlock);
    emit(rewriter);
    triton::gpu::WarpReturnOp::create(rewriter, loc);
  }

  rewriter.eraseOp(op);
  return success();
}

struct ClusterArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerClusterSyncForAllWarps(op, rewriter, [&](OpBuilder &b) {
      createClusterArrive(b, op.getLoc(), op.getRelaxed());
    });
  }
};

struct ClusterWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerClusterSyncForAllWarps(
        op, rewriter, [&](OpBuilder &b) { createClusterWait(b, op.getLoc()); });
  }
};

struct ClusterBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterBarrierOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerClusterSyncForAllWarps(op, rewriter, [&](OpBuilder &b) {
      createClusterArrive(b, op.getLoc(), op.getRelaxed());
      createClusterWait(b, op.getLoc());
    });
  }
};

struct ClusterSize1DOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterSize1DOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterSize1DOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterSize1DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<NVVM::ClusterDim>(op, i32_ty);
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
    auto remotePtr = NVVM::MapaOp::create(rewriter, loc, convertedPtrTy,
                                          srcSmemPtr, adaptor.getCtaRank());

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
  patterns.add<ClusterSize1DOpConversion>(typeConverter, benefit);
  patterns.add<MapToRemoteBufferOpConversion>(typeConverter, benefit);
  patterns.add<ClusterBarrierOpConversion>(typeConverter, benefit);
  return;
}
