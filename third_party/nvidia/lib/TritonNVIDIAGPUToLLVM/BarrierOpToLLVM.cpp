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
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "Utility.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {
struct FenceAsyncSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = NVVM::ProxyKind::async_shared;
    auto space = op.getBCluster() ? NVVM::SharedSpace::shared_cluster
                                  : NVVM::SharedSpace::shared_cta;
    auto ctx = rewriter.getContext();
    auto spaceAttr = NVVM::SharedSpaceAttr::get(ctx, space);
    rewriter.replaceOpWithNewOp<NVVM::FenceProxyOp>(op, kind, spaceAttr);
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    auto pred = b.icmp_eq(id, b.i32_val(0));
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                            std::to_string(op.getCount()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InvalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InvalBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InvalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierExpectConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::BarrierExpectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::BarrierExpectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto id = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(id, b.i32_val(0));
    pred = b.and_(pred, adaptor.getPred());
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx =
        "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], " +
        std::to_string(op.getSize()) + ";";
    auto &barSyncOp = *ptxBuilder.create<>(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(smemObj.getBase(), "r")},
              /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WaitBarrierOp> {
  const NVIDIA::TargetInfo *targetInfo;
  WaitBarrierOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  static std::optional<int64_t> getTimeoutNs() {
    std::string envVal = tools::getStrEnv("TRITON_MBAR_WAIT_TIMEOUT_SEC");
    if (envVal.empty())
      return std::nullopt;
    // Env var is in seconds, convert to nanoseconds
    char *end;
    double seconds = std::strtod(envVal.c_str(), &end);
    if (end == envVal.c_str() || seconds < 0)
      return std::nullopt;
    return static_cast<int64_t>(seconds * 1e9);
  }

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto loc = op.getLoc();
    bool predicated =
        adaptor.getPred() && !matchPattern(op.getPred(), m_NonZero());

    std::optional<int64_t> timeoutNs = getTimeoutNs();
    bool useTimeout =
        timeoutNs.has_value() && targetInfo->getComputeCapability() >= 90;

    std::string ptx;
    if (targetInfo->getComputeCapability() < 90) {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
}
)";
      } else {
        ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    } else {
      if (useTimeout) {
        // Timeout path: trap on timeout
        std::string timeoutStr = std::to_string(*timeoutNs);
        if (!predicated) {
          ptx = R"(
{
	.reg .pred complete, timedOut;
	.reg .u64 startTime, currentTime, elapsed;
	mov.u64 startTime, %globaltimer;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@complete bra.uni done;
	mov.u64 currentTime, %globaltimer;
	sub.u64 elapsed, currentTime, startTime;
	setp.ge.u64 timedOut, elapsed, )" +
                timeoutStr + R"(;
	@timedOut trap;
	bra.uni waitLoop;
	done:
}
)";
        } else {
          ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete, timedOut;
	.reg .u64 startTime, currentTime, elapsed;
	mov.u64 startTime, %globaltimer;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@complete bra.uni done;
	mov.u64 currentTime, %globaltimer;
	sub.u64 elapsed, currentTime, startTime;
	setp.ge.u64 timedOut, elapsed, )" +
                timeoutStr + R"(;
	@timedOut trap;
	bra.uni waitLoop;
	done:
	skipWait:
}
)";
        }
      } else {
        // No timeout
        if (!predicated) {
          ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
}
)";
        } else {
          ptx = R"(
{
	@!$2 bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
        }
      }
    }

    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create<>(ptx);
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 4> operands = {
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};
    if (predicated)
      operands.push_back(ptxBuilder.newOperand(adaptor.getPred(), "b"));

    waitLoop(operands, /*onlyAttachMLIRArgs=*/true);

    // If timeout is enabled, print debug info BEFORE executing the wait loop.
    // This way, if the kernel traps due to timeout, the last printed message
    // identifies which barrier caused the timeout.
    // Only print from thread 0 in each block to avoid flooding the output.
    if (useTimeout) {
      // Guard: only thread 0 prints
      Value tid = getThreadId(rewriter, loc);
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Value isThread0 = b.icmp_eq(tid, b.i32_val(0));

      // Create blocks for conditional printf
      Block *currentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();
      Block *continueBlock = currentBlock->splitBlock(insertPt);
      Block *printfBlock = rewriter.createBlock(
          currentBlock->getParent(), Region::iterator(continueBlock));

      // Conditional branch: if thread0, print; else skip
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<LLVM::CondBrOp>(loc, isThread0, printfBlock,
                                      continueBlock);

      // Printf block
      rewriter.setInsertionPointToStart(printfBlock);

      // Get block IDs for debug output
      Value blockIdX = rewriter.create<NVVM::BlockIdXOp>(loc, i32_ty);
      Value blockIdY = rewriter.create<NVVM::BlockIdYOp>(loc, i32_ty);
      Value blockIdZ = rewriter.create<NVVM::BlockIdZOp>(loc, i32_ty);

      // Convert barrier address (ptr) to i64 for printing
      Value barAddrInt =
          rewriter.create<LLVM::PtrToIntOp>(loc, i64_ty, smemObj.getBase());

      // Convert phase to i32 for printing (if it's not already)
      Value phase = adaptor.getPhase();
      if (phase.getType() != i32_ty) {
        phase = rewriter.create<LLVM::ZExtOp>(loc, i32_ty, phase);
      }

      // Print debug info identifying this barrier wait
      targetInfo->printf(
          rewriter,
          "MBAR_WAIT: addr=0x%llx phase=%d block=(%d,%d,%d) timeout_sec=%lld",
          {barAddrInt, phase, blockIdX, blockIdY, blockIdZ,
           rewriter.create<LLVM::ConstantOp>(
               loc, i64_ty,
               rewriter.getI64IntegerAttr(*timeoutNs / 1000000000LL))},
          {/*isSigned=*/false, true, true, true, true, true});

      // Branch to continue
      rewriter.create<LLVM::BrOp>(loc, continueBlock);

      // Continue in continueBlock
      rewriter.setInsertionPointToStart(continueBlock);
    }

    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool isRemoteBarrier = false;
    if (auto barType = dyn_cast<ttg::MemDescType>(op.getAlloc().getType())) {
      isRemoteBarrier =
          isa<ttng::SharedClusterMemorySpaceAttr>(barType.getMemorySpace());
    }

    // TODO: Add phase result as needed.
    std::stringstream ptxAsm;
    ptxAsm << "@$0 mbarrier.arrive.shared::";
    if (isRemoteBarrier)
      ptxAsm << "cluster";
    else
      ptxAsm << "cta";
    ptxAsm << ".b64 _, [$1]";
    if (op.getCount() > 1) {
      ptxAsm << ", " << op.getCount();
    }
    ptxAsm << ";";

    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value id = getThreadId(rewriter, op.getLoc());
    Value pred = b.icmp_eq(id, b.i32_val(0));
    if (op.getPred())
      pred = b.and_(pred, adaptor.getPred());

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 2> operands = {
        ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(adaptor.getAlloc(), "r")};

    auto arriveOp = *ptxBuilder.create<>(ptxAsm.str());
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct NamedBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    std::string ptxAsm = "bar.arrive $0, $1;";

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 2> operands = {
        ptxBuilder.newOperand(adaptor.getBar(), "r"),
        ptxBuilder.newOperand(adaptor.getNumThreads(), "r")};

    auto arriveOp = *ptxBuilder.create<>(ptxAsm);
    arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct NamedBarrierWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::NamedBarrierWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::NamedBarrierWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::NamedBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    std::string ptxAsm = "bar.sync $0, $1;";

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 2> operands = {
        ptxBuilder.newOperand(adaptor.getBar(), "r"),
        ptxBuilder.newOperand(adaptor.getNumThreads(), "r")};

    auto waitOp = *ptxBuilder.create<>(ptxAsm);
    waitOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCLCTryCancelOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::AsyncCLCTryCancelOp> {
  // TODO. check target infor for compute capability >= 100
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::AsyncCLCTryCancelOp>::ConvertOpToLLVMPattern;

  // clc response is 16-byte opaque object available at the location specified
  // by the 16-byte wide shared memory address (i.e. 1st operand of PTX inst)
  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncCLCTryCancelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto tid = getThreadId(rewriter, loc);
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value pred = b.icmp_eq(tid, b.i32_val(0));

    std::string ptx = R"(
    {
      .reg .u32 first_cta_in_cluster;
      .reg .pred pred_first_cta_in_cluster;
      .reg .pred pred_issue;
      mov.u32  first_cta_in_cluster, %cluster_ctaid.x;
      setp.u32.eq pred_first_cta_in_cluster, first_cta_in_cluster, 0x0;
      and.pred pred_issue, $2, pred_first_cta_in_cluster;
      @pred_issue clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [$0], [$1];
    }
    )";

    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *, 3> operands = {
        ptxBuilder.newOperand(adaptor.getClcResAlloc(), "r"),
        ptxBuilder.newOperand(adaptor.getMbarAlloc(), "r"),
        ptxBuilder.newOperand(pred, "b")};

    auto clcOp = *ptxBuilder.create<>(ptx);
    clcOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, op.getLoc(), voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct CLCQueryCancelOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCQueryCancelOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::CLCQueryCancelOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCQueryCancelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    TritonLLVMOpBuilder b(op.getLoc(), rewriter);

    std::string ptx = R"(
    {
      .reg .b128 clc_result;
      .reg .pred p1;
      mov.s32 $0, -1;
      ld.shared.b128 clc_result, [$1];
      clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
      @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, _, _, _}, clc_result;
    }
    )";

    PTXBuilder builder;
    auto queryOp = *builder.create<>(ptx);

    SmallVector<PTXBuilder::Operand *, 2> operands = {
        builder.newOperand("=r", true),
        builder.newOperand(adaptor.getClcResAlloc(), "r")};
    queryOp(operands, /*onlyAttachMLIRArgs=*/true);

    Value ctaId = builder.launch(rewriter, op.getLoc(), i32_ty, false);

    rewriter.replaceOp(op, ctaId);

    return success();
  }
};

struct VoteBallotSyncOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::VoteBallotSyncOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::VoteBallotSyncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::VoteBallotSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type predType = op.getPred().getType();

    // Scalar case: simple pass-through to NVVM
    if (!isa<RankedTensorType>(predType)) {
      Value result = rewriter.create<NVVM::VoteSyncOp>(
          loc, rewriter.getI32Type(), adaptor.getMask(), adaptor.getPred(),
          NVVM::VoteSyncKind::ballot);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Tensor case: unpack elements, apply ballot to each, pack results
    auto predTensorType = cast<RankedTensorType>(predType);
    auto resultType = op.getResult().getType();

    // Unpack the tensor predicate elements - each thread owns some elements
    SmallVector<Value> predElems =
        unpackLLElements(loc, adaptor.getPred(), rewriter);

    // For vote_ballot_sync with tensor predicates:
    // 1. First, OR all local predicate elements together to get a single bool
    // 2. Apply the ballot operation once with the combined predicate
    // 3. Replicate the result to all elements of the output tensor

    TritonLLVMOpBuilder b(loc, rewriter);

    // Combine all local predicate elements with OR
    Value combinedPred;
    if (predElems.empty()) {
      combinedPred = b.i1_val(false);
    } else {
      combinedPred = predElems[0];
      for (size_t i = 1; i < predElems.size(); ++i) {
        combinedPred = b.or_(combinedPred, predElems[i]);
      }
    }

    // Perform the warp-level ballot with the combined predicate
    Value ballot = rewriter.create<NVVM::VoteSyncOp>(
        loc, rewriter.getI32Type(), adaptor.getMask(), combinedPred,
        NVVM::VoteSyncKind::ballot);

    // Replicate the ballot result to all elements of the output tensor
    SmallVector<Value> resultElems(predElems.size(), ballot);

    // Pack results back into tensor
    Value packedResult = packLLElements(loc, getTypeConverter(), resultElems,
                                        rewriter, resultType);
    rewriter.replaceOp(op, packedResult);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(typeConverter,
                                                                  benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCLCTryCancelOpConversion>(typeConverter, benefit);
  patterns.add<CLCQueryCancelOpConversion>(typeConverter, benefit);
  patterns.add<VoteBallotSyncOpConversion>(typeConverter, benefit);
}
