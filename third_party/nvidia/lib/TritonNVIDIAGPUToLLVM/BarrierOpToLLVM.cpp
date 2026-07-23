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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace ttg = mlir::triton::gpu;

namespace {

struct PhysicalClusterInfo {
  SmallVector<int, 3> dims;
  int size;
  bool hasPerCTAProgramIds;
};

// num-ctas describes Triton's logical/layout cluster model. ctas_per_cga keeps
// num-ctas == 1 and records the physical CUDA cluster in ttg.cluster-dim-*.
// Keep those concepts separate: changing lookupNumCTAs would alter layout and
// program-id semantics throughout the compiler.
PhysicalClusterInfo getPhysicalClusterInfo(Operation *op) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  assert(mod && "expected CLC op inside a module");
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  SmallVector<int, 3> dims = ttg::TritonGPUDialect::getClusterDims(mod);
  int explicitSize = dims[0] * dims[1] * dims[2];
  if (explicitSize > 1)
    return {std::move(dims), explicitSize, numCTAs == 1};
  return {{numCTAs, 1, 1}, numCTAs, false};
}

Value getElectWarp0OrThread0(const NVIDIA::TargetInfo &targetInfo,
                             TritonLLVMOpBuilder &b) {
  if (targetInfo.getComputeCapability() >= 90) {
    return LLVM::NVIDIA::createElectPredicateWarp0(b.loc, *b.builder);
  } else {
    auto tid = getThreadId(*b.builder, b.loc);
    return b.icmp_eq(tid, b.i32_val(0));
  }
}

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

struct FenceOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::FenceOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto scope = op.getScope();
    // "gpu" -> syncscope("device"), "sys" -> syncscope("") (system scope)
    StringRef syncscope = scope == "gpu" ? "device" : "";
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, LLVM::AtomicOrdering::acq_rel, StringAttr::get(ctx, syncscope));
    return success();
  }
};

struct FenceMBarrierInitReleaseClusterOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp>::
      ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceMBarrierInitReleaseClusterOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Only one thread needs to issue the fence, just like mbarrier.init.
    Value tid = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(tid, b.i32_val(0));

    PTXBuilder ptxBuilder;
    auto &fence = *ptxBuilder.create("fence.mbarrier_init.release.cluster");
    fence().predicate(pred);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  const NVIDIA::TargetInfo *targetInfo;
  InitBarrierOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit,
                          NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);

    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);

    auto numCTAs = triton::gpu::lookupNumCTAs(op);
    auto initCount = op.getCount();
    // The lead barrier accounts for all arrives from CTAs that broadcast into
    // the same barrier.
    initCount *= numCTAs / barrierTy.getNumElements();

    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                            std::to_string(initCount) + ";";
    auto &barSyncOp = *ptxBuilder.create(ptx);
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
  const NVIDIA::TargetInfo *targetInfo;
  InvalBarrierOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit,
                           NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InvalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // We use an elect predicate to tell ptxas that the operation is uniform,
    // which results in better codegen.
    Value pred = getElectWarp0OrThread0(*targetInfo, b);
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);
    Value barrierPtr = LLVM::NVIDIA::getLeaderAddress(
        loc, rewriter, smemObj.getBase(), barrierTy);
    ::mlir::triton::PTXBuilder ptxBuilder;
    const std::string ptx = "@$0 mbarrier.inval.shared::cta.b64 [$1];";
    auto &barSyncOp = *ptxBuilder.create(ptx);
    barSyncOp({ptxBuilder.newOperand(pred, "b"),
               ptxBuilder.newOperand(barrierPtr, "r")},
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
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    // If several CTAs cast to the same barrier, that barrier will receive all
    // the bytes from its broadcast group
    auto numCTAs = triton::gpu::lookupNumCTAs(rewriter);
    auto expectedBytes = op.getSize() * (numCTAs / barrierTy.getNumElements());

    auto id = getThreadId(rewriter, loc);
    Value basePred = b.icmp_eq(id, b.i32_val(0));
    basePred = b.and_(basePred, adaptor.getPred());
    auto leaderCTAPred =
        LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy);
    bool crossCluster = leaderCTAPred.has_value();
    Value leaderPred =
        leaderCTAPred ? b.and_(basePred, *leaderCTAPred) : basePred;
    Value leaderBarrierPtr = LLVM::NVIDIA::getLeaderAddress(
        loc, rewriter, smemObj.getBase(), barrierTy);

    ::mlir::triton::PTXBuilder expectPtxBuilder;
    const std::string expectPtx =
        "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], " +
        std::to_string(expectedBytes) + ";";
    auto &expectOp = *expectPtxBuilder.create(expectPtx);
    expectOp({expectPtxBuilder.newOperand(leaderPred, "b"),
              expectPtxBuilder.newOperand(leaderBarrierPtr, "r")},
             /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    expectPtxBuilder.launch(rewriter, loc, voidTy);

    if (crossCluster) {
      // Non-leader CTAs still contribute one arrival to the lead CTA barrier.
      auto nonLeaderPred = b.and_(basePred, b.xor_(leaderPred, b.true_val()));
      ::mlir::triton::PTXBuilder arrivePtxBuilder;
      const std::string arrivePtx =
          "@$0 mbarrier.arrive.shared::cluster.b64 _, [$1], 1;";
      auto &arriveOp = *arrivePtxBuilder.create(arrivePtx);
      arriveOp({arrivePtxBuilder.newOperand(nonLeaderPred, "b"),
                arrivePtxBuilder.newOperand(leaderBarrierPtr, "r")},
               /*onlyAttachMLIRArgs=*/true);
      arrivePtxBuilder.launch(rewriter, loc, voidTy);
    }

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

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto barrierTy = op.getAlloc().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto pred = adaptor.getPred();
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);

    bool predicated = pred && !matchPattern(pred, m_NonZero());
    int suspendNs = 0;
    if (targetInfo->getComputeCapability() >= 100) {
      auto mod = op->getParentOfType<ModuleOp>();
      if (auto attr = mod->getAttrOfType<IntegerAttr>(
              "tlx.mbarrier_try_wait_suspend_ns"))
        suspendNs = attr.getInt();
    }
    bool useSuspendHint = suspendNs > 0;
    std::string ptx;
    if (targetInfo->getComputeCapability() < 90) {
      if (!predicated) {
        ptx = R"(
{
	.reg .pred complete;
	waitLoop:
	mbarrier.test_wait.parity.shared::cta.b64 complete, [$0], $1;
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
	mbarrier.test_wait.parity.shared::cta.b64 complete, [$0], $1;
	@!complete nanosleep.u32 20;
	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    } else {
      // SM90+ polls with try_wait in a spin loop. Blackwell can opt into the
      // four-operand form, whose suspend hint lowers to NANOSLEEP.SYNCS.
      std::string tryWait = useSuspendHint
                                ? "\tmbarrier.try_wait.parity.shared::cta.b64 "
                                  "complete, [$0], $1, $2;\n"
                                : "\tmbarrier.try_wait.parity.shared::cta.b64 "
                                  "complete, [$0], $1;\n";
      if (!predicated) {
        ptx = std::string(R"(
{
	.reg .pred complete;
	waitLoop:
)") + tryWait +
              R"(	@!complete bra.uni waitLoop;
}
)";
      } else {
        std::string predOperand = useSuspendHint ? "$3" : "$2";
        ptx = std::string("\n{\n\t@!") + predOperand +
              R"( bra.uni skipWait;
	.reg .pred complete;
	waitLoop:
)" + tryWait +
              R"(	@!complete bra.uni waitLoop;
	skipWait:
}
)";
      }
    }
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto &waitLoop = *ptxBuilder.create(ptx);
    SmallVector<::mlir::triton::PTXBuilder::Operand *, 4> operands = {
        ptxBuilder.newOperand(smemObj.getBase(), "r"),
        ptxBuilder.newOperand(adaptor.getPhase(), "r")};
    if (useSuspendHint)
      operands.push_back(ptxBuilder.newOperand(b.i32_val(suspendNs), "r"));
    if (predicated)
      operands.push_back(ptxBuilder.newOperand(pred, "b"));

    waitLoop(operands, /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(ctx));
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
    auto loc = op.getLoc();
    bool isPerThread = op.getPerThread();
    auto barrierTy = op.getAlloc().getType();
    // Compute the shared-memory base before any barrier so the lead (non-
    // perThread) path emits the barrier sync after the address extraction, as
    // the lit expectations require.
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);

    // A (non-perThread) arrive has block-level semantics, so we must
    // synchronize the CTA before it. Technically this should be MemBar's job,
    // but an arrive can follow TMEM accesses which have no MemBar equivalent.
    // perThread arrives are per-thread (no leader pattern) and must NOT emit a
    // CTA-wide barrier -- doing so would deadlock under warp specialization
    // where not all threads of the CTA reach the arrive.
    if (!isPerThread)
      ttg::BarrierOp::create(rewriter, loc, ttg::AddrSpace::Local);

    // A barrier physically in cluster shared memory must be signalled with
    // cluster scope regardless of broadcast.
    bool isRemoteBarrier = false;
    if (auto barType = dyn_cast<ttg::MemDescType>(barrierTy)) {
      isRemoteBarrier =
          isa<ttng::SharedClusterMemorySpaceAttr>(barType.getMemorySpace());
    }

    if (isPerThread) {
      // Warp arrive: every thread arrives independently, no leader pattern.
      // perThread arrives are warp-specialization (single CTA) and never
      // broadcast across CTAs, so no lead-CTA redirection applies here.
      bool hasPred = !!op.getPred();
      std::stringstream ptxAsm;
      if (hasPred) {
        ptxAsm << "@$0 ";
      }
      ptxAsm << "mbarrier.arrive.shared::"
             << (isRemoteBarrier ? "cluster" : "cta") << ".b64 _, ["
             << (hasPred ? "$1" : "$0") << "]";
      if (op.getCount() > 1) {
        ptxAsm << ", " << op.getCount();
      }
      ptxAsm << ";";

      PTXBuilder ptxBuilder;
      SmallVector<PTXBuilder::Operand *, 2> operands;
      if (hasPred) {
        operands.push_back(ptxBuilder.newOperand(adaptor.getPred(), "b"));
      }
      operands.push_back(ptxBuilder.newOperand(adaptor.getAlloc(), "r"));

      auto arriveOp = *ptxBuilder.create<>(ptxAsm.str());
      arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
      auto voidTy = void_ty(getContext());
      ptxBuilder.launch(rewriter, loc, voidTy);
    } else {
      // Leader pattern: only thread 0 arrives.
      //
      // MultiCTA broadcast (#9475): when several CTAs broadcast into the same
      // barrier, each CTA's leader arrives on the lead CTA's barrier with
      // cluster scope. getLeaderCTAPredicate / getLeaderAddress are no-ops for
      // CTA-local barriers, so the common path is unchanged.
      TritonLLVMOpBuilder b(loc, rewriter);
      bool isCrossCluster =
          LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy)
              .has_value();
      Value barrierPtr = LLVM::NVIDIA::getLeaderAddress(
          loc, rewriter, smemObj.getBase(), barrierTy);

      std::stringstream ptxAsm;
      ptxAsm << "@$0 mbarrier.arrive.shared::"
             << (isRemoteBarrier || isCrossCluster ? "cluster" : "cta")
             << ".b64 _, [$1]";
      if (op.getCount() > 1) {
        ptxAsm << ", " << op.getCount();
      }
      ptxAsm << ";";

      Value id = getThreadId(rewriter, loc);
      Value pred = b.icmp_eq(id, b.i32_val(0));
      if (op.getPred())
        pred = b.and_(pred, adaptor.getPred());

      PTXBuilder ptxBuilder;
      SmallVector<PTXBuilder::Operand *, 2> operands = {
          ptxBuilder.newOperand(pred, "b"),
          ptxBuilder.newOperand(barrierPtr, "r")};

      auto arriveOp = *ptxBuilder.create<>(ptxAsm.str());
      arriveOp(operands, /*onlyAttachMLIRArgs=*/true);
      auto voidTy = void_ty(getContext());
      ptxBuilder.launch(rewriter, loc, voidTy);
    }

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
    // Use the NVVM intrinsic which has IntrConvergent, preventing LLVM from
    // duplicating this barrier across control flow (e.g., jump threading).
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.nvvm.barrier.cta.arrive.aligned.count",
        TypeRange{}, {adaptor.getBar(), adaptor.getNumThreads()});
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
    // Use the NVVM intrinsic which has IntrConvergent, preventing LLVM from
    // duplicating this barrier across control flow (e.g., jump threading).
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.nvvm.barrier.cta.sync.aligned.count", TypeRange{},
        {adaptor.getBar(), adaptor.getNumThreads()});
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

    auto clcOp = *ptxBuilder.create(ptx);
    clcOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

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
      mov.s32 $1, -1;
      mov.s32 $2, -1;
      ld.shared.b128 clc_result, [$3];
      clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
      @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
    }
    )";

    PTXBuilder builder;
    auto queryOp = *builder.create<>(ptx);

    SmallVector<PTXBuilder::Operand *, 4> operands = {
        builder.newOperand("=r", true), builder.newOperand("=r", true),
        builder.newOperand("=r", true),
        builder.newOperand(adaptor.getClcResAlloc(), "r")};
    queryOp(operands, /*onlyAttachMLIRArgs=*/true);

    auto i32Ty = rewriter.getI32Type();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(getContext(), {i32Ty, i32Ty, i32Ty});
    Value result = builder.launch(rewriter, op.getLoc(), structTy, false);

    Value ctaIdX = LLVM::ExtractValueOp::create(rewriter, loc, result, 0);
    Value ctaIdY = LLVM::ExtractValueOp::create(rewriter, loc, result, 1);
    Value ctaIdZ = LLVM::ExtractValueOp::create(rewriter, loc, result, 2);

    rewriter.replaceOp(op, {ctaIdX, ctaIdY, ctaIdZ});

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
      Value result = NVVM::VoteSyncOp::create(
          rewriter, loc, rewriter.getI32Type(), adaptor.getMask(),
          adaptor.getPred(), NVVM::VoteSyncKind::ballot);
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
    Value ballot = NVVM::VoteSyncOp::create(
        rewriter, loc, rewriter.getI32Type(), adaptor.getMask(), combinedPred,
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

// CLC (Cluster Launch Control) Ops - Blackwell SM100+
struct CLCTryCancelOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCTryCancelOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCTryCancelOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit,
                           NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCTryCancelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();

    // Use elect predicate - only one thread should issue CLC
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    PhysicalClusterInfo cluster = getPhysicalClusterInfo(op);
    if (cluster.size > 1) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto clusterCtaId =
          triton::nvgpu::ClusterCTAIdOp::create(rewriter, loc, i32_ty);
      pred = b.and_(pred, b.icmp_eq(clusterCtaId, b.i32_val(0)));
    }

    std::string ptxAsm = "@$2 clusterlaunchcontrol.try_cancel.async.shared::cta"
                         ".mbarrier::complete_tx::bytes";
    if (cluster.size > 1)
      ptxAsm += ".multicast::cluster::all";
    ptxAsm += ".b128 [$0], [$1];";

    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand(adaptor.getResult(), "r");
    auto *mbarOp = ptxBuilder.newOperand(adaptor.getMbarrier(), "r");
    auto *predOp = ptxBuilder.newOperand(pred, "b");
    clcOp({resultOp, mbarOp, predOp}, /*onlyAttachMLIRArgs=*/true);

    auto voidTy = void_ty(getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);

    rewriter.eraseOp(op);
    return success();
  }
};

struct CLCLoadResultOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCLoadResultOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCLoadResultOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit,
                            NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCLoadResultOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(op.getSrc().getType().getElementType()),
        rewriter);
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i128Ty = rewriter.getIntegerType(128);
    auto res = b.load(i128Ty, smemObj.getBase());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct CLCIsCanceledOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCIsCanceledOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCIsCanceledOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit,
                            NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCIsCanceledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();
    std::string ptxAsm =
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $0, $1;";
    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand("=b");
    auto *clcResultOp = ptxBuilder.newOperand(adaptor.getClcResult(), "q");
    clcOp({resultOp, clcResultOp}, /*onlyAttachMLIRArgs=*/true);

    Value result =
        ptxBuilder.launch(rewriter, loc, i1_ty, /*hasSideEffects=*/false);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct CLCGetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::CLCGetProgramIdOp> {
  const NVIDIA::TargetInfo *targetInfo;
  CLCGetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit,
                              NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::CLCGetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo->getComputeCapability() < 100) {
      return op.emitError("CLC operations require SM100+ (Blackwell)");
    }

    auto loc = op.getLoc();

    const char *dimName = [&] {
      switch (op.getDim()) {
      case ProgramIDDim::X:
        return "x";
      case ProgramIDDim::Y:
        return "y";
      case ProgramIDDim::Z:
        return "z";
      }
      llvm::llvm_unreachable_internal("Invalid program id dim");
    }();

    auto ptxAsm = ("clusterlaunchcontrol.query_cancel.get_first_ctaid::" +
                   llvm::Twine(dimName) + ".b32.b128 $0, $1;")
                      .str();

    PTXBuilder ptxBuilder;
    auto &clcOp = *ptxBuilder.create(ptxAsm);
    auto *resultOp = ptxBuilder.newOperand("=r");
    auto *clcResultOp = ptxBuilder.newOperand(adaptor.getClcResult(), "q");
    clcOp({resultOp, clcResultOp}, /*onlyAttachMLIRArgs=*/true);

    Value result =
        ptxBuilder.launch(rewriter, loc, i32_ty, /*hasSideEffects=*/false);

    PhysicalClusterInfo cluster = getPhysicalClusterInfo(op);
    if (cluster.hasPerCTAProgramIds) {
      // CLC returns the first CTA coordinate of the canceled cluster. Under
      // ctas_per_cga, every CTA is a distinct Triton program, so reconstruct
      // this CTA's coordinate from the X-major linear cluster rank.
      TritonLLVMOpBuilder b(loc, rewriter);
      Value rank = triton::nvgpu::ClusterCTAIdOp::create(rewriter, loc, i32_ty);
      int axis = static_cast<int>(op.getDim());
      int stride = 1;
      for (int i = 0; i < axis; ++i)
        stride *= cluster.dims[i];
      Value localCoord = rank;
      if (stride > 1)
        localCoord = b.udiv(localCoord, b.i32_val(stride));
      if (cluster.dims[axis] > 1)
        localCoord = b.urem(localCoord, b.i32_val(cluster.dims[axis]));
      else
        localCoord = b.i32_val(0);
      result = b.add(result, localCoord);
    } else if (op.getDim() == ProgramIDDim::X && cluster.size > 1) {
      // Triton's num-ctas model assigns one program id to the whole 1-D
      // cluster, so preserve the existing ctaid -> clusterid conversion.
      TritonLLVMOpBuilder b(loc, rewriter);
      result = b.sdiv(result, b.i32_val(cluster.size));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, NVIDIA::TargetInfo &targetInfo) {
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<FenceOpConversion>(typeConverter, benefit);
  patterns.add<FenceMBarrierInitReleaseClusterOpConversion>(typeConverter,
                                                            benefit);
  patterns.add<InitBarrierOpConversion, InvalBarrierOpConversion>(
      typeConverter, benefit, targetInfo);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  // beta Meta CLC + named-barrier + vote-ballot patterns
  patterns.add<NamedBarrierArriveOpConversion>(typeConverter, benefit);
  patterns.add<NamedBarrierWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCLCTryCancelOpConversion>(typeConverter, benefit);
  patterns.add<CLCQueryCancelOpConversion>(typeConverter, benefit);
  patterns.add<VoteBallotSyncOpConversion>(typeConverter, benefit);
  // upstream #9361 CLC patterns (distinct ops; coexist with beta's)
  patterns.add<CLCTryCancelOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCLoadResultOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCIsCanceledOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<CLCGetProgramIdOpConversion>(typeConverter, benefit, targetInfo);
}
