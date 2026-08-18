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
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_INITIALIZEWSCLUSTERBARRIERS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

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

static void createMBarrierInit(OpBuilder &b, Location loc, Value pred,
                               Value barrierPtr, int count) {
  PTXBuilder ptxBuilder;
  auto &init = *ptxBuilder.create("@$0 mbarrier.init.shared::cta.b64 [$1], " +
                                  std::to_string(count) + ";");
  init({ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newOperand(barrierPtr, "r")},
       /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

static void createMBarrierArrive(OpBuilder &b, Location loc, Value pred,
                                 Value barrierPtr, bool relaxed,
                                 Value multicastMask = {}) {
  PTXBuilder ptxBuilder;
  std::string ptx = "@$0 mbarrier.arrive." +
                    std::string(relaxed ? "relaxed" : "release") +
                    ".cluster.shared::cluster";
  if (multicastMask)
    ptx += ".multicast::cluster::32b";
  ptx += ".b64 _, [$1]";
  if (multicastMask)
    ptx += ", $2";
  ptx += ";";

  SmallVector<PTXBuilder::Operand *> operands = {
      ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(barrierPtr, "r")};
  if (multicastMask)
    operands.push_back(ptxBuilder.newOperand(multicastMask, "r"));
  auto &arrive = *ptxBuilder.create(ptx);
  arrive(operands, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

static void createMBarrierWait(OpBuilder &b, Location loc, Value barrierPtr,
                               Value parity) {
  PTXBuilder ptxBuilder;
  auto &wait =
      *ptxBuilder.create("{\n"
                         "\t.reg .pred complete;\n"
                         "waitLoop:\n"
                         "\tmbarrier.try_wait.parity.acquire.cluster.shared::"
                         "cta.b64 complete, [$0], $1;\n"
                         "\t@!complete bra.uni waitLoop;\n"
                         "}\n");
  wait({ptxBuilder.newOperand(barrierPtr, "r"),
        ptxBuilder.newOperand(parity, "r")},
       /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(b, loc, void_ty(b.getContext()));
}

static Value getClusterBarrierMbarPtr(Location loc, RewriterBase &rewriter,
                                      FunctionOpInterface func, int64_t offset,
                                      const NVIDIA::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          targetInfo.getSharedAddressSpace());
  return b.gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func),
               b.i32_val(offset));
}

static int getClusterSize(Operation *op) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  auto dims = triton::gpu::TritonGPUDialect::getClusterDims(mod);
  int physicalSize = dims[0] * dims[1] * dims[2];
  return physicalSize > 1 ? physicalSize : triton::gpu::lookupNumCTAs(op);
}

template <typename EmitFn>
void lowerClusterSyncForWarpCounts(Location loc, OpBuilder &rewriter,
                                   int defaultNumWarps, int totalNumWarps,
                                   EmitFn emit) {
  int workerNumWarps = totalNumWarps - defaultNumWarps;
  if (workerNumWarps == 0) {
    emit(rewriter);
    return;
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
  int startId = defaultNumWarps;
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
  ClusterBarrierOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit,
                             const NVIDIA::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mbarOffset = op->getAttrOfType<IntegerAttr>(
        triton::nvidia_gpu::kClusterBarrierMbarOffsetAttrName);
    if (!mbarOffset) {
      return lowerClusterSyncForAllWarps(op, rewriter, [&](OpBuilder &b) {
        createClusterArrive(b, op.getLoc(), op.getRelaxed());
        createClusterWait(b, op.getLoc());
      });
    }

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value barrierPtr0 = getClusterBarrierMbarPtr(
        loc, rewriter, func, mbarOffset.getInt(), targetInfo);
    auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr0.getType());
    Value counterPtr = b.gep(ptrTy, i8_ty, barrierPtr0, LLVM::GEPArg(8));

    NVVM::BarrierOp::create(rewriter, loc);
    Value counter = b.load(i32_ty, counterPtr);
    // A delayed CTA can miss a phase if a peer reuses one mbarrier twice
    // before it starts waiting. Alternate two slots so each slot is reused
    // only after an intervening rendezvous. The low counter bit selects the
    // slot and the high bit is that slot's parity.
    Value barrierIdx = b.and_(counter, b.i32_val(1));
    Value parity = b.lshr(counter, b.i32_val(1));
    auto barrierSlotTy = LLVM::LLVMArrayType::get(
        i8_ty, triton::nvidia_gpu::kClusterBarrierMbarSlotSize);
    Value barrierPtr = b.gep(ptrTy, barrierSlotTy, barrierPtr0, barrierIdx);
    Value threadId = getThreadId(rewriter, loc);
    Value pred = b.icmp_eq(threadId, b.i32_val(0));
    int numCTAs = getClusterSize(op);
    bool relaxed = op.getRelaxed() && targetInfo.getPtxVersion() >= 86;
    if (targetInfo.supportsMbarrierMulticast()) {
      Value ctaId = NVVM::ClusterId::create(rewriter, loc, i32_ty);
      // Exclude the issuing CTA: the mbarriers expect numCTAs - 1 arrivals.
      Value peerMask =
          b.xor_(b.i32_val((1u << numCTAs) - 1), b.shl(b.i32_val(1), ctaId));
      createMBarrierArrive(rewriter, loc, pred, barrierPtr, relaxed, peerMask);
    } else {
      Value barrierInt = b.ptrtoint(i32_ty, barrierPtr);
      Value peerId = b.add(threadId, b.i32_val(1));
      Value peerBarrierInt = b.xor_(barrierInt, b.shl(peerId, b.i32_val(24)));
      Value peerBarrierPtr = b.inttoptr(barrierPtr.getType(), peerBarrierInt);
      Value arrivePred = b.icmp_ult(threadId, b.i32_val(numCTAs - 1));
      createMBarrierArrive(rewriter, loc, arrivePred, peerBarrierPtr, relaxed);
    }
    createMBarrierWait(rewriter, loc, barrierPtr, parity);
    Value nextCounter = b.and_(b.add(counter, b.i32_val(1)), b.i32_val(3));
    targetInfo.storeShared(rewriter, loc, counterPtr, nextCounter, pred);
    NVVM::BarrierOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
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

struct InitializeWSClusterBarriers
    : public mlir::triton::impl::InitializeWSClusterBarriersBase<
          InitializeWSClusterBarriers> {
  using InitializeWSClusterBarriersBase::InitializeWSClusterBarriersBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto countAttr = mod->getAttrOfType<IntegerAttr>(
        triton::nvidia_gpu::kWSClusterBarrierCountAttrName);
    if (!countAttr || countAttr.getInt() == 0)
      return;

    auto funcs = mod.getOps<LLVM::LLVMFuncOp>();
    auto kernelIt = llvm::find_if(
        funcs, [](LLVM::LLVMFuncOp func) { return triton::isKernel(func); });
    if (kernelIt == funcs.end())
      return;
    LLVM::LLVMFuncOp kernel = *kernelIt;

    NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    Location loc = kernel.getLoc();
    TritonLLVMIRRewriter rewriter(loc, mod.getContext());
    rewriter.setInsertionPointToStart(&kernel.getBody().front());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
    Value initPred = b.icmp_eq(tid, b.i32_val(0));
    auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    int64_t shared = sharedAttr ? sharedAttr.getInt() : 0;
    int64_t count = countAttr.getInt();
    int64_t offset =
        shared - count * triton::nvidia_gpu::kClusterBarrierMbarAllocationSize;
    int numCTAs = getClusterSize(kernel);
    for (int64_t i = 0; i < count;
         ++i, offset += triton::nvidia_gpu::kClusterBarrierMbarAllocationSize) {
      Value barrierPtr0 =
          getClusterBarrierMbarPtr(loc, rewriter, kernel, offset, targetInfo);
      for (int64_t slot = 0;
           slot < triton::nvidia_gpu::kClusterBarrierMbarBufferCount; ++slot) {
        Value barrierPtr = barrierPtr0;
        if (slot != 0)
          barrierPtr = getClusterBarrierMbarPtr(
              loc, rewriter, kernel,
              offset + slot * triton::nvidia_gpu::kClusterBarrierMbarSlotSize,
              targetInfo);
        createMBarrierInit(rewriter, loc, initPred, barrierPtr, numCTAs - 1);
      }
      auto ptrTy = cast<LLVM::LLVMPointerType>(barrierPtr0.getType());
      Value counterPtr = b.gep(ptrTy, i8_ty, barrierPtr0, LLVM::GEPArg(8));
      targetInfo.storeShared(rewriter, loc, counterPtr, b.i32_val(0), initPred);
    }

    NVIDIA::createFenceMBarrierInitReleaseCluster(rewriter, loc, initPred);
    int defaultNumWarps = triton::gpu::lookupNumWarps(kernel);
    int totalNumWarps = defaultNumWarps;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
      totalNumWarps = attr.getInt();
    lowerClusterSyncForWarpCounts(
        loc, rewriter, defaultNumWarps, totalNumWarps, [&](OpBuilder &b) {
          createClusterArrive(b, loc, /*relaxed=*/true);
          createClusterWait(b, loc);
        });
  }
};

} // namespace

void mlir::triton::NVIDIA::populateClusterOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const NVIDIA::TargetInfo &targetInfo) {
  patterns.add<ClusterArriveOpConversion>(typeConverter, benefit);
  patterns.add<ClusterWaitOpConversion>(typeConverter, benefit);
  patterns.add<ClusterSize1DOpConversion>(typeConverter, benefit);
  patterns.add<MapToRemoteBufferOpConversion>(typeConverter, benefit);
  patterns.add<ClusterBarrierOpConversion>(typeConverter, benefit, targetInfo);
  return;
}
