#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace nvgpu = ::mlir::triton::nvgpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-lower-mem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

Value createBufferView(OpBuilderWithAsyncTaskIds &builder, Value alloc,
                       Value idx) {
  assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
         "Expected MemDescType");
  auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape;
  assert(allocDescType.getShape().size() > 1 &&
         "Expected multi-dimensional memdesc (e.g., Nx...) for subview");
  shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
               allocDescType.getShape().end());
  auto viewDescType = triton::gpu::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory());
  return triton::gpu::MemDescIndexOp::create(builder, alloc.getLoc(),
                                             viewDescType, alloc, idx);
}

static int getTMALoadSize(tt::DescriptorLoadOp &tmaLoad) {
  auto tensorTy = cast<RankedTensorType>(tmaLoad->getResult(0).getType());
  int loadSize = product(tensorTy.getShape());
  return loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
}

Value getBufferForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                Type loadType, Value buffer, Value bufferIdx,
                                bool mutableMem) {
  auto context = buffer.getContext();
  auto tensorType = dyn_cast<RankedTensorType>(loadType);
  assert(tensorType);

  auto order = ttg::getOrderForMemory(tensorType);
  auto CGALayout = ttg::getCGALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout =
      dyn_cast<triton::gpu::MemDescType>(buffer.getType()).getEncoding();
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      cast<ttg::MemDescType>(buffer.getType()).getMemorySpace();
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemOry=*/mutableMem);

  auto desc = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      buffer.getLoc(), subviewTy, buffer, bufferIdx);
  return desc;
}

static Value mapBarrierTo2CTALeader(OpBuilderWithAsyncTaskIds &builder,
                                    Location loc, Value barrier) {
  MLIRContext *ctx = builder.getContext();
  Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
      loc, builder.getI32Type());
  Value negTwo =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, -2, 32);
  Value leaderRank =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, ctaId, negTwo);
  auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      barrierTy.getShape(), barrierTy.getElementType(), barrierTy.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierTy.getMutableMemory(), barrierTy.getAllocShape());
  return builder.createWithAsyncTaskIds<ttng::MapToRemoteBufferOp>(
      loc, remoteTy, barrier, leaderRank);
}

static Value mapBarrierTo2CTAPeer(OpBuilderWithAsyncTaskIds &builder,
                                  Location loc, Value barrier, Value ctaId) {
  MLIRContext *ctx = builder.getContext();
  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 32);
  Value peerRank =
      builder.createWithAsyncTaskIds<arith::XOrIOp>(loc, ctaId, one);
  auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
  auto remoteTy = ttg::MemDescType::get(
      barrierTy.getShape(), barrierTy.getElementType(), barrierTy.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierTy.getMutableMemory(), barrierTy.getAllocShape());
  return builder.createWithAsyncTaskIds<ttng::MapToRemoteBufferOp>(
      loc, remoteTy, barrier, peerRank);
}

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<tt::DescriptorLoadOp> &tmaLoads,
                            SmallVector<Value> &buffers, Value barrierAlloc,
                            Value bufferIdx, Value bufferIdxExtract,
                            Value phase, Operation *headProducer,
                            Operation *headConsumer,
                            Operation *headConsumerSameLevel,
                            ArrayRef<int> additionalConsumerTaskIds,
                            DictionaryAttr consumerWaitConstraints) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int sizeInBytes = 0;
  bool twoCTA = tmaLoads.front()->hasAttr("two_cta_load");
  assert(llvm::all_of(tmaLoads,
                      [twoCTA](tt::DescriptorLoadOp load) {
                        return load->hasAttr("two_cta_load") == twoCTA;
                      }) &&
         "TMA barrier fusion cannot mix cooperative and per-CTA loads");
  for (auto &tmaLoad : tmaLoads) {
    bool cooperative = tmaLoad->hasAttr("two_cta_load");
    sizeInBytes += getTMALoadSize(tmaLoad) * (cooperative ? 2 : 1);
  }

  // For each of the following ops, we will operate on a subview of each value
  // according to the pipeline stage.

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  builder.setLoopScheduleInfoFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  Value tmaBarrier =
      twoCTA ? mapBarrierTo2CTALeader(builder, loc, prodBarrier) : prodBarrier;
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  Value expectPred = pred;
  if (twoCTA) {
    Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
        loc, builder.getI32Type());
    Value two =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 2, 32);
    Value rankInPair =
        builder.createWithAsyncTaskIds<arith::RemUIOp>(loc, ctaId, two);
    Value zero =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
    expectPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankInPair, zero);
  }
  auto expect = builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, expectPred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    builder.setInsertionPoint(tmaLoad);
    builder.setLoopScheduleInfoFromOp(tmaLoad);
    auto pipelineBuffer = getBufferForPipelineStage(builder, tmaLoad.getType(),
                                                    buffer, bufferIdx, true);
    auto copyOp =
        builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
            loc, tmaLoad.getDesc(), tmaLoad.getIndices(), tmaBarrier,
            pipelineBuffer, pred);
    if (tmaLoad->hasAttr("two_cta_load"))
      copyOp.setTwoCta(true);
    copy = copyOp;
  }

  // Create a wait_barrier before the first consumer.
  // For data-partitioned channels, shared ops (consBarrier, phase, pred)
  // need ALL consumer task IDs so they survive specializeRegion.
  builder.setInsertionPoint(headConsumerSameLevel);
  SmallVector<int> consumerTaskIds;
  for (int id : getAsyncTaskIds(headConsumer))
    consumerTaskIds.push_back(id);
  for (int id : additionalConsumerTaskIds)
    consumerTaskIds.push_back(id);
  builder.setAsynTaskIdsFromArray(consumerTaskIds);
  builder.setLoopScheduleInfoFromOp(headConsumerSameLevel);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
      loc, builder.getI32Type(), phase);
  Value waitPred =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  Value followerWaitPred;
  Value peerBarrier;
  if (twoCTA) {
    Value ctaId = builder.createWithAsyncTaskIds<nvgpu::ClusterCTAIdOp>(
        loc, builder.getI32Type());
    Value two =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 2, 32);
    Value rankInPair =
        builder.createWithAsyncTaskIds<arith::RemUIOp>(loc, ctaId, two);
    Value zero =
        builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 32);
    waitPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankInPair, zero);
    followerWaitPred = builder.createWithAsyncTaskIds<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, rankInPair, zero);
    peerBarrier = mapBarrierTo2CTAPeer(builder, loc, consBarrier, ctaId);
  }

  // Create one WaitBarrierOp per consumer task ID.
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase, waitPred, /*deps=*/ValueRange{},
      consumerWaitConstraints);
  if (twoCTA) {
    // The hardware CTA-group TMA transaction completes the leader's local
    // mbarrier. Relay that completion to the follower's corresponding local
    // barrier, then let the follower wait on it. A cluster barrier is not
    // valid here: only this warp-specialized consumer partition executes the
    // handoff, while cluster barriers require participation from every warp.
    builder.createWithAsyncTaskIds<ttng::FenceAsyncSharedOp>(
        loc, /*bCluster=*/false);
    builder.createWithAsyncTaskIds<ttng::ArriveBarrierOp>(
        loc, peerBarrier, /*count=*/1, waitPred);
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, followerWaitPred, /*deps=*/ValueRange{},
        consumerWaitConstraints);
  }
  for (int extraTaskId : additionalConsumerTaskIds) {
    builder.setAsynTaskIdsFromArray({extraTaskId});
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, waitPred,
        /*deps=*/ValueRange{}, consumerWaitConstraints);
    if (twoCTA) {
      builder.createWithAsyncTaskIds<ttng::FenceAsyncSharedOp>(
          loc, /*bCluster=*/false);
      builder.createWithAsyncTaskIds<ttng::ArriveBarrierOp>(
          loc, peerBarrier, /*count=*/1, waitPred);
      builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
          loc, consBarrier, phase, followerWaitPred, /*deps=*/ValueRange{},
          consumerWaitConstraints);
    }
  }

  // Convert all the consumers. The descriptor_load's single user is the
  // local_store into the pipeline buffer, and the async TMA copy now writes
  // that buffer directly, so erase both.
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    unsigned cnt = 0;
    Operation *localSt = nullptr;
    for (auto *usr : tmaLoad->getUsers()) {
      assert(isa<ttg::LocalStoreOp>(usr));
      localSt = usr;
      ++cnt;
    }
    assert(cnt == 1);
    localSt->erase();
    tmaLoad.erase();
  }
  builder.clearLoopScheduleInfo();
  return copy;
}

} // namespace mlir
