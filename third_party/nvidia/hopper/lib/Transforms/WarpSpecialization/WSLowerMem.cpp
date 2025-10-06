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
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-lower-mem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static std::pair<Operation *, Operation *>
createAsyncCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *c,
                Operation *op, SmallVector<AsyncTaskId> &asyncTasksPC,
                Value bufferIdx, Value bufferIdxExtract) {
  auto loadOp = cast<triton::LoadOp>(op);
  auto buffer = bufferMap.find(c)->second;
  MLIRContext *context = loadOp->getContext();
  OpBuilderWithAsyncTaskIds builder(context);
  builder.setInsertionPoint(loadOp->getParentOp());
  builder.setAsynTaskIdsFromArray(asyncTasksPC);

  builder.setInsertionPoint(loadOp);
  builder.setLoopScheduleInfo(loadOp);
  Value loadResult = loadOp.getResult();
  auto tensorType = dyn_cast<RankedTensorType>(loadResult.getType());
  if (!tensorType)
    return {nullptr, nullptr};
  // Get basic information from tensorType
  auto order = ttg::getOrderForMemory(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout =
      dyn_cast<triton::gpu::MemDescType>(buffer.getType()).getEncoding();
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemory=*/true);
  builder.setAsyncTaskIdsFromOp(loadOp);
  builder.setInsertionPointAfter(loadOp);
  auto view = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      loadOp.getLoc(), subviewTy, buffer, bufferIdx);
  // Create cp.async
  Operation *copy =
      builder.createWithAsyncTaskIds<ttg::AsyncCopyGlobalToLocalOp>(
          loadOp.getLoc(), loadOp.getPtr(), view, loadOp.getMask(),
          loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

  // Extract part.
  builder.setAsyncTaskIdsFromValueUsers(loadResult);
  builder.setInsertionPoint(c->getDstOp());
  auto viewLoad = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      loadOp.getLoc(), subviewTy, buffer, bufferIdxExtract);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      loadOp.getLoc(), loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
  // Replace all uses of loadResult
  loadResult.replaceAllUsesWith(sharedLoad.getResult());
  loadOp.erase();
  return {copy, sharedLoad};
}

// Create a local copy for a channel that is populated by the producer and
// accessed by the consumer.
// For the case where the value shared in (producer, consumer) is in tensor.
// Global buffer for the channel is already created and passed in bufferMap.
// This function creates LocalLoad at consumer and LocalStore at producer.
static std::pair<Operation *, Operation *>
createLocalCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *channel,
                Value srcBufferIdx, Value dstBufferIdx) {
  Operation *srcOp = channel->getSrcOp();
  Operation *dstOp = channel->getDstOp();
  MLIRContext *context = srcOp->getContext();
  auto buffer = bufferMap.find(channel)->second;

  Value srcValue = channel->getSrcOperand();
  auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!tensorType)
    return {nullptr, nullptr};
  // Get basic information from tensorType
  auto order = ttg::getOrderForMemory(tensorType);
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout =
      dyn_cast<triton::gpu::MemDescType>(buffer.getType()).getEncoding();
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemory=*/true);

  // Consumer part.
  OpBuilderWithAsyncTaskIds builder(dstOp);
  builder.setAsyncTaskIdsFromOp(dstOp);
  builder.setInsertionPoint(dstOp);
  builder.setLoopScheduleInfo(dstOp);
  auto dstView = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      dstOp->getLoc(), subviewTy, buffer, dstBufferIdx);
  auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
      dstOp->getLoc(), srcValue.getType(), dstView);
  srcValue.replaceAllUsesWith(sharedLoad.getResult());

  // Producer part. Create local_store for new producers.
  builder.setAsynTaskIdsFromArray(channel->relation.first);
  builder.setInsertionPoint(srcOp->getParentOp());
  builder.setInsertionPointAfter(srcOp);
  builder.setLoopScheduleInfo(srcOp);
  auto srcView = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      srcOp->getLoc(), subviewTy, buffer, srcBufferIdx);
  // Create local_alloc
  Operation *copy = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
      srcOp->getLoc(), srcValue, srcView);
  return {copy, sharedLoad};
}

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
  //    /*allocShape=*/allocDescType.getAllocShape());
  return builder.create<triton::gpu::MemDescIndexOp>(alloc.getLoc(),
                                                     viewDescType, alloc, idx);
}

// For the case where the value shared in (producer, consumer) is in smem.
static std::pair<Operation *, Operation *>
createSMEMCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *channel,
               Value srcBufferIdx, Value dstBufferIdx) {
  Operation *srcOp = channel->getSrcOp();
  Operation *dstOp = channel->getDstOp();
  auto buffer = bufferMap.find(channel)->second;
  Value srcValue = channel->getSrcOperand();
  auto memDesc = cast<triton::gpu::MemDescType>(srcValue.getType());

  // Replace original smem alloc with smem_store.
  auto oldAllocOp = cast<ttg::LocalAllocOp>(srcOp);
  auto newAllocOp = cast<ttg::LocalAllocOp>(buffer.getDefiningOp());
  OpBuilderWithAsyncTaskIds builder(oldAllocOp);
  builder.setInsertionPointAfter(oldAllocOp);

  assert(oldAllocOp.getSrc());
  auto *actualSrc = oldAllocOp.getSrc().getDefiningOp();
  builder.setLoopScheduleInfo(actualSrc);

  SmallVector<AsyncTaskId> asyncTasksSubView = getAsyncTaskIds(actualSrc);
  for (auto *user : oldAllocOp->getUsers()) {
    for (auto task : getAsyncTaskIds(user))
      if (!llvm::is_contained(asyncTasksSubView, task))
        asyncTasksSubView.push_back(task);
  }
  builder.setAsynTaskIdsFromArray(asyncTasksSubView);
  // Will be used by both produer and consumer.
  auto srcView = createBufferView(builder, newAllocOp, srcBufferIdx);

  builder.setAsyncTaskIdsFromOp(actualSrc);
  auto smemStoreOp = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
      oldAllocOp.getLoc(), oldAllocOp.getSrc(), srcView);

  // Consumer will be updated.
  oldAllocOp->getResult(0).replaceAllUsesWith(srcView);
  oldAllocOp.erase();
  // DstOp is the same, srcOp will be auto-adjusted to be the defining op of
  // srcOpnd.
  return {smemStoreOp, channel->getDstOp()};
}

static std::pair<Operation *, Operation *>
createTMEMCopy(const DenseMap<Channel *, Value> &bufferMap, Channel *channel,
               Value srcBufferIdx, Value dstBufferIdx) {
  // Replace original tmem alloc with tmem_store.
  ttng::TmemDataChannel *tmemChannel =
      static_cast<ttng::TmemDataChannel *>(channel);
  auto oldTMemAllocOp = cast<ttng::TMEMAllocOp>(tmemChannel->getAllocOp());
  auto newTMemAllocOp =
      cast<ttng::TMEMAllocOp>(bufferMap.find(channel)->second.getDefiningOp());
  OpBuilderWithAsyncTaskIds builder(oldTMemAllocOp);
  builder.setInsertionPointAfter(oldTMemAllocOp);
  builder.setLoopScheduleInfo(oldTMemAllocOp);

  // A tmemChannel is usually centered around a gen5 dotOp. There are two
  // cases, one is that the channel is for the accumulator, the other is
  // the channel is for operand A of the gen5.
  // Here we replace tmem_alloc with tmem_store when applicable and create a
  // subView that is used by tmem_store and also all users of tmem_alloc.
  // Calculate the taskIds for the subView, and tmem_store.
  // tmemStore's taskId can be the mmaOp's taskId if alloc.getSrc is available
  // for mmaOp's taskId, otherwise, it should happen in alloc.getsrc.
  Operation *opForStoreTask = tmemChannel->getMmaOp();
  if (oldTMemAllocOp.getSrc()) {
    auto taskIds = getAsyncTaskIds(opForStoreTask);
    assert(taskIds.size() == 1);
    // Check to see if alloc.getSrc is available for mmaOp's taskId.
    auto *srcOp = oldTMemAllocOp.getSrc().getDefiningOp();
    if (!hasAsyncTaskId(srcOp, taskIds[0]))
      opForStoreTask = oldTMemAllocOp.getSrc().getDefiningOp();
  }
  // TaskIds for subView should be the union of tmem_store and all users of
  // tmem_alloc.
  SmallVector<AsyncTaskId> asyncTasksSubView = getAsyncTaskIds(opForStoreTask);
  for (auto *user : oldTMemAllocOp->getUsers()) {
    for (auto task : getAsyncTaskIds(user))
      if (!llvm::is_contained(asyncTasksSubView, task))
        asyncTasksSubView.push_back(task);
  }
  builder.setAsynTaskIdsFromArray(asyncTasksSubView);

  auto srcView = createBufferView(builder, newTMemAllocOp, srcBufferIdx);
  LLVM_DEBUG({
    LDBG("createTMEMCopy: srcView ");
    srcView.dump();
  });

  if (oldTMemAllocOp.getSrc()) {
    builder.setAsyncTaskIdsFromOp(opForStoreTask);
    Value vTrue = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
        oldTMemAllocOp.getLoc(), 1, 1);
    // Promote TMEMAlloc to start, create TMEMStore.
    // auto tokType = builder.getType<AsyncTokenType>();
    // tokType, srcView, oldTMemAllocOp.getToken()
    // We used to have token from Alloc, then to other users.
    // FIXME: Type(), srcView, Value(),
    // OAI's warpspec does the above.
    auto tmemStoreOp = builder.createWithAsyncTaskIds<ttng::TMEMStoreOp>(
        oldTMemAllocOp.getLoc(), Type(), srcView, Value(),
        oldTMemAllocOp.getSrc(), vTrue);
    oldTMemAllocOp->getResult(0).replaceAllUsesWith(srcView);
    if (oldTMemAllocOp.getToken())
      oldTMemAllocOp.getToken().replaceAllUsesWith(newTMemAllocOp.getToken());
    oldTMemAllocOp.erase();
    tmemChannel->tmemProducerOp = tmemStoreOp;
    return {tmemStoreOp, channel->getDstOp()};
  }
  // Handle the case where there is no value for tmem_alloc.
  oldTMemAllocOp->getResult(0).replaceAllUsesWith(srcView);
  if (oldTMemAllocOp.getToken())
    oldTMemAllocOp.getToken().replaceAllUsesWith(newTMemAllocOp.getToken());
  oldTMemAllocOp.erase();
  // We need a new srcOp now that tmemAlloc is erased, the new SrcOp will be
  // the mmaOp.
  tmemChannel->tmemProducerOp = tmemChannel->getMmaOp();
  return {tmemChannel->getMmaOp(), channel->getDstOp()};
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
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  auto elemType = tensorType.getElementType();

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto sharedLayout =
      dyn_cast<triton::gpu::MemDescType>(buffer.getType()).getEncoding();
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemOry=*/mutableMem);

  auto desc = builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      buffer.getLoc(), subviewTy, buffer, bufferIdx);
  return desc;
}

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<tt::DescriptorLoadOp> &tmaLoads,
                            SmallVector<Value> &buffers, Value barrierAlloc,
                            Value bufferIdx, Value bufferIdxExtract,
                            Value phase, Operation *headProducer,
                            Operation *headConsumer,
                            Operation *headConsumerSameLevel, bool isPost) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int sizeInBytes = 0;
  for (auto &tmaLoad : tmaLoads) {
    sizeInBytes += getTMALoadSize(tmaLoad);
  }

  // For each of the following ops, we will operate on a subview of each value
  // according to the pipeline stage.

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  builder.setLoopScheduleInfo(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  auto expect = builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, pred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    builder.setInsertionPoint(tmaLoad);
    builder.setLoopScheduleInfo(tmaLoad);
    auto pipelineBuffer = getBufferForPipelineStage(builder, tmaLoad.getType(),
                                                    buffer, bufferIdx, true);
    // FIXME: translateTMAIndices
    copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaLoad.getDesc(), tmaLoad.getIndices(), prodBarrier,
        pipelineBuffer, pred);
  }

  // Create a wait_barrier before the first consumer.
  builder.setInsertionPoint(headConsumerSameLevel);
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.setLoopScheduleInfo(headConsumerSameLevel);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI32Type(), phase);
  auto wait = builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase);

  // Convert all the consumers to local_load
  for (auto [tmaLoad, buffer] : zip(tmaLoads, buffers)) {
    if (isPost) {
      // consumer is the user of the smem. We can't insert local_load here
      // and use the result in local_store that is the producer for the smem
      // channel. descriptor_load has a single user which is local_store.
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
      continue;
    }
    builder.setLoopScheduleInfo(tmaLoad);
    auto pipelineBuffer = getBufferForPipelineStage(
        builder, tmaLoad.getType(), buffer, bufferIdxExtract, false);
    auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
        loc, tmaLoad.getType(), pipelineBuffer);

    Value loadResult = tmaLoad.getResult();
    tmaLoad.getResult().replaceAllUsesWith(sharedLoad.getResult());
    tmaLoad.erase();
  }
  builder.clearLoopScheduleInfo();
  return copy;
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByProducers"
void insertAsyncCopy(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByProducers,
    const DenseMap<Channel *, Value> &bufferMap,
    DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &regionsWithChannels, ReuseConfig *config,
    bool isPost) {
  // For each producer op, create a async_copy or local_store from the producer
  // to the buffer. Create a local_load from the buffer at the dominating
  // consumer.
  mlir::DominanceInfo dom(funcOp);

  for (auto kv : channelsGroupedByProducers) {
    // Finding the dominating channel if possible.
    std::unordered_set<Channel *> mutuallyNonDominatingChannels;
    for (auto &c : kv.second) {
      // check if c is dominating all other previous channels.
      auto it = mutuallyNonDominatingChannels.begin();
      while (it != mutuallyNonDominatingChannels.end()) {
        auto channel = *it;
        if (dom.properlyDominates(c->getDstOp(), channel->getDstOp())) {
          it = mutuallyNonDominatingChannels.erase(it);
        } else if (dom.properlyDominates(channel->getDstOp(), c->getDstOp())) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingChannels.end())
        mutuallyNonDominatingChannels.insert(c);
    }

    assert(mutuallyNonDominatingChannels.size() == 1 &&
           "conditional consumers not supported");
    auto domininatingChannel = *mutuallyNonDominatingChannels.begin();
    auto srcOp = kv.getFirst()->getSrcOp();
    LLVM_DEBUG({
      LDBG("insertAsyncCopy handle channel ");
      srcOp->dump();
      domininatingChannel->getDstOp()->dump();
    });

    Value bufferIdx;
    Value phase = Value();
    OpBuilderWithAsyncTaskIds builder(srcOp);
    // Calculate TaskIds for bufferIdx and phase.
    SmallVector<AsyncTaskId> asyncTasksPC = getAsyncTaskIds(srcOp);
    for (auto channel : mutuallyNonDominatingChannels) {
      // bufferIdx will be used in createTMEMCopy to construct subView
      // to feed into both tmem_store and users of tmem_alloc. There are cases
      // where a TMEM channel has srcOp in task 2, dstOp in task 2, while mmaOp
      // is in task 1.
      if (channel->channelKind == DataChannelKind::TMEM) {
        ttng::TmemDataChannel *tmemChannel =
            static_cast<ttng::TmemDataChannel *>(channel);
        for (auto task : getAsyncTaskIds(tmemChannel->getMmaOp()))
          if (!llvm::is_contained(asyncTasksPC, task))
            asyncTasksPC.push_back(task);
      }
      for (auto task : getAsyncTaskIds(channel->getDstOp()))
        if (!llvm::is_contained(asyncTasksPC, task))
          asyncTasksPC.push_back(task);
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    if (auto forOp = srcOp->getParentOfType<scf::ForOp>()) {
      int reuseGrp = channelInReuseGroup(kv.getFirst(), config);
      LLVM_DEBUG({
        LDBG("call getBufferIdxAndPhase ");
        srcOp->dump();
      });
      getBufferIdxAndPhase(builder, srcOp, kv.getFirst()->getNumBuffers(),
                           regionsWithChannels, bufferIdx, phase, config,
                           reuseGrp, kv.getFirst());
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here which will
      // be used by both producer and consumers.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          srcOp->getLoc(), 0, 32);
    }

    LLVM_DEBUG({
      LDBG("-- bufferIdx ");
      bufferIdx.dump();
    });
    std::pair<Operation *, Operation *> producerConsumerOps{nullptr, nullptr};

    // No need to create async copy for TMA load which will be handled in
    // insertAsyncComm.
    if (isa<tt::DescriptorLoadOp>(srcOp)) {
      producerConsumerOps = {srcOp, domininatingChannel->getDstOp()};
    } else if (isa<triton::LoadOp>(srcOp)) {
      SmallVector<AsyncTaskId> asyncTasksPC = getAsyncTaskIds(srcOp);
      asyncTasksPC.append(getAsyncTaskIds(domininatingChannel->getDstOp()));
      // After createAsyncCopy, c->getSrcOp()/headProducer are no longer
      // valid.
      producerConsumerOps = createAsyncCopy(bufferMap, domininatingChannel,
                                            domininatingChannel->getSrcOp(),
                                            asyncTasksPC, bufferIdx, bufferIdx);
    } else if (domininatingChannel->channelKind == DataChannelKind::TMEM &&
               !isPost) {
      producerConsumerOps =
          createTMEMCopy(bufferMap, domininatingChannel, bufferIdx, bufferIdx);
    } else if (isa<ttg::LocalAllocOp>(srcOp) && !isPost) {
      producerConsumerOps =
          createSMEMCopy(bufferMap, domininatingChannel, bufferIdx, bufferIdx);
    } else if (!isPost) {
      assert(!isa<ttg::LocalLoadOp>(srcOp) &&
             "LocalLoadOp buffer should be reused");
      producerConsumerOps =
          createLocalCopy(bufferMap, domininatingChannel, bufferIdx, bufferIdx);
    }

    for (auto &channel : kv.second) {
      copyOpMap[channel] = producerConsumerOps;
    }
  }
}

} // namespace mlir
