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
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-lower-mem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

LogicalResult doConvertDescriptorLoadsToNVWS(triton::FuncOp funcOp) {
  SmallVector<tt::DescriptorLoadOp> loads;
  funcOp.walk([&](tt::DescriptorLoadOp load) { loads.push_back(load); });

  for (tt::DescriptorLoadOp load : loads) {
    auto tensorType = dyn_cast<RankedTensorType>(load.getType());
    if (!tensorType)
      return load.emitError("expected a ranked tensor descriptor load result");

    ttg::LocalStoreOp soleStore;
    ttg::LocalAllocOp soleAlloc;
    if (load->hasOneUse())
      soleStore = dyn_cast<ttg::LocalStoreOp>(*load->getUsers().begin());
    if (load->hasOneUse())
      soleAlloc = dyn_cast<ttg::LocalAllocOp>(*load->getUsers().begin());

    OpBuilderWithAsyncTaskIds builder(load);
    builder.setInsertionPoint(load);
    Value buffer;
    if (soleStore) {
      buffer = soleStore.getDst();
    } else if (soleAlloc) {
      auto oldType = cast<ttg::MemDescType>(soleAlloc.getType());
      auto bufferType = ttg::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), /*mutableMemory=*/true);
      auto newAlloc = builder.createWithAsyncTaskIds<ttg::LocalAllocOp>(
          soleAlloc.getLoc(), bufferType);
      newAlloc->setAttrs(soleAlloc->getAttrs());
      triton::replaceUsesAndPropagateType(builder, soleAlloc,
                                          newAlloc.getResult());
      buffer = newAlloc.getResult();
      builder.setInsertionPointAfter(newAlloc);
    } else {
      auto encoding =
          ttng::getEncodingFromDescriptor(load, tensorType, load.getDesc());
      auto memorySpace = ttg::SharedMemorySpaceAttr::get(load.getContext());
      auto bufferType = ttg::MemDescType::get(
          tensorType.getShape(), tensorType.getElementType(), encoding,
          memorySpace, /*mutableMemory=*/true);
      buffer = builder
                   .createWithAsyncTaskIds<ttg::LocalAllocOp>(load.getLoc(),
                                                              bufferType)
                   .getResult();
    }

    if (Operation *bufferDef = buffer.getDefiningOp();
        bufferDef && bufferDef->getBlock() == load->getBlock() &&
        load->isBeforeInBlock(bufferDef))
      bufferDef->moveBefore(load);

    int64_t txCount =
        ttng::getDescriptorLoadBytes(cast<ttg::MemDescType>(buffer.getType()));
    auto nvwsLoad = builder.createWithAsyncTaskIds<ttnvws::DescriptorLoadOp>(
        load.getLoc(), load.getDesc(), load.getIndices(), txCount, buffer,
        load.getCache(), load.getEvict());
    nvwsLoad->setAttrs(load->getAttrs());

    if (soleStore) {
      soleStore.erase();
    } else if (soleAlloc) {
      soleAlloc.erase();
    } else {
      auto localLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
          load.getLoc(), load.getType(), buffer);
      load.replaceAllUsesWith(localLoad.getResult());
    }
    load.erase();
  }

  bool hasUnconvertedLoad = false;
  funcOp.walk([&](tt::DescriptorLoadOp load) {
    load.emitError("descriptor load was not converted for AutoWS");
    hasUnconvertedLoad = true;
  });
  return failure(hasUnconvertedLoad);
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
  return triton::gpu::MemDescIndexOp::create(builder, alloc.getLoc(),
                                             viewDescType, alloc, idx);
}

static Value getTMALoadBufferForStage(OpBuilderWithAsyncTaskIds &builder,
                                      Value buffer, Value bufferIdx) {
  auto currentView = buffer.getDefiningOp<ttg::MemDescIndexOp>();
  if (!currentView)
    return buffer;
  return createBufferView(builder, currentView.getSrc(), bufferIdx);
}

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<ttnvws::DescriptorLoadOp> &tmaLoads,
                            Value barrierAlloc, Value bufferIdx,
                            Value bufferIdxExtract, Value phase,
                            Operation *headProducer, Operation *headConsumer,
                            Operation *headConsumerSameLevel,
                            ArrayRef<int> additionalConsumerTaskIds,
                            DictionaryAttr consumerWaitConstraints) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int64_t sizeInBytes = 0;
  for (auto tmaLoad : tmaLoads)
    sizeInBytes += tmaLoad.getTxCount();

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  builder.setLoopScheduleInfoFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  auto expect = builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(
      loc, prodBarrier, sizeInBytes, pred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto tmaLoad : tmaLoads) {
    builder.setInsertionPoint(tmaLoad);
    builder.setAsyncTaskIdsFromOp(tmaLoad);
    builder.setLoopScheduleInfoFromOp(tmaLoad);
    Value pipelineBuffer =
        getTMALoadBufferForStage(builder, tmaLoad.getResult(), bufferIdx);
    copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
        tmaLoad.getLoc(), tmaLoad.getDesc(), tmaLoad.getIndices(), prodBarrier,
        pipelineBuffer, pred);
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

  // Create one WaitBarrierOp per consumer task ID.
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase, waitPred, /*deps=*/ValueRange{},
      consumerWaitConstraints);
  for (int extraTaskId : additionalConsumerTaskIds) {
    builder.setAsynTaskIdsFromArray({extraTaskId});
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, waitPred,
        /*deps=*/ValueRange{}, consumerWaitConstraints);
  }

  for (auto tmaLoad : tmaLoads)
    tmaLoad.erase();
  builder.clearLoopScheduleInfo();
  return copy;
}

} // namespace mlir
