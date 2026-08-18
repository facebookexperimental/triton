#include "PlanStagingMaterializer.h"

#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include <iterator>

namespace mlir::triton::amdgpu {

LogicalResult materializeRegisterToLdsStaging(
    MutableArrayRef<PlanRegisterToLdsStaging> staging,
    PlanStagingMaterializationResult &result, std::string &error) {
  for (PlanRegisterToLdsStaging &request : staging) {
    Operation *producer = request.source.getDefiningOp();
    if (!request.loop || !producer || !request.tensorType ||
        request.logicalBytes <= 0 || request.alignment <= 0 ||
        request.consumers.empty() || request.consumerOperands.empty()) {
      error = "invalid resolved register-to-LDS staging request";
      return failure();
    }

    auto sharedEncoding = getSharedEncoding(request.tensorType);
    auto sharedMemory =
        gpu::SharedMemorySpaceAttr::get(request.loop.getContext());
    auto memdescType = gpu::MemDescType::get(
        request.tensorType.getShape(), request.tensorType.getElementType(),
        sharedEncoding, sharedMemory, /*mutableMemory=*/true);

    OpBuilder allocationBuilder(request.loop);
    request.allocation = gpu::LocalAllocOp::create(
        allocationBuilder, producer->getLoc(), memdescType);
    request.allocation->setAttr(
        "alignment", allocationBuilder.getI32IntegerAttr(request.alignment));
    ++result.newAllocations;

    OpBuilder deallocationBuilder(request.loop->getBlock(),
                                  std::next(request.loop->getIterator()));
    gpu::LocalDeallocOp::create(deallocationBuilder, request.loop.getLoc(),
                                request.allocation);

    OpBuilder storeBuilder(producer->getBlock(),
                           std::next(producer->getIterator()));
    request.store = gpu::LocalStoreOp::create(
        storeBuilder, producer->getLoc(), request.source, request.allocation);
    request.visibilityBarrier = gpu::BarrierOp::create(
        storeBuilder, producer->getLoc(), gpu::AddrSpace::Local);
    ++result.newStores;
    ++result.insertedBarriers;

    Operation *firstConsumer = request.consumers.front();
    Operation *lastConsumer = request.consumers.front();
    for (Operation *consumer : request.consumers) {
      if (consumer->isBeforeInBlock(firstConsumer))
        firstConsumer = consumer;
      if (lastConsumer->isBeforeInBlock(consumer))
        lastConsumer = consumer;
    }

    OpBuilder loadBuilder(firstConsumer);
    request.load =
        gpu::LocalLoadOp::create(loadBuilder, firstConsumer->getLoc(),
                                 request.tensorType, request.allocation);
    ++result.newLoads;
    for (OpOperand *operand : request.consumerOperands)
      operand->set(request.load);

    OpBuilder releaseBuilder(lastConsumer->getBlock(),
                             std::next(lastConsumer->getIterator()));
    request.releaseBarrier = gpu::BarrierOp::create(
        releaseBuilder, lastConsumer->getLoc(), gpu::AddrSpace::Local);
    ++result.insertedBarriers;
  }
  return success();
}

} // namespace mlir::triton::amdgpu
