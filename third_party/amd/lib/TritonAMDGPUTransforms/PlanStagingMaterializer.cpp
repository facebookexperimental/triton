#include "PlanStagingMaterializer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <iterator>

namespace mlir::triton::amdgpu {
namespace {

static void sortStagingPath(PlanLdsStaging &request) {
  llvm::sort(request.consumers, [](Operation *left, Operation *right) {
    return left->isBeforeInBlock(right);
  });
  llvm::sort(request.derivedOperations, [](Operation *left, Operation *right) {
    return left->isBeforeInBlock(right);
  });
}

static LogicalResult rewriteDerivedConsumers(PlanLdsStaging &request,
                                             Value replacement,
                                             std::string &error) {
  IRMapping mapping;
  mapping.map(request.source, replacement);
  for (Operation *operation : request.derivedOperations) {
    OpBuilder cloneBuilder(operation);
    Operation *clone = cloneBuilder.clone(*operation, mapping);
    request.clonedDerivedOperations.push_back(clone);
    ++request.derivedOperationsCloned;
    if (clone->getName() != operation->getName() ||
        clone->getAttrDictionary() != operation->getAttrDictionary() ||
        clone->getOperandTypes() != operation->getOperandTypes() ||
        clone->getResultTypes() != operation->getResultTypes()) {
      error = "cloned LDS-staging derived operation changed contract";
      return failure();
    }
  }
  for (OpOperand *operand : request.consumerOperands) {
    Value mapped = mapping.lookupOrNull(operand->get());
    if (!mapped) {
      error = "LDS-staging derived path did not map a named consumer";
      return failure();
    }
    operand->set(mapped);
    request.consumerReplacementValues.push_back(mapped);
  }

  for (Operation *operation : llvm::reverse(request.derivedOperations)) {
    if (llvm::all_of(operation->getResults(),
                     [](Value result) { return result.use_empty(); })) {
      operation->erase();
      ++request.derivedOperationsPruned;
    }
  }
  return success();
}

static LogicalResult materializeGlobalCopy(PlanLdsStaging &request,
                                           Operation *producer,
                                           std::string &error) {
  OpBuilder builder(producer);
  Location loc = producer->getLoc();
  Value destination = request.bufferView ? request.bufferView.getResult()
                                         : request.allocation.getResult();
  if (auto load = dyn_cast<triton::LoadOp>(producer)) {
    auto copy = gpu::AsyncCopyGlobalToLocalOp::create(
        builder, loc, load.getPtr(), destination, load.getMask(),
        load.getOther(), load.getCache(), load.getEvict(), load.getIsVolatile(),
        /*contiguity=*/1);
    request.globalCopy = copy;
    request.globalAccessSemanticsPreserved =
        copy.getSrc() == load.getPtr() && copy.getMask() == load.getMask() &&
        copy.getOther() == load.getOther() &&
        copy.getCache() == load.getCache() &&
        copy.getEvict() == load.getEvict() &&
        copy.getIsVolatile() == load.getIsVolatile();
  } else if (auto load = dyn_cast<BufferLoadOp>(producer)) {
    auto copy = BufferLoadToLocalOp::create(
        builder, loc, destination, load.getPtr(), load.getOffsets(),
        load.getMask(), load.getOther(), load.getStride(), load.getCache(),
        load.getContiguity());
    request.globalCopy = copy;
    request.globalAccessSemanticsPreserved =
        copy.getPtr() == load.getPtr() &&
        copy.getOffsets() == load.getOffsets() &&
        copy.getMask() == load.getMask() &&
        copy.getOther() == load.getOther() &&
        copy.getStride() == load.getStride() &&
        copy.getCache() == load.getCache() &&
        copy.getContiguity() == load.getContiguity();
  } else {
    error = "global_to_lds producer is not a supported global load";
    return failure();
  }
  if (!request.globalAccessSemanticsPreserved) {
    error = "global_to_lds changed global access semantics";
    return failure();
  }

  request.asyncCommit = gpu::AsyncCommitGroupOp::create(
      builder, loc, request.globalCopy->getResult(0));
  request.asyncWait = gpu::AsyncWaitOp::create(
      builder, loc, request.asyncCommit->getResult(0), 0);
  request.visibilityBarrier =
      gpu::BarrierOp::create(builder, loc, gpu::AddrSpace::Local);
  request.load = gpu::LocalLoadOp::create(builder, loc, request.tensorType,
                                          destination, request.asyncWait);
  request.directToLdsCopies = 1;
  request.asyncCommitsInserted = 1;
  request.asyncWaitsInserted = 1;
  return success();
}

static Value addRingIndex(PlanLdsStaging &request, scf::ForOp &loop) {
  OpBuilder outerBuilder(loop);
  Location loc = loop.getLoc();
  Value one = arith::ConstantIntOp::create(outerBuilder, loc, 1, 32);
  Value minusOne = arith::ConstantIntOp::create(outerBuilder, loc, -1, 32);
  Value depth =
      arith::ConstantIntOp::create(outerBuilder, loc, request.bufferDepth, 32);

  unsigned argumentIndex = loop.getBody()->getNumArguments();
  loop = ::mlir::addIterArgsToLoop(outerBuilder, loop, {minusOne});
  Value ring = loop.getBody()->getArgument(argumentIndex);
  OpBuilder bodyBuilder(loop.getBody(), loop.getBody()->begin());
  Value incremented = arith::AddIOp::create(bodyBuilder, loc, ring, one);
  Value wrapped =
      arith::RemUIOp::create(bodyBuilder, loc, incremented, depth);
  ::mlir::appendToForOpYield(loop, {wrapped});
  return wrapped;
}

} // namespace

LogicalResult materializeLdsStaging(MutableArrayRef<PlanLdsStaging> staging,
                                    PlanStagingMaterializationResult &result,
                                    std::string &error) {
  if (!staging.empty())
    result.loop = staging.front().loop;
  for (PlanLdsStaging &request : staging) {
    Operation *producer = request.source.getDefiningOp();
    if (!request.loop || !producer || !request.tensorType ||
        request.logicalBytes <= 0 || request.alignment <= 0 ||
        request.bufferDepth <= 0 || request.distance < 0 ||
        request.consumers.empty() || request.consumerOperands.empty()) {
      error = "invalid resolved LDS-staging request";
      return failure();
    }
    sortStagingPath(request);

    auto sharedEncoding = getSharedEncoding(request.tensorType);
    auto sharedMemory =
        gpu::SharedMemorySpaceAttr::get(request.loop.getContext());
    auto memdescType = gpu::MemDescType::get(
        request.tensorType.getShape(), request.tensorType.getElementType(),
        sharedEncoding, sharedMemory, /*mutableMemory=*/true);
    if (request.bufferDepth > 1) {
      if (request.action != "global_to_lds" || request.distance < 1 ||
          request.bufferDepth <= request.distance) {
        error = "invalid buffered global-to-LDS staging request";
        return failure();
      }
      request.ringIndex = addRingIndex(request, result.loop);
      for (PlanLdsStaging &candidate : staging)
        candidate.loop = result.loop;
      memdescType =
          triton::getMultiBufferedType(memdescType, request.bufferDepth);
    }

    OpBuilder allocationBuilder(result.loop);
    request.allocation = gpu::LocalAllocOp::create(
        allocationBuilder, producer->getLoc(), memdescType);
    request.allocation->setAttr(
        "alignment", allocationBuilder.getI32IntegerAttr(request.alignment));
    ++result.newAllocations;

    if (request.bufferDepth > 1) {
      OpBuilder viewBuilder(producer);
      request.bufferView =
          triton::createSingleBufferView(viewBuilder, request.allocation,
                                         request.ringIndex)
              .getDefiningOp<gpu::MemDescIndexOp>();
    }

    OpBuilder deallocationBuilder(result.loop->getBlock(),
                                  std::next(result.loop->getIterator()));
    gpu::LocalDeallocOp::create(deallocationBuilder, result.loop.getLoc(),
                                request.allocation);

    if (request.action == "register_to_lds") {
      OpBuilder storeBuilder(producer->getBlock(),
                             std::next(producer->getIterator()));
      request.store = gpu::LocalStoreOp::create(
          storeBuilder, producer->getLoc(), request.source, request.allocation);
      request.visibilityBarrier = gpu::BarrierOp::create(
          storeBuilder, producer->getLoc(), gpu::AddrSpace::Local);
      ++result.newStores;
      ++result.insertedBarriers;

      Operation *firstUse = request.consumers.front();
      if (!request.derivedOperations.empty() &&
          request.derivedOperations.front()->isBeforeInBlock(firstUse))
        firstUse = request.derivedOperations.front();
      OpBuilder loadBuilder(firstUse);
      request.load =
          gpu::LocalLoadOp::create(loadBuilder, firstUse->getLoc(),
                                   request.tensorType, request.allocation);
    } else if (request.action == "global_to_lds") {
      if (failed(materializeGlobalCopy(request, producer, error)))
        return failure();
      ++result.insertedBarriers;
    } else {
      error = "unknown resolved LDS-staging action";
      return failure();
    }
    ++result.newLoads;

    if (failed(
            rewriteDerivedConsumers(request, request.load.getResult(), error)))
      return failure();

    if (request.action == "global_to_lds") {
      if (!request.source.use_empty()) {
        error = "global_to_lds retained a register consumer";
        return failure();
      }
      producer->erase();
      request.source = {};
      request.globalLoadsEliminated = 1;
      request.registerSourceEliminated = true;
      request.logicalLiveRangeShortened = true;
    }

    Operation *lastConsumer = request.consumers.front();
    for (Operation *consumer : request.consumers)
      if (lastConsumer->isBeforeInBlock(consumer))
        lastConsumer = consumer;
    OpBuilder releaseBuilder(lastConsumer->getBlock(),
                             std::next(lastConsumer->getIterator()));
    request.releaseBarrier = gpu::BarrierOp::create(
        releaseBuilder, lastConsumer->getLoc(), gpu::AddrSpace::Local);
    ++result.insertedBarriers;
  }
  return success();
}

} // namespace mlir::triton::amdgpu
