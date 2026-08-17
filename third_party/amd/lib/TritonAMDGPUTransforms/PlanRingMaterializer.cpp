#include "PlanRingMaterializer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <iterator>
#include <limits>

namespace mlir::triton::amdgpu {
namespace {

static bool isLocalBarrier(Operation *operation) {
  auto barrier = dyn_cast_or_null<gpu::BarrierOp>(operation);
  return barrier && barrier.hasLocal();
}

static bool isBeforeInBlock(Operation *left, Operation *right) {
  if (!left || !right || left->getBlock() != right->getBlock())
    return false;
  return left->isBeforeInBlock(right);
}

static bool hasLocalBarrierBetween(Operation *begin, Operation *end) {
  if (!isBeforeInBlock(begin, end))
    return false;
  for (Operation *cursor = begin->getNextNode(); cursor && cursor != end;
       cursor = cursor->getNextNode())
    if (isLocalBarrier(cursor))
      return true;
  return false;
}

static bool hasLocalBarrierAfter(Operation *operation) {
  if (!operation)
    return false;
  for (Operation *cursor = operation->getNextNode(); cursor;
       cursor = cursor->getNextNode()) {
    if (isLocalBarrier(cursor))
      return true;
    if (cursor->hasTrait<OpTrait::IsTerminator>())
      break;
  }
  return false;
}

static Value buildSlotIndex(OpBuilder &builder, Location location,
                            scf::ForOp loop, int64_t depth, int64_t offset) {
  if (depth == 1)
    return arith::ConstantIntOp::create(builder, location, 0, 32);
  Value induction = arith::IndexCastOp::create(
      builder, location, builder.getI32Type(), loop.getInductionVar());
  offset %= depth;
  if (offset < 0)
    offset += depth;
  if (offset) {
    Value offsetValue =
        arith::ConstantIntOp::create(builder, location, offset, 32);
    induction =
        arith::AddIOp::create(builder, location, induction, offsetValue);
  }
  Value modulus = arith::ConstantIntOp::create(builder, location, depth, 32);
  return arith::RemUIOp::create(builder, location, induction, modulus);
}

static LogicalResult rewriteView(gpu::MemDescIndexOp view, scf::ForOp loop,
                                 int64_t depth, int64_t offset,
                                 std::string &error) {
  if (!view || view->getBlock() != loop.getBody()) {
    error = "ring slot view is not directly in the selected scf.for";
    return failure();
  }
  OpBuilder builder(view);
  Value index = buildSlotIndex(builder, view.getLoc(), loop, depth, offset);
  view.getIndexMutable().assign(index);
  return success();
}

} // namespace

LogicalResult
materializeExistingLdsRings(ArrayRef<PlanExistingLdsRingMutation> rings,
                            ArrayRef<PlanAsyncWaitMutation> waits,
                            PlanRingMaterializationResult &result,
                            std::string &error) {
  llvm::DenseSet<Operation *> resized;
  llvm::DenseMap<Operation *, int64_t> viewOffsets;
  llvm::DenseMap<Operation *, const PlanExistingLdsRingMutation *> viewRings;

  for (const PlanExistingLdsRingMutation &ring : rings) {
    gpu::LocalAllocOp allocation = ring.allocation;
    if (!ring.loop || !ring.allocation || ring.oldDepth < 1 ||
        ring.newDepth < 1 || ring.newDistance < 1 ||
        ring.newDistance > ring.newDepth) {
      error = "invalid resolved existing-LDS ring mutation";
      return failure();
    }
    if (resized.insert(allocation).second && ring.oldDepth != ring.newDepth) {
      auto oldType = cast<gpu::MemDescType>(allocation.getType());
      SmallVector<int64_t> shape(oldType.getShape());
      SmallVector<int64_t> allocShape(oldType.getAllocShape());
      if (shape.size() < 2 || allocShape.size() != shape.size() ||
          shape.front() != ring.oldDepth ||
          allocShape.front() != ring.oldDepth) {
        error = "existing LDS allocation is not a canonical leading-dimension "
                "ring";
        return failure();
      }
      shape.front() = ring.newDepth;
      allocShape.front() = ring.newDepth;
      allocation.getResult().setType(gpu::MemDescType::get(
          shape, oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), oldType.getMutableMemory(), allocShape));
      ++result.resizedAllocations;
    }

    auto recordView = [&](gpu::MemDescIndexOp view, int64_t offset) {
      auto [it, inserted] = viewOffsets.try_emplace(view, offset);
      if (!inserted && it->second != offset)
        return false;
      viewRings[view] = &ring;
      return true;
    };
    for (gpu::MemDescIndexOp view : ring.producerViews)
      if (!recordView(view, /*offset=*/0)) {
        error = "one LDS view has incompatible producer slot roles";
        return failure();
      }
    int64_t consumerOffset =
        (ring.newDepth - (ring.newDistance % ring.newDepth)) % ring.newDepth;
    for (gpu::MemDescIndexOp view : ring.consumerViews)
      if (!recordView(view, consumerOffset)) {
        error = "one LDS view has incompatible producer/consumer slot roles";
        return failure();
      }
  }

  for (const auto &[operation, offset] : viewOffsets) {
    auto view = cast<gpu::MemDescIndexOp>(operation);
    const PlanExistingLdsRingMutation &ring = *viewRings.lookup(operation);
    if (failed(rewriteView(view, ring.loop, ring.newDepth, offset, error)))
      return failure();
    ++result.rewrittenSlotIndices;
  }

  llvm::DenseSet<Operation *> updatedWaits;
  for (const PlanAsyncWaitMutation &mutation : waits) {
    gpu::AsyncWaitOp wait = mutation.wait;
    if (!wait || mutation.newRetainedGroupCount < 0 ||
        mutation.newRetainedGroupCount > std::numeric_limits<int32_t>::max()) {
      error = "invalid derived async wait count";
      return failure();
    }
    if (updatedWaits.insert(wait).second &&
        mutation.oldRetainedGroupCount != mutation.newRetainedGroupCount) {
      wait.setNum(static_cast<int32_t>(mutation.newRetainedGroupCount));
      ++result.updatedWaits;
    }

    Operation *firstConsumer = nullptr;
    for (Operation *consumer : mutation.consumers)
      if (consumer && consumer->getBlock() == wait->getBlock() &&
          isBeforeInBlock(wait, consumer) &&
          (!firstConsumer || isBeforeInBlock(consumer, firstConsumer)))
        firstConsumer = consumer;
    if (firstConsumer && !hasLocalBarrierBetween(wait, firstConsumer)) {
      OpBuilder builder(wait->getBlock(), std::next(wait->getIterator()));
      gpu::BarrierOp::create(builder, wait.getLoc(), gpu::AddrSpace::Local);
      ++result.insertedVisibilityBarriers;
    }
  }

  llvm::DenseSet<Operation *> releaseConsumers;
  for (const PlanExistingLdsRingMutation &ring : rings) {
    scf::ForOp loop = ring.loop;
    Operation *lastConsumer = nullptr;
    for (Operation *consumer : ring.consumers) {
      if (!consumer || consumer->getBlock() != loop.getBody())
        continue;
      if (!lastConsumer || isBeforeInBlock(lastConsumer, consumer))
        lastConsumer = consumer;
    }
    if (!lastConsumer || !releaseConsumers.insert(lastConsumer).second ||
        hasLocalBarrierAfter(lastConsumer))
      continue;
    OpBuilder builder(lastConsumer->getBlock(),
                      std::next(lastConsumer->getIterator()));
    gpu::BarrierOp::create(builder, lastConsumer->getLoc(),
                           gpu::AddrSpace::Local);
    ++result.insertedReleaseBarriers;
  }
  return success();
}

} // namespace mlir::triton::amdgpu
