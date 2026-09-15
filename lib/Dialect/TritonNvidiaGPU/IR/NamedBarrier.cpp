#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <algorithm>

namespace mlir::triton::nvidia_gpu {
namespace {

std::optional<int32_t> getConstantId(Value value) {
  // Resolve `ttg.warp_specialize` captures first: a statically known ID passed
  // into a partition arrives as a block argument with no defining op, and
  // reading it as dynamic would wrongly mark the whole pool unavailable.
  auto constant = gpu::resolveWarpSpecializeCapture(value)
                      .getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto valueAttr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!valueAttr)
    return std::nullopt;
  return static_cast<int32_t>(valueAttr.getInt());
}

} // namespace

NamedBarrierIdAllocator::NamedBarrierIdAllocator(ModuleOp module) {
  if (auto ids = module->getAttrOfType<DenseI32ArrayAttr>(
          kWarpSpecializeBarrierIdsAttrName))
    reserve(ids.asArrayRef());

  module.walk([&](UserNamedBarrierIdOp op) {
    std::optional<int32_t> id = getConstantId(op.getValue());
    if (!id) {
      hasDynamicUserId_ = true;
      return;
    }
    if (*id >= 0 && *id <= kLastNamedBarrierId)
      userIds_[*id] = true;
    reserve(ArrayRef<int32_t>(*id));
  });
  module.walk([&](CompilerNamedBarrierIdOp op) {
    if (std::optional<int32_t> id = getConstantId(op.getValue())) {
      if (*id >= 0 && *id <= kLastNamedBarrierId)
        compilerIds_[*id] = true;
      reserve(ArrayRef<int32_t>(*id));
    }
  });
}

bool NamedBarrierIdAllocator::isReservedByUser(int32_t id) const {
  return id >= 0 && id <= kLastNamedBarrierId && userIds_[id];
}

bool NamedBarrierIdAllocator::isReservedByCompiler(int32_t id) const {
  return id >= 0 && id <= kLastNamedBarrierId && compilerIds_[id];
}

std::optional<SmallVector<int32_t>>
NamedBarrierIdAllocator::allocate(unsigned count) {
  if (hasDynamicUserId_)
    return std::nullopt;

  SmallVector<int32_t> ids;
  for (int32_t id = kFirstAllocatableBarrierId;
       id <= kLastNamedBarrierId && ids.size() < count; ++id) {
    if (!used_[id])
      ids.push_back(id);
  }
  if (ids.size() != count)
    return std::nullopt;
  reserve(ids);
  return ids;
}

void NamedBarrierIdAllocator::reserve(ArrayRef<int32_t> ids) {
  for (int32_t id : ids) {
    if (id >= 0 && id <= kLastNamedBarrierId)
      used_[id] = true;
  }
}

SmallVector<int32_t> getWarpSpecializeBarrierIds(ModuleOp module) {
  auto attr = module->getAttrOfType<DenseI32ArrayAttr>(
      kWarpSpecializeBarrierIdsAttrName);
  if (!attr)
    return {};
  return llvm::to_vector(attr.asArrayRef());
}

static LogicalResult
ensureWarpSpecializeBarrierIdsImpl(ModuleOp module,
                                   NamedBarrierIdAllocator &allocator,
                                   bool emitDiagnostics) {
  auto fail = [&](const Twine &message) {
    if (emitDiagnostics)
      module.emitError(message);
    return failure();
  };
  unsigned maxPartitions = 0;
  module.walk([&](gpu::WarpSpecializeOp op) {
    maxPartitions = std::max<unsigned>(maxPartitions,
                                       op.getPartitionRegions().size());
  });
  if (maxPartitions == 0)
    return success();

  // A dynamic user ID is only a problem once an ID actually has to be drawn
  // from the pool. The fixed partition ID is reserved, not allocated, so a
  // kernel whose partitions it already covers stays legal -- see the dynamic
  // check at the allocation site below.
  SmallVector<int32_t> ids = getWarpSpecializeBarrierIds(module);
  if (ids.empty()) {
    ids.push_back(kFirstPartitionBarrierId);
    allocator.reserve(ids);
  }
  if (ids.front() != kFirstPartitionBarrierId)
    return fail("warp-specialize barrier ID mapping must start with "
                "reserved ID 2");
  llvm::SmallDenseSet<int32_t> uniqueIds;
  for (int32_t id : ids) {
    if (id < 0 || id > kLastNamedBarrierId)
      return fail("warp-specialize named barrier ID is outside the hardware "
                  "range [0, 15]");
    if (!uniqueIds.insert(id).second)
      return fail("warp-specialize named barrier IDs must be unique");
    if (allocator.isReservedByUser(id))
      return fail("warp-specialize named barrier ID collides with a user ID");
    if (allocator.isReservedByCompiler(id))
      return fail(
          "warp-specialize named barrier ID collides with a compiler ID");
  }
  if (ids.size() < maxPartitions) {
    auto additional = allocator.allocate(maxPartitions - ids.size());
    if (!additional) {
      if (allocator.hasDynamicUserId())
        return fail("cannot allocate warp-specialize named barriers with a "
                    "dynamic user named-barrier ID");
      return fail("not enough named barriers for warp-specialize partitions");
    }
    llvm::append_range(ids, *additional);
  }
  module->setAttr(kWarpSpecializeBarrierIdsAttrName,
                  DenseI32ArrayAttr::get(module.getContext(), ids));
  return success();
}

LogicalResult
ensureWarpSpecializeBarrierIds(ModuleOp module,
                               NamedBarrierIdAllocator &allocator) {
  return ensureWarpSpecializeBarrierIdsImpl(module, allocator,
                                            /*emitDiagnostics=*/true);
}

LogicalResult
tryEnsureWarpSpecializeBarrierIds(ModuleOp module,
                                  NamedBarrierIdAllocator &allocator) {
  return ensureWarpSpecializeBarrierIdsImpl(module, allocator,
                                            /*emitDiagnostics=*/false);
}

Value createCompilerNamedBarrierId(OpBuilder &builder, Location loc,
                                   int32_t id) {
  Value value = arith::ConstantIntOp::create(builder, loc, id, 32);
  return CompilerNamedBarrierIdOp::create(builder, loc, value);
}

} // namespace mlir::triton::nvidia_gpu
