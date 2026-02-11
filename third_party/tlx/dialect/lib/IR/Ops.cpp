#include "IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace tlx {

//-- RequireLayoutOp --

OpFoldResult RequireLayoutOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType()) {
    // no-op
    return getSrc();
  }
  return {};
}

//-- StorageAliasSpecOp --

LogicalResult StorageAliasSpecOp::verify() {
  // Verify storage kind is valid for storage alias specs (smemCluster not
  // allowed) Note: smemCluster is not in the enum, so we only check for valid
  // values
  auto storage = getStorage();
  if (storage != StorageKind::smem && storage != StorageKind::tmem) {
    return emitOpError("unsupported storage kind for storage_alias_spec, "
                       "expected smem or tmem");
  }

  // Verify buffer_size_bytes is positive if specified (null is valid)
  auto sizeAttr = getBufferSizeBytesAttr();
  if (sizeAttr) {
    int64_t size = sizeAttr.getInt();
    if (size <= 0) {
      return emitOpError("buffer_size_bytes must be positive, got ") << size;
    }
  }

  return success();
}

//-- StorageAliasLocalAllocOp --

LogicalResult StorageAliasLocalAllocOp::verify() {
  // Verify that the storage alias and result have compatible storage kinds
  auto storageAliasType =
      cast<StorageAliasSpecType>(getStorageAlias().getType());
  auto storageAliasStorage = storageAliasType.getStorage();

  auto resultType = cast<triton::gpu::MemDescType>(getResult().getType());
  auto resultMemorySpace = resultType.getMemorySpace();

  // Check consistency between storage alias storage and result memory space
  if (storageAliasStorage == StorageKind::smem) {
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(resultMemorySpace)) {
      return emitOpError(
          "storage_alias_spec has smem storage but result is not in shared "
          "memory");
    }
  } else if (storageAliasStorage == StorageKind::tmem) {
    if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(resultMemorySpace)) {
      return emitOpError(
          "storage_alias_spec has tmem storage but result is not in tensor "
          "memory");
    }
  }

  return success();
}

//-- ReuseGroupOp --

LogicalResult ReuseGroupOp::verify() {
  auto elements = getElements();

  // Must have at least one element
  if (elements.empty()) {
    return emitOpError("reuse_group requires at least one element");
  }

  // Get result type properties
  auto resultType = cast<ReuseGroupType>(getResult().getType());
  auto expectedGroupKind = resultType.getGroupKind();

  // Verify group_kind attribute matches result type
  if (getGroupKind() != expectedGroupKind) {
    return emitOpError("group_kind attribute (")
           << stringifyReuseGroupKind(getGroupKind())
           << ") doesn't match result type group kind ("
           << stringifyReuseGroupKind(expectedGroupKind) << ")";
  }

  // Note: Validation that all elements reference the same storage_alias_spec
  // is performed by the SetBufferOverlapOp verifier when the overlap scheme
  // is defined. This allows reuse_group to be spec-agnostic.

  return success();
}

//-- SetBufferOverlapOp --

// Helper function to collect all leaf memdesc values from a reuse_group tree
static void
collectReuseGroupLeaves(mlir::Value value,
                        llvm::SmallVectorImpl<mlir::Value> &leaves) {
  // Check if this is a ReuseGroupOp result (nested reuse_group)
  if (auto reuseGroupOp = value.getDefiningOp<ReuseGroupOp>()) {
    // Recursively collect leaves from all elements
    for (auto element : reuseGroupOp.getElements()) {
      collectReuseGroupLeaves(element, leaves);
    }
  } else {
    // This is a leaf (memdesc from local_alloc)
    leaves.push_back(value);
  }
}

LogicalResult SetBufferOverlapOp::verify() {
  // Get the storage_alias_spec
  auto storageAliasSpec = getStorageAliasSpec();

  // Get the overlap_def (reuse_group)
  auto overlapDef = getOverlapDef();

  if (!storageAliasSpec) {
    return emitOpError("requires a valid storage_alias_spec");
  }

  if (!overlapDef) {
    return emitOpError("requires a valid overlap_def (reuse_group)");
  }

  // Get the ReuseGroupOp that defines the overlap_def
  auto reuseGroupOp = overlapDef.getDefiningOp<ReuseGroupOp>();
  if (!reuseGroupOp) {
    return emitOpError("overlap_def must be defined by a tlx.reuse_group op");
  }

  // Collect all leaf memdesc values from the reuse_group tree
  llvm::SmallVector<mlir::Value, 8> leaves;
  collectReuseGroupLeaves(overlapDef, leaves);

  if (leaves.empty()) {
    return emitOpError("reuse_group tree must contain at least one allocation");
  }

  // Check for duplicate elements in the reuse_group tree
  llvm::SmallDenseSet<mlir::Value, 8> seenElements;
  for (auto leaf : leaves) {
    if (!seenElements.insert(leaf).second) {
      return emitOpError("reuse_group tree contains duplicate elements; "
                         "each allocation can only appear once in the tree");
    }
  }

  // Verify that all leaves were allocated from the same storage_alias_spec
  for (auto leaf : leaves) {
    // Each leaf should be a memdesc produced by StorageAliasLocalAllocOp
    auto allocOp = leaf.getDefiningOp<StorageAliasLocalAllocOp>();
    if (!allocOp) {
      return emitOpError("all elements in the reuse_group tree must be "
                         "allocated via tlx.storage_alias_local_alloc, but "
                         "found an element that is not");
    }

    // Check that this allocation uses the same storage_alias_spec
    if (allocOp.getStorageAlias() != storageAliasSpec) {
      return emitOpError("all allocations in the reuse_group must reference "
                         "the same storage_alias_spec, but found an allocation "
                         "that uses a different spec");
    }
  }

  return success();
}

} // namespace tlx
} // namespace triton
} // namespace mlir
