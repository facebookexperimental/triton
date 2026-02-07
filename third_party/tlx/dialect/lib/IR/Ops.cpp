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

} // namespace tlx
} // namespace triton
} // namespace mlir
