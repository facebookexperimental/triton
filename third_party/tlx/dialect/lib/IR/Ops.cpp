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

} // namespace tlx
} // namespace triton
} // namespace mlir
