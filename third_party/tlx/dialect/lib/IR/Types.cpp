#include "IR/Types.h"
#include "IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::tlx;

#define GET_TYPEDEF_CLASSES
#include "IR/TLXTypes.cpp.inc"

//-- StorageAliasSpecType --

LogicalResult
StorageAliasSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                             StorageKind storage,
                             std::optional<int64_t> bufferSizeBytes) {
  // smemCluster is not supported for storage_alias_spec
  // Note: smemCluster is not in the StorageKind enum, so this check
  // is a safeguard in case the enum is extended in the future
  if (storage != StorageKind::smem && storage != StorageKind::tmem) {
    return emitError() << "unsupported storage kind for storage_alias_spec, "
                       << "expected smem or tmem";
  }

  // Verify buffer_size_bytes is positive if specified
  if (bufferSizeBytes && *bufferSizeBytes <= 0) {
    return emitError() << "buffer_size_bytes must be positive, got "
                       << *bufferSizeBytes;
  }

  return success();
}

//-- ReuseGroupType --

LogicalResult
ReuseGroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ReuseGroupKind groupKind) {
  // No additional verification needed - groupKind is validated by the enum
  return success();
}

//===----------------------------------------------------------------------===//
// TLX Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::tlx::TLXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "IR/TLXTypes.cpp.inc"
      >();
}
