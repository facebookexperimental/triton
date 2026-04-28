#ifndef NVWS_TRANSFORMS_META_TO_NVWS_CONVERT_H_
#define NVWS_TRANSFORMS_META_TO_NVWS_CONVERT_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// Verify the representation contract consumed by block-local NVWS planning:
// every final managed allocation group and its complete memdesc/token use
// closure must reside in one top-level function CFG block.
LogicalResult validateNVWSManagedAllocationLocality(
    FuncOp func, StringRef diagnosticPrefix = "MetaToNVWSConvert");

} // namespace mlir::triton

#endif // NVWS_TRANSFORMS_META_TO_NVWS_CONVERT_H_
