#ifndef TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_
#define TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir::triton {

inline constexpr llvm::StringLiteral kMulticastAxesAttrName =
    "tt.multicast_axes";

inline LogicalResult verifyTMAMulticastAxes(Operation *op,
                                            DenseI32ArrayAttr axes) {
  if (!axes)
    return success();
  if (axes.empty())
    return op->emitOpError("tt.multicast_axes must not be empty");
  for (int32_t axis : axes.asArrayRef())
    if (axis < 0 || axis >= 3)
      return op->emitOpError(
          "tt.multicast_axes values must be in [0, 2]");
  return success();
}
} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITON_IR_TMAMULTICAST_H_
