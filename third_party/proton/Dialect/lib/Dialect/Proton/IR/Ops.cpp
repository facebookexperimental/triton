#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Proton/IR/Ops.cpp.inc"

#include "Dialect/Proton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {

LogicalResult RecordOp::verify() {
  if (Value predicate = getPredicate()) {
    auto predTy = dyn_cast<IntegerType>(predicate.getType());
    if (!predTy || predTy.getWidth() != 1)
      return emitOpError("predicate must be a scalar i1");
  }
  return success();
}

} // namespace proton
} // namespace triton
} // namespace mlir
