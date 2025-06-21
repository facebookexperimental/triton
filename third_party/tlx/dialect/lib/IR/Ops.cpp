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
#include "IR/OpsEnums.cpp.inc"

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

} // namespace tlx
} // namespace triton
} // namespace mlir
