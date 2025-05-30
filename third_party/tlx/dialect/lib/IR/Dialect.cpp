#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "IR/Dialect.h"
#include "IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::tlx;

void mlir::triton::tlx::TLXDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/TLXAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "IR/Ops.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "IR/TLXAttrDefs.cpp.inc"
