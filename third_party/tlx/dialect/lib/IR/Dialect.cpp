#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#include "IR/Dialect.h"
#include "IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::tlx;

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void mlir::triton::tlx::TLXDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/TLXAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
}

#define GET_ATTRDEF_CLASSES
#include "IR/TLXAttrDefs.cpp.inc"

bool mlir::triton::tlx::tlxEnablePairedMMA(Operation *op) {
  assert(op != nullptr && "expecting nonnull op for checking TLX 2cta mode");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr &&
         "expecting op nested in a module for checking TLX 2cta mode");
  auto attr = module->getAttrOfType<BoolAttr>(AttrTLXEnablePairedCTAMMAName);
  return attr != nullptr && attr.getValue() == true;
}
