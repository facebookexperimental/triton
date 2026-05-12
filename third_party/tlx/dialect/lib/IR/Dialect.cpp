#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#include "IR/Dialect.h"
#include "IR/Dialect.cpp.inc"
#include "IR/TLXTypesEnums.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::tlx;

void mlir::triton::tlx::TLXDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/TLXAttrDefs.cpp.inc"
      >();

  registerTypes();

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

bool mlir::triton::tlx::tlxExplicitClusterSync(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking explicit cluster sync");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr &&
         "expecting op nested in a module for checking explicit cluster sync");
  auto attr = module->getAttrOfType<BoolAttr>(AttrTLXExplicitClusterSyncName);
  return attr != nullptr && attr.getValue() == true;
}

bool mlir::triton::tlx::hasClusterSyncKernelInit(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking cluster sync kernel init");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr && "expecting op nested in a module for checking "
                              "cluster sync kernel init marker");
  auto attr = module->getAttrOfType<BoolAttr>(AttrClusterSyncKernelInitName);
  return attr != nullptr && attr.getValue() == true;
}

void mlir::triton::tlx::setClusterSyncKernelInitOnMod(Operation *op,
                                                      bool value) {
  assert(op != nullptr &&
         "expecting nonnull op for setting cluster sync kernel init");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr && "expecting op nested in a module for setting "
                              "cluster sync kernel init marker");
  module->setAttr(AttrClusterSyncKernelInitName,
                  BoolAttr::get(module->getContext(), value));
}

bool mlir::triton::tlx::hasClusterSyncKernelCleanup(Operation *op) {
  assert(op != nullptr &&
         "expecting nonnull op for checking cluster sync kernel cleanup");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr && "expecting op nested in a module for checking "
                              "cluster sync kernel cleanup marker");
  auto attr = module->getAttrOfType<BoolAttr>(AttrClusterSyncKernelCleanupName);
  return attr != nullptr && attr.getValue() == true;
}

void mlir::triton::tlx::setClusterSyncKernelCleanupOnMod(Operation *op,
                                                         bool value) {
  assert(op != nullptr &&
         "expecting nonnull op for setting cluster sync kernel cleanup");
  auto module = op;
  if (!isa<ModuleOp>(module)) {
    module = op->getParentOfType<ModuleOp>();
  }
  assert(module != nullptr && "expecting op nested in a module for setting "
                              "cluster sync kernel cleanup marker");
  module->setAttr(AttrClusterSyncKernelCleanupName,
                  BoolAttr::get(module->getContext(), value));
}

bool mlir::triton::tlx::tlxIsClustered(Operation *op) {
  assert(op != nullptr && "expecting nonnull op for checking cluster dims");
  auto moduleOp = op;
  if (!isa<ModuleOp>(moduleOp)) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  assert(moduleOp != nullptr &&
         "expecting op nested in a module for checking cluster dims");
  auto mod = cast<ModuleOp>(moduleOp);
  const SmallVector<int> clusterDims =
      triton::gpu::TritonGPUDialect::getClusterDims(mod);
  int clusterSize = 1;
  for (int d : clusterDims)
    clusterSize *= d;
  return clusterSize > 1;
}
