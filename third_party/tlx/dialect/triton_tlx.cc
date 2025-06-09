#include "IR/Dialect.h"
#include "ir.h" // TritonOpBuilder
#include "mlir/Pass/PassManager.h"
#include "passes.h"

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace ttg = triton::gpu;

void init_triton_tlx_ir(py::module &&m) {
  // auto *builder_cls = ir::getBuilderClass();
  // builder_cls->def(
  //     "create_require_layout",
  //     [](TritonOpBuilder &self, Value &v, Attribute &encoding) -> Value {
  //       if (auto type = dyn_cast<ttg::MemDescType>(v.getType())) {
  //         auto newType = ttg::MemDescType::get(
  //             type.getShape(), type.getElementType(), encoding,
  //             type.getMemorySpace(), type.getMutableMemory());
  //         return self.create<tlx::RequireLayoutOp>(newType, v);
  //       } else {
  //         throw std::runtime_error("Unsupported type");
  //       }
  //     });
}

void init_triton_tlx_passes(py::module &&m) {
  // TODO: add TLX passes
}

void init_triton_tlx(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tlx::TLXDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_tlx_ir(m.def_submodule("tlx_ir"));
  init_triton_tlx_passes(m.def_submodule("tlx_passes"));
}
