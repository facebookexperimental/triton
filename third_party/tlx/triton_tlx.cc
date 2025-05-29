#include "IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_tlx_passes(py::module &&m) {
  // TODO: add TLX passes
}

void init_triton_tlx(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_tlx_passes(passes.def_submodule("tlx"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tlx::TLXDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // py::class_<TritonOpBuilder>(m, "builder", py::module_local(),
  //                             py::dynamic_attr())
  //     .def(py::init<MLIRContext *>())
  //     .def("create_convert_layout",
  //          [](TritonOpBuilder &self, Value &v, Attribute &encoding) -> Value
  //          {
  //            Type newType;
  //            if (auto type = dyn_cast<ttg::MemDescType>(v.getType())) {
  //              newType = ttg::MemDescType::get(
  //                  type.getShape(), type.getElementType(), encoding,
  //                  type.getMemorySpace(), type.getMutableMemory());
  //            } else if (auto type = dyn_cast<RankedTensorType>(v.getType()))
  //            {
  //              newType = RankedTensorType::get(type.getShape(),
  //                                              type.getElementType(),
  //                                              encoding);
  //            } else {
  //              throw std::runtime_error("Unsupported type");
  //            }
  //            newType.dump();
  //            return self.create<ttg::ConvertLayoutOp>(newType, v);
  //          });
}
