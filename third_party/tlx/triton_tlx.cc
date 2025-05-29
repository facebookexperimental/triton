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
}
