// Pybind surface for the bitwise-equivalence reduction-order analysis.
// Exposes the MLIR-native checker (see lib/Analysis/ReductionOrder.cpp) to
// Python so `bitequiv/reduction_tree.py` can extract reduction-order signatures
// from a TTGIR module instead of regex-parsing the text.
#include "triton/Analysis/ReductionOrder.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace mlir;

void init_triton_bitequiv(py::module &&m) {
  m.doc() = "Bitwise-equivalence reduction-order analysis (MLIR-native).";

  // Parse a TTGIR module from text (the context must already have the dialects
  // loaded, e.g. via ir.load_dialects + nvidia.load_dialects) and return one
  // canonical reduction-order signature per tt.reduce (plus a conservative
  // entry for any MMA/dot accumulation). Two configs are bitwise-equivalent
  // reductions iff their signature lists are equal.
  m.def(
      "reduction_order_signatures",
      [](const std::string &ttgir,
         MLIRContext &ctx) -> std::vector<std::string> {
        OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ttgir, &ctx);
        if (!module)
          throw std::runtime_error("bitequiv.reduction_order_signatures: "
                                   "failed to parse TTGIR module");
        auto sigs = triton::bitequiv::getReductionOrderSignatures(*module);
        return std::vector<std::string>(sigs.begin(), sigs.end());
      },
      py::arg("ttgir"), py::arg("context"));
}
