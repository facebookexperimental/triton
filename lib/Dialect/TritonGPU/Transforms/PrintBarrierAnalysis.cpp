//===-- PrintBarrierAnalysis.cpp - Print barrier analysis ------*- C++ -*-===//
//
// This file implements a pass that runs the BarrierExecutionOrderAnalysis
// and prints the results to stderr. Useful for testing and debugging.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/BarrierAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONPRINTBARRIERANALYSIS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct PrintBarrierAnalysisPass
    : public impl::TritonPrintBarrierAnalysisBase<PrintBarrierAnalysisPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Iterate over all functions in the module
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      llvm::errs() << "\n=== Barrier Analysis for function: "
                   << funcOp.getName() << " ===\n";

      BarrierExecutionOrderAnalysis analysis(funcOp);
      analysis.run();

      // Print full analysis
      analysis.print(llvm::errs());

      // Print execution trace visualization
      analysis.printExecutionTrace(llvm::errs());

      // Print DOT graph (can be piped to graphviz)
      llvm::errs() << "\n";
      analysis.printDependencyGraph(llvm::errs());
    });
  }
};

} // namespace

} // namespace mlir::triton::gpu
