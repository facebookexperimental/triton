#pragma once

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/ProgramIdToAddressAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::test {

struct TestProgramIdAddressPass
    : public PassWrapper<TestProgramIdAddressPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProgramIdAddressPass);

  StringRef getArgument() const override {
    return "test-print-program-id-address";
  }
  StringRef getDescription() const final {
    return "print the result of the program id to address analysis pass";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Create and run the dataflow solver
    DataFlowSolver solver;
    // Load DeadCodeAnalysis first to mark blocks as executable
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<ProgramIdToAddressAnalysis>();

    if (failed(solver.initializeAndRun(moduleOp))) {
      signalPassFailure();
      return;
    }

    // Walk and print results for each value
    moduleOp.walk([&](triton::FuncOp funcOp) {
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          auto *lattice =
              solver.lookupState<dataflow::Lattice<ProgramIdAddressInfo>>(
                  result);
          if (!lattice)
            continue;

          const ProgramIdAddressInfo &info = lattice->getValue();
          InFlightDiagnostic diag = mlir::emitRemark(op->getLoc());
          std::string str;
          llvm::raw_string_ostream os(str);
          info.print(os);
          diag << str;
        }
      });
    });

    // Also print summary for load/store operations
    moduleOp.walk([&](Operation *op) {
      if (!isa<triton::LoadOp, triton::StoreOp>(op))
        return;

      Value ptrOperand = op->getOperand(0);
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ProgramIdAddressInfo>>(
              ptrOperand);
      if (!lattice)
        return;

      const ProgramIdAddressInfo &ptrInfo = lattice->getValue();

      InFlightDiagnostic diag = mlir::emitRemark(op->getLoc());
      diag << op->getName().getStringRef() << " address pattern: ";
      std::string str;
      llvm::raw_string_ostream os(str);
      ptrInfo.print(os);
      diag << str;
    });
  }
};

} // namespace mlir::test
