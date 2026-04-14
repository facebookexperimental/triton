//===-- BarrierDeadlockDetection.cpp - Deadlock detection pass ---*- C++
//-*-===//
//
// Pass entry point for barrier deadlock detection. Instantiates
// BarrierDeadlockAnalysis, runs trace extraction, and dumps a standalone
// Python Z3 script encoding the barrier constraints.
//
// See docs/barrier_deadlock_detection_design.md for design details.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/BarrierAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUBARRIERDEADLOCKDETECTION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct BarrierDeadlockDetectionPass
    : public impl::TritonGPUBarrierDeadlockDetectionBase<
          BarrierDeadlockDetectionPass> {
  using TritonGPUBarrierDeadlockDetectionBase::
      TritonGPUBarrierDeadlockDetectionBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp.walk([&](FunctionOpInterface funcOp) {
      BarrierDeadlockAnalysis analysis(funcOp, unrollBound);
      analysis.run();

      if (analysis.getTaskTraces().empty())
        return;

      // Dump Z3 script to file or stderr.
      if (!outputPath.empty()) {
        std::error_code ec;
        llvm::raw_fd_ostream fileOs(outputPath, ec);
        if (ec) {
          funcOp.emitWarning()
              << "Failed to open output file: " << ec.message();
          return;
        }
        analysis.dumpPythonZ3Script(fileOs);

        // Optionally run the solver.
        if (runSolver) {
          auto python = llvm::sys::findProgramByName("python3");
          if (!python) {
            funcOp.emitWarning() << "python3 not found; skipping solver";
            return;
          }
          llvm::StringRef args[] = {*python, outputPath};
          int exitCode = llvm::sys::ExecuteAndWait(*python, args);
          if (exitCode != 0)
            funcOp.emitWarning() << "Z3 solver exited with code " << exitCode;
        }
      } else {
        // Print to stderr with a summary header.
        llvm::errs() << "=== Barrier Deadlock Detection: " << funcOp.getName()
                     << " ===\n";
        analysis.printSummary(llvm::errs());
        llvm::errs() << "\n--- Z3 Script ---\n";
        analysis.dumpPythonZ3Script(llvm::errs());
        llvm::errs() << "--- End Z3 Script ---\n\n";
      }
    });
  }
};

} // namespace
} // namespace mlir::triton::gpu
