#include "triton/Tools/TimeTrace.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TimeProfiler.h"

namespace {

// Kept out of the Python bindings: they are built with RTTI, and subclassing
// PassInstrumentation there needs its type info, which an MLIR built without
// RTTI does not provide. MLIR verifies after a pass before runAfterPass, so
// each span includes the verifier.
struct TimeTraceInstrumentation : public mlir::PassInstrumentation {
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::timeTraceProfilerBegin(pass->getName(), op->getName().getStringRef());
  }
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::timeTraceProfilerEnd();
  }
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::timeTraceProfilerEnd();
  }
};

} // namespace

void mlir::triton::tools::addTimeTraceInstrumentation(mlir::PassManager &pm) {
  pm.addInstrumentation(std::make_unique<TimeTraceInstrumentation>());
}
