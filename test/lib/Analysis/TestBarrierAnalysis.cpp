#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/BarrierAnalysis.h"

using namespace mlir;

namespace {

struct TestBarrierAnalysisPass
    : public PassWrapper<TestBarrierAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBarrierAnalysisPass);

  TestBarrierAnalysisPass() = default;
  TestBarrierAnalysisPass(const TestBarrierAnalysisPass &other)
      : PassWrapper<TestBarrierAnalysisPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "test-print-barrier-analysis"; }
  StringRef getDescription() const final {
    return "print the result of the barrier deadlock trace extraction";
  }

  Option<int> unrollBound{*this, "unroll-bound",
                          llvm::cl::desc("Loop unrolling bound"),
                          llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      triton::BarrierDeadlockAnalysis analysis(funcOp, unrollBound);
      analysis.run();

      if (analysis.getTaskTraces().empty())
        return;

      llvm::outs() << "=== Barrier Analysis: " << funcOp.getName() << " ===\n";
      analysis.printSummary(llvm::outs());
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestBarrierAnalysisPass() {
  PassRegistration<TestBarrierAnalysisPass>();
}
} // namespace test
} // namespace mlir
