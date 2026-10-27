#include "test/include/Analysis/TestAxisInfo.h"
#include "triton/Analysis/IntegerRange.h"
#include "triton/Analysis/Utility.h"

namespace mlir {
namespace test {
namespace {
struct TestIntegerFactsPass
    : PassWrapper<TestIntegerFactsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIntegerFactsPass)
  TestIntegerFactsPass() = default;
  TestIntegerFactsPass(const TestIntegerFactsPass &other)
      : PassWrapper(other) {}
  Option<bool> generic{*this, "generic",
                       llvm::cl::desc("Use the MLIR range solver"),
                       llvm::cl::init(false)};
  StringRef getArgument() const override { return "test-integer-facts"; }
  void runOnOperation() override {
    auto solver =
        generic ? createDataFlowSolver() : triton::createIntegerRangeSolver();
    if (generic)
      solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();
    getOperation().walk([&](arith::CmpIOp op) {
      auto expected = op->getAttrOfType<StringAttr>(
          generic ? "test.generic_proof" : "test.proof");
      if (!expected)
        return;
      const auto *state =
          solver->lookupState<dataflow::IntegerValueRangeLattice>(
              op.getResult());
      std::optional<APInt> value;
      if (state && !state->getValue().isUninitialized())
        value = state->getValue().getValue().getConstantValue();
      StringRef actual = !value            ? "unknown"
                         : value->isZero() ? "disproven"
                                           : "proven";
      if (actual != expected.getValue()) {
        op.emitError() << "expected " << expected.getValue() << ", got "
                       << actual;
        signalPassFailure();
      }
    });
  }
};
} // namespace

void registerTestAlignmentPass() {
  PassRegistration<TestAxisInfoPass>();
  PassRegistration<TestIntegerFactsPass>();
}
} // namespace test
} // namespace mlir
