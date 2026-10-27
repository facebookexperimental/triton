#include "test/include/Analysis/TestAxisInfo.h"
#include "triton/Analysis/IntegerRange.h"

namespace mlir {
namespace test {
namespace {
struct TestIntegerFactsPass
    : PassWrapper<TestIntegerFactsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIntegerFactsPass)
  StringRef getArgument() const override { return "test-integer-facts"; }
  void runOnOperation() override {
    auto solver = triton::createIntegerRangeSolver(getOperation());
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();
    getOperation().walk([&](arith::CmpIOp op) {
      auto expected = op->getAttrOfType<StringAttr>("test.proof");
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
