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
    triton::IntegerFactAnalysis facts;
    if (failed(facts.initialize(getOperation())))
      return signalPassFailure();
    getOperation().walk([&](arith::CmpIOp op) {
      auto expected = op->getAttrOfType<StringAttr>("test.proof");
      if (!expected)
        return;
      triton::IntegerProof proof =
          facts.prove(op.getPredicate(), op.getLhs(), op.getRhs());
      StringRef actual = proof == triton::IntegerProof::Proven ? "proven"
                         : proof == triton::IntegerProof::Disproven
                             ? "disproven"
                             : "unknown";
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
