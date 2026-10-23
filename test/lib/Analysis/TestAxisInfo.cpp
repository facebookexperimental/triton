#include "test/include/Analysis/TestAxisInfo.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Uniformity.h"

namespace mlir {
namespace test {
namespace {
struct TestUniformityPass
    : PassWrapper<TestUniformityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUniformityPass)
  StringRef getArgument() const override { return "test-uniformity"; }
  StringRef getDescription() const override {
    return "check common-dialect uniformity without changing the input IR";
  }
  void runOnOperation() override {
    getOperation().walk([&](FunctionOpInterface function) {
      auto arguments =
          function->getAttrOfType<ArrayAttr>("test.uniform_arguments");
      if (!arguments || function.getFunctionBody().empty())
        return;
      UniformityAnalysis::Options options;
      options.linearThreadIdX = function->hasAttr("test.linear_thread_id_x");
      options.isUniformArgument = [&](BlockArgument arg) {
        return arg.getArgNumber() < arguments.size() &&
               cast<BoolAttr>(arguments[arg.getArgNumber()]).getValue();
      };
      options.getRule =
          [&](Operation *op,
              unsigned) -> std::optional<UniformityAnalysis::Rule> {
        if (!function->isAncestor(op)) {
          function.emitError("uniformity model received a cloned operation");
          signalPassFailure();
          return UniformityAnalysis::Rule::Unknown;
        }
        if (op->hasAttr("test.uniform_model"))
          return UniformityAnalysis::Rule::Uniform;
        return std::nullopt;
      };
      std::string before;
      llvm::raw_string_ostream beforeStream(before);
      function->print(beforeStream);
      UniformityAnalysis analysis(function, std::move(options));
      if (failed(analysis.getStatus())) {
        function.emitError("uniformity analysis failed");
        signalPassFailure();
        return;
      }
      function->walk([&](Operation *op) {
        if (auto expected = op->getAttrOfType<BoolAttr>("test.uniform")) {
          if (op->getNumResults() != 1 ||
              analysis.isUniform(op->getResult(0)) != expected.getValue()) {
            op->emitError("unexpected result uniformity; expected ")
                << expected;
            signalPassFailure();
          }
        }
        if (auto expected =
                op->getAttrOfType<ArrayAttr>("test.uniform_operands")) {
          if (expected.size() != op->getNumOperands()) {
            op->emitError("incorrect number of operand expectations");
            signalPassFailure();
            return;
          }
          for (auto [i, operand] : llvm::enumerate(op->getOpOperands())) {
            bool want = cast<BoolAttr>(expected[i]).getValue();
            if (analysis.isUniform(operand) != want) {
              op->emitError("unexpected operand uniformity at index ") << i;
              signalPassFailure();
            }
          }
        }
      });
      std::string after;
      llvm::raw_string_ostream afterStream(after);
      function->print(afterStream);
      if (before != after) {
        function.emitError("uniformity analysis changed its input");
        signalPassFailure();
      }
    });
  }
};
} // namespace
void registerTestAlignmentPass() {
  PassRegistration<TestAxisInfoPass>();
  PassRegistration<TestUniformityPass>();
}
} // namespace test
} // namespace mlir
