#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir;

namespace {

struct TestBackwardSliceWithWSPass
    : public PassWrapper<TestBackwardSliceWithWSPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBackwardSliceWithWSPass);

  StringRef getArgument() const final {
    return "test-print-backward-slice-with-ws";
  }
  StringRef getDescription() const final {
    return "print the backward slice (WS-aware) for each "
           "map_to_remote_buffer result";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    AsmState state(mod);

    mod.walk([&](triton::nvidia_gpu::MapToRemoteBufferOp mapaOp) {
      SetVector<Operation *> slice;
      triton::nvidia_gpu::getBackwardSliceWithWS(mapaOp.getResult(), &slice);

      for (auto *op : slice) {
        std::string resultStr;
        llvm::raw_string_ostream os(resultStr);
        for (auto result : op->getResults()) {
          os << " ";
          result.printAsOperand(os, state);
        }
        mlir::emitRemark(op->getLoc())
            << "in_slice: " << op->getName() << resultStr;
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestBackwardSliceWithWSPass() {
  PassRegistration<TestBackwardSliceWithWSPass>();
}
} // namespace test
} // namespace mlir
