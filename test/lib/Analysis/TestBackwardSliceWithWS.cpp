#include "mlir/Pass/Pass.h"
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
    return "print the backward slice (WS-aware) for ops marked with "
           "a 'slice_target' attribute";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](Operation *op) {
      if (!op->hasAttr("slice_target"))
        return;

      for (auto result : op->getResults()) {
        SetVector<Operation *> slice;
        triton::nvidia_gpu::getBackwardSliceWithWS(result, &slice);

        for (auto *sliceOp : slice)
          mlir::emitRemark(sliceOp->getLoc()) << "in_slice";
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
