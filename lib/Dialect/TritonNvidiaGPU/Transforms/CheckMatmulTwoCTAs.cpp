#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class TritonNvidiaGPUCheckMatmulTwoCTAPass
    : public impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
          TritonNvidiaGPUCheckMatmulTwoCTAPass> {
public:
  using impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
      TritonNvidiaGPUCheckMatmulTwoCTAPass>::
      TritonNvidiaGPUCheckMatmulTwoCTAPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Operation *firstMatmul = nullptr;
    bool firstTwoCTA = false;

    auto checkTwoCTA = [&](Operation *op, bool currentTwoCTA) -> WalkResult {
      if (!firstMatmul) {
        firstMatmul = op;
        firstTwoCTA = currentTwoCTA;
        return WalkResult::advance();
      }
      if (currentTwoCTA != firstTwoCTA) {
        auto diag = op->emitError()
                    << "inconsistent two_ctas setting across matmuls; "
                       "expected all matmuls to "
                    << (firstTwoCTA ? "enable" : "disable") << " two_ctas.";
        diag.attachNote(firstMatmul->getLoc())
            << "first matmul here has two_ctas="
            << (firstTwoCTA ? "true" : "false") << ".";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    WalkResult result = mod.walk([&](Operation *op) {
      if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op))
        return checkTwoCTA(op, mmaOp.getTwoCtas());
      if (auto scaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op))
        return checkTwoCTA(op, scaledOp.getTwoCtas());
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    bool twoCTAValue = firstMatmul ? firstTwoCTA : false;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
