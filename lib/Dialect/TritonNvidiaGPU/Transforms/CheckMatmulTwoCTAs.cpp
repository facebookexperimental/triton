#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"

namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

tt::DotOp getDependentDotProducerImpl(Value value, DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return nullptr;

  Operation *def = value.getDefiningOp();
  if (!def)
    return nullptr;

  if (auto dotOp = dyn_cast<tt::DotOp>(def))
    return dotOp;

  for (Value operand : def->getOperands()) {
    if (auto producer = getDependentDotProducerImpl(operand, visited))
      return producer;
  }
  return nullptr;
}

tt::DotOp getDependentDotProducer(Value value) {
  DenseSet<Value> visited;
  return getDependentDotProducerImpl(value, visited);
}

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

    auto checkNoDependentTwoCTADot = [&](tt::DotOp op) -> WalkResult {
      if (!op.getTwoCtas())
        return WalkResult::advance();
      for (Value operand : {op.getA(), op.getB()}) {
        auto producer = getDependentDotProducer(operand);
        if (!producer || producer == op || !producer.getTwoCtas())
          continue;
        auto diag = op->emitError()
                    << "two_ctas=True does not currently support dependent "
                       "matmul chains where one 2-CTA dot consumes a value "
                       "derived from another 2-CTA dot result.";
        diag.attachNote(producer->getLoc())
            << "producer 2-CTA dot result is consumed by this dot.";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    WalkResult result = mod.walk([&](Operation *op) {
      if (auto dotOp = dyn_cast<tt::DotOp>(op))
        return checkTwoCTA(op, dotOp.getTwoCtas());
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

    result =
        mod.walk([&](tt::DotOp op) { return checkNoDependentTwoCTADot(op); });
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
