#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// `two_ctas` lives on the concrete source-level dot ops rather than on
// DotOpInterface, so read it per op kind. Returns nullopt when `op` is not a
// source-level dot, which also keeps the already-lowered tcgen05 ops out of the
// dependent-chain walk.
std::optional<bool> getSourceDotTwoCTAs(Operation *op) {
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return dotOp.getTwoCtas();
  if (auto dotScaledOp = dyn_cast<tt::DotScaledOp>(op))
    return dotScaledOp.getTwoCtas();
  return std::nullopt;
}

// Nearest source-level dot (plain or scaled) that produces `value`. Both forms
// must be recognized: a dependent chain can mix them in either direction.
tt::DotOpInterface getDependentDotProducerImpl(Value value,
                                               DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return {};

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  // Any source-level dot terminates the search, whether or not it is 2-CTA;
  // the caller decides what to do with a non-2-CTA producer. The cast is safe:
  // a non-nullopt result means `def` is tt.dot or tt.dot_scaled, and both
  // implement DotOpInterface.
  if (getSourceDotTwoCTAs(def).has_value())
    return cast<tt::DotOpInterface>(def);

  for (Value operand : def->getOperands()) {
    if (auto producer = getDependentDotProducerImpl(operand, visited))
      return producer;
  }
  return {};
}

tt::DotOpInterface getDependentDotProducer(Value value) {
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

    // Both source-level dot forms must be walked: `tt.dot_scaled` also carries
    // `two_ctas`, and a dependent chain can mix the two in either direction.
    // Already-lowered tcgen05 ops that implement the interface fall out through
    // getSourceDotTwoCTAs returning nullopt.
    auto checkNoDependentTwoCTADot = [&](tt::DotOpInterface op) -> WalkResult {
      Operation *dotOp = op.getOperation();
      if (!getSourceDotTwoCTAs(dotOp).value_or(false))
        return WalkResult::advance();
      for (Value operand : {op.getA(), op.getB()}) {
        auto producer = getDependentDotProducer(operand);
        if (!producer || producer == dotOp ||
            !getSourceDotTwoCTAs(producer).value_or(false))
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
      // Runs before tt.dot_scaled is lowered, so match the scaled op here too.
      // Otherwise ttng.two-ctas stays false and every downstream tcgen05
      // lowering that reads getModuleTwoCTAs sees the wrong value.
      if (auto dotScaledOp = dyn_cast<tt::DotScaledOp>(op))
        return checkTwoCTA(op, dotScaledOp.getTwoCtas());
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

    if (!allowDependentChains) {
      result = mod.walk(
          [&](tt::DotOpInterface op) { return checkNoDependentTwoCTADot(op); });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }

    // FPSAN rewrites all `tcgen05` MMA ops but sets the flag so it can be
    // propagated.
    if (!firstMatmul && mod->hasAttr(AttrTwoCTAsName))
      return;
    bool twoCTAValue = firstMatmul ? firstTwoCTA : false;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
