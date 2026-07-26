#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_NVGPUANALYZE2CTADEPENDENCIES
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

constexpr StringLiteral DependencyAttr = "ttng.two_cta_dependency";
constexpr StringLiteral CollectiveContraction = "collective_contraction";
constexpr StringLiteral RequiresPeerGather = "requires_peer_gather";

static bool dependsOnTwoCTAMMA(Value root, Operation *consumer) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    Operation *def = value.getDefiningOp();
    if (!def)
      continue;
    if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(def)) {
      if (mma.getTwoCtas() && def != consumer)
        return true;
    }
    llvm::append_range(worklist, def->getOperands());
  }
  return false;
}

static bool isTransposedMemDesc(Value value) {
  while (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>())
    value = subview.getSrc();
  return value.getDefiningOp<ttg::MemDescTransOp>() != nullptr;
}

class Analyze2CTADependencies
    : public impl::NVGPUAnalyze2CTADependenciesBase<Analyze2CTADependencies> {
public:
  using impl::NVGPUAnalyze2CTADependenciesBase<
      Analyze2CTADependencies>::NVGPUAnalyze2CTADependenciesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!ttng::is2CTA(module) || module->hasAttr("tlx.has_tlx_ops"))
      return;

    module.walk([&](ttng::TCGen5MMAOp mma) {
      if (!mma.getTwoCtas() || !dependsOnTwoCTAMMA(mma.getA(), mma))
        return;

      StringRef kind = isTransposedMemDesc(mma.getA()) ? RequiresPeerGather
                                                       : CollectiveContraction;
      mma->setAttr(DependencyAttr, StringAttr::get(mma.getContext(), kind));
    });
  }
};

} // namespace
} // namespace mlir
