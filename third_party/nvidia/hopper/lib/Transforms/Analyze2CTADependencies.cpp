// Classify dependent 2-CTA MMA operand chains before warp specialization.
//
// A collective contraction keeps its CTA-local operand; an operand that is a
// transposed view of another 2-CTA MMA result cannot be re-loaded and must be
// gathered from the peer CTA. See
// WarpSpecialization/docs/AutoWS2CTABackwardPlan.md, "Dependency
// classification and peer exchange".
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

static bool requiresPeerGather(Value value) {
  while (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>())
    value = subview.getSrc();
  if (value.getDefiningOp<ttg::MemDescTransOp>())
    return true;

  auto alloc = value.getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc || !alloc.getSrc())
    return false;

  SmallVector<Value> worklist{alloc.getSrc()};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    Operation *def = current.getDefiningOp();
    if (!def)
      continue;
    if (auto trans = dyn_cast<triton::TransOp>(def)) {
      // Rank-3 transposes are also used to expose a size-two axis to tt.split
      // when a collective contraction is statically subtiled. They only
      // repack the contraction dimension and do not transpose the logical MMA
      // operand across CTAs. Peer gathering is required for the rank-2 matrix
      // transpose used by dQ-style dependent MMAs.
      auto resultType = dyn_cast<RankedTensorType>(trans.getType());
      if (resultType && resultType.getRank() == 2)
        return true;
    }
    if (isa<ttng::TCGen5MMAOp>(def))
      continue;
    llvm::append_range(worklist, def->getOperands());
  }
  return false;
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

      StringRef kind = requiresPeerGather(mma.getA()) ? RequiresPeerGather
                                                      : CollectiveContraction;
      mma->setAttr(DependencyAttr, StringAttr::get(mma.getContext(), kind));
    });
  }
};

} // namespace
} // namespace mlir
