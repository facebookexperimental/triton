#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_NVGPUPLAN2CTAEXCHANGE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

constexpr StringLiteral DependencyAttr = "ttng.two_cta_dependency";
constexpr StringLiteral RequiresPeerGather = "requires_peer_gather";

static FailureOr<std::pair<ttg::LocalAllocOp, unsigned>>
findGatherSource(ttng::TCGen5MMAOp mma) {
  Value value = mma.getA();
  while (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>())
    value = subview.getSrc();

  auto trans = value.getDefiningOp<ttg::MemDescTransOp>();
  if (!trans || trans.getOrder().size() != 2)
    return failure();

  auto alloc = trans.getSrc().getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc || !alloc.getSrc())
    return failure();

  // The dependent MMA needs a complete A contraction dimension. Map A
  // dimension 1 through the memdesc transpose to the source tensor dimension
  // distributed by the producing collective MMA.
  return std::make_pair(alloc, static_cast<unsigned>(trans.getOrder()[1]));
}

class Plan2CTAExchange
    : public impl::NVGPUPlan2CTAExchangeBase<Plan2CTAExchange> {
public:
  using impl::NVGPUPlan2CTAExchangeBase<
      Plan2CTAExchange>::NVGPUPlan2CTAExchangeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!ttng::is2CTA(module) || module->hasAttr("tlx.has_tlx_ops"))
      return;

    WalkResult result = module.walk([&](ttng::TCGen5MMAOp mma) {
      auto dependency = mma->getAttrOfType<StringAttr>(DependencyAttr);
      if (!dependency || dependency.getValue() != RequiresPeerGather)
        return WalkResult::advance();

      auto source = findGatherSource(mma);
      if (failed(source)) {
        mma.emitError("requires_peer_gather expects a transposed local_alloc "
                      "operand with a tensor source");
        return WalkResult::interrupt();
      }

      ttg::LocalAllocOp alloc = source->first;
      unsigned splitDim = source->second;
      Value tensor = alloc.getSrc();
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      if (splitDim >= tensorType.getRank() ||
          tensorType.getShape()[splitDim] % 2 != 0) {
        mma.emitError("peer-gather source dimension must be evenly divisible "
                      "by two CTAs");
        return WalkResult::interrupt();
      }

      if (tensor.getDefiningOp<ttng::TwoCTAPeerGatherOp>())
        return WalkResult::advance();

      OpBuilder builder(alloc);
      auto gather = ttng::TwoCTAPeerGatherOp::create(
          builder, alloc.getLoc(), tensorType, tensor,
          builder.getI32IntegerAttr(splitDim), builder.getI32IntegerAttr(2));
      alloc->setOperand(0, gather.getResult());
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
