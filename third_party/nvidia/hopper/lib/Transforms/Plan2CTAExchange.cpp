#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_NVGPUPLAN2CTAEXCHANGE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

constexpr StringLiteral DependencyAttr = "ttng.two_cta_dependency";
constexpr StringLiteral RequiresPeerGather = "requires_peer_gather";

struct GatherSource {
  ttg::LocalAllocOp alloc;
  Value tensor;
  RankedTensorType resultType;
  unsigned splitDim;
};

static Value createExchangeTensor(OpBuilder &builder, Location loc,
                                  Value source, unsigned splitDim) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<int64_t> reshapedShape;
  for (auto [dim, size] : llvm::enumerate(sourceType.getShape())) {
    if (dim == splitDim) {
      reshapedShape.push_back(2);
      reshapedShape.push_back(size / 2);
    } else {
      reshapedShape.push_back(size);
    }
  }
  Value reshaped = tt::ReshapeOp::create(builder, loc, reshapedShape, source,
                                         /*allowReorder=*/false);
  SmallVector<int32_t> order;
  for (unsigned dim = 0; dim < reshapedShape.size(); ++dim)
    if (dim != splitDim)
      order.push_back(dim);
  order.push_back(splitDim);
  Value transposed = tt::TransOp::create(builder, loc, reshaped, order);

  // With eight base warps layout propagation may distribute the factor-of-two
  // axis across warps. Split requires that axis in registers. Materialize the
  // same split-compatible layout used by the later exchange lowering so the
  // logical packing remains valid independently of the base warp count.
  auto transposedType = cast<RankedTensorType>(transposed.getType());
  SmallVector<unsigned> layoutOrder(transposedType.getRank());
  std::iota(layoutOrder.begin(), layoutOrder.end(), 0);
  std::reverse(layoutOrder.begin(), layoutOrder.end());
  SmallVector<unsigned> sizePerThread(transposedType.getRank(), 1);
  sizePerThread.back() = 2;
  sizePerThread[sizePerThread.size() - 2] = 2;
  int numWarps =
      ttg::lookupNumWarps(builder.getInsertionBlock()->getParentOp());
  auto ctaLayout = ttg::CGAEncodingAttr::fromSplitParams(
      builder.getContext(), /*CTAsPerCGA=*/{1, 1, 1},
      /*CTASplitNum=*/{1, 1, 1}, /*CTAOrder=*/{2, 1, 0});
  auto splitEncoding = ttg::BlockedEncodingAttr::get(
      builder.getContext(), sizePerThread,
      /*threadsPerWarp=*/{1, 32, 1},
      /*warpsPerCTA=*/{static_cast<unsigned>(numWarps), 1, 1}, layoutOrder,
      ctaLayout);
  auto splitInputType =
      RankedTensorType::get(transposedType.getShape(),
                            transposedType.getElementType(), splitEncoding);
  Value splitInput =
      ttg::ConvertLayoutOp::create(builder, loc, splitInputType, transposed);
  return tt::SplitOp::create(builder, loc, splitInput).getOutLHS();
}

static FailureOr<GatherSource> findGatherSource(ttng::TCGen5MMAOp mma) {
  Value value = mma.getA();
  while (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>())
    value = subview.getSrc();

  if (auto trans = value.getDefiningOp<ttg::MemDescTransOp>()) {
    if (trans.getOrder().size() != 2)
      return failure();
    auto alloc = trans.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (!alloc || !alloc.getSrc())
      return failure();

    // The dependent MMA needs a complete A contraction dimension. Map A
    // dimension 1 through the memdesc transpose to the source tensor dimension
    // distributed by the producing collective MMA.
    return GatherSource{alloc, alloc.getSrc(),
                        cast<RankedTensorType>(alloc.getSrc().getType()),
                        static_cast<unsigned>(trans.getOrder()[1])};
  }

  // BM128 expresses the TLX physical dQ layout before shared-memory
  // allocation: reshape(transpose(dS), [M/2, 2*N]). The peer gather consumes
  // the untransposed [N,M] dS tensor and materializes that physical result.
  auto alloc = value.getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc || !alloc.getSrc())
    return failure();
  auto reshape = alloc.getSrc().getDefiningOp<tt::ReshapeOp>();
  if (!reshape)
    return failure();
  auto trans = reshape.getSrc().getDefiningOp<tt::TransOp>();
  if (!trans || trans.getOrder().size() != 2)
    return failure();

  auto sourceType = dyn_cast<RankedTensorType>(trans.getSrc().getType());
  auto resultType = dyn_cast<RankedTensorType>(alloc.getSrc().getType());
  if (!sourceType || !resultType || sourceType.getRank() != 2 ||
      resultType.getRank() != 2)
    return failure();

  unsigned splitDim = static_cast<unsigned>(trans.getOrder()[0]);
  unsigned otherDim = splitDim == 0 ? 1 : 0;
  if (sourceType.getShape()[splitDim] != resultType.getShape()[0] * 2 ||
      resultType.getShape()[1] != sourceType.getShape()[otherDim] * 2)
    return failure();
  return GatherSource{alloc, trans.getSrc(), resultType, splitDim};
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

      ttg::LocalAllocOp alloc = source->alloc;
      unsigned splitDim = source->splitDim;
      Value tensor = source->tensor;
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
          builder, alloc.getLoc(), source->resultType, tensor,
          builder.getI32IntegerAttr(splitDim), builder.getI32IntegerAttr(2));
      // BM64's direct completion protocol is already validated. Match the TLX
      // relay only for a combined BM128 query tile.
      unsigned queryDim = splitDim == 0 ? 1 : 0;
      if (tensorType.getShape()[queryDim] >= 128) {
        // Keep the peer gather's logical [M/2,2*N] result, but allocate its
        // transpose as physical [2*N,M/2] shared memory.  The dependent MMA
        // consumes a transposed memdesc view, matching TLX's operand-A layout.
        SmallVector<int32_t> transposeOrder{1, 0};
        Value physicalTensor = tt::TransOp::create(
            builder, alloc.getLoc(), gather.getResult(), transposeOrder);
        alloc->setOperand(0, physicalTensor);

        auto oldAllocType = cast<ttg::MemDescType>(alloc.getType());
        auto physicalTensorType =
            cast<RankedTensorType>(physicalTensor.getType());
        auto physicalAllocType = ttg::MemDescType::get(
            physicalTensorType.getShape(), oldAllocType.getElementType(),
            oldAllocType.getEncoding(), oldAllocType.getMemorySpace(),
            oldAllocType.getMutableMemory());
        alloc.getResult().setType(physicalAllocType);

        builder.setInsertionPointAfter(alloc);
        Value mmaView = ttg::MemDescTransOp::create(
            builder, alloc.getLoc(), alloc.getResult(), transposeOrder);
        mma.getAMutable().assign(mmaView);

        builder.setInsertionPoint(alloc);
        Value exchangeTensor =
            createExchangeTensor(builder, alloc.getLoc(), tensor, splitDim);
        auto exchangeTensorType =
            cast<RankedTensorType>(exchangeTensor.getType());
        auto exchangeAllocType = ttg::MemDescType::get(
            exchangeTensorType.getShape(), oldAllocType.getElementType(),
            oldAllocType.getEncoding(), oldAllocType.getMemorySpace(),
            oldAllocType.getMutableMemory());
        auto exchangeAlloc = ttg::LocalAllocOp::create(
            builder, alloc.getLoc(), exchangeAllocType, exchangeTensor);

        builder.setInsertionPointAfter(alloc);
        ttng::TwoCTAPeerRelayOp::create(builder, alloc.getLoc(),
                                        exchangeAlloc.getResult());
      } else {
        alloc->setOperand(0, gather.getResult());
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
