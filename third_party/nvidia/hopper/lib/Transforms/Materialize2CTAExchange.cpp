#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <algorithm>
#include <numeric>

namespace mlir {

#define GEN_PASS_DEF_NVGPUMATERIALIZE2CTAEXCHANGE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace nvgpu = triton::nvgpu;

static FailureOr<ttg::LocalStoreOp>
findDestinationStore(ttng::TwoCTAPeerGatherOp gather) {
  ttg::LocalStoreOp destination;
  for (Operation *user : gather.getResult().getUsers()) {
    auto store = dyn_cast<ttg::LocalStoreOp>(user);
    if (!store || destination)
      return failure();
    destination = store;
  }
  if (!destination)
    return failure();
  return destination;
}

static FailureOr<ttng::ArriveBarrierOp>
findDestinationFullArrival(ttg::LocalStoreOp store) {
  for (Operation *op = store->getNextNode(); op; op = op->getNextNode()) {
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
      return arrive;
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;
  }
  return failure();
}

static Value unwrapWarpSpecializeCapture(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(parentOp);
    if (!partitions ||
        blockArg.getArgNumber() >= partitions.getExplicitCaptures().size())
      break;
    value = partitions.getExplicitCaptures()[blockArg.getArgNumber()];
  }
  return value;
}

static LogicalResult addExpectedArrival(Value barrier) {
  Value base = barrier;
  while (auto index = base.getDefiningOp<ttg::MemDescIndexOp>())
    base = unwrapWarpSpecializeCapture(index.getSrc());
  base = unwrapWarpSpecializeCapture(base);

  ttng::InitBarrierOp initializer;
  for (Operation *user : base.getUsers()) {
    auto index = dyn_cast<ttg::MemDescIndexOp>(user);
    if (!index)
      continue;
    for (Operation *indexUser : index.getResult().getUsers()) {
      if (auto init = dyn_cast<ttng::InitBarrierOp>(indexUser)) {
        if (initializer)
          return failure();
        initializer = init;
      }
    }
  }
  if (!initializer)
    return failure();
  if (initializer.getCount() > 2)
    return failure();
  initializer->setAttr(
      "count", IntegerAttr::get(initializer.getCountAttr().getType(), 2));
  return success();
}

static std::pair<Value, Value> splitTensor(OpBuilder &builder, Location loc,
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
  auto split = tt::SplitOp::create(builder, loc, splitInput);
  return {split.getOutLHS(), split.getOutRHS()};
}

static LogicalResult materializeGather(ttng::TwoCTAPeerGatherOp gather) {
  auto destinationStore = findDestinationStore(gather);
  if (failed(destinationStore))
    return gather.emitError(
        "expected exactly one AutoWS local_store gather consumer");
  auto destinationFullArrival = findDestinationFullArrival(*destinationStore);
  if (failed(destinationFullArrival))
    return gather.emitError(
        "expected an AutoWS full-barrier arrival after the gather store");

  Value source = gather.getSrc();
  unsigned splitDim = gather.getSplitDim();
  auto sourceType = cast<RankedTensorType>(source.getType());
  if (gather.getNumCTAs() != 2 || sourceType.getRank() != 2 ||
      splitDim >= sourceType.getRank() ||
      sourceType.getShape()[splitDim] % 2 != 0)
    return gather.emitError("only an even rank-2, two-CTA peer gather is "
                            "supported");

  // AutoWS materializes the destination buffer view between the abstract
  // gather and its local_store. Insert at the store so both the register
  // source and the planned destination dominate the DSMEM lowering.
  OpBuilder builder(*destinationStore);
  Location loc = gather.getLoc();
  auto [lhs, rhs] = splitTensor(builder, loc, source, splitDim);

  Value destination = destinationStore->getDst();
  auto destinationType = cast<ttg::MemDescType>(destination.getType());
  SmallVector<int64_t> halfShape(destinationType.getShape());
  halfShape[splitDim] /= 2;
  auto halfType = ttg::MemDescType::get(
      halfShape, destinationType.getElementType(),
      destinationType.getEncoding(), destinationType.getMemorySpace(),
      destinationType.getMutableMemory(), destinationType.getAllocShape());
  SmallVector<int32_t> firstOffset(sourceType.getRank(), 0);
  SmallVector<int32_t> secondOffset(sourceType.getRank(), 0);
  secondOffset[splitDim] = halfShape[splitDim];
  Value firstDestination = ttg::MemDescSubsliceOp::create(
      builder, loc, halfType, destination, firstOffset);
  Value secondDestination = ttg::MemDescSubsliceOp::create(
      builder, loc, halfType, destination, secondOffset);

  // The existing AutoWS empty-barrier wait before destinationStore protects
  // buffer reuse. Extend its full barrier with remote completion bytes so the
  // existing consumer wait also observes the peer half. This preserves the
  // barrier's pipelined phase rotation and avoids adding planner-visible state.
  int64_t expectedBytes =
      cast<RankedTensorType>(lhs.getType()).getNumElements() *
      sourceType.getElementTypeBitWidth() / 8;
  Value predTrue = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value fullBarrier = (*destinationFullArrival).getAlloc();
  if (failed(addExpectedArrival(fullBarrier)))
    return gather.emitError(
        "could not update the AutoWS full-barrier arrival count");
  Operation *fullBarrierDef = fullBarrier.getDefiningOp();
  Operation *destinationStoreOp = (*destinationStore).getOperation();
  if (fullBarrierDef &&
      fullBarrierDef->getBlock() == destinationStoreOp->getBlock() &&
      destinationStoreOp->isBeforeInBlock(fullBarrierDef)) {
    Operation *clonedDef = builder.clone(*fullBarrierDef);
    fullBarrier = clonedDef->getResult(0);
  }
  ttng::BarrierExpectOp::create(builder, loc, fullBarrier, expectedBytes,
                                predTrue);

  Value ctaRank =
      nvgpu::ClusterCTAIdOp::create(builder, loc, builder.getI32Type());
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value isCTAZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, ctaRank, zero);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isCTAZero,
                                /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  ttg::LocalStoreOp::create(builder, loc, lhs, firstDestination);
  ttg::AsyncRemoteShmemStoreOp::create(builder, loc, lhs, firstDestination, one,
                                       fullBarrier);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  ttg::LocalStoreOp::create(builder, loc, rhs, secondDestination);
  ttg::AsyncRemoteShmemStoreOp::create(builder, loc, rhs, secondDestination,
                                       zero, fullBarrier);

  builder.setInsertionPointAfter(ifOp);
  destinationStore->erase();
  gather->erase();
  return success();
}

class Materialize2CTAExchange
    : public impl::NVGPUMaterialize2CTAExchangeBase<Materialize2CTAExchange> {
public:
  using impl::NVGPUMaterialize2CTAExchangeBase<
      Materialize2CTAExchange>::NVGPUMaterialize2CTAExchangeBase;

  void runOnOperation() override {
    SmallVector<ttng::TwoCTAPeerGatherOp> gathers;
    getOperation().walk(
        [&](ttng::TwoCTAPeerGatherOp gather) { gathers.push_back(gather); });
    for (auto gather : gathers) {
      if (failed(materializeGather(gather))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir
