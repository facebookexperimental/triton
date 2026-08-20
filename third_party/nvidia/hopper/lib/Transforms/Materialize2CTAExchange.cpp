// Lower planned 2-CTA peer gathers to DSMEM stores and barrier publication.
//
// Runs after warp specialization and software pipelining. Writes the CTA-owned
// half locally and sends the peer half with an async remote shared-memory
// store. Two empty/full lifetimes are involved -- the final dQ destination and
// the temporary exchange staging slot -- and both are required. BM128 routes
// completion through a one-warp relay rather than a cluster barrier, which
// would deadlock inside a warp-specialized partition. See
// WarpSpecialization/docs/AutoWS2CTABackwardPlan.md, "Dependency
// classification and peer exchange" and "Synchronization lifetime".
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

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

static FailureOr<int32_t> getFirstFreeWarpSpecializeBarrierId(Operation *op) {
  auto ws = op->getParentOfType<ttg::WarpSpecializeOp>();
  if (!ws)
    return failure();

  constexpr int32_t numReservedWarpSpecializeBarriers = 2;
  constexpr int32_t maxNamedBarriers = 16;
  int32_t totalPartitionWarps = ws.getTotalPartitionWarps();
  int32_t maxPartitionWarpGroups = 0;
  ws->getParentOfType<ModuleOp>().walk([&](ttg::WarpSpecializeOp candidate) {
    maxPartitionWarpGroups =
        std::max(maxPartitionWarpGroups,
                 static_cast<int32_t>(llvm::divideCeil(
                     candidate.getTotalPartitionWarps(), 4)));
  });
  // AllocateWarpGroups pads this region up to the module-wide warp-group count
  // and assigns one named barrier per partition, so the first free id depends
  // on how many padding partitions it creates. That count is not exposed, so it
  // is re-derived here: padToMaxWarpGroups (AllocateWarpGroups.cpp) fills the
  // shortfall with successive powers of two, making the partition count equal
  // to the shortfall's popcount. Keep the two in sync -- if that filling
  // strategy changes, this silently returns an id that collides with a real
  // partition barrier rather than failing. The assert below is the
  // strategy-independent bound: no filling can use more partitions than warps.
  int32_t paddedPartitionWarps = maxPartitionWarpGroups * 4;
  int32_t warpShortfall = paddedPartitionWarps - totalPartitionWarps;
  int32_t paddingPartitions =
      llvm::popcount(static_cast<uint32_t>(warpShortfall));
  assert(warpShortfall >= 0 && paddingPartitions <= warpShortfall &&
         "padding partition count must not exceed the warp shortfall");
  int32_t firstFree = numReservedWarpSpecializeBarriers +
                      ws.getPartitionRegions().size() + paddingPartitions;
  if (firstFree >= maxNamedBarriers)
    return failure();
  return firstFree;
}

static FailureOr<ttg::LocalStoreOp>
findDestinationStore(ttng::TwoCTAPeerGatherOp gather) {
  Value value = gather.getResult();
  if (value.hasOneUse())
    if (auto trans = dyn_cast<tt::TransOp>(*value.getUsers().begin()))
      value = trans.getResult();
  ttg::LocalStoreOp destination;
  for (Operation *user : value.getUsers()) {
    auto store = dyn_cast<ttg::LocalStoreOp>(user);
    if (!store || destination)
      return failure();
    destination = store;
  }
  if (!destination)
    return failure();
  return destination;
}

static SmallVector<ttng::ArriveBarrierOp>
findDestinationFullArrivals(ttg::LocalStoreOp store) {
  SmallVector<ttng::ArriveBarrierOp> arrivals;
  for (Operation *op = store->getNextNode(); op; op = op->getNextNode()) {
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
      arrivals.push_back(arrive);
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;
  }
  return arrivals;
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

static Value getMemDescBase(Value value) {
  while (true) {
    value = unwrapWarpSpecializeCapture(value);
    if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = index.getSrc();
      continue;
    }
    if (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = subview.getSrc();
      continue;
    }
    return unwrapWarpSpecializeCapture(value);
  }
}

static FailureOr<ttg::LocalStoreOp>
findExchangeStore(ttng::TwoCTAPeerGatherOp gather,
                  ttng::TwoCTAPeerRelayOp relay) {
  Value exchangeBase = getMemDescBase(relay.getSrc());
  ttg::LocalStoreOp exchangeStore;
  // The exchange buffer must have exactly one writer for the redirect below to
  // be unambiguous. Count rather than toggle: toggling re-selects on the third
  // match and would silently return the last of an odd number of writers.
  unsigned numStores = 0;
  gather->getParentRegion()->walk([&](ttg::LocalStoreOp store) {
    if (getMemDescBase(store.getDst()) != exchangeBase)
      return;
    ++numStores;
    exchangeStore = store;
  });
  if (numStores != 1)
    return failure();
  return exchangeStore;
}

static Value getLocalMemDescBase(Value value) {
  while (true) {
    if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = index.getSrc();
      continue;
    }
    if (auto subview = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = subview.getSrc();
      continue;
    }
    return value;
  }
}

static ttng::WaitBarrierOp findRelayWait(ttng::TwoCTAPeerRelayOp relay) {
  for (Operation *op = relay->getPrevNode(); op; op = op->getPrevNode())
    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op))
      return wait;
  return {};
}

static Value findCaptureInRelay(ttng::TwoCTAPeerRelayOp relay,
                                Value outerValue) {
  auto wsOp = relay->getParentOfType<ttg::WarpSpecializeOp>();
  if (!wsOp)
    return {};
  Block *partitionBlock = nullptr;
  for (Region *region : wsOp.getPartitionRegions()) {
    if (region->isAncestor(relay->getParentRegion())) {
      partitionBlock = &region->front();
      break;
    }
  }
  if (!partitionBlock)
    return {};
  if (auto arg = dyn_cast<BlockArgument>(getLocalMemDescBase(outerValue))) {
    if (arg.getArgNumber() < partitionBlock->getNumArguments())
      return partitionBlock->getArgument(arg.getArgNumber());
  }
  auto partitions = wsOp.getPartitionOp();
  Value outerBase = getMemDescBase(outerValue);
  for (auto [index, capture] :
       llvm::enumerate(partitions.getExplicitCaptures())) {
    if (getMemDescBase(capture) != outerBase ||
        index >= partitionBlock->getNumArguments())
      continue;
    return partitionBlock->getArgument(index);
  }
  return {};
}

static Value indexBarrierLike(OpBuilder &builder, Location loc, Value barrier,
                              Value indexedLike) {
  auto likeIndex = indexedLike.getDefiningOp<ttg::MemDescIndexOp>();
  if (!likeIndex)
    return barrier;
  auto barrierType = cast<ttg::MemDescType>(barrier.getType());
  SmallVector<int64_t> shape(barrierType.getShape().drop_front());
  auto viewType = ttg::MemDescType::get(
      shape, barrierType.getElementType(), barrierType.getEncoding(),
      barrierType.getMemorySpace(), barrierType.getMutableMemory());
  return ttg::MemDescIndexOp::create(builder, loc, viewType, barrier,
                                     likeIndex.getIndex());
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
                                           Value source, unsigned splitDim,
                                           bool vectorizeOtherDim = false) {
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
  if (vectorizeOtherDim) {
    unsigned otherSourceDim = splitDim == 0 ? 1 : 0;
    unsigned reshapedOtherDim =
        otherSourceDim < splitDim ? otherSourceDim : otherSourceDim + 1;
    auto position = std::find(order.begin(), order.end(), reshapedOtherDim);
    assert(position != order.end());
    sizePerThread[std::distance(order.begin(), position)] = 2;
  }
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

// Resolve a pipelined dS value to the concrete value that seeded its loop
// carry. The producer stores every carried version into the same single-ring
// TMEM allocation before the peer exchange consumes it.
static Value resolveLoopCarryInit(Value value) {
  while (true) {
    if (auto result = dyn_cast<OpResult>(value)) {
      if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
        value = forOp.getInitArgs()[result.getResultNumber()];
        continue;
      }
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      if (forOp && arg.getArgNumber() > 0) {
        value = forOp.getInitArgs()[arg.getArgNumber() - 1];
        continue;
      }
    }
    return value;
  }
}

// TLX reloads the two rank-owned dS halves from the reused dp/dQ TMEM slot.
// Do the same when AutoWS has already materialized that store. This avoids a
// full 128x128 register relayout scratch solely to make SplitOp legal at eight
// warps.
static std::optional<std::pair<Value, Value>>
splitStoredTmem(OpBuilder &builder, Location loc, Value source,
                unsigned splitDim) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType || sourceType.getRank() != 2 || splitDim != 1 ||
      sourceType.getShape()[1] % 2 != 0)
    return std::nullopt;

  Value concreteSource = resolveLoopCarryInit(source);
  ttng::TMEMStoreOp store;
  for (Operation *user : concreteSource.getUsers()) {
    if (auto candidate = dyn_cast<ttng::TMEMStoreOp>(user)) {
      store = candidate;
      break;
    }
  }
  if (!store)
    return std::nullopt;

  int64_t halfN = sourceType.getShape()[1] / 2;
  int numWarps =
      ttg::lookupNumWarps(builder.getInsertionBlock()->getParentOp());
  auto loadHalf = [&](int64_t offset) -> Value {
    Value slice = ttng::TMEMSubSliceOp::create(builder, loc, store.getDst(),
                                               offset, halfN);
    auto sliceType = cast<ttg::MemDescType>(slice.getType());
    Attribute layout = ttng::getDefaultLayoutForTmemLdSt(sliceType, numWarps);
    auto loadType = RankedTensorType::get({sourceType.getShape()[0], halfN},
                                          sourceType.getElementType(), layout);
    return ttng::TMEMLoadOp::create(builder, loc, loadType, slice);
  };
  return std::pair<Value, Value>{loadHalf(0), loadHalf(halfN)};
}

static LogicalResult materializeGather(ttng::TwoCTAPeerGatherOp gather,
                                       ttng::TwoCTAPeerRelayOp relay,
                                       Value &mmaFullBase) {
  auto destinationStore = findDestinationStore(gather);
  if (failed(destinationStore))
    return gather.emitError(
        "expected exactly one AutoWS local_store gather consumer");
  FailureOr<ttg::LocalStoreOp> exchangeStore;
  if (relay) {
    exchangeStore = findExchangeStore(gather, relay);
    if (failed(exchangeStore))
      return gather.emitError(
          "expected exactly one AutoWS peer-exchange staging store");
  }
  auto destinationFullArrivals = findDestinationFullArrivals(*destinationStore);
  if (destinationFullArrivals.empty())
    return gather.emitError(
        "expected an AutoWS full-barrier arrival after the gather store");

  ttng::WaitBarrierOp relayWait = relay ? findRelayWait(relay) : nullptr;
  Value relayFullBase =
      relayWait ? getMemDescBase(relayWait.getAlloc()) : Value();
  ttng::ArriveBarrierOp relayFullArrival;
  ttng::ArriveBarrierOp destinationFullArrival;
  for (ttng::ArriveBarrierOp arrival : destinationFullArrivals) {
    if (!destinationFullArrival)
      destinationFullArrival = arrival;
  }
  if (relay) {
    for (ttng::ArriveBarrierOp arrival :
         findDestinationFullArrivals(*exchangeStore)) {
      if (relayFullBase && getMemDescBase(arrival.getAlloc()) == relayFullBase) {
        relayFullArrival = arrival;
        break;
      }
    }
  }
  if (!destinationFullArrival)
    return gather.emitError("could not identify the dQ MMA full barrier");
  if (relay && !relayFullArrival)
    return gather.emitError("could not identify the relay full barrier");

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
  auto resultType = cast<RankedTensorType>(gather.getType());
  bool shapeChanging = sourceType != resultType;

  Value destination = destinationStore->getDst();
  auto destinationType = cast<ttg::MemDescType>(destination.getType());
  bool physicalTranspose =
      shapeChanging &&
      destinationType.getShape() ==
          ArrayRef<int64_t>{resultType.getShape()[1], resultType.getShape()[0]};
  std::pair<Value, Value> halves;
  if (shapeChanging) {
    auto tmemHalves = splitStoredTmem(builder, loc, source, splitDim);
    halves = tmemHalves ? *tmemHalves
                        : splitTensor(builder, loc, source, splitDim,
                                      /*vectorizeOtherDim=*/true);
  } else {
    halves = splitTensor(builder, loc, source, splitDim);
  }
  auto [lhs, rhs] = halves;
  unsigned concatDim = splitDim;
  Value lhsPayload = lhs;
  Value rhsPayload = rhs;
  if (shapeChanging && !physicalTranspose) {
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    if (sourceType.getNumElements() != resultType.getNumElements())
      return gather.emitError(
          "shape-changing peer gather must preserve the element count");

    // BM128 gathers rank-owned source M halves from adjacent CTA N domains.
    // splitTensor assigns adjacent source-N pairs to each thread. Transposing
    // [N,M/2] to [M/2,N] preserves those pairs as whole b32 DSMEM payloads.
    SmallVector<int32_t> transposeOrder{1, 0};
    lhsPayload = tt::TransOp::create(builder, loc, lhs, transposeOrder);
    rhsPayload = tt::TransOp::create(builder, loc, rhs, transposeOrder);
    bool foundConcatDim = false;
    SmallVector<int64_t> transposedPayloadShape{
        lhsType.getShape()[1], lhsType.getShape()[0]};
    for (unsigned dim = 0; dim < resultType.getRank(); ++dim) {
      SmallVector<int64_t> candidateShape(resultType.getShape());
      candidateShape[dim] /= 2;
      if (candidateShape == transposedPayloadShape) {
        concatDim = dim;
        foundConcatDim = true;
        break;
      }
    }
    if (!foundConcatDim || lhsType.getRank() != 2)
      return gather.emitError(
          "unsupported shape-changing peer gather destination");
  } else if (physicalTranspose) {
    // The planner exposed TLX's physical [2*N,M/2] allocation.  Source M
    // halves are already native [N,M/2] payloads; concatenate them along the
    // physical N dimension and let memdesc_trans present [M/2,2*N] to MMA.
    concatDim = 0;
  }

  SmallVector<int64_t> halfShape(destinationType.getShape());
  halfShape[concatDim] /= 2;
  auto halfType = ttg::MemDescType::get(
      halfShape, destinationType.getElementType(),
      destinationType.getEncoding(), destinationType.getMemorySpace(),
      destinationType.getMutableMemory(), destinationType.getAllocShape());
  SmallVector<int32_t> firstOffset(sourceType.getRank(), 0);
  SmallVector<int32_t> secondOffset(sourceType.getRank(), 0);
  secondOffset[concatDim] = halfShape[concatDim];
  Value firstDestination = ttg::MemDescSubsliceOp::create(
      builder, loc, halfType, destination, firstOffset);
  Value secondDestination = ttg::MemDescSubsliceOp::create(
      builder, loc, halfType, destination, secondOffset);
  Value firstStoreDestination = firstDestination;
  Value secondStoreDestination = secondDestination;
  Value exchangeDestination =
      relay ? (*exchangeStore).getDst() : Value();

  // The existing AutoWS empty-barrier wait before destinationStore protects
  // buffer reuse. Extend its full barrier with remote completion bytes so the
  // existing consumer wait also observes the peer half. This preserves the
  // barrier's pipelined phase rotation and avoids adding planner-visible state.
  int64_t expectedBytes =
      cast<RankedTensorType>(lhsPayload.getType()).getNumElements() *
      sourceType.getElementTypeBitWidth() / 8;
  Value predTrue = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value fullBarrier =
      relay ? relayFullArrival.getAlloc() : destinationFullArrival.getAlloc();
  if (!relay && failed(addExpectedArrival(fullBarrier)))
    return gather.emitError(
        "could not update the AutoWS full-barrier arrival count");
  if (relay && failed(addExpectedArrival(destinationFullArrival.getAlloc())))
    return gather.emitError("could not update the dQ MMA full-barrier count");
  mmaFullBase = getLocalMemDescBase(destinationFullArrival.getAlloc());
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
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value isCTAZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, ctaRank, zero);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isCTAZero,
                                /*withElseRegion=*/true);

  if (shapeChanging) {
    // Store the rank-owned half in the final dQ destination and stage the
    // other half in a dedicated exchange buffer, matching TLX.
    // A CTA barrier makes all distributed local_store writes visible before
    // the elected thread issues cp.async.bulk through the async proxy.
    builder.setInsertionPoint(ifOp);
    FailureOr<int32_t> stagingBarrierId =
        getFirstFreeWarpSpecializeBarrierId(gather);
    if (failed(stagingBarrierId))
      return gather.emitError(
          "no named-barrier slot remains for the 2-CTA staging rendezvous");
    Value stagingBarrier = arith::ConstantIntOp::create(
        builder, loc, *stagingBarrierId, 32);
    int numThreads =
        32 * ttg::lookupNumWarps(builder.getInsertionBlock()->getParentOp());
    Value stagingThreads =
        arith::ConstantIntOp::create(builder, loc, numThreads, 32);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    ttg::LocalStoreOp::create(builder, loc, lhsPayload,
                              firstStoreDestination);
    ttg::LocalStoreOp::create(builder, loc, rhsPayload, exchangeDestination);
    ttng::NamedBarrierWaitOp::create(builder, loc, stagingBarrier,
                                     stagingThreads);
    ttng::FenceAsyncSharedOp::create(builder, loc, /*bCluster=*/false);
    ttg::AsyncRemoteShmemCopyOp::create(
        builder, loc, exchangeDestination, firstStoreDestination, one,
        fullBarrier);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    ttg::LocalStoreOp::create(builder, loc, rhsPayload,
                              secondStoreDestination);
    ttg::LocalStoreOp::create(builder, loc, lhsPayload, exchangeDestination);
    ttng::NamedBarrierWaitOp::create(builder, loc, stagingBarrier,
                                     stagingThreads);
    ttng::FenceAsyncSharedOp::create(builder, loc, /*bCluster=*/false);
    ttg::AsyncRemoteShmemCopyOp::create(
        builder, loc, exchangeDestination, secondStoreDestination, zero,
        fullBarrier);
  } else {
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    ttg::LocalStoreOp::create(builder, loc, lhsPayload,
                              firstStoreDestination);
    ttg::AsyncRemoteShmemStoreOp::create(
        builder, loc, lhsPayload, firstStoreDestination, one, fullBarrier);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    ttg::LocalStoreOp::create(builder, loc, rhsPayload,
                              secondStoreDestination);
    ttg::AsyncRemoteShmemStoreOp::create(
        builder, loc, rhsPayload, secondStoreDestination, zero, fullBarrier);
  }

  builder.setInsertionPointAfter(ifOp);
  if (relay)
    relayFullArrival.erase();
  if (relay)
    exchangeStore->erase();
  destinationStore->erase();
  if (gather->hasOneUse())
    gather->getUsers().begin()->erase();
  gather->erase();
  return success();
}

static LogicalResult materializeRelay(ttng::TwoCTAPeerRelayOp relay,
                                      Value mmaFullBase) {
  ttng::WaitBarrierOp relayWait = findRelayWait(relay);
  if (!relayWait)
    return relay.emitError("expected an AutoWS relay-channel wait");
  Value capturedMmaFull = findCaptureInRelay(relay, mmaFullBase);
  if (!capturedMmaFull)
    return relay.emitError("expected the dQ MMA full barrier to be captured");

  OpBuilder builder(relay);
  Location loc = relay.getLoc();
  ttng::FenceAsyncSharedOp::create(builder, loc, /*bCluster=*/false);
  Value mmaFull =
      indexBarrierLike(builder, loc, capturedMmaFull, relayWait.getAlloc());
  ttng::ArriveBarrierOp::create(builder, loc, mmaFull, 1);
  relay.erase();
  return success();
}

class Materialize2CTAExchange
    : public impl::NVGPUMaterialize2CTAExchangeBase<Materialize2CTAExchange> {
public:
  using impl::NVGPUMaterialize2CTAExchangeBase<
      Materialize2CTAExchange>::NVGPUMaterialize2CTAExchangeBase;

  void runOnOperation() override {
    SmallVector<ttng::TwoCTAPeerGatherOp> gathers;
    SmallVector<ttng::TwoCTAPeerRelayOp> relays;
    getOperation().walk(
        [&](ttng::TwoCTAPeerGatherOp gather) { gathers.push_back(gather); });
    getOperation().walk(
        [&](ttng::TwoCTAPeerRelayOp relay) { relays.push_back(relay); });
    if (relays.size() > 1) {
      relays.front().emitError("expected at most one 2-CTA relay per kernel");
      return signalPassFailure();
    }
    ttng::TwoCTAPeerRelayOp relay = relays.empty() ? nullptr : relays.front();
    Value mmaFullBase;
    for (auto gather : gathers) {
      if (failed(materializeGather(gather, relay, mmaFullBase))) {
        signalPassFailure();
        return;
      }
    }
    if (relay &&
        (gathers.empty() || failed(materializeRelay(relay, mmaFullBase))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
