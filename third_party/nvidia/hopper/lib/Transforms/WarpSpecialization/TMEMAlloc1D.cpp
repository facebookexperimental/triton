#include "TMEMUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-1D-tmem-alloc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

FailureOr<ttng::TMEMAllocOp> TMEM1DAllocator::alloc1DTMEMBuffer() {
  auto expandedInput = getExpandedInput();
  FailureOr<RankedTensorType> oldRetTypeOr =
      getResultTensorType(expandedInput->getResult(0), 2);
  if (failed(oldRetTypeOr))
    return failure();
  RankedTensorType oldRetType = *oldRetTypeOr;
  auto shape = oldRetType.getShape();
  auto tmemDesc = createTMEMDesc(builder, oldRetType, shape[0], shape[1]);
  auto allocCall =
      ttng::TMEMAllocOp::create(builder, expandedInput->getLoc(), tmemDesc,
                                builder.getType<ttg::AsyncTokenType>(),
                                /*src=*/Value());
  return allocCall;
}

LogicalResult TMEM1DAllocator::TMEMStore1D(OpResult producer,
                                           AsyncTaskId producerTaskId,
                                           Operation *allocOpBuffer) {
  // Expand from 1D -> 2D
  auto producerOp = producer.getDefiningOp();
  FailureOr<RankedTensorType> oldRetTypeOr = getResultTensorType(producer, 1);
  if (failed(oldRetTypeOr))
    return failure();
  RankedTensorType oldRetType = *oldRetTypeOr;
  builder.setInsertionPointAfter(producerOp);
  auto originTaskIds = builder.getAsyncTaskIds();
  builder.setAsynTaskIdsFromArray(producerTaskId);
  auto originLoopScheduleInfo = builder.getLoopScheduleInfo();
  builder.setLoopScheduleInfoFromOp(producerOp);
  auto encoding = oldRetType.getEncoding();
  Value expandDimsInput = producer;
  unsigned axis = 1;
  auto context = builder.getContext();
  // Handle blocked encoding which isn't a slice attribute.
  if (auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    auto rank = oldRetType.getRank();
    if (rank == 1) {
      // create return encoding with rank 2
      auto retSizePerThread = llvm::to_vector(blockedEnc.getSizePerThread());
      retSizePerThread.insert(retSizePerThread.begin() + axis, 1);
      auto retThreadsPerWarp = to_vector(blockedEnc.getThreadsPerWarp());
      retThreadsPerWarp.insert(retThreadsPerWarp.begin() + axis, 1);
      auto retWarpsPerCTA = to_vector(blockedEnc.getWarpsPerCTA());
      retWarpsPerCTA.insert(retWarpsPerCTA.begin() + axis, 1);
      auto retOrder = {axis, blockedEnc.getOrder()[0]};
      auto argCTALayout = blockedEnc.getCGALayout();
      auto retCTAsPerCGA = {argCTALayout.getCTAsPerCGA()[0], axis};
      auto retCTASplitNum = {argCTALayout.getCTASplitNum()[0], axis};
      auto retCTAOrder = {axis, argCTALayout.getCTAOrder()[0]};
      auto retCTALayout = triton::gpu::CGAEncodingAttr::fromSplitParams(
          context, retCTAsPerCGA, retCTASplitNum, retCTAOrder);
      blockedEnc = triton::gpu::BlockedEncodingAttr::get(
          context, retSizePerThread, retThreadsPerWarp, retWarpsPerCTA,
          retOrder, retCTALayout);
    }
    auto sliceEnc = ttg::SliceEncodingAttr::get(
        builder.getContext(), oldRetType.getRank(), blockedEnc);
    auto sliceType = oldRetType.cloneWithEncoding(sliceEnc);
    expandDimsInput = builder.createWithAsyncTaskIds<ttg::ConvertLayoutOp>(
        expandDimsInput.getLoc(), sliceType, expandDimsInput);
  }
  auto expandDims = builder.createWithAsyncTaskIds<tt::ExpandDimsOp>(
      producerOp->getLoc(), expandDimsInput, axis);
  setExpandedInput(expandDims);
  Operation *allocOp;
  if (allocOpBuffer) {
    allocOp = allocOpBuffer;
  } else {
    FailureOr<ttng::TMEMAllocOp> allocOpOr = alloc1DTMEMBuffer();
    if (failed(allocOpOr))
      return failure();
    allocOp = *allocOpOr;
  }
  setAllocOp(allocOp);

  // Verify that these layouts are compatible.
  auto tmemDesc = dyn_cast<ttg::MemDescType>(allocOp->getResult(0).getType());
  assert(tmemDesc && "Expected MemDescType");
  auto expandType = expandDims.getType();
  bool layoutTmemCompatible =
      ttng::isDistributedLayoutTMemCompatible(expandDims, expandType, tmemDesc);
  auto oldLayout = expandDims.getType().getEncoding();
  auto newLayout = oldLayout;
  if (!layoutTmemCompatible) {
    auto tmemEnc = cast<ttng::TensorMemoryEncodingAttr>(tmemDesc.getEncoding());
    auto compatibleLayouts =
        ttng::getTmemCompatibleLayouts(expandDims, expandType, tmemDesc);
    assert(!compatibleLayouts.empty() && "No TMEM-compatible layout found");
    newLayout = compatibleLayouts.front();
  }
  mlir::Operation *src = expandDims;
  if (newLayout != oldLayout) {
    auto ty = cast<RankedTensorType>(expandType);
    auto newTy = ty.cloneWithEncoding(newLayout);
    src = builder.createWithAsyncTaskIds<ttg::ConvertLayoutOp>(
        expandDims.getLoc(), newTy, expandDims);
  }
  // Generate the store
  Value trueVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(src->getLoc(), 1, 1);
  auto storeOp = builder.createWithAsyncTaskIds<ttng::TMEMStoreOp>(
      src->getLoc(), allocOp->getResult(0), src->getResult(0), trueVal);
  builder.setAsynTaskIdsFromArray(originTaskIds);
  builder.setLoopScheduleInfoFromInfo(originLoopScheduleInfo);
  return success();
}

Value TMEM1DAllocator::TMEMLoad1D(OpResult producer, Operation *consumer) {
  auto allocOp = getAllocOp();
  auto producerOp = producer.getDefiningOp();
  auto oldInputType = dyn_cast<RankedTensorType>(producer.getType());
  auto targetEncoding = oldInputType.getEncoding();
  auto oldExpandType = getExpandedInput().getType();
  auto allocDesc = dyn_cast<ttg::MemDescType>(allocOp->getResult(0).getType());
  auto tmemEnc = cast<ttng::TensorMemoryEncodingAttr>(allocDesc.getEncoding());
  auto compatibleLayouts =
      ttng::getTmemCompatibleLayouts(allocOp, oldExpandType, allocDesc);
  assert(!compatibleLayouts.empty() && "No TMEM-compatible layout found");
  auto newDistributedEncoding = compatibleLayouts.front();
  auto newExpandType = oldExpandType.cloneWithEncoding(newDistributedEncoding);
  // Generate the load
  auto originTaskIds = builder.getAsyncTaskIds();
  auto originLoopScheduleInfo = builder.getLoopScheduleInfo();
  builder.setAsyncTaskIdsFromOp(consumer);
  builder.setInsertionPoint(consumer);
  builder.setLoopScheduleInfoFromOp(consumer);
  auto loadOp = builder.createWithAsyncTaskIds<ttng::TMEMLoadOp>(
      consumer->getLoc(), newExpandType, builder.getType<ttg::AsyncTokenType>(),
      allocOp->getResult(0), Value());
  // Generate the reshape
  auto reshape = builder.createWithAsyncTaskIds<tt::ReshapeOp>(
      consumer->getLoc(), oldInputType.getShape(), loadOp);
  // Generate a convert layout.
  auto newInput = builder.createWithAsyncTaskIds<ttg::ConvertLayoutOp>(
      consumer->getLoc(), oldInputType, reshape);
  // Replace the uses in the consumer
  consumer->replaceUsesOfWith(producer, newInput);
  builder.setAsynTaskIdsFromArray(originTaskIds);
  builder.setLoopScheduleInfoFromInfo(originLoopScheduleInfo);
  return newInput.getResult();
}

LogicalResult generate1DAllocations(OpBuilderWithAsyncTaskIds &builder,
                                    Operation *producer,
                                    llvm::SmallVector<Operation *> &allocOps) {
  assert(producer->hasAttr("tmem.start") && "Expected tmem.start");
  Operation *allocOpBuffer = nullptr;
  auto producerTMEMStart =
      mlir::cast<mlir::IntegerAttr>(producer->getAttr("tmem.start")).getInt();
  // If producerTMEMStart < allocOps.size() then we will be testing reusing
  // an existing allocation. Otherwise we will be testing a new allocation.
  if (producerTMEMStart < allocOps.size()) {
    auto allocOp = allocOps[producerTMEMStart];
    auto allocShape =
        dyn_cast<ttg::MemDescType>(allocOp->getResult(0).getType()).getShape();
    if (allocShape[allocShape.size() - 1] != 1) {
      builder.setInsertionPointAfter(allocOp);
      // Hardcode allocShape[0] / 2 for testing.
      allocOpBuffer = sliceAndReinterpretTMEMBuffer(
          builder, allocOp, allocShape[allocShape.size() - 2] / 2, 1);
    } else {
      allocOpBuffer = allocOp;
    }
  }
  auto producerPartition = getAsyncTaskIds(producer);
  for (auto consumer : producer->getUsers()) {
    auto consumerParition = getAsyncTaskIds(consumer);
    if (producerPartition != consumerParition) {
      Value newProducer = TMEM1DAllocator(builder).replaceWith1DTMEM(
          producer->getOpResult(0), producerPartition.front(), consumer,
          allocOpBuffer);
      if (!newProducer) {
        return failure();
      }
    }
  }
  // Delete tmem.start
  producer->removeAttr("tmem.start");
  return success();
}

Operation *sliceAndReinterpretMDTMEM(OpBuilderWithAsyncTaskIds &builder,
                                     Operation *allocOp, Operation *newAlloc,
                                     Operation *user, int offset) {
  // This function is TMEM-specific - verify both allocations are TMEM
  if (!isa<ttng::TMEMAllocOp>(allocOp)) {
    LDBG("sliceAndReinterpretMDTMEM called with non-TMEM allocOp");
    return nullptr;
  }
  if (!isa<ttng::TMEMAllocOp>(newAlloc)) {
    LDBG("sliceAndReinterpretMDTMEM called with non-TMEM newAlloc");
    return nullptr;
  }

  // user is the index into newAlloc.
  // create a new index based on allocOp to reduce from 1xMxN to MxN.
  // then subslice + interpret
  // or subslice on 3D, then interpret then index
  auto index = cast<ttg::MemDescIndexOp>(user);
  auto allocResult = allocOp->getResult(0);
  auto allocType = cast<ttg::MemDescType>(allocResult.getType());
  auto shape = allocType.getShape();
  // We can have 3D shapes: 1x64x128, shape[0] will be "1".
  // This assumes a 2D shape, maybe we should start with the index and
  // reinterpet the index.
  auto oldBlockN = shape[shape.size() - 1];

  auto newResult = newAlloc->getResult(0);
  auto newType = cast<ttg::MemDescType>(newResult.getType());
  auto newShape = newType.getShape();
  auto blockN = newShape[shape.size() - 1];

  // `oldBlockN` and `offset` are measured in the source allocation's physical
  // TMEM columns, while `blockN` is the destination tile width.  A 16-bit
  // destination packed into a 32-bit source occupies blockN / 2 source
  // columns (the f32 -> f16/bf16 reuse case), so validate the effective width
  // before constructing the parent view and converted subslice below.
  auto elemTyWidth = newType.getElementType().getIntOrFloatBitWidth();
  auto oldElemTyWidth = allocType.getElementType().getIntOrFloatBitWidth();
  int64_t sliceCols = blockN;
  if (oldElemTyWidth == elemTyWidth * 2) {
    sliceCols = blockN / 2;
  } else if (elemTyWidth != oldElemTyWidth) {
    // Unsupported element type conversion
    return nullptr;
  }

  // Validate the allocation before materializing the parent view and subslice.
  if (sliceCols <= 0 || oldBlockN < sliceCols || oldBlockN % sliceCols != 0 ||
      (offset + sliceCols) > oldBlockN) {
    LDBG("TMEM allocation validation failed: oldBlockN="
         << oldBlockN << ", blockN=" << blockN << ", sliceCols=" << sliceCols
         << ", offset=" << offset);
    return nullptr;
  }

  // Reinterpret the complete representative allocation first.  Reinterpreting
  // the subslice directly hides its offset in the source type and is rejected
  // by MemDescReinterpretOp's parent-only verifier rule.  For f16 packed in an
  // f32 allocation, the parent destination is twice as wide in logical
  // elements and the subslice offset is converted to those units.
  int64_t parentBlockN = oldBlockN;
  int64_t subsliceOffset = offset;
  if (oldElemTyWidth == elemTyWidth * 2) {
    subsliceOffset *= 2;
  }
  // Keep the destination tile's blockN/colStride encoding while extending its
  // logical shape for the parent view.  Passing parentBlockN to
  // createTMEMDesc would also enlarge the encoding blockN, which changes the
  // physical footprint instead of merely exposing the full parent.
  auto newTmemEncoding =
      cast<ttng::TensorMemoryEncodingAttr>(newType.getEncoding());
  auto getStorageBits = [](ttg::MemDescType ty) {
    auto rank = cast<ttg::LayoutEncodingTrait>(ty.getEncoding()).getRank();
    auto shape = ty.getAllocShape().take_back(rank);
    auto layout = ttg::toLinearLayout(shape, ty.getEncoding());
    int64_t copies = 1;
    for (int64_t dim : ty.getAllocShape().drop_back(rank))
      copies *= dim;
    auto col = StringAttr::get(ty.getContext(), "col");
    return copies * layout.getInDimSize(col) * ty.getElementTypeBitWidth();
  };

  auto makeParentDesc = [&](int64_t width) {
    auto parentEncoding = ttng::TensorMemoryEncodingAttr::get(
        builder.getContext(), newTmemEncoding.getBlockM(),
        newTmemEncoding.getBlockN(), newTmemEncoding.getColStride(),
        newTmemEncoding.getCGALayout(), newTmemEncoding.getTwoCTAs(),
        newTmemEncoding.getCtaMode());
    SmallVector<int64_t> parentShape(newShape);
    parentShape.back() = width;
    return ttg::MemDescType::get(parentShape, newType.getElementType(),
                                 parentEncoding, newType.getMemorySpace(),
                                 newType.getMutableMemory(), parentShape);
  };

  // The destination encoding may use a different packing convention (for
  // example an f16 LHS intentionally using colStride=1).  Choose the smallest
  // parent logical width that has exactly the representative's physical
  // footprint under that encoding; this is the descriptor that can safely be
  // reinterpreted before taking the requested view.
  auto sourceStorageBits = getStorageBits(allocType);
  ttg::MemDescType parentDesc;
  for (int64_t width = parentBlockN; width <= oldBlockN * 16; width *= 2) {
    auto candidate = makeParentDesc(width);
    if (getStorageBits(candidate) == sourceStorageBits) {
      parentBlockN = width;
      parentDesc = candidate;
      break;
    }
  }
  if (!parentDesc) {
    LDBG("unable to construct a same-footprint TMEM parent view");
    return nullptr;
  }
  Value parent = allocResult;
  if (parentDesc != allocType) {
    parent = ttg::MemDescReinterpretOp::create(builder, allocOp->getLoc(),
                                               parentDesc, allocResult);
    // Keep the scheduling metadata carried by the original index user on the
    // parent view.  The final subslice below gets the same metadata as well.
    parent.getDefiningOp()->setAttrs(user->getAttrDictionary());
  }

  // `user` indexes away a leading multibuffer dimension.  Indexing a subslice
  // is rejected by MemDescIndexOp, so index the reinterpreted parent instead,
  // widen the temporary index result to the parent width, and subslice that
  // rank-reduced value.  External users continue to see the requested target
  // tile after the replacement below.
  auto parentIndexType = ttg::MemDescType::get(
      parentDesc.getShape().drop_front(), parentDesc.getElementType(),
      parentDesc.getEncoding(), parentDesc.getMemorySpace(),
      parentDesc.getMutableMemory(), parentDesc.getAllocShape().drop_front());
  user->getOpOperand(0).set(parent);
  user->getResult(0).setType(parentIndexType);
  builder.setInsertionPointAfter(user);
  auto subSlice = ttng::TMEMSubSliceOp::create(
      builder, allocOp->getLoc(), user->getResult(0), subsliceOffset, blockN);
  subSlice->setAttrs(user->getAttrDictionary());
  user->getResult(0).replaceAllUsesExcept(subSlice.getResult(),
                                          subSlice.getOperation());
  return subSlice.getOperation();
}

Operation *sliceAndReinterpretTMEMBuffer(OpBuilder &builder, Operation *allocOp,
                                         int offset, size_t blockN) {
  auto allocResult = allocOp->getResult(0);
  auto allocType = cast<ttg::MemDescType>(allocResult.getType());
  auto shape = allocType.getShape();
  auto oldBlockN = shape[1];

  // Validate the allocation is valid before attempting to create subslice
  if (oldBlockN < blockN || oldBlockN % blockN != 0 ||
      (offset + blockN) > oldBlockN) {
    // Cannot use this TMEM allocation - return nullptr to signal failure
    // Caller should try another TMEM allocation or fall back to SMEM
    LDBG("TMEM buffer allocation validation failed: oldBlockN="
         << oldBlockN << ", blockN=" << blockN << ", offset=" << offset);
    return nullptr;
  }

  // The allocation is already the parent descriptor.  Form the subview
  // directly; reinterpreting a sliced descriptor is invalid.
  auto subSlice = ttng::TMEMSubSliceOp::create(builder, allocOp->getLoc(),
                                               allocResult, offset, blockN);
  return subSlice.getOperation();
}

ttg::MemDescType createTMEMDesc(OpBuilder &builder, Type inputType,
                                int64_t blockM, int64_t blockN) {
  size_t elemBitWidth = 0;
  Type elemType;
  Attribute memSpace;
  Attribute encoding;
  auto context = builder.getContext();
  unsigned highShape = 0;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    elemBitWidth = tensorType.getElementTypeBitWidth();
    elemType = tensorType.getElementType();
    memSpace = ttng::TensorMemorySpaceAttr::get(context);
    encoding = tensorType.getEncoding();
  } else if (auto allocType = dyn_cast<ttg::MemDescType>(inputType)) {
    elemBitWidth = allocType.getElementTypeBitWidth();
    elemType = allocType.getElementType();
    memSpace = allocType.getMemorySpace();
    encoding = allocType.getEncoding();
    if (allocType.getShape().size() == 3)
      highShape = allocType.getShape()[0];
  } else {
    assert(false && "Expected RankedTensorType or ttg::MemDescType");
  }
  assert((elemBitWidth == 16 || elemBitWidth == 32) &&
         "TMEM Layout don't support fp8");
  unsigned colStride = 32 / elemBitWidth;
  // TODO(njriasan): Do we need to handle the ScaleDotElemType::E2M1 && transA
  // case at all from TCGen5MMAScaledOp::getBlockM?
  ttg::CGAEncodingAttr cgaLayout;
  bool twoCTAs = false;
  if (auto tmemLayout =
          mlir::dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
              encoding)) {
    cgaLayout = tmemLayout.getCGALayout();
    twoCTAs = tmemLayout.getTwoCTAs();
  } else if (auto ttgLayout =
                 mlir::dyn_cast<ttg::LayoutEncodingTrait>(encoding)) {
    cgaLayout = ttg::getCGALayout(encoding);
  } else {
    assert(false && "Unsupported encoding");
  }
  auto outputEncoding = ttng::TensorMemoryEncodingAttr::get(
      context, blockM, blockN, colStride, cgaLayout, twoCTAs,
      ttng::TensorMemoryCTAMode::DEFAULT);
  if (highShape > 0) {
    llvm::SmallVector<int64_t> shapeVec{highShape, blockM, blockN};
    llvm::ArrayRef<int64_t> shape(shapeVec);
    return ttg::MemDescType::get(shape, elemType, outputEncoding, memSpace,
                                 /*mutableMemory=*/true);
  }
  llvm::SmallVector<int64_t> shapeVec{blockM, blockN};
  llvm::ArrayRef<int64_t> shape(shapeVec);
  return ttg::MemDescType::get(shape, elemType, outputEncoding, memSpace,
                               /*mutableMemory=*/true);
}

#define GEN_PASS_DEF_NVGPUTEST1DTMEMALLOC
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTest1DTMEMAllocPass
    : public impl::NVGPUTest1DTMEMAllocBase<NVGPUTest1DTMEMAllocPass> {
public:
  using impl::NVGPUTest1DTMEMAllocBase<
      NVGPUTest1DTMEMAllocPass>::NVGPUTest1DTMEMAllocBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilderWithAsyncTaskIds builder(moduleOp.getContext());
    llvm::SmallVector<Operation *> allocOps;
    moduleOp->walk([&](Operation *allocOp) {
      if (allocOp->hasAttr("tmem.start_buffer")) {
        allocOps.push_back(allocOp);
      }
    });
    WalkResult result = moduleOp->walk([&](mlir::Operation *irOp) {
      if (irOp->hasAttr("tmem.start")) {
        if (failed(generate1DAllocations(builder, irOp, allocOps)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir
