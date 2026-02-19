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

ttng::TMEMAllocOp TMEM1DAllocator::alloc1DTMEMBuffer() {
  auto expandedInput = getExpandedInput();
  auto oldRetType = getResultTensorType(expandedInput->getResult(0), 2);
  auto shape = oldRetType.getShape();
  auto tmemDesc = createTMEMDesc(builder, oldRetType, shape[0], shape[1]);
  auto allocCall = builder.create<ttng::TMEMAllocOp>(
      expandedInput->getLoc(), tmemDesc, builder.getType<ttg::AsyncTokenType>(),
      /*src=*/Value());
  return allocCall;
}

void TMEM1DAllocator::TMEMStore1D(OpResult producer, AsyncTaskId producerTaskId,
                                  Operation *allocOpBuffer) {
  // Expand from 1D -> 2D
  auto producerOp = producer.getDefiningOp();
  auto oldRetType = getResultTensorType(producer, 1);
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
      auto argCTALayout = blockedEnc.getCTALayout();
      auto retCTAsPerCGA = {argCTALayout.getCTAsPerCGA()[0], axis};
      auto retCTASplitNum = {argCTALayout.getCTASplitNum()[0], axis};
      auto retCTAOrder = {axis, argCTALayout.getCTAOrder()[0]};
      auto retCTALayout = triton::gpu::CTALayoutAttr::get(
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
    allocOp = alloc1DTMEMBuffer();
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
    newLayout = ttng::getTmemCompatibleLayout(
        tmemDesc.getShape()[0], tmemDesc.getShape()[1], expandType, numWarps);
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
}

Value TMEM1DAllocator::TMEMLoad1D(OpResult producer, Operation *consumer) {
  auto allocOp = getAllocOp();
  auto producerOp = producer.getDefiningOp();
  auto oldInputType = dyn_cast<RankedTensorType>(producer.getType());
  auto targetEncoding = oldInputType.getEncoding();
  auto oldExpandType = getExpandedInput().getType();
  Attribute newDistributedEncoding = ttng::getTmemCompatibleLayout(
      oldExpandType.getShape()[0], oldExpandType.getShape()[1], oldExpandType,
      numWarps);
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

void generate1DAllocations(OpBuilderWithAsyncTaskIds &builder,
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
      TMEM1DAllocator(builder).replaceWith1DTMEM(producer->getOpResult(0),
                                                 producerPartition.front(),
                                                 consumer, allocOpBuffer);
    }
  }
  // Delete tmem.start
  producer->removeAttr("tmem.start");
}

ttg::MemDescReinterpretOp
sliceAndReinterpretMDTMEM(OpBuilderWithAsyncTaskIds &builder,
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

  // Validate the allocation is valid before attempting to create subslice
  if (oldBlockN < blockN || oldBlockN % blockN != 0 ||
      (offset + blockN) > oldBlockN) {
    // Cannot use this TMEM allocation - return nullptr to signal failure
    // Caller should try another TMEM allocation or fall back to SMEM
    LDBG("TMEM allocation validation failed: oldBlockN="
         << oldBlockN << ", blockN=" << blockN << ", offset=" << offset);
    return nullptr;
  }

  // We convert from allocOp's type to another allocOp's type.
  // When the data type is different, we need to construct another TMEMDesc. For
  // example from 128x128xf32 to 128x128xbf16, we subslice to 128x64xf32, then
  // reinterpret to 128x64xbf16.
  auto tmemDesc = createTMEMDesc(builder, newType, newShape[shape.size() - 2],
                                 newShape[shape.size() - 1]);
  // slice from oldBlockN to blockN
  auto elemTyWidth = newType.getElementType().getIntOrFloatBitWidth();
  auto oldElemTyWidth = allocType.getElementType().getIntOrFloatBitWidth();
  if (oldElemTyWidth == elemTyWidth * 2) {
    auto subSlice = builder.create<ttng::TMEMSubSliceOp>(
        allocOp->getLoc(), allocResult, offset, blockN / 2);
    return builder.create<ttg::MemDescReinterpretOp>(allocOp->getLoc(),
                                                     tmemDesc, subSlice);
  } else if (elemTyWidth == oldElemTyWidth) {
    auto subSlice = builder.create<ttng::TMEMSubSliceOp>(
        allocOp->getLoc(), allocResult, offset, blockN);
    return builder.create<ttg::MemDescReinterpretOp>(allocOp->getLoc(),
                                                     tmemDesc, subSlice);
  } else {
    // Unsupported element type conversion
    return nullptr;
  }
}

ttg::MemDescReinterpretOp sliceAndReinterpretTMEMBuffer(OpBuilder &builder,
                                                        Operation *allocOp,
                                                        int offset,
                                                        size_t blockN) {
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

  auto tmemDesc =
      createTMEMDesc(builder, allocType, shape[shape.size() - 2], 1);
  auto subSlice = builder.create<ttng::TMEMSubSliceOp>(
      allocOp->getLoc(), allocResult, offset, blockN);
  return builder.create<ttg::MemDescReinterpretOp>(allocOp->getLoc(), tmemDesc,
                                                   subSlice);
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
  size_t CTASplitM;
  size_t CTASplitN;
  if (auto ttgLayout = mlir::dyn_cast<ttg::LayoutEncodingTrait>(encoding)) {
    ArrayRef<unsigned> CTASplitNum =
        ttg::getCTALayout(encoding).getCTASplitNum();
    CTASplitM = CTASplitNum[0];
    CTASplitN = CTASplitNum[1];
  } else if (auto tmemLayout =
                 mlir::dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
                     encoding)) {
    CTASplitM = tmemLayout.getCTASplitM();
    CTASplitN = tmemLayout.getCTASplitN();
  } else {
    assert(false && "Unsupported encoding");
  }
  auto outputEncoding = ttng::TensorMemoryEncodingAttr::get(
      context, blockM, blockN, colStride, CTASplitM, CTASplitN);
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
    moduleOp->walk([&](mlir::Operation *irOp) {
      if (irOp->hasAttr("tmem.start")) {
        generate1DAllocations(builder, irOp, allocOps);
      }
    });
  }
};

} // namespace mlir
