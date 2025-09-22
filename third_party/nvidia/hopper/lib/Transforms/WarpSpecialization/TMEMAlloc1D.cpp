#include "TMEMUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/PassManager.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-1D-tmem-alloc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

// Wrapper class to hold the context for handling
// 1D TMEM Allocation.
class TMEM1DAllocator {
private:
  OpBuilder &builder;
  // Intermediate info to minimize code reuse across functions.
  int numWarps = -1;
  tt::ExpandDimsOp _expandedInput = nullptr;
  Operation *_allocOp = nullptr;

public:
  TMEM1DAllocator(OpBuilder &builder) : builder(builder) {}

private:
  void copyAttrs(Operation *oldOp, Operation *newOp) {
    // Right now we just copy over ttg.partition per
    // the example.
    // TODO: Should we copy over loop information?
    auto partitionAttr = oldOp->getAttr("ttg.partition");
    newOp->setAttr("ttg.partition", partitionAttr);
  }

  void setExpandedInput(tt::ExpandDimsOp expandedInput) {
    this->_expandedInput = expandedInput;
  }

  tt::ExpandDimsOp getExpandedInput() {
    assert(_expandedInput != nullptr && "Must call setExpandedInput");
    return _expandedInput;
  }

  void setAllocOp(Operation *allocOp) { this->_allocOp = allocOp; }

  Operation *getAllocOp() {
    assert(_allocOp != nullptr && "Must call getAllocOp()");
    return _allocOp;
  }

  RankedTensorType getResultTensorType(Operation *op, size_t expectedSize) {
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType || outputType.getShape().size() != 2) {
      assert("Invalid tensor input");
    }
    return outputType;
  }

  ttng::TMEMAllocOp alloc1DTMEMBuffer() {
    auto expandedInput = getExpandedInput();
    auto oldRetType = getResultTensorType(expandedInput, 2);
    auto shape = oldRetType.getShape();
    auto tmemDesc = createTMEMDesc(builder, oldRetType, shape[0], shape[1]);
    auto allocCall = builder.create<ttng::TMEMAllocOp>(
        expandedInput->getLoc(), tmemDesc,
        builder.getType<ttg::AsyncTokenType>(),
        /*src=*/Value());
    return allocCall;
  }

  void TMEMStore1D(Operation *producer,
                   std::optional<Operation *> allocOpBuffer) {
    // Expand from 1D -> 2D
    auto oldRetType = getResultTensorType(producer, 1);
    builder.setInsertionPointAfter(producer);
    // TODO(njriasan): This only works because
    // producer->getResult(0) already has a ttg.slice attribute.
    // We will need to update this to work more generally.
    auto expandDims = builder.create<tt::ExpandDimsOp>(
        producer->getLoc(), producer->getResult(0), 1);
    copyAttrs(producer, expandDims);
    setExpandedInput(expandDims);
    Operation *allocOp;
    if (allocOpBuffer.has_value()) {
      allocOp = allocOpBuffer.value();
    } else {
      allocOp = alloc1DTMEMBuffer();
    }
    setAllocOp(allocOp);

    // Verify that these layouts are compatible.
    auto tmemDesc = dyn_cast<ttg::MemDescType>(allocOp->getResult(0).getType());
    auto expandType = expandDims.getType();
    bool layoutTmemCompatible = ttng::isDistributedLayoutTMemCompatible(
        expandDims, expandType, tmemDesc);
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
      src = builder.create<ttg::ConvertLayoutOp>(expandDims.getLoc(), newTy,
                                                 expandDims);
      copyAttrs(producer, src);
    }
    // Generate the store
    Value trueVal = builder.create<arith::ConstantIntOp>(src->getLoc(), 1, 1);
    auto storeOp = builder.create<ttng::TMEMStoreOp>(
        src->getLoc(), allocOp->getResult(0), src->getResult(0), trueVal);
    copyAttrs(producer, storeOp);
  }

  void TMEMLoad1D(Operation *producer, Operation *consumer) {
    auto allocOp = getAllocOp();
    auto producerOutput = producer->getResult(0);
    auto oldInputType = dyn_cast<RankedTensorType>(producerOutput.getType());
    auto targetEncoding = oldInputType.getEncoding();
    auto oldExpandType = getExpandedInput().getType();
    Attribute newDistributedEncoding = ttng::getTmemCompatibleLayout(
        oldExpandType.getShape()[0], oldExpandType.getShape()[1], oldExpandType,
        numWarps);
    auto newExpandType =
        oldExpandType.cloneWithEncoding(newDistributedEncoding);
    // Generate the load
    builder.setInsertionPoint(consumer);
    auto loadOp = builder.create<ttng::TMEMLoadOp>(
        consumer->getLoc(), newExpandType,
        builder.getType<ttg::AsyncTokenType>(), allocOp->getResult(0), Value());
    copyAttrs(consumer, loadOp);
    // Generate the reshape
    auto reshape = builder.create<tt::ReshapeOp>(
        consumer->getLoc(), oldInputType.getShape(), loadOp);
    copyAttrs(consumer, reshape);
    // Generate a convert layout.
    auto newInput = builder.create<ttg::ConvertLayoutOp>(consumer->getLoc(),
                                                         oldInputType, reshape);
    copyAttrs(consumer, newInput);
    // Replace the uses in the consumer
    consumer->replaceUsesOfWith(producerOutput, newInput);
  }

public:
  void
  replaceWith1DTMEM(Operation *producer, Operation *consumer,
                    std::optional<Operation *> allocOpBuffer = std::nullopt) {
    this->numWarps = ttg::lookupNumWarps(producer);
    assert((numWarps == 4 || numWarps == 8) && "Only support 4 or 8 warps");
    TMEMStore1D(producer, allocOpBuffer);
    TMEMLoad1D(producer, consumer);
  }
};

void generate1DAllocations(OpBuilder &builder, Operation *producer,
                           llvm::SmallVector<ttng::TMEMAllocOp> &allocOps) {
  assert(producer->hasAttr("tmem.start") && "Expected tmem.start");
  std::optional<Operation *> allocOpBuffer = std::nullopt;
  auto producerTMEMStart =
      mlir::cast<mlir::IntegerAttr>(producer->getAttr("tmem.start")).getInt();
  if (producerTMEMStart < allocOps.size()) {
    auto allocOp = allocOps[producerTMEMStart];
    auto allocShape = allocOp.getType().getShape();
    if (allocShape[1] != 1) {
      builder.setInsertionPointAfter(allocOp);
      // Hardcode allocShape[0] / 2 for testing.
      allocOpBuffer =
          sliceAndReinterpretTMEMBuffer(builder, allocOp, allocShape[0] / 2, 1);
    } else {
      allocOpBuffer = allocOp;
    }
  }
  auto producerPartition =
      mlir::cast<mlir::IntegerAttr>(producer->getAttr("ttg.partition"))
          .getInt();
  for (auto consumer : producer->getUsers()) {
    auto consumerParition =
        mlir::cast<mlir::IntegerAttr>(consumer->getAttr("ttg.partition"))
            .getInt();
    if (producerPartition != consumerParition) {
      TMEM1DAllocator(builder).replaceWith1DTMEM(producer, consumer,
                                                 allocOpBuffer);
    }
  }
  // Delete tmem.start
  producer->removeAttr("tmem.start");
}

ttg::MemDescReinterpretOp
sliceAndReinterpretTMEMBuffer(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                              int offset, size_t blockN) {
  auto allocType = allocOp.getType();
  auto shape = allocType.getShape();
  auto oldBlockN = shape[1];
  assert(oldBlockN >= blockN && "Invalid blockN size");
  assert(oldBlockN % blockN == 0 && "Invalid blockN divisibility");
  assert((offset + blockN) <= oldBlockN && "Invalid offset");
  auto tmemDesc = createTMEMDesc(builder, allocType, shape[0], 1);
  auto subSlice = builder.create<ttng::TMEMSubSliceOp>(allocOp->getLoc(),
                                                       allocOp, offset, blockN);
  return builder.create<ttg::MemDescReinterpretOp>(allocOp->getLoc(), tmemDesc,
                                                   subSlice);
}

ttg::MemDescType createTMEMDesc(OpBuilder &builder, Type inputType,
                                size_t blockM, size_t blockN) {
  size_t elemBitWidth = 0;
  llvm::ArrayRef<int64_t> shape;
  Type elemType;
  Attribute memSpace;
  Attribute encoding;
  auto context = builder.getContext();
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    elemBitWidth = tensorType.getElementTypeBitWidth();
    shape = tensorType.getShape();
    elemType = tensorType.getElementType();
    memSpace = ttng::TensorMemorySpaceAttr::get(context);
    encoding = tensorType.getEncoding();
  } else if (auto allocType = dyn_cast<ttg::MemDescType>(inputType)) {
    elemBitWidth = allocType.getElementTypeBitWidth();
    shape = allocType.getShape();
    elemType = allocType.getElementType();
    memSpace = allocType.getMemorySpace();
    encoding = allocType.getEncoding();
  } else {
    assert(false && "Expected RankedTensorType or ttg::MemDescType");
  }
  assert(shape.size() == 2 && "Expected 2D shape");
  ArrayRef<unsigned> CTASplitNum = ttg::getCTALayout(encoding).getCTASplitNum();
  assert((elemBitWidth == 16 || elemBitWidth == 32) &&
         "TMEM Layout don't support fp8");
  auto unpacked = elemBitWidth != 16;
  // TODO(njriasan): Do we need to handle the ScaleDotElemType::E2M1 && transA
  // case at all from TCGen5MMAScaledOp::getBlockM?
  auto outputEncoding = ttng::TensorMemoryEncodingAttr::get(
      context, blockM, blockN,
      /*unpacked=*/unpacked, CTASplitNum[0], CTASplitNum[1]);
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
    OpBuilder builder(moduleOp.getContext());
    llvm::SmallVector<ttng::TMEMAllocOp> allocOps;
    moduleOp->walk([&](ttng::TMEMAllocOp allocOp) {
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
