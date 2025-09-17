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
  ttng::TMEMAllocOp _allocOp = nullptr;

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

  void setAllocOp(ttng::TMEMAllocOp allocOp) { this->_allocOp = allocOp; }

  ttng::TMEMAllocOp getAllocOp() {
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
    SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                  oldRetType.getShape().end()};
    auto context = builder.getContext();
    auto oldEncoding = oldRetType.getEncoding();
    Attribute tensorMemorySpace =
        ttng::TensorMemorySpaceAttr::get(builder.getContext());
    // TODO(njriasan): Do we need to handle the ScaleDotElemType::E2M1 && transA
    // case at all from TCGen5MMAScaledOp::getBlockM?
    auto blockM = shape[0];
    auto elemType = oldRetType.getElementType();
    unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
    assert((elemBitWidth == 16 || elemBitWidth == 32) &&
           "TMEM Layout don't support fp8");
    bool unpacked = elemBitWidth != 16;
    ArrayRef<unsigned> CTASplitNum =
        ttg::getCTALayout(oldEncoding).getCTASplitNum();
    auto encoding = ttng::TensorMemoryEncodingAttr::get(
        builder.getContext(), blockM, shape[1],
        /*unpacked=*/unpacked, CTASplitNum[0], CTASplitNum[1]);
    auto tmemDesc =
        ttg::MemDescType::get(shape, elemType, encoding, tensorMemorySpace,
                              /*mutableMemory=*/true);

    auto allocCall = builder.create<ttng::TMEMAllocOp>(
        expandedInput->getLoc(), tmemDesc,
        builder.getType<ttg::AsyncTokenType>(),
        /*src=*/Value());
    return allocCall;
  }

  void TMEMStore1D(Operation *producer,
                   std::optional<ttng::TMEMAllocOp> allocOpBuffer) {
    // Expand from 1D -> 2D
    auto oldRetType = getResultTensorType(producer, 1);
    builder.setInsertionPointAfter(producer);
    auto expandDims = builder.create<tt::ExpandDimsOp>(
        producer->getLoc(), producer->getResult(0), 1);
    copyAttrs(producer, expandDims);
    setExpandedInput(expandDims);
    ttng::TMEMAllocOp allocOp;
    if (allocOpBuffer.has_value()) {
      allocOp = allocOpBuffer.value();
    } else {
      allocOp = alloc1DTMEMBuffer();
    }
    setAllocOp(allocOp);

    // Verify that these layouts are compatible.
    auto tmemDesc = allocOp.getType();
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
        src->getLoc(), allocOp, src->getResult(0), trueVal);
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
        builder.getType<ttg::AsyncTokenType>(), allocOp, Value());
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
  void replaceWith1DTMEM(
      Operation *producer, Operation *consumer,
      std::optional<ttng::TMEMAllocOp> allocOpBuffer = std::nullopt) {
    this->numWarps = ttg::lookupNumWarps(producer);
    TMEMStore1D(producer, allocOpBuffer);
    TMEMLoad1D(producer, consumer);
  }
};

void generate1DAllocations(OpBuilder &builder, Operation *producer) {
  assert(producer->hasAttr("tmem.start") && "Expected tmem.start");
  auto producerPartition =
      mlir::cast<mlir::IntegerAttr>(producer->getAttr("ttg.partition"))
          .getInt();
  for (auto consumer : producer->getUsers()) {
    auto consumerParition =
        mlir::cast<mlir::IntegerAttr>(consumer->getAttr("ttg.partition"))
            .getInt();
    if (producerPartition != consumerParition) {
      TMEM1DAllocator(builder).replaceWith1DTMEM(producer, consumer);
    }
  }
  // Delete tmem.start
  producer->removeAttr("tmem.start");
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
    moduleOp->walk([&](mlir::Operation *irOp) {
      if (irOp->hasAttr("tmem.start")) {
        generate1DAllocations(builder, irOp);
      }
    });
  }
};

} // namespace mlir
