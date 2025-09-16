#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <iostream>

#define DEBUG_TYPE "nvgpu-1D-tmem-alloc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define GEN_PASS_DEF_NVGPUTEST1DTMEMALLOC
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

static ttng::TMEMAllocOp alloc1DTMEMBuffer(OpBuilder &builder,
                                           ModuleOp &moduleOp,
                                           Operation *expandedInput) {
  // Assume we are only every transferring single output ops.
  auto oldRetType =
      dyn_cast<RankedTensorType>(expandedInput->getResult(0).getType());
  if (!oldRetType || oldRetType.getShape().size() != 2) {
    assert("Producer must be expanded to a 2D Tensor");
  }
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  auto context = builder.getContext();
  auto oldEncoding = oldRetType.getEncoding();
  Attribute tensorMemorySpace =
      ttng::TensorMemorySpaceAttr::get(builder.getContext());
  // TODO(njriasan): Do we need to handle the ScaleDotElemType::E2M1 && transA
  // case at all from TCGen5MMAScaledOp::getBlockM? My assumption is no
  // because we are just using TMEM to transfer data and not necessarily
  // in a MMA operation.
  auto blockM = shape[0];
  auto elemType = oldRetType.getElementType();
  unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
  assert((elemBitWidth == 16 || elemBitWidth == 32) &&
         "TMEM Layout don't support fp8");
  // TODO: Does this matter? I assume this is only relevant for
  // selecting valid layouts.
  bool unpacked = elemBitWidth != 16;
  ArrayRef<unsigned> CTASplitNum =
      ttg::getCTALayout(oldEncoding).getCTASplitNum();
  auto encoding = ttng::TensorMemoryEncodingAttr::get(
      builder.getContext(), blockM, shape[1],
      /*unpacked=*/unpacked, CTASplitNum[0], CTASplitNum[1]);
  auto tmemDesc =
      ttg::MemDescType::get(shape, elemType, encoding, tensorMemorySpace,
                            /*mutableMemory=*/true);

  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  auto allocCall = builder.create<ttng::TMEMAllocOp>(
      expandedInput->getLoc(), tmemDesc, builder.getType<gpu::AsyncTokenType>(),
      /*src=*/Value());
  return allocCall;
}

static ttng::TMEMAllocOp
TMEMStore1D(OpBuilder &builder, ModuleOp &moduleOp, Operation *producer,
            std::optional<ttng::TMEMAllocOp> allocOpBuffer = std::nullopt) {
  assert(producer->hasAttr("tmem.start") &&
         "Producer must have tmem.start. We only support 1 producer per "
         "consumer.");

  // Expand from 1D -> 2D
  auto producerOutput = producer->getResult(0);
  auto oldRetType = dyn_cast<RankedTensorType>(producerOutput.getType());
  if (!oldRetType || oldRetType.getShape().size() != 1) {
    assert("Producer should only be a 1D Tensor");
  }
  builder.setInsertionPointAfter(producer);
  auto expandDims =
      builder.create<tt::ExpandDimsOp>(producer->getLoc(), producerOutput, 1);
  ttng::TMEMAllocOp allocOp;
  if (allocOpBuffer.has_value()) {
    allocOp = allocOpBuffer.value();
  } else {
    allocOp = alloc1DTMEMBuffer(builder, moduleOp, expandDims);
  }
  auto tmemDesc = allocOp.getType();
  auto expandType = expandDims.getType();

  // Verify that these layouts are compatible.
  bool layoutTmemCompatible =
      ttng::isDistributedLayoutTMemCompatible(expandDims, expandType, tmemDesc);
  auto oldLayout = expandDims.getType().getEncoding();
  auto newLayout = oldLayout;
  if (!layoutTmemCompatible) {
    // Is this necessary?
    int numWarps = ttg::lookupNumWarps(expandDims);
    newLayout = ttng::getTmemCompatibleLayout(
        tmemDesc.getShape()[0], tmemDesc.getShape()[1], expandType, numWarps);
  }
  mlir::Operation *src = expandDims;
  if (newLayout != oldLayout) {
    auto ty = cast<RankedTensorType>(expandType);
    auto newTy = ty.cloneWithEncoding(newLayout);
    builder.setInsertionPointAfter(expandDims);
    src = builder.create<ttg::ConvertLayoutOp>(expandDims.getLoc(), newTy,
                                               expandDims);
  }
  // Generate the store
  // TODO: What value should the predciate be?
  Value trueVal = builder.create<arith::ConstantIntOp>(src->getLoc(), 1, 1);
  builder.create<ttng::TMEMStoreOp>(src->getLoc(), src->getResult(0), allocOp,
                                    trueVal);
  // Remove the attribute. There may be multiple users, possibly within
  // the same partition, so we can't remove the OP entirely.
  producer->removeAttr("tmem.start");
  return allocOp;
}

static void TMEMLoad1D(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                       Operation *producer, Operation *consumer) {
  assert(consumer->hasAttr("tmem.end") &&
         "Consumer must have tmem.start. We only support 1 consumer per "
         "producer.");
  // Generate the load
  builder.setInsertionPoint(consumer);
  auto producerOutput = producer->getResult(0);
  auto oldInputType = dyn_cast<RankedTensorType>(producerOutput.getType());
  auto targetEncoding = oldInputType.getEncoding();
  // auto ld = rewriter.create<triton::nvidia_gpu::TMEMLoadOp>(
  //   loc, newAccType, tokType, acc, /*dep=*/mma.getToken());
  // TODO: Determine what to set for result type.
  auto loadOp = builder.create<ttng::TMEMLoadOp>(
      consumer->getLoc(), nullptr, builder.getType<gpu::AsyncTokenType>(),
      allocOp);
  // Generate the reshape
  auto reshape = builder.create<tt::ReshapeOp>(consumer->getLoc(),
                                               oldInputType.getShape(), loadOp);
  // Generate a convert layout.
  auto newInput = builder.create<ttg::ConvertLayoutOp>(consumer->getLoc(),
                                                       targetEncoding, reshape);
  // Replace the uses in the consumer
  size_t numOperands = consumer->getNumOperands();
  for (unsigned i = 0; i < numOperands; i++) {
    if (consumer->getOperand(i) == producerOutput) {
      consumer->setOperand(i, newInput);
    }
  }
  // Remove the attribute since we reuse the consumer.
  consumer->removeAttr("tmem.end");
}

static void replaceWith1DTMEM(OpBuilder &builder, ModuleOp &moduleOp,
                              Operation *producer, Operation *consumer) {
  auto allocOp = TMEMStore1D(builder, moduleOp, producer);
  TMEMLoad1D(builder, allocOp, producer, consumer);
}

class NVGPUTest1DTMEMAllocPass
    : public impl::NVGPUTest1DTMEMAllocBase<NVGPUTest1DTMEMAllocPass> {
public:
  using impl::NVGPUTest1DTMEMAllocBase<
      NVGPUTest1DTMEMAllocPass>::NVGPUTest1DTMEMAllocBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    // Collect all pairs of (tmem.start, tmem.end)
    SmallVector<Operation *> tmemStarts;
    SmallVector<Operation *> tmemEnds;
    moduleOp->walk([&](mlir::Operation *irOp) {
      if (irOp->hasAttr("tmem.start")) {
        auto id =
            mlir::cast<mlir::IntegerAttr>(irOp->getAttr("tmem.start")).getInt();
        while (tmemStarts.size() <= id)
          tmemStarts.push_back(nullptr);
        assert(tmemStarts[id] == nullptr);
        tmemStarts[id] = irOp;
      }
      if (irOp->hasAttr("tmem.end")) {
        auto id =
            mlir::cast<mlir::IntegerAttr>(irOp->getAttr("tmem.end")).getInt();
        while (tmemEnds.size() <= id)
          tmemEnds.push_back(nullptr);
        assert(tmemEnds[id] == nullptr);
        tmemEnds[id] = irOp;
      }
    });
    assert(tmemStarts.size() == tmemEnds.size());
    // Generate the actual TMEM operations.
    OpBuilder builder(moduleOp.getContext());
    for (size_t i = 0; i < tmemStarts.size(); i++) {
      auto producer = tmemStarts[i];
      auto consumer = tmemEnds[i];
      assert(producer != nullptr);
      assert(consumer != nullptr);
      builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
      // Actual generate the TMEM operations.
      replaceWith1DTMEM(builder, moduleOp, producer, consumer);
    }
    std::cout << "REACHED END" << std::endl;
  }
};

} // namespace mlir
