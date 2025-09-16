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
                                           Operation *expandedInput) {
  // Assume we are only every transferring single output ops.
  auto oldRetType =
      dyn_cast<RankedTensorType>(expandedInput->getResult(0).getType());
  if (!oldRetType || oldRetType.getShape().size() != 2) {
    assert("Producer must be expanded to a 2D Tensor");
  }
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  Attribute tensorMemorySpace =
      triton::nvidia_gpu::TensorMemorySpaceAttr::get(builder.getContext());
  // TODO: FIXME
  Attribute encoding = oldRetType.getEncoding();
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), encoding, tensorMemorySpace,
      /*mutableMemory=*/true);
  auto allocCall = builder.create<ttng::TMEMAllocOp>(
      expandedInput->getLoc(), accMemDescType,
      builder.getType<gpu::AsyncTokenType>(), /*src=*/Value());
  return allocCall;
}

static ttng::TMEMAllocOp TMEMStore1D(OpBuilder &builder, Operation *producer) {
  assert(producer->hasAttr("tmem.start") &&
         "Producer must have tmem.start. We only support 1 producer per "
         "consumer.");

  // Expand from 1D -> 2D
  auto producerOutput = producer->getResult(0);
  auto oldRetType = dyn_cast<RankedTensorType>(producerOutput.getType());
  if (!oldRetType || oldRetType.getShape().size() != 1) {
    assert("Producer should only be a 1D Tensor");
  }
  builder.setInsertionPoint(producer);
  auto expandDims = builder.create<triton::ExpandDimsOp>(producer->getLoc(),
                                                         producerOutput, 1);
  auto alloc = alloc1DTMEMBuffer(builder, expandDims);
  // Remove the attribute. There may be multiple users, possibly within
  // the same partition, so we can't remove the OP entirely.
  producer->removeAttr("tmem.start");
  return alloc;
}

static void TMEMLoad1D(OpBuilder &builder, ttng::TMEMAllocOp allocOP,
                       Operation *consumer) {
  assert(consumer->hasAttr("tmem.end") &&
         "Consumer must have tmem.start. We only support 1 consumer per "
         "producer.");
  return;
}

static void replaceWith1DTMEM(OpBuilder &builder, Operation *producer,
                              Operation *consumer) {
  auto allocOp = TMEMStore1D(builder, producer);
  TMEMLoad1D(builder, allocOp, consumer);
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
    OpBuilder builder(moduleOp);
    for (size_t i = 0; i < tmemStarts.size(); i++) {
      auto producer = tmemStarts[i];
      auto consumer = tmemEnds[i];
      assert(producer != nullptr);
      assert(consumer != nullptr);
      // Actual generate the TMEM operations.
      replaceWith1DTMEM(builder, producer, consumer);
    }
  }
};

} // namespace mlir
