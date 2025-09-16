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

static mlir::Operation *alloc1DTMEMBuffer(mlir::ModuleOp moduleOp,
                                          mlir::Operation *producer,
                                          mlir::Operation *consumer) {
  return nullptr;
}

static void TMEMStore1D(mlir::Operation *allocOP, mlir::Operation *producer) {
  return;
}

static void TMEMLoad1D(mlir::Operation *allocOP, mlir::Operation *consumer) {
  return;
}

static void replaceWith1DTMEM(mlir::ModuleOp moduleOp,
                              mlir::Operation *producer,
                              mlir::Operation *consumer) {
  auto allocOp = alloc1DTMEMBuffer(moduleOp, producer, consumer);
  TMEMStore1D(allocOp, producer);
  TMEMLoad1D(allocOp, consumer);
}

class NVGPUTest1DTMEMAllocPass
    : public impl::NVGPUTest1DTMEMAllocBase<NVGPUTest1DTMEMAllocPass> {
public:
  using impl::NVGPUTest1DTMEMAllocBase<
      NVGPUTest1DTMEMAllocPass>::NVGPUTest1DTMEMAllocBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    // Collect all pairs of (tmem.start, tmem.end)
    mlir::SmallVector<mlir::Operation *> tmemStarts;
    mlir::SmallVector<mlir::Operation *> tmemEnds;
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
    for (size_t i = 0; i < tmemStarts.size(); i++) {
      auto producer = tmemStarts[i];
      auto consumer = tmemEnds[i];
      assert(producer != nullptr);
      assert(consumer != nullptr);
      // Actual generate the TMEM operations.
      replaceWith1DTMEM(moduleOp, producer, consumer);
    }
  }
};

} // namespace mlir
