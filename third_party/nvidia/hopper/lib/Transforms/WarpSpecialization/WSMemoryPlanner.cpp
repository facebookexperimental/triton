#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-ws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

void doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers) {

  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }
  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.
}

#define GEN_PASS_DEF_NVGPUTESTWSMEMORYPLANNER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSMemoryPlannerPass
    : public impl::NVGPUTestWSMemoryPlannerBase<NVGPUTestWSMemoryPlannerPass> {
public:
  using impl::NVGPUTestWSMemoryPlannerBase<
      NVGPUTestWSMemoryPlannerPass>::NVGPUTestWSMemoryPlannerBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numBuffers > 1)
      doMemoryPlanner(funcOp, numBuffers);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
