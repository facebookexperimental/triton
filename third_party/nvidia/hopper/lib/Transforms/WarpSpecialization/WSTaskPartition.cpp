#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-ws-task-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

// Compute a partition schedule for later passes to actually partition the
// program into partitions.
void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups) {
  if (numWarpGroups <= 1)
    return;

  // Bail out in the presence of user annotations.
  DenseSet<int> allPartitions;
  funcOp->walk([&](Operation *op) {
    allPartitions.insert_range(getWSPartitionIds(op));
  });

  if (!allPartitions.empty())
    return;

  SmallVector<LoopLikeOpInterface> loops;
  bool hasLoad = false;
  SmallVector<Operation *> stores;
  SmallVector<Operation *> dots;

  funcOp.walk([&](Operation *op) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op))
      loops.push_back(loop);
    else if (isa<ttng::WarpGroupDotOp>(op))
      dots.push_back(op);
    else if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      hasLoad = true;
    else if (isa<tt::StoreOp, tt::DescriptorStoreOp>(op))
      stores.push_back(op);
  });

  if (loops.empty() || !hasLoad || dots.empty())
    return;

  // Step 1. Select loads into the first task, which is the producer task by
  // default. Place dots into the second task, which is the consumer.
  // Only consider loads that are connected to a dot op in a loop.
  DenseSet<Operation *> producerOps;
  SmallVector<Operation *> consumerOps;
  BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.inclusive = true;

  for (auto op : dots) {
    consumerOps.push_back(op);
    auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(op);
    if (!dotOp)
      continue;
    SetVector<Operation *> backwardSlice;
    (void)getBackwardSlice(dotOp.getA(), &backwardSlice, opt);
    (void)getBackwardSlice(dotOp.getB(), &backwardSlice, opt);
    for (auto depOp : backwardSlice) {
      if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(depOp)) {
        producerOps.insert(depOp);
      } else if (isa<tt::LoadOp>(depOp) && isExpensiveLoadOrStore(depOp)) {
        producerOps.insert(depOp);
      }
    }
  }

  LLVM_DEBUG({
    LDBG("Producer ops:\n");
    for (auto op : producerOps) {
      op->dump();
    }

    LDBG("\n");
    LDBG("Consumer ops:\n");
    for (auto op : consumerOps) {
      op->dump();
    }

    LDBG("\n");
  });

  if (consumerOps.empty() || producerOps.empty())
    return;

  // Annotate the program with partition ids
  SmallVector<WSPartitionId, 1> producerTaskIds{0};
  SmallVector<WSPartitionId, 2> consumerTaskIds;
  for (unsigned i = 0; i < numWarpGroups - 1; ++i) {
    consumerTaskIds.push_back(i + producerTaskIds.size());
  }

  for (auto op : producerOps) {
    setWSPartitionIds(op, producerTaskIds);
  }

  for (auto op : consumerOps) {
    setWSPartitionIds(op, consumerTaskIds);
  }

  // All stores go with the consumers.
  for (auto op : stores) {
    setWSPartitionIds(op, consumerTaskIds);
  }

  LLVM_DEBUG({
    LDBG("After WS task partition");
    funcOp.dump();
    LDBG("\n");
  });
}

#define GEN_PASS_DEF_NVGPUTESTWSTASKPARTITION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSTaskPartitionPass
    : public impl::NVGPUTestWSTaskPartitionBase<NVGPUTestWSTaskPartitionPass> {
public:
  using impl::NVGPUTestWSTaskPartitionBase<
      NVGPUTestWSTaskPartitionPass>::NVGPUTestWSTaskPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numWarpGroups > 1)
      doTaskPartition(funcOp, numWarpGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
