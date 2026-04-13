#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered across generic and async proxy.
//
//===----------------------------------------------------------------------===//

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct FenceInsertionPass
    : public impl::TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  using impl::TritonGPUFenceInsertionBase<
      FenceInsertionPass>::TritonGPUFenceInsertionBase;
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](DotOpInterface dotOp) {
      Value a = dotOp.getA();
      Value b = dotOp.getB();
      SmallVector<Operation *> copyRegToSharedOpsA = findCopyRegToSharedOps(a);
      SmallVector<Operation *> copyRegToSharedOpsB = findCopyRegToSharedOps(b);
      if (copyRegToSharedOpsA.empty() && copyRegToSharedOpsB.empty())
        return WalkResult::advance();

      OpBuilder builder(dotOp);
      auto fence = FenceAsyncSharedOp::create(builder, dotOp.getLoc(),
                                              /*bCluster=*/false);
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (!copyRegToSharedOpsA.empty() &&
            llvm::any_of(copyRegToSharedOpsA, [&](Operation *op) {
              return shouldPreventFenceHoist(op, loopOp);
            }))
          break;
        if (!copyRegToSharedOpsB.empty() &&
            llvm::any_of(copyRegToSharedOpsB, [&](Operation *op) {
              return shouldPreventFenceHoist(op, loopOp);
            }))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      // If the previous op is already a fence, this one isn't needed.
      if (auto lastFence =
              dyn_cast_or_null<FenceAsyncSharedOp>(fence->getPrevNode())) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
      }

      return WalkResult::advance();
    });

    // AsyncTMACopyLocalToGlobalOp reads shared memory via the async proxy.
    // If the SMEM was written via the generic proxy (e.g. LocalAllocOp with a
    // source), we need a fence between the write and the TMA store.
    mod.walk([&](AsyncTMACopyLocalToGlobalOp tmaStoreOp) {
      Value src = tmaStoreOp.getSrc();
      SmallVector<Operation *> copyRegToSharedOps = findCopyRegToSharedOps(src);
      if (copyRegToSharedOps.empty())
        return WalkResult::advance();

      OpBuilder builder(tmaStoreOp);
      auto fence = FenceAsyncSharedOp::create(builder, tmaStoreOp.getLoc(),
                                              /*bCluster=*/false);
      // Try to hoist the fence out of loops if all dependencies are outside.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (llvm::any_of(copyRegToSharedOps, [&](Operation *op) {
              return shouldPreventFenceHoist(op, loopOp);
            }))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      // If the previous op is already a fence, this one isn't needed.
      if (auto lastFence =
              dyn_cast_or_null<FenceAsyncSharedOp>(fence->getPrevNode())) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
      }

      return WalkResult::advance();
    });
  }

private:
  // Walk users of `root` transitively through memdesc view ops, collecting
  // any LocalStoreOp found into `result`.
  void findLocalStoresThroughViews(Value root,
                                   llvm::SetVector<Operation *> &result) {
    SmallVector<Value> worklist = {root};
    DenseSet<Value> seen;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      for (auto *user : v.getUsers()) {
        if (isa<ttg::LocalStoreOp>(user)) {
          result.insert(user);
        } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
          for (auto res : user->getResults())
            worklist.push_back(res);
        }
      }
    }
  }

  // Return true if the fence should NOT be hoisted past `loopOp` because
  // `writeOp` (a generic-proxy SMEM write) executes concurrently with the
  // loop in a different region of the same warp_specialize.
  bool shouldPreventFenceHoist(Operation *writeOp, LoopLikeOpInterface loopOp) {
    if (loopOp->isAncestor(writeOp))
      return true;
    // Don't hoist if the write and the loop are in different concurrent
    // regions of the same warp_specialize (default body vs partition, or
    // different partitions). These regions execute in parallel, so the
    // write happens each loop iteration and the fence must too.
    auto writeWsPartitions =
        writeOp->getParentOfType<ttg::WarpSpecializePartitionsOp>();
    auto loopWsPartitions =
        loopOp->getParentOfType<ttg::WarpSpecializePartitionsOp>();
    if (writeWsPartitions && writeWsPartitions == loopWsPartitions)
      return true;
    // Check for default body vs partition: one has a
    // WarpSpecializePartitionsOp parent and the other doesn't, but both
    // are inside the same WarpSpecializeOp.
    if (bool(writeWsPartitions) != bool(loopWsPartitions)) {
      auto writeWs = writeOp->getParentOfType<ttg::WarpSpecializeOp>();
      if (writeWs &&
          writeWs == loopOp->getParentOfType<ttg::WarpSpecializeOp>())
        return true;
    }
    return false;
  }

  // Return true if the operand depends on a copy from register to shared.
  SmallVector<Operation *> findCopyRegToSharedOps(Value operand) {
    DenseSet<Value> visited;
    llvm::SetVector<Operation *> result;
    findCopyRegToSharedOps(operand, visited, result);
    return result.takeVector();
  }

  void findCopyRegToSharedOps(Value operand, DenseSet<Value> &visited,
                              llvm::SetVector<Operation *> &result) {
    // If the value has already been visited we can safely return false as we
    // would early return when true.
    if (visited.count(operand))
      return;
    visited.insert(operand);
    if (!isa<triton::gpu::MemDescType>(operand.getType()))
      return;

    // Check if any user of this memdesc is a LocalStoreOp, indicating
    // a generic-proxy write to this buffer. This handles the case where
    // the buffer was pre-allocated (e.g. by NVGPUWSTMAStoreLowering) and
    // written via a separate local_store rather than local_alloc with source.
    for (auto *user : operand.getUsers()) {
      if (isa<ttg::LocalStoreOp>(user)) {
        result.insert(user);
        return;
      }
    }

    auto op = operand.getDefiningOp();
    if (op) {
      // reach an alloc copying from register, we need a fence.
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (localAlloc.getSrc()) {
          result.insert(op);
        }
        // Check if there are local_store ops that write to that buffer,
        // following through memdesc view ops (which may have multiple users
        // e.g. when EPILOGUE_SUBTILE > 1 writes multiple sub-tiles).
        findLocalStoresThroughViews(localAlloc.getResult(), result);
        if (!result.empty())
          return;
        // When the alloc is captured by a warp_specialize op, check all
        // partition regions for local_store ops to the corresponding block
        // arg. This handles the case where early TMA store lowering creates
        // a local_alloc + async_tma_copy in the epilogue partition, and
        // code partitioning splits the alloc: the local_store ends up in
        // the computation partition while the TMA copy stays in the
        // epilogue partition.
        for (auto *user : localAlloc.getResult().getUsers()) {
          auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user);
          if (!wsOp)
            continue;
          auto captures = wsOp.getExplicitCaptures();
          for (unsigned i = 0; i < captures.size(); i++) {
            if (captures[i] != localAlloc.getResult())
              continue;
            for (Region *region : wsOp.getPartitionRegions()) {
              Value blockArg = region->getArgument(i);
              findLocalStoresThroughViews(blockArg, result);
              if (!result.empty())
                return;
            }
            break;
          }
        }
      }
      // if it is not an alloc, iterate over the operands.
      for (auto v : op->getOperands()) {
        findCopyRegToSharedOps(v, visited, result);
      }
      return;
    }

    // reach BlockArgument
    BlockArgument arg = cast<BlockArgument>(operand);
    unsigned argNum = arg.getArgNumber();
    Operation *argOwner = arg.getOwner()->getParentOp();
    // look through ForOp iter argument
    if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
      assert(argNum != 0 && "induction var cannot be memdesc type");
      --argNum;
      // prologue
      findCopyRegToSharedOps(forOp.getInitArgs()[argNum], visited, result);
      // yield
      auto yieldOp = forOp.getBody()->getTerminator();
      Value v = yieldOp->getOperand(argNum);
      findCopyRegToSharedOps(v, visited, result);
      return;
    }

    // look through `ttg.warp_specialize`.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(argOwner)) {
      findCopyRegToSharedOps(wsOp.getParentOp().getExplicitCaptures()[argNum],
                             visited, result);
      return;
    }

    // Conservatively return true for other ops
    result.insert(argOwner);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
