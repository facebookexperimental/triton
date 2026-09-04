#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/STLExtras.h"
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

      eraseIfDuplicateFence(fence);

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

      eraseIfDuplicateFence(fence);

      return WalkResult::advance();
    });

    // AsyncTMAReduceOp also reads shared memory via the async proxy.
    // Same fence logic as AsyncTMACopyLocalToGlobalOp.
    mod.walk([&](AsyncTMAReduceOp tmaReduceOp) {
      Value src = tmaReduceOp.getSrc();
      SmallVector<Operation *> copyRegToSharedOps = findCopyRegToSharedOps(src);
      if (copyRegToSharedOps.empty())
        return WalkResult::advance();

      OpBuilder builder(tmaReduceOp);
      auto fence = FenceAsyncSharedOp::create(builder, tmaReduceOp.getLoc(),
                                              /*bCluster=*/false);
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (llvm::any_of(copyRegToSharedOps, [&](Operation *op) {
              return shouldPreventFenceHoist(op, loopOp);
            }))
          break;
        loopOp.moveOutOfLoop(fence);
      }

      eraseIfDuplicateFence(fence);

      return WalkResult::advance();
    });
  }

private:
  static bool isLogicalLifetimeBoundary(Operation *op) {
    return op->hasAttr("tlx.logical_lifetime_boundary");
  }

  // Erase `fence` if a matching FenceAsyncSharedOp already exists earlier
  // in the same block, with only pure (memory-effect-free) ops in between.
  void eraseIfDuplicateFence(FenceAsyncSharedOp fence) {
    Operation *prev = fence->getPrevNode();
    while (prev) {
      if (auto lastFence = dyn_cast<FenceAsyncSharedOp>(prev)) {
        if (lastFence.getBCluster() == fence.getBCluster())
          fence.erase();
        break;
      }
      if (!isMemoryEffectFree(prev))
        break;
      prev = prev->getPrevNode();
    }
  }

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
        } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          Operation *parent = yield->getParentOp();
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
            if (yielded != v)
              continue;
            if (auto ifOp = dyn_cast<scf::IfOp>(parent))
              worklist.push_back(ifOp.getResult(index));
            else if (auto forOp = dyn_cast<scf::ForOp>(parent))
              worklist.push_back(forOp.getResult(index));
            else
              result.insert(user);
          }
        } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
            if (initArg == v)
              worklist.push_back(forOp.getBody()->getArgument(index + 1));
          }
        } else if (auto partOp =
                       dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
          auto captures = partOp.getExplicitCaptures();
          auto wsOp = cast<ttg::WarpSpecializeOp>(partOp->getParentOp());
          for (unsigned i = 0; i < captures.size(); ++i) {
            if (captures[i] != v)
              continue;
            for (Region *region : wsOp.getPartitionRegions())
              worklist.push_back(region->getArgument(i));
          }
        } else if (isa<triton::ReturnOp, CallOpInterface,
                       RegionBranchOpInterface,
                       RegionBranchTerminatorOpInterface>(user)) {
          result.insert(user);
        } else if (user->hasTrait<OpTrait::MemDescViewTrait>() ||
                   isMemoryEffectFree(user)) {
          // A local_alias starts a new logical use of reused physical storage.
          // Stores in that lifetime do not populate the preceding lifetime.
          if (isLogicalLifetimeBoundary(user))
            continue;
          for (Value res : user->getResults()) {
            if (isa<ttg::MemDescType>(res.getType()))
              worklist.push_back(res);
          }
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
      // Do not trace a local_alias back into the physical backing allocation:
      // the alias is a distinct logical lifetime. Scan forward from the alias
      // so generic writes within this lifetime are still fenced.
      if (isLogicalLifetimeBoundary(op)) {
        findLocalStoresThroughViews(operand, result);
        return;
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        unsigned resultNum = cast<OpResult>(operand).getResultNumber();
        for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
          if (region->empty()) {
            result.insert(op);
            continue;
          }
          auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
          if (!yield || resultNum >= yield.getNumOperands()) {
            result.insert(op);
            continue;
          }
          findCopyRegToSharedOps(yield.getOperand(resultNum), visited, result);
        }
        return;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        unsigned resultNum = cast<OpResult>(operand).getResultNumber();
        findCopyRegToSharedOps(forOp.getInitArgs()[resultNum], visited, result);
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        findCopyRegToSharedOps(yieldOp.getOperand(resultNum), visited, result);
        return;
      }

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
      }
      if (isa<CallOpInterface, RegionBranchOpInterface>(op)) {
        result.insert(op);
        return;
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
      findCopyRegToSharedOps(wsOp.getExplicitCaptures()[argNum], visited,
                             result);
      return;
    }

    // Conservatively return true for other ops
    result.insert(argOwner);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
