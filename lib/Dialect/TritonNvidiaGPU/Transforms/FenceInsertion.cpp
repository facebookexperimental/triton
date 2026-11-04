#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.h"
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
    // Computed before any fence is inserted, so the result does not depend on
    // the order in which consumers are visited.
    DenseMap<Operation *, UnfencedWrites> unfencedWrites;
    mod.walk([&](FunctionOpInterface func) {
      unfencedWrites[func] = collectUnfencedWrites(func);
    });
    // Every generic-proxy write to the allocations `consumer` reads was already
    // fenced by the kernel. A writer in the same thread fenced it in program
    // order; a writer in another thread or partition fenced it before the
    // synchronization the consumer must wait on, so a fence here is redundant.
    auto producerFenced = [&](Operation *consumer, ValueRange operands,
                              ArrayRef<Operation *> writers) {
      auto it =
          unfencedWrites.find(consumer->getParentOfType<FunctionOpInterface>());
      if (it == unfencedWrites.end())
        return false;
      const UnfencedWrites &unfenced = it->second;
      return !unfenced.unknownRoot && llvm::all_of(writers, [](Operation *op) {
        return !getGenericDataWrites(op).empty() && isFencedBeforeUse(op);
      }) && llvm::all_of(operands, [&](Value operand) {
        Operation *root = getRootAlloc(operand);
        return root && !unfenced.roots.contains(root);
      });
    };
    mod.walk([&](DotOpInterface dotOp) {
      SmallVector<Operation *> copyRegToSharedOps;
      SmallVector<Value> operands;
      for (const auto &access : getMemoryAccesses(
               dotOp.getOperation(), ttg::SharedKind::Async, RW::Read)) {
        operands.push_back(access.value);
        llvm::append_range(copyRegToSharedOps,
                           findCopyRegToSharedOps(access.value));
      }
      if (copyRegToSharedOps.empty() ||
          producerFenced(dotOp, operands, copyRegToSharedOps))
        return WalkResult::advance();

      OpBuilder builder(dotOp);
      auto fence = FenceAsyncSharedOp::create(builder, dotOp.getLoc(),
                                              /*bCluster=*/false);
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
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

    // AsyncTMACopyLocalToGlobalOp reads shared memory via the async proxy.
    // If the SMEM was written via the generic proxy (e.g. LocalAllocOp with a
    // source), we need a fence between the write and the TMA store.
    mod.walk([&](AsyncTMACopyLocalToGlobalOp tmaStoreOp) {
      Value src = tmaStoreOp.getSrc();
      SmallVector<Operation *> copyRegToSharedOps = findCopyRegToSharedOps(src);
      if (copyRegToSharedOps.empty() ||
          producerFenced(tmaStoreOp, src, copyRegToSharedOps))
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
      if (copyRegToSharedOps.empty() ||
          producerFenced(tmaReduceOp, src, copyRegToSharedOps))
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
  // True if `op` may sit between a generic-proxy write and its fence: it
  // neither publishes the write to another thread (barrier arrive, atomic,
  // global store) nor reads shared memory through the async proxy.
  static bool canPrecedeFence(Operation *op) {
    if (isMemoryEffectFree(op))
      return true;
    if (op->getNumRegions() != 0)
      return false;
    auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffects)
      return false;
    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffects.getEffects(effects);
    return llvm::all_of(
        effects, [](const MemoryEffects::EffectInstance &effect) {
          auto shared = dyn_cast<ttg::SharedMemoryEffect>(&effect);
          bool generic = shared && shared.getKind() == ttg::SharedKind::Generic;
          if (isa<MemoryEffects::Write>(effect.getEffect()))
            return generic;
          if (isa<MemoryEffects::Read>(effect.getEffect()))
            return !shared || generic;
          return true;
        });
  }

  // True if a fence_async_shared follows `write` before any op that could
  // publish it or read it through the async proxy. Looks past the end of an
  // scf.if branch, but not past a loop body or partition boundary.
  static bool isFencedBeforeUse(Operation *write) {
    for (Operation *op = write; op;) {
      for (Operation *next = op->getNextNode(); next;
           next = next->getNextNode()) {
        if (isa<FenceAsyncSharedOp>(next))
          return true;
        if (!canPrecedeFence(next))
          return false;
      }
      op = op->getBlock()->getParentOp();
      if (!isa_and_nonnull<scf::IfOp>(op))
        return false;
    }
    return false;
  }

  // The allocation `v` is a view of, looking through memdesc views (TLX
  // local_alias is already rewritten to memdesc_reinterpret by this point),
  // warp_specialize captures and loop arguments whose initial and yielded
  // values have the same allocation. Null if it cannot be determined. A view
  // of another CTA's buffer is not resolved: its writes need a cluster-scope
  // fence, which this analysis does not distinguish.
  static Operation *getRootAlloc(Value v) {
    DenseSet<BlockArgument> activeLoopArgs;
    return getRootAlloc(v, activeLoopArgs);
  }

  static Operation *getRootAlloc(Value v,
                                 DenseSet<BlockArgument> &activeLoopArgs) {
    while (true) {
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        Operation *owner = arg.getOwner()->getParentOp();
        if (auto partitions =
                dyn_cast<ttg::WarpSpecializePartitionsOp>(owner)) {
          v = partitions.getExplicitCaptures()[arg.getArgNumber()];
          continue;
        }
        auto forOp = dyn_cast<scf::ForOp>(owner);
        // A loop argument reached again while resolving itself is a cycle.
        if (!forOp || arg.getArgNumber() == 0 ||
            !activeLoopArgs.insert(arg).second)
          return nullptr;
        unsigned i = arg.getArgNumber() - 1;
        Operation *init = getRootAlloc(forOp.getInitArgs()[i], activeLoopArgs);
        Value yielded = forOp.getBody()->getTerminator()->getOperand(i);
        bool sameRoot =
            yielded == arg || getRootAlloc(yielded, activeLoopArgs) == init;
        activeLoopArgs.erase(arg);
        return sameRoot ? init : nullptr;
      }
      Operation *op = v.getDefiningOp();
      if (isa<ttg::LocalAllocOp>(op))
        return op;
      if (!op->hasTrait<OpTrait::MemDescViewTrait>() ||
          isa<MapToRemoteBufferOp>(op))
        return nullptr;
      v = op->getOperand(0);
    }
  }

  // Allocations with a generic-proxy data write that the kernel did not fence,
  // and whether any write, fenced or not, goes to an allocation that cannot be
  // determined (a fence does not help if its scope is wrong, e.g. a write to
  // another CTA's buffer). Writes through a memdesc_reinterpret alias resolve
  // to the same allocation, so they count.
  struct UnfencedWrites {
    DenseSet<Operation *> roots;
    bool unknownRoot = false;
  };

  // The shared-memory values `op` writes through the generic proxy. An
  // immutable local_alloc with a source declares no effects before allocation,
  // so it is listed explicitly. mbarrier.init is a generic write, but barriers
  // are never read through the async proxy as MMA or TMA operands.
  static SmallVector<Value> getGenericDataWrites(Operation *op) {
    SmallVector<Value> writes;
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (alloc.getSrc())
        writes.push_back(alloc.getResult());
      return writes;
    }
    SmallVector<Value> barriers;
    if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op))
      llvm::append_range(barriers, barrierOp.getBarriers());
    for (const auto &access :
         getMemoryAccesses(op, ttg::SharedKind::Generic, RW::Write)) {
      if (access.isShared() && !llvm::is_contained(barriers, access.value))
        writes.push_back(access.value);
    }
    return writes;
  }

  static UnfencedWrites collectUnfencedWrites(FunctionOpInterface func) {
    UnfencedWrites unfenced;
    func.walk([&](Operation *op) {
      SmallVector<Value> writes = getGenericDataWrites(op);
      if (writes.empty())
        return;
      SmallVector<Operation *> roots;
      for (Value write : writes) {
        Operation *root = getRootAlloc(write);
        if (!root) {
          unfenced.unknownRoot = true;
          return;
        }
        roots.push_back(root);
      }
      if (!isFencedBeforeUse(op))
        unfenced.roots.insert(roots.begin(), roots.end());
    });
    return unfenced;
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
        if (hasSharedAccess(user, ttg::SharedKind::Generic, RW::Write)) {
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
      if (hasSharedAccess(user, ttg::SharedKind::Generic, RW::Write)) {
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
        // Walk through memdesc view ops (e.g. memdesc_index) since the
        // warp_specialize may capture a view of the alloc rather than the
        // alloc directly.
        SmallVector<Value> wsWorklist = {localAlloc.getResult()};
        DenseSet<Value> wsSeen;
        while (!wsWorklist.empty()) {
          Value v = wsWorklist.pop_back_val();
          if (!wsSeen.insert(v).second)
            continue;
          for (auto *user : v.getUsers()) {
            if (auto partOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
              auto captures = partOp.getExplicitCaptures();
              auto wsOp = cast<ttg::WarpSpecializeOp>(partOp->getParentOp());
              for (unsigned i = 0; i < captures.size(); i++) {
                if (captures[i] != v)
                  continue;
                for (Region *region : wsOp.getPartitionRegions()) {
                  Value blockArg = region->getArgument(i);
                  findLocalStoresThroughViews(blockArg, result);
                  if (!result.empty())
                    return;
                }
              }
            } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
              for (auto res : user->getResults())
                wsWorklist.push_back(res);
            }
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
