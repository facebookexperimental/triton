#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

struct TMAStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TMAStore> getTMAStores(scf::ForOp forOp) {
  SmallVector<TMAStore> tmaStores;

  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::DescriptorStoreLikeOpInterface>(op)) {
      tmaStores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
      // Don't walk into nested loops.
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  return tmaStores;
}

static Value createAlloc(scf::ForOp &forOp, const TMAStore &store) {
  OpBuilder builder(forOp);
  RankedTensorType ty = store.src.getType();
  auto encoding =
      triton::nvidia_gpu::getEncodingFromDescriptor(store.op, ty, store.desc);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      ttg::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                            sharedMemorySpace, /*mutableMemory*/ true);
  Value alloc =
      ttg::LocalAllocOp::create(builder, store.op->getLoc(), memdescType);
  return alloc;
}

static void createTMAAsyncCopy(scf::ForOp forOp, const TMAStore &store,
                               Value alloc) {
  OpBuilder builder(store.op);
  Location loc = store.op->getLoc();
  RankedTensorType ty = store.src.getType();

  // Put wait before the local_store make the store truly async. We know
  // that we are the only user of the CopyLocalToGlobal.
  ttng::TMAStoreWaitOp::create(builder, loc, 0);
  ttg::LocalStoreOp::create(builder, loc, store.src, alloc);
  ttng::FenceAsyncSharedOp::create(builder, loc, false);
  auto desc = store.desc;
  if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(store.op)) {
    auto indices = ttng::translateTMAIndices(
        builder, storeOp.getLoc(),
        storeOp.getDesc().getType().getBlockType().getEncoding(),
        storeOp.getIndices());
    ttng::AsyncTMACopyLocalToGlobalOp::create(builder, loc, desc,
                                              storeOp.getIndices(), alloc);
  } else if (auto reduceOp = dyn_cast<tt::DescriptorReduceOp>(store.op)) {
    auto indices = ttng::translateTMAIndices(
        builder, reduceOp.getLoc(),
        reduceOp.getDesc().getType().getBlockType().getEncoding(),
        reduceOp.getIndices());
    ttng::AsyncTMAReduceOp::create(builder, loc, reduceOp.getKind(), desc,
                                   reduceOp.getIndices(), alloc,
                                   triton::EvictionPolicy::NORMAL);
  } else {
    auto scatterOp = cast<tt::DescriptorScatterOp>(store.op);
    ttng::AsyncTMAScatterOp::create(builder, loc, desc, scatterOp.getXOffsets(),
                                    scatterOp.getYOffset(), alloc);
  }

  store.op->erase();
}

static void lowerTMADescriptorCreation(scf::ForOp forOp) {
  // Use max_stage=3 to double buffer the descriptor.
  triton::CoarseSchedule schedule(3);
  triton::lowerTMADescriptors(forOp, schedule);
}

bool mlir::triton::mergeEarlyLoweredTMAStoreAllocs(scf::ForOp forOp) {
  // Merge LocalAllocOp buffers used by early-lowered TMA stores that have
  // the same shape and element type. This reduces SMEM usage and frees
  // budget for double-buffering channel buffers.
  SmallVector<ttng::AsyncTMACopyLocalToGlobalOp> tmaStores;
  forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      if (copyOp.getToken())
        tmaStores.push_back(copyOp);
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  if (tmaStores.empty())
    return false;

  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> sharedAllocs;
  bool changed = false;

  for (auto copyOp : tmaStores) {
    Value origAlloc = copyOp.getSrc();
    auto allocOp = origAlloc.getDefiningOp<ttg::LocalAllocOp>();
    if (!allocOp)
      continue;
    auto memDescType = cast<ttg::MemDescType>(origAlloc.getType());
    auto key =
        std::make_pair(memDescType.getShape(), memDescType.getElementType());
    Value &sharedAlloc = sharedAllocs[key];
    if (!sharedAlloc) {
      sharedAlloc = origAlloc;
      continue;
    }
    if (sharedAlloc == origAlloc)
      continue;

    // Redirect all in-loop users of this alloc to the shared one.
    origAlloc.replaceAllUsesWith(sharedAlloc);
    allocOp->erase();
    changed = true;
  }

  return changed;
}

bool mlir::triton::pipelineEarlyLoweredTMAStores(scf::ForOp forOp) {
  // Convert early-lowered TMA stores from the token-based wait pattern to
  // the pendings-based pattern. At this point the LocalAllocOp has been
  // split (LocalAllocOp() outside loop + LocalStoreOp inside loop) and
  // the alloc hoisted before the loop by the memory planner.
  //
  // Input pattern (inside loop):
  //   LocalStoreOp(src, alloc)
  //   AsyncTMACopyLocalToGlobalOp(desc, coord, alloc) -> token
  //   TMAStoreTokenWaitOp(token)
  //
  // Output pattern (inside loop):
  //   TMAStoreWaitOp(0)
  //   LocalStoreOp(src, alloc)
  //   FenceAsyncSharedOp
  //   AsyncTMACopyLocalToGlobalOp(desc, coord, alloc)  // no token
  //
  // Plus TMAStoreWaitOp(0) after the loop.

  SmallVector<ttng::AsyncTMACopyLocalToGlobalOp> tmaStores;
  forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      if (copyOp.getToken())
        tmaStores.push_back(copyOp);
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  if (tmaStores.empty())
    return false;

  // Reuse a single alloc per shape/element-type, matching pipelineTMAStores.
  // Since each subtile store does TMAStoreWaitOp(0) before overwriting,
  // sharing is safe.
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> sharedAllocs;
  DenseSet<Operation *> origAllocsToErase;

  for (auto copyOp : tmaStores) {
    Location loc = copyOp.getLoc();
    Value origAlloc = copyOp.getSrc();
    auto allocOp = origAlloc.getDefiningOp<ttg::LocalAllocOp>();
    auto memDescType = cast<ttg::MemDescType>(origAlloc.getType());
    auto key =
        std::make_pair(memDescType.getShape(), memDescType.getElementType());
    Value &sharedAlloc = sharedAllocs[key];
    if (!sharedAlloc) {
      if (allocOp && !allocOp->getParentOfType<scf::ForOp>()) {
        sharedAlloc = origAlloc;
      } else {
        OpBuilder hoistBuilder(forOp);
        sharedAlloc = ttg::LocalAllocOp::create(hoistBuilder, loc, memDescType);
      }
    }

    // Find the LocalStoreOp that writes to this alloc.
    ttg::LocalStoreOp localStore = nullptr;
    for (auto user : origAlloc.getUsers()) {
      if (auto s = dyn_cast<ttg::LocalStoreOp>(user)) {
        if (s->getParentOfType<scf::ForOp>() == forOp) {
          localStore = s;
          break;
        }
      }
    }

    // Insert TMAStoreWaitOp(0) before the LocalStoreOp.
    if (localStore) {
      OpBuilder builder(localStore);
      ttng::TMAStoreWaitOp::create(builder, loc, 0);
      // Redirect LocalStoreOp to the shared alloc.
      if (sharedAlloc != origAlloc)
        localStore.getDstMutable().assign(sharedAlloc);
    }

    // Insert FenceAsyncSharedOp before the TMA copy.
    {
      OpBuilder builder(copyOp);
      ttng::FenceAsyncSharedOp::create(builder, loc, false);
    }

    // Remove the token wait.
    for (auto user : llvm::make_early_inc_range(copyOp->getUsers())) {
      if (auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(user))
        waitOp->erase();
    }

    // Replace with fire-and-forget copy using the shared alloc.
    {
      OpBuilder builder(copyOp);
      ttng::AsyncTMACopyLocalToGlobalOp::create(builder, loc, copyOp.getDesc(),
                                                copyOp.getCoord(), sharedAlloc,
                                                copyOp.getEvict());
      copyOp->erase();
    }

    // Mark the original alloc for removal if it's not the shared one.
    if (allocOp && sharedAlloc != origAlloc)
      origAllocsToErase.insert(allocOp);
  }

  // Erase unused original allocs.
  for (auto *op : origAllocsToErase) {
    if (op->use_empty())
      op->erase();
  }

  // Final wait after the loop.
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  ttng::TMAStoreWaitOp::create(builder, forOp->getLoc(), 0);

  return true;
}

bool mlir::triton::pipelineTMAStores(scf::ForOp forOp) {
  SmallVector<TMAStore> tmaStores = getTMAStores(forOp);
  if (tmaStores.empty())
    return false;

  DenseMap<Operation *, Value> storeToAlloc;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> allocs;
  for (const TMAStore &store : tmaStores) {
    // Reuse allocations for stores of the same shape and types. This allows
    // saving shared memory usage. It is valid since we have a wait 0 before
    // every local_store. We could pipeline more aggressively if we didn't
    // reuse but there is a tradeoff with shared memory usage.
    RankedTensorType srcTy = store.src.getType();
    auto key = std::make_pair(srcTy.getShape(), srcTy.getElementType());
    Value &alloc = allocs[key];
    if (!alloc) {
      alloc = createAlloc(forOp, store);
    }
    storeToAlloc[store.op] = alloc;
  }

  bool hasDeviceSideTMA = llvm::any_of(tmaStores, [](const TMAStore &store) {
    return !triton::isHostSideDescriptor(store.desc);
  });
  for (const TMAStore &store : tmaStores) {
    createTMAAsyncCopy(forOp, store, storeToAlloc[store.op]);
  }

  // Deallocate shared memory buffers.
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  ttng::TMAStoreWaitOp::create(builder, forOp->getLoc(), 0);
  for (auto it : storeToAlloc) {
    ttg::LocalDeallocOp::create(builder, forOp->getLoc(), it.second);
  }

  if (hasDeviceSideTMA) {
    // This is a bit coarse as it would multibuffer any descriptor in the loop
    // but it likely to not have a big impact.
    lowerTMADescriptorCreation(forOp);
  }
  return true;
}
