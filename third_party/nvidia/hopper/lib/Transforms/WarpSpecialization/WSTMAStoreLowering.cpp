#include "CodePartitionUtility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-tma-store-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

void doTMAStoreLowering(triton::FuncOp &funcOp) {
  SmallVector<tt::DescriptorStoreOp> storeOps;
  funcOp.walk([&](tt::DescriptorStoreOp op) {
    // Skip stores with non-trivial reduce semantics.
    if (op.getReduceKind() != tt::DescriptorReduceKind::NONE)
      return;
    storeOps.push_back(op);
  });

  if (storeOps.empty())
    return;

  LDBG("Lowering " << storeOps.size() << " DescriptorStoreOp(s)");

  MLIRContext *ctx = funcOp.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);

  for (auto storeOp : storeOps) {
    auto loc = storeOp.getLoc();
    auto asyncTaskIds = getAsyncTaskIds(storeOp);

    OpBuilderWithAsyncTaskIds builder(storeOp);
    builder.setInsertionPoint(storeOp);

    auto src = storeOp.getSrc();
    auto desc = storeOp.getDesc();
    auto tensorType = src.getType();

    // Compute shared encoding from the descriptor.
    auto encoding = ttng::getEncodingFromDescriptor(storeOp, tensorType, desc);
    ttg::MemDescType memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutableMemory=*/true);

    // Allocate SMEM and copy register data into it in one step.
    auto alloc = builder.create<ttg::LocalAllocOp>(loc, memDescType, src);

    // Translate indices for TMA.
    auto indices = ttng::translateTMAIndices(
        builder, loc, desc.getType().getBlockType().getEncoding(),
        storeOp.getIndices());

    // Async TMA copy from local (SMEM) to global, producing a token.
    auto tokenType = ttg::AsyncTokenType::get(ctx);
    auto tmaStore = builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, tokenType, desc, indices, alloc, tt::EvictionPolicy::NORMAL);

    // Wait for this specific TMA store to finish reading from SMEM.
    builder.create<ttng::TMAStoreTokenWaitOp>(loc, tmaStore.getToken(),
                                              ValueRange{}, ValueRange{},
                                              ValueRange{}, ValueRange{});

    storeOp.erase();
  }
}

// ---------------------------------------------------------------------------
// Standalone pass wrapper
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUWSTMASTORELOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

struct NVGPUWSTMAStoreLoweringPass
    : public impl::NVGPUWSTMAStoreLoweringBase<NVGPUWSTMAStoreLoweringPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { doTMAStoreLowering(funcOp); });
  }
};

// ---------------------------------------------------------------------------
// Reorder buffer_ordered TMAStoreTokenWaitOps based on SMEM buffer reuse
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITREORDER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

// For a TMAStoreTokenWaitOp, return the SMEM memdesc that the corresponding
// TMA store reads from (i.e., the src operand of AsyncTMACopyLocalToGlobalOp).
static Value getTMAStoreBuffer(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();
  if (auto copyOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
    return copyOp.getSrc();
  }
  return nullptr;
}

// Find the K-th write to `buffer` after `start` in the same block. With K
// buffer copies the same index repeats every K writes, so we skip K-1 writes
// and return the K-th. Returns nullptr if fewer than K writes exist.
static Operation *findKthBufferWrite(Operation *start, Value buffer, int k) {
  int count = 0;
  for (auto it = std::next(start->getIterator()),
            end = start->getBlock()->end();
       it != end; ++it) {
    // Only count ops that WRITE to the buffer (e.g., local_store).
    // Reads (e.g., async_tma_copy_local_to_global) don't conflict.
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(&*it)) {
      if (storeOp.getDst() == buffer) {
        if (++count == k)
          return &*it;
      }
    }
  }
  return nullptr;
}

struct NVGPUTMAStoreTokenWaitReorderPass
    : public impl::NVGPUTMAStoreTokenWaitReorderBase<
          NVGPUTMAStoreTokenWaitReorderPass> {
  void runOnOperation() override {
    getOperation()->walk([&](Block *block) {
      // Collect buffer_ordered waits in block order.
      SmallVector<ttng::TMAStoreTokenWaitOp> waits;
      for (auto &op : *block) {
        auto waitOp = dyn_cast<ttng::TMAStoreTokenWaitOp>(&op);
        if (!waitOp || !waitOp->hasAttr("buffer_ordered"))
          continue;
        waits.push_back(waitOp);
      }
      if (waits.empty())
        return;

      for (auto waitOp : waits) {
        Value buffer = getTMAStoreBuffer(waitOp);
        if (!buffer)
          continue;

        // buffer_ordered = K means there are K buffer copies. The same index
        // repeats every K uses, so we move the wait to just before the K-th
        // subsequent use of the buffer (after K-1 intervening uses).
        auto attr = waitOp->getAttrOfType<IntegerAttr>("buffer_ordered");
        if (!attr)
          continue;
        int k = attr.getInt();

        Operation *kthWrite = findKthBufferWrite(waitOp, buffer, k);
        if (kthWrite) {
          waitOp->moveBefore(kthWrite);
        } else {
          // Fewer than K subsequent uses — sink to just before the terminator.
          waitOp->moveBefore(block->getTerminator());
        }

        // Remove the attribute after processing.
        waitOp->removeAttr("buffer_ordered");
      }
    });
  }
};

// ---------------------------------------------------------------------------
// Lower TMAStoreTokenWaitOp with barriers into TMAStoreWaitOp + ArriveBarrierOp
// ---------------------------------------------------------------------------
#define GEN_PASS_DEF_NVGPUTMASTORETOKENWAITLOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

// Count AsyncTMACopyLocalToGlobalOp ops in [from, to) within a block.
static int countTMAStoresInRange(Block::iterator from, Block::iterator to) {
  int count = 0;
  for (auto it = from; it != to; ++it) {
    if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(&*it))
      ++count;
  }
  return count;
}

// Compute the pendings value for a TMAStoreTokenWaitOp.
// pendings = number of AsyncTMACopyLocalToGlobalOp ops issued after the token's
// defining store and before this wait, in program execution order.
static int computePendings(ttng::TMAStoreTokenWaitOp waitOp) {
  Value token = waitOp.getToken();

  // Direct case: token defined by AsyncTMACopyLocalToGlobalOp in same block.
  if (auto defOp = token.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>()) {
    if (defOp->getBlock() == waitOp->getBlock()) {
      // Count TMA stores strictly between def and wait.
      return countTMAStoresInRange(std::next(defOp->getIterator()),
                                   waitOp->getIterator());
    }
    return 0;
  }

  // Loop-carried case: token is a block argument of an scf.for body.
  if (auto blockArg = dyn_cast<BlockArgument>(token)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return 0;

    // Block args for scf.for body are [iv, iter_arg0, iter_arg1, ...].
    // The iter_arg index is blockArg.getArgNumber() - 1 (subtract the IV).
    unsigned iterArgIdx = blockArg.getArgNumber() - 1;

    // Find the corresponding yield operand.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldedVal = yieldOp.getOperand(iterArgIdx);

    // Trace the yielded value to its defining AsyncTMACopyLocalToGlobalOp.
    auto defOp = yieldedVal.getDefiningOp<ttng::AsyncTMACopyLocalToGlobalOp>();
    if (!defOp || defOp->getBlock() != forOp.getBody())
      return 0;

    Block *body = forOp.getBody();

    // Stores after the def until end of loop body (excluding yield).
    int storesAfterDef =
        countTMAStoresInRange(std::next(defOp->getIterator()), body->end());

    // Stores from start of loop body until the wait.
    int storesBeforeWait =
        countTMAStoresInRange(body->begin(), waitOp->getIterator());

    return storesAfterDef + storesBeforeWait;
  }

  // Fallback: unknown pattern, drain all stores.
  return 0;
}

struct NVGPUTMAStoreTokenWaitLoweringPass
    : public impl::NVGPUTMAStoreTokenWaitLoweringBase<
          NVGPUTMAStoreTokenWaitLoweringPass> {
  void runOnOperation() override {
    SmallVector<ttng::TMAStoreTokenWaitOp> opsToLower;
    getOperation()->walk(
        [&](ttng::TMAStoreTokenWaitOp op) { opsToLower.push_back(op); });
    for (auto op : opsToLower) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      int pendings = computePendings(op);
      builder.create<ttng::TMAStoreWaitOp>(loc, pendings);
      for (auto barrier : op.getBarriers()) {
        builder.create<ttng::ArriveBarrierOp>(loc, barrier, /*count=*/1);
      }
      op.erase();
    }
  }
};

} // namespace mlir
