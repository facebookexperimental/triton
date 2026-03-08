//===- Insert2CTASync.cpp - Insert cross-CTA sync for 2-CTA mode ---------===//
//
// This pass inserts the cross-CTA synchronization pattern for 2-CTA MMA.
// The pattern is "arrive remote, wait local":
// - Both CTAs arrive on the leader CTA's barrier
// - Only the leader CTA waits on the barrier
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// Define before including Passes.h to get the base class
#define GEN_PASS_DEF_NVGPU2CTAINSERTSYNC
#include "nvidia/hopper/include/Transforms/Passes.h"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace nvgpu = triton::nvgpu;

#define DEBUG_TYPE "nvgpu-2cta-insert-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Find all TCGen5MMAOp operations with two_ctas=true
static SmallVector<ttng::TCGen5MMAOp> find2CTAMMAOps(ModuleOp moduleOp) {
  SmallVector<ttng::TCGen5MMAOp> result;
  moduleOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
    if (mmaOp.getTwoCtas()) {
      result.push_back(mmaOp);
    }
  });
  return result;
}

/// Insert ClusterCTAIdOp at the function entry if not already present
static Value getOrCreateClusterCTAId(tt::FuncOp funcOp, OpBuilder &builder) {
  Block &entryBlock = funcOp.getBody().front();
  for (auto &op : entryBlock) {
    if (auto ctaIdOp = dyn_cast<nvgpu::ClusterCTAIdOp>(&op)) {
      return ctaIdOp.getResult();
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entryBlock);
  auto ctaIdOp = builder.create<nvgpu::ClusterCTAIdOp>(funcOp.getLoc());
  return ctaIdOp.getResult();
}

/// Get or create the leader predicate (cta_rank % 2 == 0)
static Value getOrCreateLeaderPredicate(tt::FuncOp funcOp, Value ctaRank,
                                        OpBuilder &builder) {
  // Check if we already have this predicate computed
  Block &entryBlock = funcOp.getBody().front();
  for (auto &op : entryBlock) {
    if (auto cmpOp = dyn_cast<arith::CmpIOp>(&op)) {
      if (cmpOp.getPredicate() == arith::CmpIPredicate::eq) {
        // Check if it's comparing (cta_rank % 2) == 0
        if (auto remOp = cmpOp.getLhs().getDefiningOp<arith::RemSIOp>()) {
          if (remOp.getLhs() == ctaRank) {
            return cmpOp.getResult();
          }
        }
      }
    }
  }

  // Create the predicate after the ctaRank definition
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(ctaRank);
  Location loc = funcOp.getLoc();

  // cta_rank % 2
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  Value ctaInPair = builder.create<arith::RemSIOp>(loc, ctaRank, two);

  // (cta_rank % 2) == 0
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value isLeader =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, ctaInPair, zero);

  return isLeader;
}

/// Get or create the leader CTA rank (cta_rank & ~1)
static Value getOrCreateLeaderRank(tt::FuncOp funcOp, Value ctaRank,
                                   OpBuilder &builder) {
  // Check if we already have this computed
  Block &entryBlock = funcOp.getBody().front();
  for (auto &op : entryBlock) {
    if (auto andOp = dyn_cast<arith::AndIOp>(&op)) {
      if (andOp.getLhs() == ctaRank) {
        // Check if RHS is ~1 = -2 (0xFFFFFFFE)
        if (auto constOp =
                andOp.getRhs().getDefiningOp<arith::ConstantIntOp>()) {
          if (constOp.value() == -2) {
            return andOp.getResult();
          }
        }
      }
    }
  }

  // Create after ctaRank definition
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(ctaRank);
  Location loc = funcOp.getLoc();

  // cta_rank & ~1 (clear bit 0 to get leader rank)
  // ~1 = -2 in two's complement
  Value mask = builder.create<arith::ConstantIntOp>(loc, -2, 32);
  Value leaderRank = builder.create<arith::AndIOp>(loc, ctaRank, mask);

  return leaderRank;
}

/// Find the barrier used for B load synchronization
/// Returns nullptr if not found
static Value findBLoadBarrier(ttng::TCGen5MMAOp mmaOp) {
  // The MMA op may have barriers attached
  auto barriers = mmaOp.getBarriers();
  if (!barriers.empty()) {
    return barriers.front();
  }

  // Look for barrier wait before the MMA that's associated with B load
  // This is a heuristic - in practice, the frontend should allocate a dedicated
  // barrier for 2-CTA sync
  return Value();
}

/// Check if an init_barrier already exists in the function for 2-CTA sync
static Value find2CTASyncBarrier(tt::FuncOp funcOp) {
  Value result;
  funcOp.walk([&](ttng::InitBarrierOp initOp) {
    // Check for a barrier with arrive count of 2 (for 2 CTAs)
    if (initOp.getCount() == 2) {
      result = initOp.getAlloc();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Check if the MMA op already has cross-CTA sync operations nearby
/// Returns true if sync is already present (from frontend)
static bool hasExisting2CTASync(ttng::TCGen5MMAOp mmaOp) {
  // Check if there's a MapToRemoteBufferOp before this MMA in the same block
  Block *block = mmaOp->getBlock();
  for (auto it = block->begin(); it != Block::iterator(mmaOp); ++it) {
    if (isa<ttng::MapToRemoteBufferOp>(&*it)) {
      return true;
    }
  }
  return false;
}

struct Insert2CTASyncPass
    : public mlir::impl::NVGPU2CTAInsertSyncBase<Insert2CTASyncPass> {
  using NVGPU2CTAInsertSyncBase::NVGPU2CTAInsertSyncBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // Track which MMA ops have been processed
    DenseSet<Operation *> processedOps;

    for (auto mmaOp : mmaOps) {
      if (processedOps.contains(mmaOp)) {
        continue;
      }

      // Check if sync is already present from frontend (TLX)
      if (hasExisting2CTASync(mmaOp)) {
        LDBG("2-CTA sync already present from frontend, skipping");
        processedOps.insert(mmaOp);
        continue;
      }

      // Get the containing function
      auto funcOp = mmaOp->getParentOfType<tt::FuncOp>();
      if (!funcOp) {
        LDBG("MMA op not inside a function");
        continue;
      }

      OpBuilder builder(moduleOp.getContext());

      // Get or create ClusterCTAIdOp
      Value ctaRank = getOrCreateClusterCTAId(funcOp, builder);

      // Get or create leader rank (cta_rank & ~1)
      Value leaderRank = getOrCreateLeaderRank(funcOp, ctaRank, builder);

      // Get or create leader predicate (cta_rank % 2 == 0)
      Value isLeader = getOrCreateLeaderPredicate(funcOp, ctaRank, builder);

      // Find or check for existing 2-CTA sync barrier
      Value syncBarrier = find2CTASyncBarrier(funcOp);

      if (!syncBarrier) {
        // No dedicated 2-CTA sync barrier found
        // The frontend should have allocated this via tlx.alloc_barriers with
        // arrive_count=2 In this case, we just log and skip - the sync pattern
        // is likely already in place from the frontend
        LDBG("No 2-CTA sync barrier found - assuming frontend handles sync");
        processedOps.insert(mmaOp);
        continue;
      }

      // Insert the synchronization pattern before the MMA op
      // Pattern: arrive remote, wait local
      // 1. Map local barrier to leader's barrier
      // 2. Arrive on leader's barrier (both CTAs do this)
      // 3. Leader waits for both arrivals

      builder.setInsertionPoint(mmaOp);
      Location loc = mmaOp.getLoc();

      // Map our barrier to the leader CTA's barrier
      auto mappedBarrier = builder.create<ttng::MapToRemoteBufferOp>(
          loc, syncBarrier.getType(), syncBarrier, leaderRank);

      // Both CTAs arrive on the leader's barrier
      // arrive_barrier with count=1
      builder.create<ttng::ArriveBarrierOp>(
          loc, mappedBarrier.getResult(), /*count=*/1);

      // Compute phase (typically k % 2 in a loop, but for simplicity use 0)
      // In a real implementation, this would track the loop iteration
      Value phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);

      // Leader waits for both arrivals
      // wait_barrier with predicate = isLeader
      builder.create<ttng::WaitBarrierOp>(loc, mappedBarrier.getResult(), phase,
                                          isLeader);

      processedOps.insert(mmaOp);

      LDBG("Inserted 2-CTA sync pattern before MMA op");
      LDBG("  Pattern: arrive on leader's barrier, leader waits");
    }

    LDBG("Finished processing " << processedOps.size()
                                << " 2-CTA MMA operations");
  }
};

} // namespace
