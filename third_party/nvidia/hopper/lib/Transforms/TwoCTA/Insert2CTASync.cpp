//===- Insert2CTASync.cpp - Insert cross-CTA sync for 2-CTA mode ---------===//
//
// This pass inserts the cross-CTA synchronization pattern for 2-CTA MMA.
// The pattern is "arrive remote, wait local":
// - Both CTAs arrive on the leader CTA's barrier
// - Only the leader CTA waits on the barrier
//
// This pass also allocates the barrier with arrive_count=2 if not present.
//
// NOTE: This pass is skipped when warp specialization is enabled because the
// WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
// For WS+2CTA, use TLX which has native support.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

/// Check if warp specialization is enabled on any loop
static bool isWarpSpecializationEnabled(ModuleOp moduleOp) {
  bool wsEnabled = false;
  moduleOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("tt.warp_specialize")) {
      wsEnabled = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return wsEnabled;
}

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

/// Allocate a barrier for 2-CTA synchronization at function entry
/// Returns the barrier view (single barrier with arrive_count=2)
static Value create2CTASyncBarrier(tt::FuncOp funcOp, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));

  Location loc = funcOp.getLoc();
  auto context = funcOp.getContext();

  // Create shared memory space attribute
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(context);

  // Create barrier encoding (similar to WSCodePartition)
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);

  // Allocate space for 1 barrier (we only need one for 2-CTA sync)
  Type barrierMemDescType = ttg::MemDescType::get(
      {1, 1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);

  // Allocate the barrier in shared memory
  Value barrierAlloc = builder.create<ttg::LocalAllocOp>(
      loc, barrierMemDescType, Value());

  // Get a view to the single barrier
  Value idx = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value barrierView = builder.create<ttg::MemDescIndexOp>(
      loc, singleBarrierMemDescType, barrierAlloc, idx);

  // Initialize the barrier with arrive_count=2 (for 2 CTAs)
  builder.create<ttng::InitBarrierOp>(loc, barrierView, /*count=*/2);

  // Add a GPU barrier to ensure all threads see the initialization
  builder.create<mlir::gpu::BarrierOp>(loc);

  LDBG("Created 2-CTA sync barrier with arrive_count=2");

  return barrierView;
}

/// Compute phase from loop induction variable: phase = (iteration & 1)
/// Similar to WS's getBufferIdxAndPhase but simplified for single barrier
static Value computePhaseFromLoopIV(OpBuilder &builder, Location loc,
                                    Value loopIV) {
  // phase = loopIV & 1
  Value one;
  if (loopIV.getType().isIndex()) {
    one = builder.create<arith::ConstantIndexOp>(loc, 1);
  } else {
    auto intType = cast<IntegerType>(loopIV.getType());
    one = builder.create<arith::ConstantIntOp>(loc, 1, intType.getWidth());
  }
  Value phaseVal = builder.create<arith::AndIOp>(loc, loopIV, one);

  // Convert to i32 for barrier wait
  if (phaseVal.getType().isIndex()) {
    phaseVal = builder.create<arith::IndexCastOp>(
        loc, builder.getI32Type(), phaseVal);
  } else if (phaseVal.getType().getIntOrFloatBitWidth() != 32) {
    phaseVal = builder.create<arith::TruncIOp>(
        loc, builder.getI32Type(), phaseVal);
  }

  return phaseVal;
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

/// Find existing 2-CTA sync barrier if already allocated
static Value findExisting2CTASyncBarrier(tt::FuncOp funcOp) {
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

struct Insert2CTASyncPass
    : public mlir::impl::NVGPU2CTAInsertSyncBase<Insert2CTASyncPass> {
  using NVGPU2CTAInsertSyncBase::NVGPU2CTAInsertSyncBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Skip if warp specialization is enabled
    // WS passes restructure the IR in ways that are incompatible with 2-CTA sync.
    // For WS+2CTA, use TLX which has native support.
    if (isWarpSpecializationEnabled(moduleOp)) {
      LDBG("Warp specialization enabled (tt.warp_specialize attr found)");
      LDBG("Skipping 2-CTA sync insertion - use TLX for WS+2CTA");
      return;
    }

    // Find all 2-CTA MMA operations
    SmallVector<ttng::TCGen5MMAOp> mmaOps = find2CTAMMAOps(moduleOp);

    if (mmaOps.empty()) {
      LDBG("No 2-CTA MMA operations found, nothing to do");
      return;
    }

    LDBG("Found " << mmaOps.size() << " 2-CTA MMA operations");

    // Process each function that contains 2-CTA MMA ops
    DenseMap<tt::FuncOp, Value> funcToBarrier;

    for (auto mmaOp : mmaOps) {
      // Check if sync is already present from frontend (TLX)
      if (hasExisting2CTASync(mmaOp)) {
        LDBG("2-CTA sync already present from frontend, skipping");
        continue;
      }

      // Get the containing function
      auto funcOp = mmaOp->getParentOfType<tt::FuncOp>();
      if (!funcOp) {
        LDBG("MMA op not inside a function");
        continue;
      }

      // Check if MMA is inside a scf.for loop
      auto forOp = mmaOp->getParentOfType<scf::ForOp>();
      if (!forOp) {
        LDBG("MMA op not inside scf.for loop - skipping 2-CTA sync");
        continue;
      }

      OpBuilder builder(moduleOp.getContext());

      // Get or create the 2-CTA sync barrier for this function
      Value syncBarrier;
      auto it = funcToBarrier.find(funcOp);
      if (it != funcToBarrier.end()) {
        syncBarrier = it->second;
      } else {
        // Check if barrier already exists
        syncBarrier = findExisting2CTASyncBarrier(funcOp);
        if (!syncBarrier) {
          // Allocate a new barrier at function entry
          syncBarrier = create2CTASyncBarrier(funcOp, builder);
        }
        funcToBarrier[funcOp] = syncBarrier;
      }

      // Get or create ClusterCTAIdOp
      Value ctaRank = getOrCreateClusterCTAId(funcOp, builder);

      // Get or create leader rank (cta_rank & ~1)
      Value leaderRank = getOrCreateLeaderRank(funcOp, ctaRank, builder);

      // Get or create leader predicate (cta_rank % 2 == 0)
      Value isLeader = getOrCreateLeaderPredicate(funcOp, ctaRank, builder);

      // Get the loop induction variable for phase computation
      Value loopIV = forOp.getInductionVar();

      // Insert the synchronization pattern before the MMA op
      builder.setInsertionPoint(mmaOp);
      Location loc = mmaOp.getLoc();
      auto context = moduleOp.getContext();

      // Compute phase from loop iteration (phase = iteration & 1)
      Value phase = computePhaseFromLoopIV(builder, loc, loopIV);

      // Map our barrier to the leader CTA's barrier
      // Result type needs SharedClusterMemorySpaceAttr for remote access
      auto srcType = cast<ttg::MemDescType>(syncBarrier.getType());
      Attribute clusterMemorySpace =
          ttng::SharedClusterMemorySpaceAttr::get(context);
      auto remoteType = ttg::MemDescType::get(
          srcType.getShape(), srcType.getElementType(), srcType.getEncoding(),
          clusterMemorySpace, srcType.getMutableMemory());

      auto mappedBarrier = builder.create<ttng::MapToRemoteBufferOp>(
          loc, remoteType, syncBarrier, leaderRank);

      // Both CTAs arrive on the leader's barrier with count=1
      builder.create<ttng::ArriveBarrierOp>(
          loc, mappedBarrier.getResult(), /*count=*/1);

      // Leader waits for both arrivals with the computed phase
      builder.create<ttng::WaitBarrierOp>(loc, mappedBarrier.getResult(), phase,
                                          isLeader);

      LDBG("Inserted 2-CTA sync pattern before MMA op");
      LDBG("  Pattern: arrive on leader's barrier, leader waits");
      LDBG("  Phase computed from loop induction variable");
    }

    LDBG("Finished processing 2-CTA MMA operations");
  }
};

} // namespace
