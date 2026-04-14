// Insert cross-CTA synchronization for 2-CTA MMA operations.
//
// This pass implements the "arrive remote, wait local" pattern for 2-CTA
// TCGen5 MMA operations. When two CTAs cooperatively execute an MMA instruction
// (tcgen05.mma.cta_group::2), each CTA loads half of the B operand. Before
// issuing the MMA, the leader CTA (even-ranked) must know that both CTAs have
// finished loading their B halves.
//
// The pattern:
//   1. Both CTAs arrive on the leader CTA's cross-CTA barrier
//   2. Only the leader CTA waits on the barrier
//   3. Both CTAs issue the 2-CTA MMA (hardware synchronizes execution)
//
// Pipeline placement: This pass runs BEFORE the WS pipeline so that
// WSTaskIdPropagate naturally assigns async_task_id attributes to the
// cross-CTA sync ops, and WSCodePartition correctly partitions them
// into the consumer warp group.
//
// Reference: fbcode/generative_recommenders/ops/triton/triton_addmm.py

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace nvgpu = triton::nvgpu;

#define DEBUG_TYPE "nvgpu-insert-2cta-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_NVGPUINSERT2CTASYNC
#include "nvidia/hopper/include/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

// Insert the "arrive remote, wait local" cross-CTA sync ops before a 2-CTA
// MMA. The barrier must be allocated externally (before the containing loop
// if the MMA is in a loop).
static void insertSyncBeforeMMA(ttng::TCGen5MMAOp mma, Value barrierAlloc) {
  MLIRContext *ctx = mma.getContext();
  Location loc = mma.getLoc();
  OpBuilder builder(mma);
  auto i32Ty = builder.getI32Type();

  // Get barrier view (index 0 of the single-buffered allocation).
  Value barrierView = triton::createSingleBufferView(builder, barrierAlloc, 0);

  // Get CTA rank within the cluster.
  Value ctaRank = builder.create<nvgpu::ClusterCTAIdOp>(loc, i32Ty);

  // Compute leader CTA rank: leader = ctaRank & ~1 (even-ranked CTA in the
  // pair). For a cluster with dims [2,1,1], CTA 0 is leader for CTAs {0,1}.
  Value negTwo = builder.create<arith::ConstantIntOp>(loc, -2, 32);
  Value leaderRank = builder.create<arith::AndIOp>(loc, ctaRank, negTwo);

  // Map barrier to leader CTA's shared memory via mapa instruction.
  // The result type uses SharedClusterMemorySpace to indicate it refers
  // to another CTA's shared memory.
  auto barrierDescType = cast<ttg::MemDescType>(barrierView.getType());
  auto remoteBarType = ttg::MemDescType::get(
      barrierDescType.getShape(), barrierDescType.getElementType(),
      barrierDescType.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierDescType.getMutableMemory(), barrierDescType.getAllocShape());
  Value remoteBar = builder.create<ttng::MapToRemoteBufferOp>(
      loc, remoteBarType, barrierView, leaderRank);

  // Both CTAs arrive on leader's barrier (count=1 each, total=2).
  builder.create<ttng::ArriveBarrierOp>(loc, remoteBar, /*count=*/1u);

  // Compute phase from loop induction variable.
  // WaitBarrierOp expects I32 for the phase parameter.
  Value phase;
  if (auto forOp = mma->getParentOfType<scf::ForOp>()) {
    // Compute iteration index: (iv - lb) / step, then phase = iter % 2.
    // Division by step ensures correct alternation for non-unit steps.
    Value iv = forOp.getInductionVar();
    Value lb = forOp.getLowerBound();
    Value offset = builder.create<arith::SubIOp>(loc, iv, lb);
    Value step = forOp.getStep();
    Value iterIdx = builder.create<arith::DivUIOp>(loc, offset, step);

    if (iv.getType().isIndex()) {
      Value twoIV = builder.create<arith::ConstantIndexOp>(loc, 2);
      Value rem = builder.create<arith::RemUIOp>(loc, iterIdx, twoIV);
      phase = builder.create<arith::IndexCastOp>(loc, i32Ty, rem);
    } else {
      Value twoIV = builder.create<arith::ConstantIntOp>(
          loc, 2, cast<IntegerType>(iv.getType()).getWidth());
      phase = builder.create<arith::RemUIOp>(loc, iterIdx, twoIV);
    }
  } else {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  }

  // Only leader CTA waits: pred = (ctaRank % 2 == 0).
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value ctaMod2 = builder.create<arith::RemUIOp>(loc, ctaRank, two);
  Value isLeader = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 ctaMod2, zero);

  // Leader waits on LOCAL barrier (not the remote-mapped one).
  // PTX mbarrier.try_wait only supports .shared (local), not .shared::cluster.
  // The local barrier IS the leader's barrier — both CTAs arrived on it via
  // the remote mapping, so the leader can wait on it locally.
  builder.create<ttng::WaitBarrierOp>(loc, barrierView, phase, isLeader);

  LDBG("Inserted cross-CTA sync before MMA at " << loc);
}

struct Insert2CTASync : public impl::NVGPUInsert2CTASyncBase<Insert2CTASync> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Check if any cluster dimension >= 2 (needed for 2-CTA).
    bool hasCluster = false;
    for (auto attr :
         {"ttg.cluster-dim-x", "ttg.cluster-dim-y", "ttg.cluster-dim-z"}) {
      if (auto intAttr = moduleOp->getAttrOfType<IntegerAttr>(attr)) {
        if (intAttr.getInt() >= 2)
          hasCluster = true;
      }
    }
    if (!hasCluster)
      return;

    // Skip TLX kernels — they manage their own cross-CTA sync via
    // explicit barrier ops in the kernel.
    if (moduleOp->hasAttr("tlx.has_tlx_ops"))
      return;

    // Collect 2-CTA MMA ops that need cross-CTA sync insertion.
    SmallVector<ttng::TCGen5MMAOp> twoCTAMMAOps;
    moduleOp->walk([&](ttng::TCGen5MMAOp mma) {
      if (mma.getTwoCtas())
        twoCTAMMAOps.push_back(mma);
    });

    if (twoCTAMMAOps.empty())
      return;

    LDBG("Found " << twoCTAMMAOps.size() << " 2-CTA MMA ops");

    // Group MMAs by their containing scf.for loop. Allocate one cross-CTA
    // barrier per loop, shared by all MMAs in that loop.
    DenseMap<Operation *, SmallVector<ttng::TCGen5MMAOp>> loopToMMAs;
    SmallVector<ttng::TCGen5MMAOp> nonLoopMMAs;

    for (auto mma : twoCTAMMAOps) {
      auto forOp = mma->getParentOfType<scf::ForOp>();
      if (forOp)
        loopToMMAs[forOp.getOperation()].push_back(mma);
      else
        nonLoopMMAs.push_back(mma);
    }

    // Process MMAs inside loops.
    for (auto &[loopOp, mmas] : loopToMMAs) {
      auto forOp = cast<scf::ForOp>(loopOp);

      // Currently only a single 2-CTA MMA per loop is supported. Multiple
      // MMAs would require separate barriers (one per MMA) to avoid phase
      // conflicts on the shared barrier.
      assert(mmas.size() == 1 &&
             "Multiple 2-CTA MMAs in the same loop not yet supported. "
             "Each MMA needs its own cross-CTA barrier to avoid phase "
             "deadlock.");

      // Allocate cross-CTA barrier. In the post-WS path (Meta WS),
      // the loop is nested inside a WarpSpecializeOp partition. The barrier
      // alloc+init must be placed BEFORE the WarpSpecializeOp (so thread 0
      // from the producer warp group initializes it), then captured into the
      // consumer partition.
      Value barrierAlloc;
      auto wsOp = forOp->getParentOfType<ttg::WarpSpecializeOp>();
      if (!wsOp) {
        // Pre-WS path: standard alloc before the for loop.
        barrierAlloc = triton::createBarrierAlloc(forOp, /*numBarriers=*/1,
                                                  /*arriveCount=*/2);
      } else {
        // Post-WS path: allocate BEFORE the WarpSpecializeOp.
        Location loc = wsOp->getLoc();
        ImplicitLocOpBuilder rewriter(loc, wsOp);
        barrierAlloc =
            triton::createScalarAlloc(rewriter, rewriter.getI64Type(), 1);
        Value initView =
            triton::createSingleBufferView(rewriter, barrierAlloc, 0);
        rewriter.create<ttng::InitBarrierOp>(initView, /*arriveCount=*/2);

        // Inval and dealloc AFTER the WarpSpecializeOp.
        rewriter.setInsertionPointAfter(wsOp);
        Value invalView =
            triton::createSingleBufferView(rewriter, barrierAlloc, 0);
        rewriter.create<ttng::InvalBarrierOp>(invalView);
        rewriter.create<ttg::LocalDeallocOp>(barrierAlloc);

        // Capture barrier into WarpSpecializeOp partition regions.
        wsOp->insertOperands(wsOp->getNumOperands(), barrierAlloc);
        Value capturedBarrier;
        for (Region *region : wsOp.getPartitionRegions()) {
          BlockArgument arg = region->addArgument(barrierAlloc.getType(), loc);
          if (region->isAncestor(mmas[0]->getParentRegion()))
            capturedBarrier = arg;
        }
        assert(capturedBarrier && "MMA not found in any partition region");
        barrierAlloc = capturedBarrier;
      }

      insertSyncBeforeMMA(mmas[0], barrierAlloc);
    }

    // Process standalone MMAs (rare: single-iteration epilogue).
    for (auto mma : nonLoopMMAs) {
      Value barrierAlloc =
          triton::createBarrierAlloc(mma, /*numBarriers=*/1, /*arriveCount=*/2);
      insertSyncBeforeMMA(mma, barrierAlloc);
    }
  }
};

} // anonymous namespace

namespace mlir {

// Public entry point for calling Insert2CTASync logic from other passes
// (e.g., from NVGPUWarpSpecialization after code partitioning).
void doInsert2CTASync(triton::FuncOp funcOp) {
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return;

  // Check if any cluster dimension >= 2 (needed for 2-CTA).
  bool hasCluster = false;
  for (auto attr :
       {"ttg.cluster-dim-x", "ttg.cluster-dim-y", "ttg.cluster-dim-z"}) {
    if (auto intAttr = moduleOp->getAttrOfType<IntegerAttr>(attr)) {
      if (intAttr.getInt() >= 2)
        hasCluster = true;
    }
  }
  if (!hasCluster)
    return;

  // Skip TLX kernels.
  if (moduleOp->hasAttr("tlx.has_tlx_ops"))
    return;

  // Collect 2-CTA MMA ops.
  SmallVector<ttng::TCGen5MMAOp> twoCTAMMAOps;
  funcOp->walk([&](ttng::TCGen5MMAOp mma) {
    if (mma.getTwoCtas())
      twoCTAMMAOps.push_back(mma);
  });
  if (twoCTAMMAOps.empty())
    return;

  LDBG("doInsert2CTASync: Found " << twoCTAMMAOps.size() << " 2-CTA MMA ops");

  // Group MMAs by their containing scf.for loop.
  DenseMap<Operation *, SmallVector<ttng::TCGen5MMAOp>> loopToMMAs;
  SmallVector<ttng::TCGen5MMAOp> nonLoopMMAs;
  for (auto mma : twoCTAMMAOps) {
    auto forOp = mma->getParentOfType<scf::ForOp>();
    if (forOp)
      loopToMMAs[forOp.getOperation()].push_back(mma);
    else
      nonLoopMMAs.push_back(mma);
  }

  // Helper: allocate barrier outside WarpSpecializeOp (if present) and pass
  // it as a capture into the isolated partition regions.
  //
  // Root cause for needing this: InitBarrierOp lowers to PTX predicated on
  // threadIdx.x == 0. In WS, thread 0 belongs to the PRODUCER warp group.
  // If the barrier init is inside the CONSUMER partition region, no consumer
  // thread has threadIdx.x == 0, so the barrier is never initialized →
  // deadlock. Placing alloc+init BEFORE the WarpSpecializeOp ensures
  // thread 0 (producer) initializes it. The barrier is then captured into
  // the consumer region via WarpSpecializeOp's explicit capture mechanism.
  auto allocBarrierForMMA = [&](Operation *anchorOp,
                                ttng::TCGen5MMAOp mma) -> Value {
    auto wsOp = anchorOp->getParentOfType<ttg::WarpSpecializeOp>();
    if (!wsOp) {
      // Pre-WS path: standard alloc before the anchor op.
      return triton::createBarrierAlloc(anchorOp, /*numBarriers=*/1,
                                        /*arriveCount=*/2);
    }

    // Post-WS path: allocate and init BEFORE the WarpSpecializeOp.
    Location loc = wsOp->getLoc();
    ImplicitLocOpBuilder rewriter(loc, wsOp);
    Value barrierAlloc =
        triton::createScalarAlloc(rewriter, rewriter.getI64Type(), 1);
    Value initView = triton::createSingleBufferView(rewriter, barrierAlloc, 0);
    rewriter.create<ttng::InitBarrierOp>(initView, /*arriveCount=*/2);

    // Inval and dealloc AFTER the WarpSpecializeOp.
    rewriter.setInsertionPointAfter(wsOp);
    Value invalView = triton::createSingleBufferView(rewriter, barrierAlloc, 0);
    rewriter.create<ttng::InvalBarrierOp>(invalView);
    rewriter.create<ttg::LocalDeallocOp>(barrierAlloc);

    // Add barrier alloc as a capture to WarpSpecializeOp.
    // WarpSpecializePartitionsOp is IsolatedFromAbove, so external values
    // must be passed as explicit captures (operands → block arguments).
    wsOp->insertOperands(wsOp->getNumOperands(), barrierAlloc);
    Value capturedBarrier;
    for (Region *region : wsOp.getPartitionRegions()) {
      BlockArgument arg = region->addArgument(barrierAlloc.getType(), loc);
      // Find the partition region containing the MMA.
      if (region->isAncestor(mma->getParentRegion()))
        capturedBarrier = arg;
    }
    assert(capturedBarrier && "MMA not found in any partition region");
    return capturedBarrier;
  };

  for (auto &[loopOp, mmas] : loopToMMAs) {
    auto forOp = cast<scf::ForOp>(loopOp);
    assert(mmas.size() == 1 &&
           "Multiple 2-CTA MMAs in the same loop not yet supported.");
    Value barrierAlloc = allocBarrierForMMA(forOp, mmas[0]);
    insertSyncBeforeMMA(mmas[0], barrierAlloc);
  }

  for (auto mma : nonLoopMMAs) {
    Value barrierAlloc = allocBarrierForMMA(mma, mma);
    insertSyncBeforeMMA(mma, barrierAlloc);
  }
}

} // namespace mlir
