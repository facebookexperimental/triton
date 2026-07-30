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
// Pipeline placement: This pass runs AFTER all WS-related passes
// (pipeline, optimize_partition_warps, hoist_tmem_alloc, etc.) to avoid
// scheduling/pipeline interference — the barrier ops won't be reordered
// or erased by subsequent WS passes.
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

static Value castToI64(OpBuilder &builder, Location loc, Value value) {
  auto i64Ty = builder.getI64Type();
  Type type = value.getType();
  if (type.isIndex())
    return arith::IndexCastOp::create(builder, loc, i64Ty, value);
  auto intTy = dyn_cast<IntegerType>(type);
  assert(intTy && "expected index or integer loop value");
  unsigned width = intTy.getWidth();
  if (width == 64)
    return value;
  if (width < 64)
    return arith::ExtUIOp::create(builder, loc, i64Ty, value);
  return arith::TruncIOp::create(builder, loc, i64Ty, value);
}

static Value computeLoopIterIndex(OpBuilder &builder, Location loc,
                                  scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value offset = arith::SubIOp::create(builder, loc, iv, lb);
  Value iterIdx = arith::DivUIOp::create(builder, loc, offset, step);
  return castToI64(builder, loc, iterIdx);
}

static Value computeLoopTripCount(OpBuilder &builder, Location loc,
                                  scf::ForOp forOp) {
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value distance = arith::SubIOp::create(builder, loc, ub, lb);
  Value one;
  if (step.getType().isIndex())
    one = arith::ConstantIndexOp::create(builder, loc, 1);
  else
    one = arith::ConstantIntOp::create(
        builder, loc, 1, cast<IntegerType>(step.getType()).getWidth());
  Value numerator = arith::AddIOp::create(
      builder, loc, distance, arith::SubIOp::create(builder, loc, step, one));
  Value tripCount = arith::DivUIOp::create(builder, loc, numerator, step);
  return castToI64(builder, loc, tripCount);
}

static Value computeLinearizedLoopPhase(OpBuilder &builder, Location loc,
                                        scf::ForOp forOp) {
  Value linearIter = computeLoopIterIndex(builder, loc, forOp);
  Value stride = computeLoopTripCount(builder, loc, forOp);
  for (auto parentFor = forOp->getParentOfType<scf::ForOp>(); parentFor;
       parentFor = parentFor->getParentOfType<scf::ForOp>()) {
    Value parentIter = computeLoopIterIndex(builder, loc, parentFor);
    Value scaledParent =
        arith::MulIOp::create(builder, loc, parentIter, stride);
    linearIter = arith::AddIOp::create(builder, loc, scaledParent, linearIter);
    Value parentTripCount = computeLoopTripCount(builder, loc, parentFor);
    stride = arith::MulIOp::create(builder, loc, stride, parentTripCount);
  }

  Value two = arith::ConstantIntOp::create(builder, loc, 2, 64);
  Value rem = arith::RemUIOp::create(builder, loc, linearIter, two);
  return arith::TruncIOp::create(builder, loc, builder.getI32Type(), rem);
}

// Insert the "arrive remote, wait local" cross-CTA sync ops before a 2-CTA
// MMA. The barrier must be allocated externally (before the containing loop
// if the MMA is in a loop).
static void insertSyncBeforeMMA(Operation *mma, Value barrierAlloc,
                                unsigned barrierIdx = 0) {
  MLIRContext *ctx = mma->getContext();
  Location loc = mma->getLoc();
  OpBuilder builder(mma);
  auto i32Ty = builder.getI32Type();

  // Get this MMA's barrier view from the per-loop barrier allocation.
  Value barrierView =
      triton::createSingleBufferView(builder, barrierAlloc, barrierIdx);

  // Get CTA rank within the cluster.
  Value ctaRank = nvgpu::ClusterCTAIdOp::create(builder, loc, i32Ty);

  // Compute leader CTA rank: leader = ctaRank & ~1 (even-ranked CTA in the
  // pair). For a cluster with dims [2,1,1], CTA 0 is leader for CTAs {0,1}.
  Value negTwo = arith::ConstantIntOp::create(builder, loc, -2, 32);
  Value leaderRank = arith::AndIOp::create(builder, loc, ctaRank, negTwo);

  // Map barrier to leader CTA's shared memory via mapa instruction.
  // The result type uses SharedClusterMemorySpace to indicate it refers
  // to another CTA's shared memory.
  auto barrierDescType = cast<ttg::MemDescType>(barrierView.getType());
  auto remoteBarType = ttg::MemDescType::get(
      barrierDescType.getShape(), barrierDescType.getElementType(),
      barrierDescType.getEncoding(),
      ttng::SharedClusterMemorySpaceAttr::get(ctx),
      barrierDescType.getMutableMemory(), barrierDescType.getAllocShape());
  Value remoteBar = ttng::MapToRemoteBufferOp::create(
      builder, loc, remoteBarType, barrierView, leaderRank);

  // Both CTAs arrive on leader's barrier (count=1 each, total=2).
  ttng::ArriveBarrierOp::create(builder, loc, remoteBar, /*count=*/1u);

  // Compute phase from loop induction variable.
  // WaitBarrierOp expects I32 for the phase parameter.
  Value phase;
  if (auto forOp = mma->getParentOfType<scf::ForOp>()) {
    phase = computeLinearizedLoopPhase(builder, loc, forOp);
  } else {
    phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
  }

  // Only leader CTA waits: pred = (ctaRank % 2 == 0).
  Value two = arith::ConstantIntOp::create(builder, loc, 2, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value ctaMod2 = arith::RemUIOp::create(builder, loc, ctaRank, two);
  Value isLeader = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         ctaMod2, zero);

  // Leader waits on LOCAL barrier (not the remote-mapped one).
  // PTX mbarrier.try_wait only supports .shared (local), not .shared::cluster.
  // The local barrier IS the leader's barrier — both CTAs arrived on it via
  // the remote mapping, so the leader can wait on it locally.
  ttng::WaitBarrierOp::create(builder, loc, barrierView, phase, isLeader);

  LDBG("Inserted cross-CTA sync before MMA at " << loc);
}

// Synchronize the 2-CTA block-scale copy feeding a scaled MMA. lower-mma stages
// the scales into TMEM via a leader-issued tcgen05.cp.cta_group::2 that writes
// both CTAs' scale TMEM; nothing orders that cross-CTA write before the
// follower CTA's MMA reads its scale TMEM (single-CTA is ordered by the
// in-order tcgen05 pipeline, 2-CTA is not). Commit the copy and wait on both
// CTAs before the MMA.
static void insertScaleCpSyncBeforeMMA(Operation *mma, Value barrierAlloc,
                                       unsigned barrierIdx = 0) {
  Location loc = mma->getLoc();
  OpBuilder builder(mma);
  Value barrierView =
      triton::createSingleBufferView(builder, barrierAlloc, barrierIdx);

  // Commit the scale cp(s). Under 2-CTA this lowers to a leader-issued
  // multicast::cluster commit that arrives on both CTAs' barrier.
  ttng::TCGen5CommitOp::create(builder, loc, barrierView, /*pred=*/Value(),
                               /*descs=*/ValueRange{});

  // Both CTAs wait for the copy to complete before issuing the MMA.
  Value phase;
  if (auto forOp = mma->getParentOfType<scf::ForOp>())
    phase = computeLinearizedLoopPhase(builder, loc, forOp);
  else
    phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value truePred = arith::ConstantIntOp::create(builder, loc, 1, 1);
  ttng::WaitBarrierOp::create(builder, loc, barrierView, phase, truePred);

  LDBG("Inserted cross-CTA scale-cp sync before scaled MMA at " << loc);
}

// Allocate a cross-CTA barrier with `numBarriers` slots and the given
// `arriveCount`. It is hoisted and init'd before the loop / WarpSpecializeOp so
// the one-time cluster mbarrier-init fence covers it. Handles the three
// placement cases (pre-WS, post-WS default region, post-WS partition capture);
// `anchorOp` locates the partition region to capture into. Returns the barrier
// usable at that site.
static Value allocateLoopCrossCTABarrier(scf::ForOp forOp, Operation *anchorOp,
                                         unsigned numBarriers,
                                         unsigned arriveCount) {
  auto isInDefaultRegion = [](Operation *op,
                              ttg::WarpSpecializeOp wsOp) -> bool {
    Region *defaultRegion = &wsOp.getDefaultRegion();
    return defaultRegion->isAncestor(op->getParentRegion());
  };

  auto wsOp = forOp->getParentOfType<ttg::WarpSpecializeOp>();
  if (!wsOp) {
    // Pre-WS path: standard alloc before the for loop.
    return triton::createBarrierAlloc(forOp, numBarriers, arriveCount);
  }

  // Post-WS path: alloc+init BEFORE the WarpSpecializeOp (so thread 0 of the
  // producer warp group initializes it, covered by the one-time cluster fence),
  // inval+dealloc AFTER.
  Location loc = wsOp->getLoc();
  ImplicitLocOpBuilder rewriter(loc, wsOp);
  Value barrierAlloc =
      triton::createScalarAlloc(rewriter, rewriter.getI64Type(), numBarriers);
  for (unsigned i = 0; i < numBarriers; ++i) {
    Value initView = triton::createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InitBarrierOp>(initView, arriveCount);
  }
  rewriter.setInsertionPointAfter(wsOp);
  for (unsigned i = 0; i < numBarriers; ++i) {
    Value invalView = triton::createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InvalBarrierOp>(invalView);
  }
  rewriter.create<ttg::LocalDeallocOp>(barrierAlloc);

  if (isInDefaultRegion(anchorOp, wsOp)) {
    // The default region implicitly captures values defined before wsOp.
    return barrierAlloc;
  }

  // Partition region (IsolatedFromAbove): capture the barrier explicitly.
  auto partOp = wsOp.getPartitionOp();
  partOp->insertOperands(partOp->getNumOperands(), barrierAlloc);
  Value capturedBarrier;
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg = region->addArgument(barrierAlloc.getType(), loc);
    if (region->isAncestor(anchorOp->getParentRegion()))
      capturedBarrier = arg;
  }
  assert(capturedBarrier && "anchor op not found in any partition region");
  return capturedBarrier;
}

struct Insert2CTASync : public impl::NVGPUInsert2CTASyncBase<Insert2CTASync> {
  using impl::NVGPUInsert2CTASyncBase<Insert2CTASync>::NVGPUInsert2CTASyncBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (!ttng::is2CTA(moduleOp))
      return;

    // Skip TLX kernels — they manage their own cross-CTA sync via
    // explicit barrier ops in the kernel.
    if (moduleOp->hasAttr("tlx.has_tlx_ops"))
      return;

    // Two modes (see the pass doc): the default MMA rendezvous, and the
    // scale-copy completion sync for scaled MMAs (run after lower-mma).
    bool scaleMode = syncScaleCopy;

    // Collect 2-CTA MMA ops that need sync. In scale-copy mode only scaled MMAs
    // carry a block-scale tmem_copy, so restrict to those.
    SmallVector<ttng::MMAv5OpInterface> twoCTAMMAOps;
    moduleOp->walk([&](ttng::MMAv5OpInterface mma) {
      if (!mma.getTwoCtas())
        return;
      if (scaleMode && !isa<ttng::TCGen5MMAScaledOp>(mma.getOperation()))
        return;
      twoCTAMMAOps.push_back(mma);
    });

    if (twoCTAMMAOps.empty())
      return;

    LDBG("Found " << twoCTAMMAOps.size() << " 2-CTA "
                  << (scaleMode ? "scaled " : "") << "MMA ops");

    // Group MMAs by their containing scf.for loop. Allocate one cross-CTA
    // barrier slot per MMA in each loop.
    DenseMap<Operation *, SmallVector<ttng::MMAv5OpInterface>> loopToMMAs;
    SmallVector<ttng::MMAv5OpInterface> nonLoopMMAs;

    for (auto mma : twoCTAMMAOps) {
      auto forOp = mma->getParentOfType<scf::ForOp>();
      if (forOp)
        loopToMMAs[forOp.getOperation()].push_back(mma);
      else
        nonLoopMMAs.push_back(mma);
    }

    // MMA rendezvous barrier: both CTAs thread-arrive (count=2), leader waits.
    // Scale-cp barrier: one leader-issued multicast commit arrives on both
    // CTAs (count=1), both CTAs wait.
    unsigned arriveCount = scaleMode ? 1 : 2;
    auto insertSync = [&](Operation *mma, Value bar, unsigned idx) {
      if (scaleMode)
        insertScaleCpSyncBeforeMMA(mma, bar, idx);
      else
        insertSyncBeforeMMA(mma, bar, idx);
    };

    // Process MMAs inside loops.
    for (auto &[loopOp, mmas] : loopToMMAs) {
      auto forOp = cast<scf::ForOp>(loopOp);
      unsigned numBarriers = mmas.size();
      Value barrierAlloc = allocateLoopCrossCTABarrier(
          forOp, mmas[0].getOperation(), numBarriers, arriveCount);
      for (unsigned i = 0; i < numBarriers; ++i)
        insertSync(mmas[i].getOperation(), barrierAlloc, i);
    }

    // Process standalone MMAs (rare: single-iteration epilogue).
    for (auto mma : nonLoopMMAs) {
      Value barrierAlloc = triton::createBarrierAlloc(
          mma.getOperation(), /*numBarriers=*/1, arriveCount);
      insertSync(mma.getOperation(), barrierAlloc, 0);
    }
  }
};

} // anonymous namespace
