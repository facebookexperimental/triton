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
static void insertSyncBeforeMMA(ttng::TCGen5MMAOp mma, Value barrierAlloc) {
  MLIRContext *ctx = mma.getContext();
  Location loc = mma.getLoc();
  OpBuilder builder(mma);
  auto i32Ty = builder.getI32Type();

  // Get barrier view (index 0 of the single-buffered allocation).
  Value barrierView = triton::createSingleBufferView(builder, barrierAlloc, 0);

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

struct Insert2CTASync : public impl::NVGPUInsert2CTASyncBase<Insert2CTASync> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (!ttng::is2CTA(moduleOp))
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

    // Helper: check if an op is inside the default region of a
    // WarpSpecializeOp (as opposed to a partition region).
    auto isInDefaultRegion = [](Operation *op,
                                ttg::WarpSpecializeOp wsOp) -> bool {
      Region *defaultRegion = &wsOp.getDefaultRegion();
      return defaultRegion->isAncestor(op->getParentRegion());
    };

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
      // the loop is nested inside a WarpSpecializeOp. The barrier
      // alloc+init must be placed BEFORE the WarpSpecializeOp (so thread 0
      // from the producer warp group initializes it).
      Value barrierAlloc;
      auto wsOp = forOp->getParentOfType<ttg::WarpSpecializeOp>();
      if (!wsOp) {
        // Pre-WS path: standard alloc before the for loop.
        barrierAlloc = triton::createBarrierAlloc(forOp, /*numBarriers=*/1,
                                                  /*arriveCount=*/2);
      } else if (isInDefaultRegion(mmas[0], wsOp)) {
        // Post-WS path, MMA in default region: The default region can
        // implicitly capture values defined before the WarpSpecializeOp,
        // so no explicit capture is needed. Just allocate+init before wsOp
        // and inval+dealloc after.
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
      } else {
        // Post-WS path, MMA in a partition region (IsolatedFromAbove):
        // Must capture the barrier explicitly into the partition.
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
        auto partOp = wsOp.getPartitionOp();
        partOp->insertOperands(partOp->getNumOperands(), barrierAlloc);
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
