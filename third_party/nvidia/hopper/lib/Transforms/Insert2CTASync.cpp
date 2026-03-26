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
    Value twoIdx = builder.create<arith::ConstantIndexOp>(loc, 2);
    Value phaseIdx = builder.create<arith::RemUIOp>(loc, iterIdx, twoIdx);
    phase = builder.create<arith::IndexCastOp>(loc, i32Ty, phaseIdx);
  } else {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  }

  // Only leader CTA waits: pred = (ctaRank % 2 == 0).
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value ctaMod2 = builder.create<arith::RemUIOp>(loc, ctaRank, two);
  Value isLeader = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 ctaMod2, zero);

  // Leader waits for both CTAs' arrivals before issuing the 2-CTA MMA.
  builder.create<ttng::WaitBarrierOp>(loc, remoteBar, phase, isLeader);

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
    // Skip async MMAs (is_async) — those are TLX-managed and already have
    // their own cross-CTA sync via explicit barrier ops in the kernel.
    SmallVector<ttng::TCGen5MMAOp> twoCTAMMAOps;
    moduleOp->walk([&](ttng::TCGen5MMAOp mma) {
      if (mma.getTwoCtas() && !mma.getIsAsync())
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

      // Allocate cross-CTA barrier. createBarrierAlloc places alloc+init
      // before forOp and inval+dealloc after forOp.
      //
      // The barrier init must be in the function's entry block for remote
      // barrier support. In post-pipeline
      // IR, the loop is at the top level of the function, so placing before
      // forOp is in the entry block. Assert this invariant.
      assert(forOp->getParentRegion() ==
                 &forOp->getParentOfType<triton::FuncOp>().getBody() &&
             "Expected scf.for to be in the function entry region. "
             "Nested loops may require barrier init at function entry.");
      Value barrierAlloc = triton::createBarrierAlloc(forOp, /*numBarriers=*/1,
                                                      /*arriveCount=*/2);

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

} // namespace
