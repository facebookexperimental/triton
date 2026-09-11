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

// Find an scf.while after-region argument which advances by one on every
// iteration. AutoWS adds such an accumulation counter to persistent loops.
// The condition can forward only an ordered subset of the before-region
// arguments, so map the after argument back through scf.condition before
// looking up its corresponding scf.yield operand.
static Value findWhileIterationCounter(scf::WhileOp whileOp) {
  auto forwarded = whileOp.getConditionOp().getArgs();
  auto yielded = whileOp.getYieldedValues();
  for (BlockArgument afterArg : whileOp.getAfterArguments()) {
    unsigned afterIdx = afterArg.getArgNumber();
    if (afterIdx >= forwarded.size())
      continue;
    auto beforeArg = dyn_cast<BlockArgument>(forwarded[afterIdx]);
    if (!beforeArg || beforeArg.getOwner() != whileOp.getBeforeBody() ||
        beforeArg.getArgNumber() >= yielded.size())
      continue;

    auto add = yielded[beforeArg.getArgNumber()].getDefiningOp<arith::AddIOp>();
    if (!add)
      continue;
    Value increment;
    if (add.getLhs() == afterArg)
      increment = add.getRhs();
    else if (add.getRhs() == afterArg)
      increment = add.getLhs();
    else
      continue;
    auto one = increment.getDefiningOp<arith::ConstantIntOp>();
    if (one && one.value() == 1)
      return afterArg;
  }
  return {};
}

// Linearized iteration index across the loops that execute within the lifetime
// of the barrier. phaseScope is the operation containing the barrier
// allocation; loops outside it reinitialize the barrier and must not
// contribute. Returns null when no enclosing loop contributes an index.
static Value computeLinearBarrierIter(OpBuilder &builder, Location loc,
                                      Operation *mma, Operation *phaseScope) {
  Value linearIter;
  Value stride = arith::ConstantIntOp::create(builder, loc, 1, 64);
  for (Operation *parent = mma->getParentOp(); parent && parent != phaseScope;
       parent = parent->getParentOp()) {
    Value iter;
    Value tripCount;
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      iter = computeLoopIterIndex(builder, loc, forOp);
      tripCount = computeLoopTripCount(builder, loc, forOp);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      Value counter = findWhileIterationCounter(whileOp);
      if (!counter) {
        LDBG("Could not derive persistent while iteration for MMA at " << loc);
        break;
      }
      iter = castToI64(builder, loc, counter);
    } else {
      continue;
    }

    Value contribution = arith::MulIOp::create(builder, loc, iter, stride);
    linearIter = linearIter ? arith::AddIOp::create(builder, loc, contribution,
                                                    linearIter)
                                  .getResult()
                            : contribution;
    if (!tripCount)
      break;
    stride = arith::MulIOp::create(builder, loc, stride, tripCount);
  }

  return linearIter;
}

// Slot and phase for a `depth`-deep barrier array, matching how the operand
// SMEM buffers rotate: slot = iter % depth, phase = (iter / depth) & 1.
struct SlotAndPhase {
  Value slot;  // i32 in [0, depth); null when depth == 1 (always slot 0)
  Value phase; // i32, 0 or 1
};

static SlotAndPhase computeSlotAndPhase(OpBuilder &builder, Location loc,
                                        Value linearIter, unsigned depth) {
  auto i32Ty = builder.getI32Type();
  if (!linearIter) {
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    return {zero, zero};
  }
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 64);
  // Single-buffered: the only slot is 0, left null so the caller uses the base
  // index directly, and the phase is the raw parity. Emit it exactly as the
  // single-slot barrier always did, so depth-1 kernels keep identical IR.
  if (depth == 1) {
    Value two = arith::ConstantIntOp::create(builder, loc, 2, 64);
    Value rem = arith::RemUIOp::create(builder, loc, linearIter, two);
    return {/*slot=*/Value(),
            arith::TruncIOp::create(builder, loc, i32Ty, rem)};
  }
  Value depthVal = arith::ConstantIntOp::create(builder, loc, depth, 64);
  Value slot = arith::RemUIOp::create(builder, loc, linearIter, depthVal);
  Value gen = arith::DivUIOp::create(builder, loc, linearIter, depthVal);
  Value phase = arith::AndIOp::create(builder, loc, gen, one);
  return {arith::TruncIOp::create(builder, loc, i32Ty, slot),
          arith::TruncIOp::create(builder, loc, i32Ty, phase)};
}

// Walk a memdesc value back to the MemDescIndexOp that selects its buffer,
// looking through the view ops that preserve the underlying allocation. The set
// matches the other view walkers in this backend -- memDescRoot in
// WSCodePartition.cpp and getRootBuffer in CodePartitionUtility.cpp -- and it
// has to: a transposed B operand reaches the MMA through a memdesc_trans, and
// stopping there would report it as unbuffered and shrink the barrier back to
// one slot, which is the deadlock this pass exists to prevent.
static ttg::MemDescIndexOp findBufferIndex(Value v) {
  while (v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return {};
    if (auto idxOp = dyn_cast<ttg::MemDescIndexOp>(def))
      return idxOp;
    if (isa<ttg::MemDescTransOp, ttg::MemDescSubsliceOp, ttg::MemDescReshapeOp,
            ttg::MemDescReinterpretOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    return {};
  }
  return {};
}

// Buffer count of the allocation behind `v`. Returns 0 when it is not a view
// into a buffer array -- i.e. a single, non-rotating buffer.
static unsigned getAllocDepth(Value v) {
  auto idxOp = findBufferIndex(v);
  if (!idxOp)
    return 0;
  auto srcTy = cast<ttg::MemDescType>(idxOp.getSrc().getType());
  return srcTy.getRank() > 0 ? srcTy.getShape()[0] : 0;
}

// The cross-CTA barrier must rotate over as many slots as the MMA's operand
// SMEM buffers: the follower CTA can run ahead by up to that depth (its loads
// are gated on the leader's MMA releasing a slot), and a shallower barrier
// would let it lap the phase bit and deadlock the pair. Mirrors TLX, which
// allocates its 2-CTA barrier with NUM_SMEM_BUFFERS slots.
static unsigned getOperandPipelineDepth(ttng::MMAv5OpInterface mma) {
  // Floor of 1: operands that are not multi-buffered cannot be produced ahead,
  // so the follower cannot drift and a single slot is correct.
  unsigned depth = 1;
  auto consider = [&](Value v) { depth = std::max(depth, getAllocDepth(v)); };
  consider(mma.getA());
  consider(mma.getB());
  // The MMA's completion barriers are allocated at the same channel depth.
  for (Value bar : mma.getCompletionBarriers())
    consider(bar);

  // Cross-check the planner's own annotation when it is still present; they
  // are emitted together, so a mismatch means something rewrote one of them.
  if (auto idxOp = findBufferIndex(mma.getB())) {
    if (auto *allocOp = idxOp.getSrc().getDefiningOp()) {
      if (auto copies = allocOp->getAttrOfType<IntegerAttr>("buffer.copy")) {
        unsigned annotated = copies.getInt();
        if (annotated > depth) {
          // Should not happen (both are emitted from the planner's numCopies),
          // but trust the larger value: too few slots deadlocks the CTA pair.
          allocOp->emitWarning()
              << "buffer.copy=" << annotated << " exceeds allocation depth "
              << depth << "; 2-CTA sync uses buffer.copy";
          depth = annotated;
        }
      }
    }
  }
  return depth;
}

// Insert the "arrive remote, wait local" cross-CTA sync ops before a 2-CTA
// MMA. The barrier must be allocated externally (before the containing loop
// if the MMA is in a loop).
static void insertSyncBeforeMMA(ttng::MMAv5OpInterface mma, Value barrierAlloc,
                                Operation *phaseScope, unsigned baseIdx,
                                unsigned depth) {
  MLIRContext *ctx = mma->getContext();
  Location loc = mma->getLoc();
  OpBuilder builder(mma);
  auto i32Ty = builder.getI32Type();

  // This MMA owns `depth` consecutive slots starting at baseIdx, rotating with
  // its operand buffers so a follower CTA that runs ahead lands on a distinct
  // slot instead of lapping a single barrier's phase bit.
  Value linearIter = computeLinearBarrierIter(builder, loc, mma, phaseScope);
  SlotAndPhase sp = computeSlotAndPhase(builder, loc, linearIter, depth);
  Value idx = arith::ConstantIntOp::create(builder, loc, baseIdx, 32);
  if (depth > 1)
    idx = arith::AddIOp::create(builder, loc, idx, sp.slot);
  Value barrierView =
      triton::createSingleBufferView(builder, barrierAlloc, idx);

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
  ttng::WaitBarrierOp::create(builder, loc, barrierView, sp.phase, isLeader);

  LDBG("Inserted cross-CTA sync before MMA at " << loc);
}

struct Insert2CTASync : public impl::NVGPUInsert2CTASyncBase<Insert2CTASync> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (!ttg::isPhysicalCluster(moduleOp))
      return;

    // Skip TLX kernels — they manage their own cross-CTA sync via
    // explicit barrier ops in the kernel.
    if (moduleOp->hasAttr("tlx.has_tlx_ops"))
      return;

    // Collect 2-CTA MMA ops that need cross-CTA sync insertion.
    SmallVector<ttng::MMAv5OpInterface> twoCTAMMAOps;
    moduleOp->walk([&](ttng::MMAv5OpInterface mma) {
      if (mma.getTwoCtas())
        twoCTAMMAOps.push_back(mma);
    });

    if (twoCTAMMAOps.empty())
      return;

    LDBG("Found " << twoCTAMMAOps.size() << " 2-CTA MMA ops");

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

      // Allocate cross-CTA barrier. In the post-WS path (Meta WS),
      // the loop is nested inside a WarpSpecializeOp. The barrier
      // alloc+init must be placed BEFORE the WarpSpecializeOp (so thread 0
      // from the producer warp group initializes it).
      Value barrierAlloc;
      Operation *phaseScope;
      // Each MMA gets `depth` consecutive slots, where depth is its operand
      // pipeline depth: that is the furthest a follower CTA can drift.
      SmallVector<unsigned> depths;
      unsigned numBarriers = 0;
      for (auto mma : mmas) {
        unsigned d = getOperandPipelineDepth(mma);
        depths.push_back(d);
        numBarriers += d;
      }
      auto wsOp = forOp->getParentOfType<ttg::WarpSpecializeOp>();
      if (!wsOp) {
        // Pre-WS path: standard alloc before the for loop.
        barrierAlloc =
            triton::createBarrierAlloc(forOp, numBarriers, /*arriveCount=*/2);
        phaseScope = forOp->getParentOp();
      } else if (isInDefaultRegion(mmas[0], wsOp)) {
        // Post-WS path, MMA in default region: The default region can
        // implicitly capture values defined before the WarpSpecializeOp,
        // so no explicit capture is needed. Just allocate+init before wsOp
        // and inval+dealloc after.
        Location loc = wsOp->getLoc();
        ImplicitLocOpBuilder rewriter(loc, wsOp);
        barrierAlloc = triton::createScalarAlloc(
            rewriter, rewriter.getI64Type(), numBarriers);
        for (unsigned i = 0; i < numBarriers; ++i) {
          Value initView =
              triton::createSingleBufferView(rewriter, barrierAlloc, i);
          rewriter.create<ttng::InitBarrierOp>(initView, /*arriveCount=*/2);
        }

        // Inval and dealloc AFTER the WarpSpecializeOp.
        rewriter.setInsertionPointAfter(wsOp);
        for (unsigned i = 0; i < numBarriers; ++i) {
          Value invalView =
              triton::createSingleBufferView(rewriter, barrierAlloc, i);
          rewriter.create<ttng::InvalBarrierOp>(invalView);
        }
        rewriter.create<ttg::LocalDeallocOp>(barrierAlloc);
        phaseScope = wsOp->getParentOp();
      } else {
        // Post-WS path, MMA in a partition region (IsolatedFromAbove):
        // Must capture the barrier explicitly into the partition.
        Location loc = wsOp->getLoc();
        ImplicitLocOpBuilder rewriter(loc, wsOp);
        barrierAlloc = triton::createScalarAlloc(
            rewriter, rewriter.getI64Type(), numBarriers);
        for (unsigned i = 0; i < numBarriers; ++i) {
          Value initView =
              triton::createSingleBufferView(rewriter, barrierAlloc, i);
          rewriter.create<ttng::InitBarrierOp>(initView, /*arriveCount=*/2);
        }

        // Inval and dealloc AFTER the WarpSpecializeOp.
        rewriter.setInsertionPointAfter(wsOp);
        for (unsigned i = 0; i < numBarriers; ++i) {
          Value invalView =
              triton::createSingleBufferView(rewriter, barrierAlloc, i);
          rewriter.create<ttng::InvalBarrierOp>(invalView);
        }
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
        phaseScope = wsOp->getParentOp();
      }

      unsigned baseIdx = 0;
      for (auto [mma, depth] : llvm::zip(mmas, depths)) {
        LDBG("  MMA slots [" << baseIdx << ", " << baseIdx + depth << ")");
        insertSyncBeforeMMA(mma, barrierAlloc, phaseScope, baseIdx, depth);
        baseIdx += depth;
      }
    }

    // Process standalone MMAs (rare: single-iteration epilogue).
    for (auto mma : nonLoopMMAs) {
      Value barrierAlloc;
      Operation *phaseScope;
      auto wsOp = mma->getParentOfType<ttg::WarpSpecializeOp>();
      if (wsOp) {
        // A software-pipelined loop can leave prologue/epilogue MMA copies
        // directly in a partition region.  They still need a barrier created
        // at kernel-entry scope; a partition-local init can race a peer CTA's
        // first remote arrival. Find the top-level operation containing the
        // WarpSpecializeOp so a surrounding loop, when present, is included in
        // the barrier lifetime.
        Location loc = wsOp->getLoc();
        Operation *lifetimeAnchor = wsOp;
        auto funcOp = wsOp->getParentOfType<triton::FuncOp>();
        while (lifetimeAnchor->getParentOp() != funcOp)
          lifetimeAnchor = lifetimeAnchor->getParentOp();
        barrierAlloc = triton::createBarrierAlloc(
            lifetimeAnchor, /*numBarriers=*/1, /*arriveCount=*/2);
        phaseScope = lifetimeAnchor->getParentOp();

        if (!isInDefaultRegion(mma, wsOp)) {
          auto partOp = wsOp.getPartitionOp();
          partOp->insertOperands(partOp->getNumOperands(), barrierAlloc);
          Value capturedBarrier;
          for (Region *region : wsOp.getPartitionRegions()) {
            BlockArgument arg =
                region->addArgument(barrierAlloc.getType(), loc);
            if (region->isAncestor(mma->getParentRegion()))
              capturedBarrier = arg;
          }
          assert(capturedBarrier && "MMA not found in any partition region");
          barrierAlloc = capturedBarrier;
        }
      } else {
        barrierAlloc = triton::createBarrierAlloc(mma, /*numBarriers=*/1,
                                                  /*arriveCount=*/2);
        phaseScope = mma->getParentOp();
      }
      insertSyncBeforeMMA(mma, barrierAlloc, phaseScope, /*baseIdx=*/0,
                          /*depth=*/1);
    }
  }
};

} // anonymous namespace
