//===- DotDecomposeAndSchedule.cpp ----------------------------------------===//
//
// TTGIR-level replacement for the LLVM-IR `LLIRSchedule` pass on AMD MFMA
// matmul kernels (v8 / v10 family). See:
//   ~/MetaMain2/triton/claude/llir_sched_at_ttgir_design.md
//   ~/MetaMain2/triton/claude/llir_sched_at_ttgir_plan.md
//
// Status
// ------
//  - Phase 0 (cbfbda28f): no-op scaffold; emits remark per candidate inner
//    loop.
//  - Phase 1a (280370f92): compute the M-split partition plan per dot.
//  - Phase 1b (this commit): backward walker — for each plan, walk back
//    from the dot's M-dim operand (A) and accumulator (C), collect the
//    producer ops that need co-partitioning with the same dim. Skips the
//    B operand entirely (N-dim, doesn't get sliced under M-split). Adapted
//    from `WSDataPartition::getBackwardSliceToPartition`
//    (third_party/nvidia/hopper/.../WSDataPartition.cpp:291), stripped of
//    Hopper-only ops (WarpGroupDotOp / TCGen5MMAOp / TMEMAllocOp / async-
//    task-ID tracking). **Still does not mutate IR.**
//  - Phase 1c (next): forward walker for user ops.
//  - Phase 1d: sliceOp — clone + retype.
//  - Phases 2-6: N-split, schedule recipe + sched.barrier lowering, etc.
//
// Opt-in behind `TRITON_ENABLE_TTGIR_SCHED=1`.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "tritonamdgpu-dot-decompose-and-schedule"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDOTDECOMPOSEANDSCHEDULE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Partition plan (Phase 1a)
//===----------------------------------------------------------------------===//

/// Per-dot partition plan derived from the dot's result encoding and tensor
/// shape. Phase 1a only: just the M dimension. N is added in Phase 2; K is
/// deferred to a follow-up (would also require decomposing local_load to
/// per-vector grain — see design doc).
struct DotPartitionPlan {
  triton::DotOp dotOp;
  unsigned blockM;          // BLOCK_M; e.g. 256 for v8/v10.
  unsigned instrM;          // MFMA instr M; e.g. 16 for [16,16,32].
  unsigned numPartitions;   // blockM / instrM.
};

static triton::gpu::AMDMfmaEncodingAttr getMfmaEncoding(triton::DotOp dotOp) {
  auto resultType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
  if (!resultType)
    return nullptr;
  return dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(resultType.getEncoding());
}

static std::optional<DotPartitionPlan> planMSplit(triton::DotOp dotOp) {
  auto mfmaEnc = getMfmaEncoding(dotOp);
  if (!mfmaEnc)
    return std::nullopt;
  auto resultType = cast<RankedTensorType>(dotOp.getResult().getType());
  auto resultShape = resultType.getShape();
  if (resultShape.size() != 2)
    return std::nullopt;
  auto instrShape = mfmaEnc.getInstrShape();
  if (instrShape.size() < 2)
    return std::nullopt;
  unsigned blockM = static_cast<unsigned>(resultShape[0]);
  unsigned instrM = instrShape[0];
  if (instrM == 0 || blockM % instrM != 0)
    return std::nullopt;
  unsigned numPartitions = blockM / instrM;
  if (numPartitions < 2)
    return std::nullopt;
  return DotPartitionPlan{dotOp, blockM, instrM, numPartitions};
}

//===----------------------------------------------------------------------===//
// Backward walker (Phase 1b)
//===----------------------------------------------------------------------===//

/// Per-op partition decision: which dim of which Value gets sliced when this
/// op is co-partitioned with the dot. Phase 1b only records what we *would*
/// partition; Phase 1d will use this to drive the actual slice/clone.
///
/// Stripped subset of `WSDataPartition::DataPartitionScheme` (line 93):
/// no rematerialization, no opsToSkip, no async-task-ID tracking, no
/// funcArgPartitionDims, no operand-side index tracking (M-split only
/// touches dot's A operand, never B — the caller passes the right side in).
struct DataPartitionScheme {
  unsigned numPartitions = 0;
  /// Ops that would be cloned per-partition.
  llvm::SetVector<Operation *> ops;
  /// Which dim of the op's result tensor to partition (0 for M-split).
  llvm::DenseMap<Operation *, unsigned> opPartitionDims;
};

/// True if a tensor value's `dim` is large enough to actually split into
/// `numPartitions` pieces. A 1-D scalar broadcast through `numPartitions=1`
/// wouldn't need partitioning. Mirrors the predicate in
/// `WSDataPartition::needToSlice` (line 220).
static bool needToSlice(Value v, unsigned dim, unsigned numPartitions) {
  auto rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt)
    return false;
  if (dim >= rtt.getRank())
    return false;
  int64_t extent = rtt.getShape()[dim];
  return extent >= static_cast<int64_t>(numPartitions);
}

/// True if `op` is in the same region as `boundary` — i.e. inside the same
/// scf.for body we're partitioning. Stops the backward walk from escaping
/// into the prologue or surrounding function.
static bool isInsideRegion(Operation *op, Region *boundary) {
  return op->getParentRegion() == boundary;
}

/// Adapted from `WSDataPartition::getBackwardSliceToPartition` (line 291).
/// Recursively walks `v`'s defining-op chain. Each visited op is recorded in
/// `scheme.ops` with `opPartitionDims = currentDim`. Returns false on
/// incompatible / unsupported op (the caller should treat this as "skip the
/// plan").
///
/// What's kept from WSDataPartition's allowed-op list:
///   * `Elementwise` trait (matches arith / math / triton elementwise ops)
///   * `arith::ConstantOp`, `Ext*Op`, `triton::SplatOp`, `triton::AddPtrOp`
///   * `ttg::ConvertLayoutOp`, `ttg::LocalAllocOp`, `ttg::LocalLoadOp`
///   * `triton::LoadOp` (global load, leaf of the chain)
///   * `triton::amdgpu::BufferLoadToLocalOp` (AMD-specific shmem DMA — leaf)
///
/// What's stripped (Hopper-only or not seen in v8/v10 inner loop):
///   * `WarpGroupDotOp`, `TCGen5MMAOp`, `TMEMAllocOp/LoadOp/StoreOp`
///   * `DescriptorLoadOp` (TMA — Phase 4+ if needed)
///   * `BroadcastOp`, `ExpandDimsOp`, `TransOp`, `ReshapeOp` — these need
///     dim-flipping logic; deferred to Phase 1c+ if the chain demands them.
///     For v8/v10 the M-chain from dot.A back to local_load → buffer_load is
///     simple (no transpose / reshape / broadcast), so this works as-is.
///
/// Returns false (= unsupported) if the chain hits anything not on the list
/// while still inside the region. Stops cleanly at the region boundary
/// (function args / iter-args / values from before the for-loop).
static bool getBackwardSliceToPartition(Value v, unsigned currentDim,
                                        DataPartitionScheme &scheme,
                                        Region *boundary) {
  // Don't recurse into a value whose tensor extent is too small to slice.
  if (!needToSlice(v, currentDim, scheme.numPartitions))
    return true;

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return true; // block argument — leaf (iter-arg / func arg).

  // Don't escape the loop body boundary.
  if (!isInsideRegion(defOp, boundary))
    return true; // value was defined outside; leaf.

  // Already visited (compatible dim, since we only have one dim today).
  if (scheme.ops.contains(defOp)) {
    auto it = scheme.opPartitionDims.find(defOp);
    if (it != scheme.opPartitionDims.end() && it->second != currentDim) {
      // Cross-dim conflict — would need rematerialize (Phase 2+). Bail.
      return false;
    }
    return true;
  }

  // Op-type guard list (stripped of Hopper-only branches).
  bool isAllowed =
      defOp->hasTrait<OpTrait::Elementwise>() ||
      isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
          triton::SplatOp, triton::AddPtrOp, triton::LoadOp,
          triton::gpu::ConvertLayoutOp, triton::gpu::LocalAllocOp,
          triton::gpu::LocalLoadOp,
          triton::amdgpu::BufferLoadToLocalOp>(defOp);
  if (!isAllowed) {
    LLVM_DEBUG(llvm::dbgs() << "ttgir-sched: backward walker bailing on "
                            << defOp->getName() << "\n");
    return false;
  }

  scheme.ops.insert(defOp);
  scheme.opPartitionDims[defOp] = currentDim;

  // Recurse into each operand of `defOp`. For Phase 1b we propagate the same
  // `currentDim` to every operand. Dim-flipping through TransOp /
  // ExpandDimsOp will be added in Phase 1c if any v8/v10 chain needs it.
  for (Value operand : defOp->getOperands()) {
    if (!getBackwardSliceToPartition(operand, currentDim, scheme, boundary))
      return false;
  }
  return true;
}

/// Run the backward walker for one plan. Returns the populated scheme on
/// success, std::nullopt on infeasibility (e.g. unsupported op in chain).
///
/// For M-split, the walker is launched on:
///   * dot's accumulator C (output dim 0)
///   * dot's operand A    (LHS dim 0 — same M as the result's M)
/// We deliberately do NOT walk operand B (RHS dim 0 is K, not M).
static std::optional<DataPartitionScheme>
backwardSliceForMSplit(const DotPartitionPlan &plan) {
  DataPartitionScheme scheme;
  scheme.numPartitions = plan.numPartitions;

  triton::DotOp dot = plan.dotOp;
  Region *boundary = dot->getParentRegion();

  // Walk operand A (LHS, M dim is dim 0 of the operand tensor too).
  if (!getBackwardSliceToPartition(dot.getA(), /*dim=*/0, scheme, boundary))
    return std::nullopt;

  // Walk accumulator C (output, M dim is dim 0).
  if (!getBackwardSliceToPartition(dot.getC(), /*dim=*/0, scheme, boundary))
    return std::nullopt;

  return scheme;
}

//===----------------------------------------------------------------------===//
// Loop-level driver
//===----------------------------------------------------------------------===//

static void collectPlans(scf::ForOp forOp,
                         SmallVectorImpl<DotPartitionPlan> &plans,
                         unsigned &numMfmaDotsSeen,
                         unsigned &numMfmaDotsSkipped) {
  forOp.getBody()->walk([&](triton::DotOp dotOp) {
    if (!getMfmaEncoding(dotOp))
      return;
    ++numMfmaDotsSeen;
    if (auto plan = planMSplit(dotOp))
      plans.push_back(*plan);
    else
      ++numMfmaDotsSkipped;
  });
}

struct TritonAMDGPUDotDecomposeAndSchedulePass
    : impl::TritonAMDGPUDotDecomposeAndScheduleBase<
          TritonAMDGPUDotDecomposeAndSchedulePass> {

  void runOnOperation() override {
    if (!triton::tools::getBoolEnv("TRITON_ENABLE_TTGIR_SCHED"))
      return;

    ModuleOp module = getOperation();
    unsigned numForOps = 0;
    unsigned numCandidateLoops = 0;
    unsigned totalMfmaDots = 0;
    unsigned totalSkipped = 0;
    unsigned totalPlans = 0;
    unsigned totalBwdInfeasible = 0;

    module.walk([&](scf::ForOp forOp) {
      ++numForOps;

      SmallVector<DotPartitionPlan> plans;
      unsigned mfmaDotsSeen = 0;
      unsigned mfmaDotsSkipped = 0;
      collectPlans(forOp, plans, mfmaDotsSeen, mfmaDotsSkipped);
      if (mfmaDotsSeen == 0)
        return;

      ++numCandidateLoops;
      totalMfmaDots += mfmaDotsSeen;
      totalSkipped += mfmaDotsSkipped;
      totalPlans += plans.size();

      // Run the backward walker for each plan.
      unsigned loopBwdInfeasible = 0;
      unsigned loopTotalProducerOps = 0;
      for (const auto &plan : plans) {
        auto schemeOpt = backwardSliceForMSplit(plan);
        triton::DotOp dot = plan.dotOp;
        if (!schemeOpt) {
          ++loopBwdInfeasible;
          ++totalBwdInfeasible;
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / instrM=" << plan.instrM
              << "), but backward walker bailed (phase 1b: plan only)";
          continue;
        }
        loopTotalProducerOps += schemeOpt->ops.size();
        dot.emitRemark()
            << "ttgir-sched: would M-split this dot into "
            << plan.numPartitions << " (blockM=" << plan.blockM
            << " / instrM=" << plan.instrM << "), co-partitioning "
            << schemeOpt->ops.size()
            << " producer op(s) (phase 1b: plan only)";
      }

      forOp.emitRemark()
          << "ttgir-sched: candidate inner loop with " << mfmaDotsSeen
          << " MFMA tt.dot op(s); plans " << plans.size()
          << ", skipped " << mfmaDotsSkipped << ", bwd-infeasible "
          << loopBwdInfeasible << ", co-partition producer-ops total "
          << loopTotalProducerOps << " (phase 1b: plan only)";
    });

    module.emitRemark()
        << "ttgir-sched: visited " << numForOps << " scf.for op(s), "
        << numCandidateLoops << " candidate(s), " << totalMfmaDots
        << " MFMA tt.dot op(s), " << totalPlans << " planned M-split(s), "
        << totalSkipped << " skipped, " << totalBwdInfeasible
        << " bwd-infeasible (phase 1b: plan only)";
  }
};

} // namespace

} // namespace mlir
