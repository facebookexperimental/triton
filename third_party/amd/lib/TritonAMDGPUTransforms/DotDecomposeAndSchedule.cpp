//===- DotDecomposeAndSchedule.cpp ----------------------------------------===//
//
// TTGIR-level replacement for the LLVM-IR `LLIRSchedule` pass on AMD MFMA
// matmul kernels (v8 / v10 family). See:
//   ~/AMD/triton/claude/llir_sched_at_ttgir_design.md
//   ~/AMD/triton/claude/llir_sched_at_ttgir_plan.md
//
// Status
// ------
//  - Phase 0 (af881d529 / cbfbda28f): no-op scaffold; emits remark per
//    candidate inner loop.
//  - Phase 1a (this commit): compute the M-split partition plan per dot
//    (read AMDMfmaEncodingAttr::getInstrShape, derive numPartitions from
//    BLOCK_M / instrM, run feasibility checks), emit a richer remark
//    showing the plan. **Still does not mutate IR.**
//  - Phase 1b (next): backward walker to collect operand-producing ops to
//    co-partition.
//  - Phase 1c: forward walker for user ops.
//  - Phase 1d: sliceOp to actually clone + retype.
//  - Phases 2-6: N-split, schedule recipe + sched.barrier lowering,
//    coverage matrix, default-disable LLIR, docs.
//
// Opt-in behind `TRITON_ENABLE_TTGIR_SCHED=1`.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "tritonamdgpu-dot-decompose-and-schedule"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDOTDECOMPOSEANDSCHEDULE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

/// Per-dot partition plan derived from the dot's result encoding and tensor
/// shape. Phase 1a only: just the M dimension. N and K splits are added in
/// Phases 2 and (deferred) 7 respectively. See
/// `~/AMD/triton/claude/llir_sched_at_ttgir_design.md` for the rationale on
/// granularity.
struct DotPartitionPlan {
  /// The dot we plan to split.
  triton::DotOp dotOp;
  /// Total tensor extent along M (BLOCK_M); e.g. 256 for v8/v10.
  unsigned blockM;
  /// MFMA instruction extent along M; e.g. 16 for instr_shape=[16,16,32].
  unsigned instrM;
  /// numPartitions = blockM / instrM; e.g. 16 for v8/v10.
  unsigned numPartitions;
};

/// Mfma-typed `tt.dot` predicate. We require the result tensor's encoding to
/// be `AMDMfmaEncodingAttr` — that's the only `tt.dot` shape the LLIR
/// scheduler matches today, and the only one that lowers 1-to-1 to MFMA
/// intrinsics in a known tile pattern.
static triton::gpu::AMDMfmaEncodingAttr getMfmaEncoding(triton::DotOp dotOp) {
  auto resultType = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
  if (!resultType)
    return nullptr;
  return dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(resultType.getEncoding());
}

/// Build a partition plan for `dotOp` along dim M. Returns std::nullopt if the
/// dot doesn't match the expected shape constraints (so the caller can skip
/// without aborting the whole loop).
static std::optional<DotPartitionPlan> planMSplit(triton::DotOp dotOp) {
  auto mfmaEnc = getMfmaEncoding(dotOp);
  if (!mfmaEnc)
    return std::nullopt;

  auto resultType = cast<RankedTensorType>(dotOp.getResult().getType());
  auto resultShape = resultType.getShape();
  if (resultShape.size() != 2)
    return std::nullopt; // 3-D dots not supported in Phase 1.

  // instrShape is [M, N, K]; we only need M for this split direction.
  auto instrShape = mfmaEnc.getInstrShape();
  if (instrShape.size() < 2)
    return std::nullopt;

  unsigned blockM = static_cast<unsigned>(resultShape[0]);
  unsigned instrM = instrShape[0];

  // Feasibility: need clean divisibility (no remainder tiles), and need at
  // least 2 partitions for splitting to be worthwhile.
  if (instrM == 0 || blockM % instrM != 0)
    return std::nullopt;
  unsigned numPartitions = blockM / instrM;
  if (numPartitions < 2)
    return std::nullopt;

  DotPartitionPlan plan{dotOp, blockM, instrM, numPartitions};
  return plan;
}

/// Collect partition plans for every MFMA-typed `tt.dot` inside this for-op's
/// body. Mismatched / unsupported dots are silently skipped (the caller still
/// counts them, for diagnostics).
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
    // Belt-and-suspenders: pass body re-checks the env var so a lit
    // invocation is opt-in too. compiler.py also gates the pass-add call.
    if (!triton::tools::getBoolEnv("TRITON_ENABLE_TTGIR_SCHED"))
      return;

    ModuleOp module = getOperation();
    unsigned numForOps = 0;
    unsigned numCandidateLoops = 0;
    unsigned totalMfmaDots = 0;
    unsigned totalSkipped = 0;
    unsigned totalPlans = 0;

    module.walk([&](scf::ForOp forOp) {
      ++numForOps;

      SmallVector<DotPartitionPlan> plans;
      unsigned mfmaDotsSeen = 0;
      unsigned mfmaDotsSkipped = 0;
      collectPlans(forOp, plans, mfmaDotsSeen, mfmaDotsSkipped);
      if (mfmaDotsSeen == 0)
        return; // not a candidate

      ++numCandidateLoops;
      totalMfmaDots += mfmaDotsSeen;
      totalSkipped += mfmaDotsSkipped;
      totalPlans += plans.size();

      forOp.emitRemark()
          << "ttgir-sched: candidate inner loop with " << mfmaDotsSeen
          << " MFMA tt.dot op(s); plans " << plans.size()
          << ", skipped " << mfmaDotsSkipped << " (phase 1a: plan only)";

      // Emit a per-plan remark so the bench / lit test can see exactly what
      // would be split. This is the foundation for Phase 1b-d.
      for (const auto &plan : plans) {
        triton::DotOp dot = plan.dotOp;
        dot.emitRemark()
            << "ttgir-sched: would M-split this dot into " << plan.numPartitions
            << " (blockM=" << plan.blockM << " / instrM=" << plan.instrM
            << ") (phase 1a: plan only)";
      }
    });

    module.emitRemark()
        << "ttgir-sched: visited " << numForOps << " scf.for op(s), "
        << numCandidateLoops << " candidate(s), " << totalMfmaDots
        << " MFMA tt.dot op(s), " << totalPlans << " planned M-split(s), "
        << totalSkipped << " skipped (phase 1a: plan only)";
  }
};

} // namespace

} // namespace mlir
