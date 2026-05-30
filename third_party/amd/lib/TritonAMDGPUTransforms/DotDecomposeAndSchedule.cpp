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
//  - Phase 1b (a2c8db5b0): backward walker — collect producer ops to
//    co-partition along the M dim. Walks from dot.getA() and dot.getC()
//    with dim=0; deliberately skips dot.getB() (N-dim).
//  - Phase 1c (this commit): forward walker — collect *user* ops on the
//    dot's result chain. For v8/v10's in-loop chain this just picks up
//    `scf.yield`; richer test cases also exercise `arith.truncf` /
//    `ttg.convert_layout` / `amdgpu.buffer_store`. Adapted from
//    `WSDataPartition::getForwardSliceToPartition`
//    (third_party/nvidia/hopper/.../WSDataPartition.cpp:441), stripped
//    of Hopper-only branches. **Still does not mutate IR.**
//  - Phase 1d (this commit): actual SSA mutation. Gated behind a separate
//    env var `TRITON_TTGIR_SCHED_APPLY=1` (default off) so the existing
//    `TRITON_ENABLE_TTGIR_SCHED=1` keeps planning-only behavior. When
//    applied, each plan's dot is replaced by N small dots sliced along M,
//    glued back via `amdgpu.concat`. The producer chain is NOT modified
//    in this sub-step (extract_slice is a no-op at the CTA-tile level, so
//    the upstream layout is preserved). This is the minimal mutation that
//    exposes N dots at TTGIR for later scheduling. Plan-level `numPartitions`
//    is now CTA-tile-aware: `blockM / (instrM * warpsPerCTA[0])` instead
//    of `blockM / instrM`, to satisfy the extract_slice verifier's
//    "source/destination must have matching CTA-tile layouts" constraint.
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

struct DotPartitionPlan {
  triton::DotOp dotOp;
  unsigned blockM;          // BLOCK_M; e.g. 256 for v8/v10.
  unsigned instrM;          // MFMA instr M; e.g. 16 for [16,16,32].
  unsigned ctaTileM;        // instrM * warpsPerCTA[0]; e.g. 32 for v8/v10.
  unsigned numPartitions;   // blockM / ctaTileM; e.g. 8 for v8/v10.
  unsigned tileM;           // ctaTileM (i.e. M dim of each sub-dot result).
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
  // ctaTileM = instrM * warpsPerCTA[0]; required so that `amdgpu.extract_slice`
  // sees source/destination tensors with matching CTA-tile layouts (the op
  // verifier rejects sub-CTA-tile slices). For v8/v10 with warpsPerCTA=[2,2]
  // and instr=16, ctaTileM = 32, so numPartitions = BLOCK_M(256) / 32 = 8.
  auto warpsPerCTA = mfmaEnc.getWarpsPerCTA();
  if (warpsPerCTA.size() < 2)
    return std::nullopt;
  unsigned blockM = static_cast<unsigned>(resultShape[0]);
  unsigned instrM = instrShape[0];
  unsigned warpsM = warpsPerCTA[0];
  if (instrM == 0 || warpsM == 0)
    return std::nullopt;
  unsigned ctaTileM = instrM * warpsM;
  if (blockM % ctaTileM != 0)
    return std::nullopt;
  unsigned numPartitions = blockM / ctaTileM;
  if (numPartitions < 2)
    return std::nullopt;
  return DotPartitionPlan{dotOp, blockM, instrM, ctaTileM, numPartitions,
                          ctaTileM};
}

//===----------------------------------------------------------------------===//
// Partition scheme + backward walker (Phase 1b)
//===----------------------------------------------------------------------===//

struct DataPartitionScheme {
  unsigned numPartitions = 0;
  llvm::SetVector<Operation *> ops;
  llvm::DenseMap<Operation *, unsigned> opPartitionDims;
};

static bool needToSlice(Value v, unsigned dim, unsigned numPartitions) {
  auto rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt)
    return false;
  if (dim >= rtt.getRank())
    return false;
  int64_t extent = rtt.getShape()[dim];
  return extent >= static_cast<int64_t>(numPartitions);
}

static bool isInsideRegion(Operation *op, Region *boundary) {
  return op->getParentRegion() == boundary;
}

/// Forward-declared so backward and forward walkers can share the same
/// scheme + op-classification helpers.
static bool isAllowedProducerOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
             triton::SplatOp, triton::AddPtrOp, triton::LoadOp,
             triton::gpu::ConvertLayoutOp, triton::gpu::LocalAllocOp,
             triton::gpu::LocalLoadOp,
             triton::amdgpu::BufferLoadToLocalOp>(op);
}

/// Adapted from `WSDataPartition::getBackwardSliceToPartition` (line 291).
static bool getBackwardSliceToPartition(Value v, unsigned currentDim,
                                        DataPartitionScheme &scheme,
                                        Region *boundary) {
  if (!needToSlice(v, currentDim, scheme.numPartitions))
    return true;

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return true;

  if (!isInsideRegion(defOp, boundary))
    return true;

  if (scheme.ops.contains(defOp)) {
    auto it = scheme.opPartitionDims.find(defOp);
    if (it != scheme.opPartitionDims.end() && it->second != currentDim)
      return false;
    return true;
  }

  if (!isAllowedProducerOp(defOp)) {
    LLVM_DEBUG(llvm::dbgs() << "ttgir-sched: backward walker bailing on "
                            << defOp->getName() << "\n");
    return false;
  }

  scheme.ops.insert(defOp);
  scheme.opPartitionDims[defOp] = currentDim;

  for (Value operand : defOp->getOperands()) {
    if (!getBackwardSliceToPartition(operand, currentDim, scheme, boundary))
      return false;
  }
  return true;
}

static std::optional<DataPartitionScheme>
backwardSliceForMSplit(const DotPartitionPlan &plan) {
  DataPartitionScheme scheme;
  scheme.numPartitions = plan.numPartitions;

  triton::DotOp dot = plan.dotOp;
  Region *boundary = dot->getParentRegion();

  if (!getBackwardSliceToPartition(dot.getA(), /*dim=*/0, scheme, boundary))
    return std::nullopt;
  if (!getBackwardSliceToPartition(dot.getC(), /*dim=*/0, scheme, boundary))
    return std::nullopt;
  return scheme;
}

//===----------------------------------------------------------------------===//
// Forward walker (Phase 1c)
//===----------------------------------------------------------------------===//

/// Op allow-list for forward walker. Producer-side ops are still allowed
/// (some user-side chains pass through `ConvertLayoutOp` etc.), plus the
/// extra user-side ops that don't appear as producers: truncating casts,
/// stores. Mirrors WSDataPartition's forward-side allow-set sans Hopper
/// branches.
static bool isAllowedUserOp(Operation *op) {
  if (isAllowedProducerOp(op))
    return true;
  return isa<arith::TruncFOp, arith::TruncIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
             triton::StoreOp, triton::gpu::LocalStoreOp,
             triton::amdgpu::BufferStoreOp,
             scf::YieldOp>(op);
}

/// Adapted from `WSDataPartition::getForwardSliceToPartition` (line 441),
/// stripped of:
///   * AtomicRMWOp / DescriptorReduceOp `onlyUsedByAtomicStore` special case
///   * AsyncTaskId tracking
///   * BroadcastOp / ExpandDimsOp / TransOp dim-flipping (deferred to
///     Phase 1c-ext if a v8/v10 user chain needs it)
///   * MultiResult `scf.if` follow-through (not used by v8/v10 main loop)
///
/// Walks `v.getUsers()` recursively. Stops at:
///   * `scf::YieldOp` — recorded in scheme, but no further recursion (the
///     iter-arg back-edge is already covered by the backward walk on
///     `dot.getC()` in the next iteration).
///   * ops outside `boundary`
///   * `tt.return` / `func.return` terminators (no results to follow)
///   * unsupported op → bail with false.
///
/// `seen` guards against cyclic SSA (multi-use values) — same trick as the
/// WSDataPartition walker.
static bool getForwardSliceToPartition(Value v, unsigned currentDim,
                                       DataPartitionScheme &scheme,
                                       Region *boundary,
                                       llvm::DenseSet<Value> &seen) {
  if (!seen.insert(v).second)
    return true;
  if (!needToSlice(v, currentDim, scheme.numPartitions))
    return true;

  for (Operation *userOp : v.getUsers()) {
    // Cross-region: don't walk into ops outside the current loop body. This
    // catches the dot's result being captured by an op in a nested scf.if /
    // scf.for region — handled in Phase 1c-ext if needed.
    if (!isInsideRegion(userOp, boundary)) {
      // It's safe to ignore; the value escapes the loop body, which means
      // the cross-iteration users will be handled by the backward walk on
      // the iter-arg / result.
      continue;
    }

    // Already classified? Check compatibility.
    if (scheme.ops.contains(userOp)) {
      auto it = scheme.opPartitionDims.find(userOp);
      if (it != scheme.opPartitionDims.end() && it->second != currentDim) {
        return false;
      }
      // scf.yield is recorded but we don't recurse through it (see above).
      if (isa<scf::YieldOp>(userOp))
        continue;
      // Compatible existing classification — recurse into its results.
      for (Value res : userOp->getResults()) {
        if (!getForwardSliceToPartition(res, currentDim, scheme, boundary,
                                        seen))
          return false;
      }
      continue;
    }

    if (!isAllowedUserOp(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "ttgir-sched: forward walker bailing on "
                              << userOp->getName() << "\n");
      return false;
    }

    scheme.ops.insert(userOp);
    scheme.opPartitionDims[userOp] = currentDim;

    // scf.yield: stop here (the iter-arg cycle is closed by the backward
    // walker hitting dot.getC() next iteration).
    if (isa<scf::YieldOp>(userOp))
      continue;

    // Recurse into each result of the user op.
    for (Value res : userOp->getResults()) {
      if (!getForwardSliceToPartition(res, currentDim, scheme, boundary, seen))
        return false;
    }
  }
  return true;
}

/// Driver: extend an existing scheme by walking forward from the dot's
/// result. Returns false on infeasibility (unsupported op or dim conflict).
static bool extendSliceForwardFromDot(const DotPartitionPlan &plan,
                                      DataPartitionScheme &scheme) {
  triton::DotOp dot = plan.dotOp;
  Region *boundary = dot->getParentRegion();
  llvm::DenseSet<Value> seen;
  return getForwardSliceToPartition(dot.getResult(), /*dim=*/0, scheme,
                                    boundary, seen);
}

//===----------------------------------------------------------------------===//
// Apply M-split (Phase 1d)
//===----------------------------------------------------------------------===//

/// Compute the sliced tensor type for `original` along `dim`, taking
/// `tileExtent` elements out of the original `dim`. Preserves the encoding.
static RankedTensorType sliceTensorType(RankedTensorType original, unsigned dim,
                                        unsigned tileExtent) {
  SmallVector<int64_t> shape(original.getShape().begin(),
                             original.getShape().end());
  shape[dim] = tileExtent;
  return RankedTensorType::get(shape, original.getElementType(),
                               original.getEncoding());
}

/// Build `amdgpu.extract_slice src[offsetsAlongDim, 0]` returning a tensor
/// of the same rank with `tileExtent` along `dim`.
static Value buildExtractSliceAlongDim(OpBuilder &builder, Location loc,
                                       Value src, unsigned dim, int64_t offset,
                                       unsigned tileExtent) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto sliceType = sliceTensorType(srcType, dim, tileExtent);
  SmallVector<int64_t> offsets(srcType.getRank(), 0);
  offsets[dim] = offset;
  return triton::amdgpu::ExtractSliceOp::create(
      builder, loc, sliceType, src, builder.getDenseI64ArrayAttr(offsets));
}

/// Apply the M-split rewrite for one plan. Replaces
///   %D = tt.dot %A, %B, %C
/// with
///   %A_0  = amdgpu.extract_slice %A [0, 0]
///   %C_0  = amdgpu.extract_slice %C [0, 0]
///   %D_0  = tt.dot %A_0, %B, %C_0
///   %A_1  = amdgpu.extract_slice %A [ctaTileM, 0]
///   ...
///   %D = amdgpu.concat %D_0, %D_1, ..., %D_{N-1}
///
/// `%B` is shared across all partitions (M-split doesn't touch the B/N dim).
/// Producer ops are NOT modified here — `extract_slice` is a no-op at the
/// CTA-tile level so the upstream tile layout is preserved. Phase 2 will
/// also slice along N (composing with this).
static LogicalResult applyMSplit(const DotPartitionPlan &plan) {
  triton::DotOp dot = plan.dotOp;
  OpBuilder builder(dot);
  Location loc = dot.getLoc();
  unsigned n = plan.numPartitions;
  unsigned tile = plan.tileM;

  SmallVector<Value> partialResults;
  partialResults.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    int64_t off = static_cast<int64_t>(i) * tile;
    Value aSlice = buildExtractSliceAlongDim(builder, loc, dot.getA(),
                                             /*dim=*/0, off, tile);
    Value cSlice = buildExtractSliceAlongDim(builder, loc, dot.getC(),
                                             /*dim=*/0, off, tile);
    // Result type of the small dot mirrors the sliced C.
    auto smallResultType = cast<RankedTensorType>(cSlice.getType());
    auto smallDot = triton::DotOp::create(
        builder, loc, smallResultType, aSlice, dot.getB(), cSlice,
        dot.getInputPrecisionAttr(), dot.getMaxNumImpreciseAccAttr());
    partialResults.push_back(smallDot.getResult());
  }

  // Concat all sub-dot results back into the original tensor shape.
  auto fullType = cast<RankedTensorType>(dot.getResult().getType());
  auto concat = triton::amdgpu::ConcatOp::create(builder, loc, fullType,
                                                 partialResults);
  dot.getResult().replaceAllUsesWith(concat.getResult());
  dot.erase();
  return success();
}

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

    // Phase 1d gate: when set, the pass mutates the IR (replaces each
    // candidate dot with N sliced sub-dots + concat). Default off so the
    // existing TRITON_ENABLE_TTGIR_SCHED=1 keeps planning-only behavior.
    bool apply = triton::tools::getBoolEnv("TRITON_TTGIR_SCHED_APPLY");
    const char *modeSuffix = apply ? "phase 1d: applied" : "phase 1d: plan only";

    ModuleOp module = getOperation();
    unsigned numForOps = 0;
    unsigned numCandidateLoops = 0;
    unsigned totalMfmaDots = 0;
    unsigned totalSkipped = 0;
    unsigned totalPlans = 0;
    unsigned totalBwdInfeasible = 0;
    unsigned totalFwdInfeasible = 0;
    unsigned totalApplied = 0;

    // Collect plans first so we don't iterate while we mutate.
    struct LoopWork {
      scf::ForOp forOp;
      SmallVector<DotPartitionPlan> plans;
      unsigned mfmaDotsSeen = 0;
      unsigned mfmaDotsSkipped = 0;
    };
    SmallVector<LoopWork> work;

    module.walk([&](scf::ForOp forOp) {
      ++numForOps;
      LoopWork lw;
      lw.forOp = forOp;
      collectPlans(forOp, lw.plans, lw.mfmaDotsSeen, lw.mfmaDotsSkipped);
      if (lw.mfmaDotsSeen == 0)
        return;
      ++numCandidateLoops;
      totalMfmaDots += lw.mfmaDotsSeen;
      totalSkipped += lw.mfmaDotsSkipped;
      totalPlans += lw.plans.size();
      work.push_back(std::move(lw));
    });

    for (auto &lw : work) {
      unsigned loopBwdInfeasible = 0;
      unsigned loopFwdInfeasible = 0;
      unsigned loopApplied = 0;
      unsigned loopTotalProducerOps = 0;
      unsigned loopTotalUserOps = 0;

      // Process plans in reverse program order so erasing the dot doesn't
      // invalidate later plans' references. (Each plan is independent — the
      // dot's operands are siblings, not children — but reverse-order is the
      // safe pattern.)
      for (auto it = lw.plans.rbegin(); it != lw.plans.rend(); ++it) {
        const auto &plan = *it;
        triton::DotOp dot = plan.dotOp;
        auto schemeOpt = backwardSliceForMSplit(plan);
        if (!schemeOpt) {
          ++loopBwdInfeasible;
          ++totalBwdInfeasible;
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / ctaTileM=" << plan.ctaTileM
              << "), but backward walker bailed (" << modeSuffix << ")";
          continue;
        }
        unsigned producerCount = schemeOpt->ops.size();
        loopTotalProducerOps += producerCount;

        if (!extendSliceForwardFromDot(plan, *schemeOpt)) {
          ++loopFwdInfeasible;
          ++totalFwdInfeasible;
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / ctaTileM=" << plan.ctaTileM
              << "), backward found " << producerCount
              << " producer op(s), forward walker bailed (" << modeSuffix << ")";
          continue;
        }
        unsigned totalOps = schemeOpt->ops.size();
        unsigned userCount = totalOps - producerCount;
        loopTotalUserOps += userCount;

        if (!apply) {
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / ctaTileM=" << plan.ctaTileM << "), co-partitioning "
              << producerCount << " producer op(s) + " << userCount
              << " user op(s) (phase 1d: plan only)";
          continue;
        }

        // APPLY path: mutate IR.
        if (failed(applyMSplit(plan))) {
          ++loopFwdInfeasible; // bookkeeping: treat apply failure as infeasible
          ++totalFwdInfeasible;
          continue;
        }
        ++loopApplied;
        ++totalApplied;
      }

      // (modeSuffix declared at outer scope)
      lw.forOp.emitRemark()
          << "ttgir-sched: candidate inner loop with " << lw.mfmaDotsSeen
          << " MFMA tt.dot op(s); plans " << lw.plans.size()
          << ", skipped " << lw.mfmaDotsSkipped << ", bwd-infeasible "
          << loopBwdInfeasible << ", fwd-infeasible " << loopFwdInfeasible
          << ", applied " << loopApplied << ", co-partition producer-ops "
          << loopTotalProducerOps << " + user-ops " << loopTotalUserOps
          << " (" << modeSuffix << ")";
    }

    module.emitRemark()
        << "ttgir-sched: visited " << numForOps << " scf.for op(s), "
        << numCandidateLoops << " candidate(s), " << totalMfmaDots
        << " MFMA tt.dot op(s), " << totalPlans << " planned M-split(s), "
        << totalSkipped << " skipped, " << totalBwdInfeasible
        << " bwd-infeasible, " << totalFwdInfeasible
        << " fwd-infeasible, " << totalApplied << " applied ("
        << modeSuffix << ")";
  }
};

} // namespace

} // namespace mlir
