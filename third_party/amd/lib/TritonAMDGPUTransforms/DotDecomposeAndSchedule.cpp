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
//  - Phase 1d (next): sliceOp — clone + retype the SSA chain.
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
    unsigned totalFwdInfeasible = 0;

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

      unsigned loopBwdInfeasible = 0;
      unsigned loopFwdInfeasible = 0;
      unsigned loopTotalProducerOps = 0;
      unsigned loopTotalUserOps = 0;

      for (const auto &plan : plans) {
        triton::DotOp dot = plan.dotOp;
        auto schemeOpt = backwardSliceForMSplit(plan);
        if (!schemeOpt) {
          ++loopBwdInfeasible;
          ++totalBwdInfeasible;
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / instrM=" << plan.instrM
              << "), but backward walker bailed (phase 1c: plan only)";
          continue;
        }
        unsigned producerCount = schemeOpt->ops.size();
        loopTotalProducerOps += producerCount;

        // Phase 1c: extend the scheme forward from the dot's result.
        if (!extendSliceForwardFromDot(plan, *schemeOpt)) {
          ++loopFwdInfeasible;
          ++totalFwdInfeasible;
          dot.emitRemark()
              << "ttgir-sched: would M-split this dot into "
              << plan.numPartitions << " (blockM=" << plan.blockM
              << " / instrM=" << plan.instrM << "), backward walker found "
              << producerCount
              << " producer op(s), forward walker bailed (phase 1c: plan only)";
          continue;
        }
        // Total includes producers + users; user count = total - producer.
        unsigned totalOps = schemeOpt->ops.size();
        unsigned userCount = totalOps - producerCount;
        loopTotalUserOps += userCount;

        dot.emitRemark()
            << "ttgir-sched: would M-split this dot into "
            << plan.numPartitions << " (blockM=" << plan.blockM
            << " / instrM=" << plan.instrM << "), co-partitioning "
            << producerCount << " producer op(s) + " << userCount
            << " user op(s) (phase 1c: plan only)";
      }

      forOp.emitRemark()
          << "ttgir-sched: candidate inner loop with " << mfmaDotsSeen
          << " MFMA tt.dot op(s); plans " << plans.size()
          << ", skipped " << mfmaDotsSkipped << ", bwd-infeasible "
          << loopBwdInfeasible << ", fwd-infeasible " << loopFwdInfeasible
          << ", co-partition producer-ops " << loopTotalProducerOps
          << " + user-ops " << loopTotalUserOps << " (phase 1c: plan only)";
    });

    module.emitRemark()
        << "ttgir-sched: visited " << numForOps << " scf.for op(s), "
        << numCandidateLoops << " candidate(s), " << totalMfmaDots
        << " MFMA tt.dot op(s), " << totalPlans << " planned M-split(s), "
        << totalSkipped << " skipped, " << totalBwdInfeasible
        << " bwd-infeasible, " << totalFwdInfeasible
        << " fwd-infeasible (phase 1c: plan only)";
  }
};

} // namespace

} // namespace mlir
