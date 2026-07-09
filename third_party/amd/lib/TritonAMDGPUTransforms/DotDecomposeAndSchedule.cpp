//===- DotDecomposeAndSchedule.cpp ----------------------------------------===//
//
// TTGIR-level replacement for the LLVM-IR `LLIRSchedule` pass on AMD MFMA
// matmul kernels.
//
// What this pass does
// -------------------
// For each MFMA-typed `tt.dot` in an inner `scf.for`:
//   1. Compute the M × N partition plan from
//      `AMDMfmaEncodingAttr::getInstrShape()` and `warpsPerCTA`:
//        ctaTileM = instrShape[0] * warpsPerCTA[0]    (32 for v8/v10)
//        ctaTileN = instrShape[1] * warpsPerCTA[1]    (32 for v8/v10)
//        numPartitionsM = blockM / ctaTileM           (8 for v8/v10)
//        numPartitionsN = blockN / ctaTileN           (4 for v8/v10)
//   2. Walk backward from `dot.getA()` and `dot.getC()` to collect
//      producer ops that would need co-partitioning (adapted from
//      `WSDataPartition::getBackwardSliceToPartition`,
//      `third_party/nvidia/hopper/.../WSDataPartition.cpp:291`, stripped
//      of Hopper-only branches).
//   3. Walk forward from `dot.getResult()` to collect user ops
//      (`WSDataPartition::getForwardSliceToPartition`:441, same strip).
//   4. (APPLY only) Replace the dot with `M × N` sub-dots via
//      `amdgpu.extract_slice` on each operand, then re-aggregate with a
//      single `amdgpu.concat` in row-major (M outer, N inner) order.
//   5. (APPLY only) Insert a `ROCDL::SchedBarrier(0)` after every k-th
//      sub-dot (default k = numPartitionsN, i.e. one barrier per M-row)
//      so LLVM's misched can optimize within row-bounded regions
//      without reordering across boundaries.
//
// Producer / user chains are not modified — `extract_slice` is a no-op
// at the CTA-tile level, so upstream layouts are preserved.
//
// Env-var contract
// ----------------
//   TRITON_ENABLE_TTGIR_SCHED          : enable the pass (planning-only)
//   TRITON_TTGIR_SCHED_APPLY           : also mutate IR
//   TRITON_TTGIR_SCHED_BARRIER_STRIDE  : barrier stride
//                                          unset → numPartitionsN
//                                          0     → no barriers
//                                          1     → between every sub-dot
//                                          k     → every k-th
//
// e2e results (stand-alone autotuned matmul, K sweep on gfx950):
//   K=1024 → -1.4 %, K=2048 → +2.4 %, K=4096 → +4.1 %, K=8192 → +11.0 %
// FA-fwd tutorial (the kernel that crashes the matmul_4waves LLIR pass
// with `Instruction does not dominate all uses!`):
//   runs correctly under TTGIR-SCHED with output within ±2 % of baseline.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "Utility.h" // AMD transforms: composePaddedLayout (+ TargetInfo)
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
// Backend-neutral modulo scheduling core (Phase A extraction). Cross-tree
// include via the global `third_party` include dir — interim until the core is
// relocated to a neutral path.
#include "mlir/IR/IRMapping.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/AMDLatencyModel.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/DataDependenceGraph.h"
#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/ModuloReservationTable.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

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
  unsigned blockM;        // BLOCK_M; e.g. 256 for v8/v10.
  unsigned instrM;        // MFMA instr M; e.g. 16 for [16,16,32].
  unsigned ctaTileM;      // instrM * warpsPerCTA[0]; e.g. 32 for v8/v10.
  unsigned numPartitions; // blockM / ctaTileM; e.g. 8 for v8/v10.
  unsigned tileM;         // ctaTileM (i.e. M dim of each sub-dot result).
  // Phase 2: N-split fields.
  unsigned blockN;         // BLOCK_N; e.g. 128 for v8/v10's left/right half.
  unsigned instrN;         // MFMA instr N; e.g. 16 for [16,16,32].
  unsigned ctaTileN;       // instrN * warpsPerCTA[1]; e.g. 32 for v8/v10.
  unsigned numPartitionsN; // blockN / ctaTileN; e.g. 4 for v8/v10. 0 = no
                           //   N-split (numPartitions < 2 for N).
  unsigned tileN;          // ctaTileN.
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
  auto warpsPerCTA = mfmaEnc.getWarpsPerCTA();
  if (warpsPerCTA.size() < 2)
    return std::nullopt;
  unsigned blockM = static_cast<unsigned>(resultShape[0]);
  unsigned blockN = static_cast<unsigned>(resultShape[1]);
  unsigned instrM = instrShape[0];
  unsigned instrN = instrShape[1];
  unsigned warpsM = warpsPerCTA[0];
  unsigned warpsN = warpsPerCTA[1];
  if (instrM == 0 || warpsM == 0 || instrN == 0 || warpsN == 0)
    return std::nullopt;

  // M split (required for the plan to be valid).
  unsigned ctaTileM = instrM * warpsM;
  if (blockM % ctaTileM != 0)
    return std::nullopt;
  unsigned numPartitions = blockM / ctaTileM;
  if (numPartitions < 2)
    return std::nullopt;

  // N split (optional — falls back to 1 partition if it can't divide).
  unsigned ctaTileN = instrN * warpsN;
  unsigned numPartitionsN = 0;
  if (blockN % ctaTileN == 0) {
    unsigned candidate = blockN / ctaTileN;
    if (candidate >= 2)
      numPartitionsN = candidate;
  }

  return DotPartitionPlan{dotOp,         blockM,         instrM,  ctaTileM,
                          numPartitions, ctaTileM,       blockN,  instrN,
                          ctaTileN,      numPartitionsN, ctaTileN};
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
             triton::gpu::LocalLoadOp, triton::amdgpu::BufferLoadToLocalOp>(op);
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
  return isa<arith::TruncFOp, arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp,
             arith::FPToSIOp, arith::FPToUIOp, triton::StoreOp,
             triton::gpu::LocalStoreOp, triton::amdgpu::BufferStoreOp,
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

/// Parse `TRITON_TTGIR_SCHED_BARRIER_STRIDE` env var. Returns:
///   * 0 → no SchedBarrier insertion (raw rewrite from Phase 2)
///   * k > 0 → insert ROCDL::SchedBarrier(0) after every k-th sub-dot
/// Default (env var unset): k = numPartitionsN (one barrier per M-row).
static int getSchedBarrierStride(unsigned defaultStride) {
  const char *env = std::getenv("TRITON_TTGIR_SCHED_BARRIER_STRIDE");
  if (!env || !*env)
    return static_cast<int>(defaultStride);
  return std::atoi(env);
}

//===----------------------------------------------------------------------===//
// Producer-chain slicing (opt-in: TRITON_TTGIR_SCHED_SLICE_LOADS)
//
// Instead of `extract_slice`-ing the dot's register operands (which leaves the
// monolithic `local_load` whole), rebuild the producer chain the backward
// walker recorded, PER PARTITION: slice the SMEM buffer at the `local_load`
// leaf (memdesc_subslice + local_load) and CLONE interior ops (convert_layout,
// casts) with sliced operands + sliced result types. This exposes per-tile load
// nodes so a downstream modulo scheduler can overlap loads with the sub-dots.
// See claude/ttgir_sched_modulo_plan.md.
//===----------------------------------------------------------------------===//

/// Leaf: slice the shared-memory buffer that `loadOp` reads, returning a
/// memdesc_subslice + local_load of the [offset, tileExtent] sub-tile along
/// `dim`. Mirrors `WGMMAPipeline.cpp::splitRhs`.
static Value buildSlicedLocalLoad(OpBuilder &builder, Location loc,
                                  triton::gpu::LocalLoadOp loadOp, unsigned dim,
                                  int64_t offset, unsigned tileExtent) {
  Value srcMem = loadOp.getSrc();
  auto srcTy = cast<triton::gpu::MemDescType>(srcMem.getType());

  SmallVector<int64_t> shape(srcTy.getShape().begin(), srcTy.getShape().end());
  shape[dim] = tileExtent;
  auto subTy = triton::gpu::MemDescType::get(
      shape, srcTy.getElementType(), srcTy.getEncoding(),
      srcTy.getMemorySpace(),
      /*mutableMemory=*/srcTy.getMutableMemory(), srcTy.getAllocShape());

  SmallVector<int32_t> offsets(srcTy.getRank(), 0);
  offsets[dim] = static_cast<int32_t>(offset);
  Value sub = triton::gpu::MemDescSubsliceOp::create(builder, loc, subTy,
                                                     srcMem, offsets);

  auto resTy = sliceTensorType(
      cast<RankedTensorType>(loadOp.getResult().getType()), dim, tileExtent);
  return triton::gpu::LocalLoadOp::create(builder, loc, resTy, sub,
                                          loadOp.getToken());
}

/// Interior ops we are willing to clone per partition (safe to clone + reset
/// the result shape along `dim`). Dim-flipping ops (broadcast/expand/trans/
/// reshape) are intentionally excluded — same limitation as the walkers.
static bool isCloneableInterior(Operation *op) {
  return isa<triton::gpu::ConvertLayoutOp>(op) ||
         op->hasTrait<mlir::OpTrait::Elementwise>() ||
         isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncFOp,
             arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op);
}

/// Materialize the [partIdx*tile : +tile] slice (along `dim`) of `v`,
/// rebuilding the producer chain recorded in `scheme`. Memoized per (value,
/// partIdx).
///   - local_load (in scheme)      -> memdesc_subslice + local_load (leaf)
///   - cloneable interior (in scheme) -> clone with sliced operands/result
///   - global tt.load (in scheme, sliceGlobalLoads) -> clone with sliced ptr
///   - anything else               -> extract_slice fallback (phase-1d
///   behavior)
///
/// `sliceGlobalLoads` (env TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS) is an
/// independent opt-in: a global `tt.load` is normally left whole (extract_slice
/// fallback) so its wide/coalesced global access is preserved; when set, the
/// load is cloned per partition with a sliced pointer (and mask/other) operand,
/// giving per-sub-tile global loads at the cost of narrower access. See
/// claude/amd_modulo_scheduling_plan.md (load-slicing is a scheduler-infra
/// knob, measured -12% standalone on the pinned matmul).
static Value
materializeSlice(OpBuilder &builder, Location loc, Value v, unsigned dim,
                 unsigned partIdx, unsigned tile,
                 const DataPartitionScheme &scheme, bool sliceLoads,
                 bool sliceGlobalLoads,
                 llvm::DenseMap<std::pair<Value, unsigned>, Value> &memo) {
  auto key = std::make_pair(v, partIdx);
  if (auto it = memo.find(key); it != memo.end())
    return it->second;

  int64_t offset = static_cast<int64_t>(partIdx) * tile;
  Operation *def = v.getDefiningOp();
  Value result;

  // A global load is cloneable (per-partition slice of its pointer operand)
  // only when explicitly opted in; otherwise it stays whole and we
  // extract_slice.
  bool cloneGlobalLoad = sliceGlobalLoads && def && isa<triton::LoadOp>(def);

  if (sliceLoads && def && scheme.ops.contains(def) &&
      isa<triton::gpu::LocalLoadOp>(def)) {
    result = buildSlicedLocalLoad(
        builder, loc, cast<triton::gpu::LocalLoadOp>(def), dim, offset, tile);
  } else if (sliceLoads && def && scheme.ops.contains(def) &&
             (isCloneableInterior(def) || cloneGlobalLoad)) {
    IRMapping map;
    for (Value operand : def->getOperands()) {
      Value newOperand =
          needToSlice(operand, dim, scheme.numPartitions)
              ? materializeSlice(builder, loc, operand, dim, partIdx, tile,
                                 scheme, sliceLoads, sliceGlobalLoads, memo)
              : operand;
      map.map(operand, newOperand);
    }
    Operation *cloned = builder.clone(*def, map);
    for (auto [oldR, newR] : llvm::zip(def->getResults(), cloned->getResults()))
      if (auto t = dyn_cast<RankedTensorType>(oldR.getType()))
        newR.setType(sliceTensorType(t, dim, tile));
    result = cloned->getResult(0);
  } else {
    result = buildExtractSliceAlongDim(builder, loc, v, dim, offset, tile);
  }

  memo[key] = result;
  return result;
}

/// Apply the M-(and optionally N-)split rewrite for one plan. Replaces
///   %D = tt.dot %A, %B, %C
/// with an M × N grid of small dots, where
///   %A_i  = amdgpu.extract_slice %A [i*ctaTileM, 0]      (M dim only)
///   %B_j  = amdgpu.extract_slice %B [0, j*ctaTileN]      (N dim only)
///   %C_ij = amdgpu.extract_slice %C [i*ctaTileM, j*ctaTileN]
///   %D_ij = tt.dot %A_i, %B_j, %C_ij
/// then
///   %D = amdgpu.concat %D_00, %D_01, ..., %D_{M-1,N-1}
/// in row-major (M outer, N inner) order.
///
/// Phase 3: after every `stride` sub-dots (default = numPartitionsN, i.e.
/// one per M-row) insert a `ROCDL::SchedBarrier(0)` so LLVM's misched can
/// reorder within a region but not across it. `stride=0` disables barriers
/// entirely (= Phase 2 behavior). Configurable via
/// `TRITON_TTGIR_SCHED_BARRIER_STRIDE`.
///
/// If `plan.numPartitionsN < 2`, the N-split is skipped (degenerates to the
/// pure-M case from Phase 1d). Producer chains are NOT modified — the
/// extract_slices are CTA-tile no-ops upstream so the existing layouts are
/// preserved.
static LogicalResult applyMSplit(const DotPartitionPlan &plan,
                                 DataPartitionScheme &scheme) {
  triton::DotOp dot = plan.dotOp;
  OpBuilder builder(dot);
  Location loc = dot.getLoc();
  unsigned nM = plan.numPartitions;
  unsigned tileM = plan.tileM;
  unsigned nN = plan.numPartitionsN >= 2 ? plan.numPartitionsN : 1;
  unsigned tileN = nN > 1 ? plan.tileN : plan.blockN;

  // Phase 3: barrier stride; default = one per M-row.
  int stride = getSchedBarrierStride(/*defaultStride=*/nN);

  // Opt-in producer-chain slicing (feeds the modulo pipeline). The driver's
  // `scheme` covers the A (and C) backward slice; build a separate scheme for B
  // (sliced along N) since backwardSliceForMSplit only walked A and C. Capture
  // operands now so the dead originals can be erased after the rewrite.
  bool sliceLoads = triton::tools::getBoolEnv("TRITON_TTGIR_SCHED_SLICE_LOADS");
  // Independent opt-in: also slice the global tt.load (per-partition cloned
  // load with a sliced pointer), instead of leaving it whole + extract_slice.
  bool sliceGlobalLoads =
      triton::tools::getBoolEnv("TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS");
  Value aOperand = dot.getA();
  Value bOperand = dot.getB();
  llvm::DenseMap<std::pair<Value, unsigned>, Value> sliceMemo;

  DataPartitionScheme schemeB;
  bool sliceB = false;
  if (sliceLoads && nN > 1) {
    schemeB.numPartitions = nN;
    sliceB = getBackwardSliceToPartition(bOperand, /*dim=*/1, schemeB,
                                         dot->getParentRegion());
  }

  SmallVector<Value> partialResults;
  partialResults.reserve(nM * nN);

  // Pre-build B slices (one per N partition), then reuse across all M-i.
  SmallVector<Value> bSlices;
  bSlices.reserve(nN);
  if (nN > 1) {
    for (unsigned j = 0; j < nN; ++j) {
      bSlices.push_back(materializeSlice(builder, loc, bOperand, /*dim=*/1, j,
                                         tileN, schemeB, sliceB,
                                         sliceGlobalLoads, sliceMemo));
    }
  } else {
    bSlices.push_back(bOperand);
  }

  unsigned dotIdx = 0;
  for (unsigned i = 0; i < nM; ++i) {
    int64_t offM = static_cast<int64_t>(i) * tileM;
    Value aSlice =
        materializeSlice(builder, loc, aOperand, /*dim=*/0, i, tileM, scheme,
                         sliceLoads, sliceGlobalLoads, sliceMemo);
    for (unsigned j = 0; j < nN; ++j) {
      Value cSlice;
      if (nN > 1) {
        auto cType = cast<RankedTensorType>(dot.getC().getType());
        auto smallCType = sliceTensorType(
            sliceTensorType(cType, /*dim=*/0, tileM), /*dim=*/1, tileN);
        SmallVector<int64_t> offsets = {offM, static_cast<int64_t>(j) * tileN};
        cSlice = triton::amdgpu::ExtractSliceOp::create(
            builder, loc, smallCType, dot.getC(),
            builder.getDenseI64ArrayAttr(offsets));
      } else {
        cSlice = buildExtractSliceAlongDim(builder, loc, dot.getC(),
                                           /*dim=*/0, offM, tileM);
      }
      auto smallResultType = cast<RankedTensorType>(cSlice.getType());
      auto smallDot = triton::DotOp::create(
          builder, loc, smallResultType, aSlice, bSlices[j], cSlice,
          dot.getInputPrecisionAttr(), dot.getMaxNumImpreciseAccAttr());
      partialResults.push_back(smallDot.getResult());
      ++dotIdx;

      // Phase 3: after every `stride` sub-dots, drop a SchedBarrier(0).
      // Don't emit one after the last sub-dot (the concat itself is an
      // implicit cap).
      if (stride > 0 && dotIdx < nM * nN &&
          (dotIdx % static_cast<unsigned>(stride)) == 0) {
        ROCDL::SchedBarrier::create(builder, loc,
                                    /*mask=*/0);
      }
    }
  }

  auto fullType = cast<RankedTensorType>(dot.getResult().getType());
  auto concat =
      triton::amdgpu::ConcatOp::create(builder, loc, fullType, partialResults);
  dot.getResult().replaceAllUsesWith(concat.getResult());
  dot.erase();

  // The original monolithic loads (and any cloned interior convert/cast) are
  // now dead — erase them so only the per-tile sliced producers remain in the
  // DDG. Fixpoint on use_empty; null out erased entries to avoid dangling
  // derefs.
  if (sliceLoads) {
    SmallVector<Operation *> chain(scheme.ops.begin(), scheme.ops.end());
    chain.append(schemeB.ops.begin(), schemeB.ops.end());
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation *&op : chain) {
        if (op && op->use_empty() &&
            isa<triton::gpu::LocalLoadOp, triton::gpu::ConvertLayoutOp,
                triton::LoadOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncFOp, arith::TruncIOp>(op)) {
          op->erase();
          op = nullptr;
          changed = true;
        }
      }
    }
    // The ops just erased were members of `scheme.ops` / `schemeB.ops`; those
    // SetVectors would now hold dangling pointers. Clear them so no later code
    // can dereference freed memory — the schemes are fully consumed here.
    scheme.ops.clear();
    schemeB.ops.clear();
  }
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

// LDS-capacity stage cap (change #3 of amd_decomp_modulo_pipeline.md). Returns
// the max pipeline depth (stageCap; numBuffers = stageCap+1) whose
// multi-buffered operand tiles fit gfx950's 160KB LDS, replacing the hardcoded
// cap. Computed from the dot's per-iteration operand tiles (A = BM×BK, B =
// BK×BN). Gated by TRITON_USE_MODULO_SCHEDULE so the existing AMD_MODULO path
// is unchanged by default.
static int computeLDSStageCap(scf::ForOp forOp, int fallback) {
  constexpr int64_t kLDSBytes = 160 * 1024; // gfx950 LDS per CU
  int64_t perBufferBytes = 0;
  forOp.getBody()->walk([&](triton::DotOp dot) {
    if (!getMfmaEncoding(dot))
      return;
    for (Value operand : {dot.getA(), dot.getB()}) {
      auto t = dyn_cast<RankedTensorType>(operand.getType());
      if (!t)
        continue;
      int64_t elems = 1;
      for (int64_t d : t.getShape())
        elems *= d;
      int64_t eb =
          std::max<int64_t>(1, t.getElementType().getIntOrFloatBitWidth() / 8);
      perBufferBytes += elems * eb;
    }
  });
  if (perBufferBytes <= 0)
    return fallback;
  int64_t maxBuffers = kLDSBytes / perBufferBytes; // floor
  if (maxBuffers < 2)
    maxBuffers = 2; // need at least a double buffer to pipeline
  return static_cast<int>(maxBuffers - 1);
}

//===----------------------------------------------------------------------===//
// Early load lowering (change #1 of amd_decomp_modulo_pipeline.md)
//
// Lower a pipelineable global tt.load to a SINGLE-buffer staged copy
//   local_alloc<1 x tile> -> async_copy_global_to_local -> commit -> wait
//   -> local_load (replacing the load's uses)
// BEFORE modulo runs, so modulo sees async_copy (GLOBAL latency) vs local_load
// (LDS) as distinct ops and can size the overlap itself. Single buffer here;
// the ModuloDotSchedule expander (change #4) grows the ring from modulo's
// stages. v1 uses a plain (unswizzled) shared encoding -- matching the real
// path's padded_shared needs targetInfo(arch) plumbed into the pass
// (follow-up).
//===----------------------------------------------------------------------===//

// True if `ld`'s result reaches an MFMA dot through cloneable interior ops
// (convert_layout / casts).
static bool loadFeedsMfmaDot(triton::LoadOp ld) {
  SmallVector<Operation *> wl(ld->getUsers().begin(), ld->getUsers().end());
  llvm::DenseSet<Operation *> seen;
  while (!wl.empty()) {
    Operation *u = wl.pop_back_val();
    if (!seen.insert(u).second)
      continue;
    if (auto dot = dyn_cast<triton::DotOp>(u))
      if (getMfmaEncoding(dot))
        return true;
    if (isCloneableInterior(u))
      for (Operation *uu : u->getUsers())
        wl.push_back(uu);
  }
  return false;
}

// The DotOperandEncoding the load `ld` feeds (through cloneable interior ops),
// or null. Drives the LDS encoding selection.
static triton::gpu::DotOperandEncodingAttr
loadDotOperandEnc(triton::LoadOp ld) {
  SmallVector<Value> wl{ld.getResult()};
  llvm::DenseSet<Operation *> seen;
  while (!wl.empty()) {
    Value v = wl.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (isa<triton::DotOp>(u))
        if (auto t = dyn_cast<RankedTensorType>(v.getType()))
          if (auto de = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                  t.getEncoding()))
            return de;
      if (seen.insert(u).second && isCloneableInterior(u))
        for (Value r : u->getResults())
          wl.push_back(r);
    }
  }
  return nullptr;
}

static void lowerLoadToStagedCopy(scf::ForOp forOp, triton::LoadOp ld,
                                  const triton::AMD::TargetInfo &targetInfo) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  if (!ty || ty.getRank() != 2)
    return;
  MLIRContext *ctx = ld.getContext();
  // Pick the LDS encoding the way the stream pipeliner does (LowerLoops):
  // padded (CDNA4 async, conflict-free) when possible, else dot-operand-driven
  // swizzled. Closes the v1 bank-conflict gap vs the plain unswizzled encoding.
  SmallVector<unsigned> sharedOrder = triton::gpu::getOrder(ty);
  auto cga = triton::gpu::getCGALayout(ty.getEncoding());
  unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
  triton::gpu::SharedEncodingTrait sharedEnc;
  if (auto dotOpEnc = loadDotOperandEnc(ld)) {
    auto srcTOM = cast<triton::gpu::TensorOrMemDesc>(ld.getType());
    sharedEnc = composePaddedLayout(targetInfo, dotOpEnc.getOpIdx(),
                                    dotOpEnc.getKWidth(), srcTOM, sharedOrder,
                                    dotOpEnc, /*useAsyncCopy=*/true);
    if (!sharedEnc)
      sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
          ctx, dotOpEnc, ty.getShape(), sharedOrder, cga, bitWidth,
          /*needTrans=*/false);
  }
  if (!sharedEnc) // no dot-operand info: plain swizzled fallback
    sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1,
                                                             sharedOrder, cga);
  Value alloc = triton::createAlloc(forOp, ty, ld.getLoc(), sharedEnc,
                                    /*distance=*/1);
  OpBuilder b(ld);
  Location loc = ld.getLoc();
  auto viewLoad = triton::createSingleBufferView(b, alloc, 0)
                      .getDefiningOp<triton::gpu::MemDescIndexOp>();
  auto copyOp = triton::gpu::AsyncCopyGlobalToLocalOp::create(
      b, loc, ld.getPtr(), viewLoad, ld.getMask(), ld.getOther(), ld.getCache(),
      ld.getEvict(), ld.getIsVolatile(), /*contiguity=*/1);
  auto commitOp =
      triton::gpu::AsyncCommitGroupOp::create(b, loc, copyOp->getResult(0));
  auto waitOp =
      triton::gpu::AsyncWaitOp::create(b, loc, commitOp->getResult(0), 0);
  triton::replaceUsesWithLocalLoad(b, ld->getResult(0), viewLoad, waitOp);
  ld.erase();
}

// Opt-in (TRITON_AMD_EARLY_LOWER): early-lower every pipelineable global load
// feeding an MFMA dot, for each inner loop.
static void runEarlyLowerLoads(ModuleOp module) {
  auto arch = mlir::getAMDArch(module);
  triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });
  for (scf::ForOp forOp : loops) {
    SmallVector<triton::LoadOp> loads;
    forOp.getBody()->walk([&](triton::LoadOp ld) {
      if (loadFeedsMfmaDot(ld))
        loads.push_back(ld);
    });
    for (triton::LoadOp ld : loads)
      lowerLoadToStagedCopy(forOp, ld, targetInfo);
  }
}

//===----------------------------------------------------------------------===//
// ModuloDotSchedule expander (change #4 of amd_decomp_modulo_pipeline.md)
//
// Takes the early-lowered loop (single-buffer async_copy/local_load) and turns
// it into a real software pipeline: (a) re-buffer single->multi + ring
// `extractIdx` (mirrors createStreamOps, LowerLoops.cpp:368-387), (b) serialize
// a CoarseSchedule (async_copy backward-slice -> stage 0 / load cluster; rest
// -> last stage / compute cluster), (c) run the general expander `expandLoops`
// (NOT SingleDotSchedule). v1: canonical depth-2 double buffer; modulo-derived
// depth / fine order is the refinement. Async-wait counts are fixed by the
// existing tritonamdgpu-update-async-wait-count pass downstream (e2e) — not in
// this pass.
//===----------------------------------------------------------------------===//
static void runModuloExpand(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });
  for (scf::ForOp forOp : loops) {
    SmallVector<triton::gpu::AsyncCopyGlobalToLocalOp> copies;
    forOp.getBody()->walk([&](triton::gpu::AsyncCopyGlobalToLocalOp cp) {
      copies.push_back(cp);
    });
    if (copies.empty())
      continue;

    const int numBuffers = 2; // v1: canonical double buffer
    IRRewriter builder(forOp);
    Location loc = forOp.getLoc();
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
    Value numBuf = arith::ConstantIntOp::create(builder, loc, numBuffers, 32);

    // Ring index iter-arg (mirrors createStreamOps).
    unsigned newArgIdx = forOp.getBody()->getNumArguments();
    forOp = addIterArgsToLoop(builder, forOp, {minusOne});
    Value ring = forOp.getBody()->getArgument(newArgIdx);
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
    ring = arith::AddIOp::create(builder, loc, ring, one);
    Value cnd = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                      ring, numBuf);
    ring = arith::SelectOp::create(builder, loc, cnd, ring, zero);
    appendToForOpYield(forOp, {ring});

    // Re-buffer each operand's alloc (depth 1 -> numBuffers), re-index by
    // `ring`. `copies` op pointers stay valid (addIterArgsToLoop moves the
    // body).
    for (auto cp : copies) {
      Value oldView = cp.getResult(); // dest memdesc = memdesc_index alloc[c0]
      auto oldViewOp = oldView.getDefiningOp<triton::gpu::MemDescIndexOp>();
      if (!oldViewOp)
        continue;
      auto oldAlloc =
          oldViewOp.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
      if (!oldAlloc)
        continue;
      auto oldTy = cast<triton::gpu::MemDescType>(oldAlloc.getType());
      auto shp = oldTy.getShape();
      if (shp.size() < 2)
        continue; // expect [1, tile...]
      SmallVector<int64_t> tile(shp.begin() + 1, shp.end());
      auto tensorTy = RankedTensorType::get(tile, oldTy.getElementType());
      auto enc =
          dyn_cast<triton::gpu::SharedEncodingTrait>(oldTy.getEncoding());
      if (!enc)
        continue;
      // createAlloc grows to numBuffers AND emits a matching local_dealloc.
      Value newAlloc = triton::createAlloc(forOp, tensorTy, oldAlloc.getLoc(),
                                           enc, numBuffers);
      OpBuilder vb(oldViewOp);
      Value newView = triton::createSingleBufferView(vb, newAlloc, ring);
      oldView.replaceAllUsesWith(newView);
      oldViewOp.erase();
      // Erase the old alloc's dangling local_dealloc, then the old alloc.
      SmallVector<Operation *> users(oldAlloc->getUsers().begin(),
                                     oldAlloc->getUsers().end());
      for (Operation *u : users)
        if (isa<triton::gpu::LocalDeallocOp>(u))
          u->erase();
      oldAlloc.erase();
    }

    // Stage assignment: async_copy backward slice (within the loop) -> stage 0
    // (load cluster); everything else -> last stage (compute cluster). The
    // backward slice keeps each load's index/pointer math co-staged (def<=use).
    llvm::DenseSet<Operation *> stage0;
    SmallVector<Operation *> wl;
    for (auto cp : copies)
      wl.push_back(cp);
    while (!wl.empty()) {
      Operation *op = wl.pop_back_val();
      if (!op || op->getBlock() != forOp.getBody())
        continue;
      if (!stage0.insert(op).second)
        continue;
      for (Value v : op->getOperands())
        if (Operation *d = v.getDefiningOp())
          wl.push_back(d);
    }
    triton::CoarseSchedule cs(/*numStages=*/numBuffers);
    auto cLoad = cs.clusters.newAtBack();
    auto cCompute = cs.clusters.newAtBack();
    for (Operation &op : forOp.getBody()->without_terminator()) {
      bool s0 = stage0.contains(&op);
      cs.insert(&op, s0 ? 0 : numBuffers - 1, s0 ? cLoad : cCompute);
    }
    cs.serialize(forOp);
  }
  // Run the general expander on every serialized loop (change #4 (b)).
  expandLoops(module);
}

// Phase E0: AMD modulo scaffold. For each inner loop, build the backend-neutral
// DDG (from TritonGPUModuloCore) using AMDLatencyModel and report the
// per-pipeline node classification. This is the scaffold for the AMD modulo
// scheduler and the runtime test gate for AMDLatencyModel.
static void runAMDModuloScaffold(ModuleOp module) {
  triton::gpu::AMDLatencyModel model;
  // Collect loops first — E2 mutates loop bodies, so don't mutate during walk.
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp f) { loops.push_back(f); });

  for (scf::ForOp forOp : loops) {
    auto ddg = triton::gpu::DataDependenceGraph::build(forOp, model);
    llvm::DenseMap<triton::gpu::HWPipeline, unsigned> counts;
    for (const auto &node : ddg.getNodes())
      ++counts[node.pipeline];
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "amd-modulo: DDG " << ddg.getNumNodes() << " nodes";
    for (auto p :
         {triton::gpu::HWPipeline::MFMA, triton::gpu::HWPipeline::LDS,
          triton::gpu::HWPipeline::GLOBAL, triton::gpu::HWPipeline::VALU,
          triton::gpu::HWPipeline::NONE})
      os << " " << triton::gpu::getPipelineName(p) << "=" << counts.lookup(p);

    // E1: run the core modulo scheduler (rau/SMS, selected by
    // TRITON_USE_MODULO_SCHEDULE; default rau) and annotate each op with its
    // stage/order so a downstream expansion can consume the schedule.
    auto schedOr = triton::gpu::runModuloScheduling(ddg);
    if (!succeeded(schedOr)) {
      os << " II=FAILED";
      forOp.emitRemark() << os.str();
      continue;
    }
    auto &sched = *schedOr;
    mlir::Builder b(module.getContext());
    for (const auto &node : ddg.getNodes()) {
      auto it = sched.nodeToCycle.find(node.idx);
      if (it == sched.nodeToCycle.end())
        continue;
      node.op->setAttr("ttg.modulo_stage",
                       b.getI32IntegerAttr(sched.getStage(node.idx)));
      node.op->setAttr("ttg.modulo_order", b.getI32IntegerAttr(it->second));
    }
    os << " II=" << sched.II << " maxStage=" << sched.getMaxStage();

    // E3: emit a serialized triton::CoarseSchedule so the EXISTING AMD pipeline
    // expander (tritonamdgpu-pipeline) does multi-buffering + loop expansion —
    // i.e. modulo replaces tritonamdgpu-schedule-loops. Map modulo stage ->
    // CoarseSchedule stage and slot (order % II) -> cluster (steady-state body
    // order). num_stages and per-buffer copy-count then follow from modulo's
    // stage assignment. This is the cross-iteration (pipelining) axis; E2 below
    // is the complementary intra-iteration interleave axis.
    if (triton::tools::getBoolEnv("TRITON_AMD_MODULO_SERIALIZE")) {
      // Phase-D-lite guardrail: cap the pipeline depth. Realistic global
      // latency
      // (~790 cyc) makes modulo want a very deep pipeline (e.g. 52 stages) to
      // fully hide it — infeasible (LDS holds ~2 stages). Until Phase D adds a
      // real LDS/register feasibility model, clamp stages to a small cap
      // (default 1 -> 2-stage pipeline, matching the stream-pipeline baseline).
      int stageCap = 1;
      // Change #3: LDS-capacity support for the AMD modulo scheduler. When the
      // decomp+modulo guard is on, set the depth to min(LDS-feasible, needed):
      //   * LDS-feasible = floor(160KB / per-iter operand bytes) - 1   (gfx950)
      //   * needed       = max(1, modulo maxStage = ceil(latency/II))
      // so small tiles can pipeline deeper (up to LDS) while large tiles stay
      // protected, and we never buffer deeper than the latency actually needs.
      if (!triton::tools::getStrEnv("TRITON_USE_MODULO_SCHEDULE").empty()) {
        int ldsCap = computeLDSStageCap(forOp, /*fallback=*/1);
        int needed = std::max(1, sched.getMaxStage());
        stageCap = std::min(ldsCap, needed);
        os << " ldsCap=" << ldsCap << " needed=" << needed;
      }
      if (const char *e = std::getenv("TRITON_AMD_MODULO_MAX_STAGE"))
        stageCap = std::atoi(e); // explicit override always wins
      triton::CoarseSchedule cs(stageCap + 1);
      // The AMD pipeline expander's SingleDotSchedule path requires exactly the
      // 2-cluster convention: cluster 0 = global load, cluster 1 = compute. The
      // *multi-buffer decision* lives in the STAGE assignment (modulo owns it);
      // the cluster is just load-vs-compute. (Finer order/slot interleave is
      // the separate E2 axis, not consumed by this path.)
      auto cLoad = cs.clusters.newAtBack();    // SCHED_GLOBAL_LOAD
      auto cCompute = cs.clusters.newAtBack(); // SCHED_COMPUTE
      for (const auto &node : ddg.getNodes()) {
        auto it = sched.nodeToCycle.find(node.idx);
        if (it == sched.nodeToCycle.end())
          continue;
        bool isGlobal = node.pipeline == triton::gpu::HWPipeline::GLOBAL;
        // Canonical double-buffer overlap: global loads are PREFETCHED one
        // stage ahead of compute (stage 0 / cluster 0), everything else lands
        // in the compute stage (stageCap / cluster 1). Modulo's raw stage =
        // ceil(latency/II) under-stages when latency < II (no prefetch), so we
        // place loads ahead explicitly; modulo still provides II (now realistic
        // via the scaled MFMA cost), schedulability, and the load/compute
        // split.
        auto cluster = isGlobal ? cLoad : cCompute;
        int stage = isGlobal ? 0 : stageCap;
        cs.insert(node.op, stage, cluster);
      }
      cs.serialize(forOp);
      os << " serialized num_stages=" << cs.getNumStages()
         << " (capped from maxStage=" << sched.getMaxStage() << ")";
      forOp.emitRemark() << os.str();
      continue;
    }

    // E2: realize the schedule in IR — reorder the loop body into modulo
    // `order` and mark stage boundaries with rocdl.sched.barrier (region hints
    // for the backend machine scheduler). Guarded: only reorder when every
    // non-terminator op is scheduled, so the move sequence stays
    // dominance-valid (modulo order respects intra-iteration deps).
    Block *body = forOp.getBody();
    Operation *term = body->getTerminator();
    SmallVector<Operation *> ops;
    bool allScheduled = true;
    for (Operation &op : body->without_terminator()) {
      if (!op.hasAttr("ttg.modulo_order")) {
        allScheduled = false;
        break;
      }
      ops.push_back(&op);
    }
    int nbar = 0;
    if (allScheduled && !ops.empty()) {
      auto orderOf = [](Operation *op) {
        return cast<IntegerAttr>(op->getAttr("ttg.modulo_order")).getInt();
      };
      auto stageOf = [](Operation *op) {
        return cast<IntegerAttr>(op->getAttr("ttg.modulo_stage")).getInt();
      };
      llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
        return orderOf(a) < orderOf(b);
      });
      for (Operation *op : ops)
        op->moveBefore(term);
      int64_t prevStage = -1;
      for (Operation *op : ops) {
        int64_t stage = stageOf(op);
        if (prevStage >= 0 && stage != prevStage) {
          OpBuilder bb(op);
          ROCDL::SchedBarrier::create(bb, op->getLoc(), /*mask=*/0);
          ++nbar;
        }
        prevStage = stage;
      }
      os << " reordered barriers=" << nbar;
    } else {
      os << " reorder=skipped";
    }
    forOp.emitRemark() << os.str();
  }
}

struct TritonAMDGPUDotDecomposeAndSchedulePass
    : impl::TritonAMDGPUDotDecomposeAndScheduleBase<
          TritonAMDGPUDotDecomposeAndSchedulePass> {
  // Inherit base constructors (incl. the Options-taking one) so the `mode`
  // pass option is wired through createTritonAMDGPUDotDecomposeAndSchedule.
  using impl::TritonAMDGPUDotDecomposeAndScheduleBase<
      TritonAMDGPUDotDecomposeAndSchedulePass>::
      TritonAMDGPUDotDecomposeAndScheduleBase;

  void runOnOperation() override {
    // Change #2: an explicit `mode` pass option overrides env-var dispatch, so
    // the same pass can sit at several pipeline slots (early-lower / decompose
    // / modulo) without the two-slot collision (env dispatch checks AMD_MODULO
    // before TTGIR_SCHED, so a post-pipeline decompose slot would re-run
    // modulo).
    StringRef m(mode);
    if (m == "early-lower") {
      runEarlyLowerLoads(getOperation());
      return;
    }
    if (m == "modulo") {
      runAMDModuloScaffold(getOperation());
      return;
    }
    if (m == "expand") {
      runModuloExpand(getOperation());
      return;
    }
    bool modeDecompose = (m == "decompose");

    if (!modeDecompose) {
      // Legacy env-var dispatch (mode unset).
      // Change #1: opt-in early load lowering (tt.load -> single-buffer
      // async_copy + local_load) so a later modulo pass sees the staged ops.
      if (triton::tools::getBoolEnv("TRITON_AMD_EARLY_LOWER")) {
        runEarlyLowerLoads(getOperation());
        return;
      }

      // Phase E0: opt-in AMD modulo scaffold (DDG build + classification).
      if (triton::tools::getBoolEnv("TRITON_ENABLE_AMD_MODULO")) {
        runAMDModuloScaffold(getOperation());
        return;
      }

      if (!triton::tools::getBoolEnv("TRITON_ENABLE_TTGIR_SCHED"))
        return;
    }

    // Phase 1d gate: when set, the pass mutates the IR (replaces each
    // candidate dot with N sliced sub-dots + concat). Default off so the
    // existing TRITON_ENABLE_TTGIR_SCHED=1 keeps planning-only behavior.
    // mode=decompose forces apply (explicit request, independent of env).
    bool apply =
        modeDecompose || triton::tools::getBoolEnv("TRITON_TTGIR_SCHED_APPLY");
    const char *modeSuffix = apply ? "phase 3: applied" : "phase 3: plan only";

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
          dot.emitRemark() << "ttgir-sched: would M-split this dot into "
                           << plan.numPartitions << " (blockM=" << plan.blockM
                           << " / ctaTileM=" << plan.ctaTileM
                           << "), but backward walker bailed (" << modeSuffix
                           << ")";
          continue;
        }
        unsigned producerCount = schemeOpt->ops.size();
        loopTotalProducerOps += producerCount;

        if (!extendSliceForwardFromDot(plan, *schemeOpt)) {
          ++loopFwdInfeasible;
          ++totalFwdInfeasible;
          dot.emitRemark() << "ttgir-sched: would M-split this dot into "
                           << plan.numPartitions << " (blockM=" << plan.blockM
                           << " / ctaTileM=" << plan.ctaTileM
                           << "), backward found " << producerCount
                           << " producer op(s), forward walker bailed ("
                           << modeSuffix << ")";
          continue;
        }
        unsigned totalOps = schemeOpt->ops.size();
        unsigned userCount = totalOps - producerCount;
        loopTotalUserOps += userCount;

        if (!apply) {
          dot.emitRemark() << "ttgir-sched: would M-split this dot into "
                           << plan.numPartitions << " (blockM=" << plan.blockM
                           << " / ctaTileM=" << plan.ctaTileM
                           << "), co-partitioning " << producerCount
                           << " producer op(s) + " << userCount
                           << " user op(s) (" << modeSuffix << ")";
          continue;
        }

        // APPLY path: mutate IR.
        if (failed(applyMSplit(plan, *schemeOpt))) {
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
          << " MFMA tt.dot op(s); plans " << lw.plans.size() << ", skipped "
          << lw.mfmaDotsSkipped << ", bwd-infeasible " << loopBwdInfeasible
          << ", fwd-infeasible " << loopFwdInfeasible << ", applied "
          << loopApplied << ", co-partition producer-ops "
          << loopTotalProducerOps << " + user-ops " << loopTotalUserOps << " ("
          << modeSuffix << ")";
    }

    module.emitRemark() << "ttgir-sched: visited " << numForOps
                        << " scf.for op(s), " << numCandidateLoops
                        << " candidate(s), " << totalMfmaDots
                        << " MFMA tt.dot op(s), " << totalPlans
                        << " planned M-split(s), " << totalSkipped
                        << " skipped, " << totalBwdInfeasible
                        << " bwd-infeasible, " << totalFwdInfeasible
                        << " fwd-infeasible, " << totalApplied << " applied ("
                        << modeSuffix << ")";
  }
};

} // namespace

} // namespace mlir
