//===- SchedGroupBarrierScheduler.cpp - AMD machine scheduling ------------===//
//
// This pass classifies relevant TTGIR operations by their eventual AMD machine
// instruction class and predicts the instruction count produced by lowering.
//
// Real LDS hazard boundaries do not exist until ModuleMembarAnalysis runs. This
// pass therefore records its decision as module/op attributes. The AMD-to-LLVM
// conversion consumes those attributes after Membar analysis, partitions each
// block at the real barriers, and materializes rocdl.sched.group.barrier plus
// hard scheduling fences in the corresponding regions.
//
//   TTGIR classification -> Membar boundaries -> hint materialization -> LLVM
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUSCHEDGROUPBARRIERSCHEDULER
#define GEN_PASS_DEF_TRITONAMDGPUINTRAWAVEPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using triton::LinearLayout;

namespace {

// LLVM AMDGPU SchedGroupMask bits (AMDGPUIGroupLP.cpp). Same encoding as the
// sched_barrier mask.
enum : int32_t {
  kMaskMFMA = 1 << 3,
  kMaskVMEMRead = 1 << 5,
  kMaskDSRead = 1 << 8,
  kMaskDSWrite = 1 << 9,
  kMaskLDSDMA = 1 << 11,
};

constexpr StringLiteral kIntraWaveMarker = "triton.intra_wave_pipeline.marker";
constexpr StringLiteral kIntraWaveLabel = "triton.intra_wave_pipeline.label";
constexpr StringLiteral kIntraWavePair = "triton.intra_wave_pipeline.pair";
constexpr StringLiteral kIntraWaveAutoInterleave =
    "triton.intra_wave_pipeline.auto_interleave";
constexpr StringLiteral kIntraWaveCoverPolicy =
    "triton.intra_wave_pipeline.cover_policy";

// --- "pretend it is decomposed": machine-instruction counts per TTGIR op -----
// The hint interface counts MACHINE instructions, so every op must be priced
// as the run it becomes. Nothing is split; this is a counting fiction.

// A block-level dot lowers to (M/(instrM*warpsM)) * (N/(instrN*warpsN)) *
// (K/instrK) MFMAs.
//
// K UNIT TRAP: for tt.dot_scaled with an e2m1 (fp4) operand the tensor's K dim
// counts BYTES -- two 4-bit values per i8 -- while instrShape[2] counts logical
// elements. They are NOT the same unit, so K must be doubled. Verified against
// the ISA: 4 dots of 256x128, A=256x128xi8 e2m1, instrShape [16,16,128],
// warpsPerCTA [2,2] -> (256/32)*(128/32)*(256/128) = 64 each = 256 total,
// which is exactly what the loop contains. Without the doubling the model says
// 32 each = 128, so half the MFMAs get no group and clump into a 50-long run.
// (The bug hid for a while because on a K=8192 shape the body has 8 dots and
// 8*32 also came to 256 -- the TOTAL matched while every per-dot count was
// half. Always check the per-op count against the ISA, not just the total.)
static unsigned mfmaCountOf(Operation *op) {
  auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rt || rt.getRank() != 2)
    return 1;
  auto mma =
      dyn_cast_or_null<triton::gpu::AMDMfmaEncodingAttr>(rt.getEncoding());
  if (!mma)
    return 1;
  auto aT = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!aT || aT.getRank() < 2)
    return 1;
  auto instr = mma.getInstrShape();
  auto warps = mma.getWarpsPerCTA();
  if (instr.size() < 3 || warps.size() < 2)
    return 1;
  int64_t k = aT.getShape()[1];
  if (auto ds = dyn_cast<triton::DotScaledOp>(op))
    if (ds.getAElemType() == triton::ScaleDotElemType::E2M1)
      k *= 2; // two fp4 values per byte
  int64_t tileM = std::max<int64_t>(1, (int64_t)instr[0] * warps[0]);
  int64_t tileN = std::max<int64_t>(1, (int64_t)instr[1] * warps[1]);
  int64_t iK = std::max<int64_t>(1, (int64_t)instr[2]);
  int64_t m = (rt.getShape()[0] + tileM - 1) / tileM;
  int64_t n = (rt.getShape()[1] + tileN - 1) / tileN;
  int64_t kc = (k + iK - 1) / iK;
  return (unsigned)std::max<int64_t>(1, m * n * kc);
}

// EXACT per-lane element count, from the layout.
//
// The count must be exact. Measured in sgb_repro, MFMA remainder always
// distributed, only the claim count varied:
//   under-claim by 1 instruction : maxMFMArun 3 -> 64   (a cliff, not a slope;
//                                  1 missing is as bad as 8)
//   over-claim by 1,2,4,8 groups : 6, 7, 10, 18         (gradual decay)
// So `bytes / 16` estimation cannot work -- there is exactly one right answer
// and both errors hurt. getTotalElemsPerThread reads it off the LinearLayout,
// which accounts for replication/broadcast by construction (a dot_operand tile
// is replicated across the warp dim it does not span, which is what made the
// arithmetic version half-count).
static unsigned accessCount(Type ty, unsigned accessBytes) {
  auto rt = dyn_cast<RankedTensorType>(ty);
  if (!rt)
    return 1;
  unsigned elems = triton::gpu::getTotalElemsPerThread(rt);
  unsigned eb =
      std::max<unsigned>(1, rt.getElementType().getIntOrFloatBitWidth() / 8);
  unsigned bytes = std::max(1u, elems * eb);
  return std::max(1u, (bytes + accessBytes - 1) / accessBytes);
}

static unsigned bytesPerThread(Type ty) {
  auto rt = dyn_cast<RankedTensorType>(ty);
  if (!rt)
    return 0;
  unsigned elems = triton::gpu::getTotalElemsPerThread(rt);
  unsigned elemBytes =
      std::max<unsigned>(1, rt.getElementType().getIntOrFloatBitWidth() / 8);
  return elems * elemBytes;
}

static unsigned dsReadCountOf(Operation *op) {
  // Access width depends on which read the backend picks, and they differ by
  // 2x. Confirmed by mapping the ISA back through its .loc directives on the
  // a4w4 body: the 64 dot-operand reads are ds_read_b128 (16 B), while all 8
  // scale reads are ds_read_b64_tr_b8 -- CDNA4's transposed read, which is
  // HALF width. A 256x8 scale tile holds 16 elems/lane, so it costs 2 of them,
  // not 1. Getting those 2 wrong is not a rounding error: leaving a single
  // instruction unclaimed takes maxMFMArun from 3 to 64 (sgb_repro).
  Type ty = op->getResult(0).getType();
  unsigned accessBytes = 16; // ds_read_b128
  if (auto rt = dyn_cast<RankedTensorType>(ty)) {
    // On gfx950, the B operand of MFMA is lowered through the transposed
    // ds_read_b64 path.  D115508145 tested "not a dot operand" here, which is
    // backwards for the target BMM and under-counts every B local_load by 2x.
    if (auto dot = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(
            rt.getEncoding()))
      if (dot.getOpIdx() == 1)
        accessBytes = 8;
  }
  return accessCount(ty, accessBytes);
}

// Mirror the generic local-store lowering's vectorisation calculation.  The
// instruction count is not bytes/16: a padded register->LDS layout can force
// scalar ds_write_b16 even when the source tensor holds many contiguous bytes.
static unsigned dsWriteCountOf(Operation *op) {
  auto store = dyn_cast<triton::gpu::LocalStoreOp>(op);
  if (!store)
    return 1;
  auto regTy = dyn_cast<RankedTensorType>(store.getSrc().getType());
  auto memTy = dyn_cast<triton::gpu::MemDescType>(store.getDst().getType());
  if (!regTy || !memTy)
    return 1;

  LinearLayout regLayout = triton::gpu::toLinearLayout(regTy);
  LinearLayout cvt = LinearLayout::empty();
  if (triton::gpu::isPaddedEncoding(memTy.getEncoding())) {
    cvt = triton::invertAndComposeBlockLocal(
        triton::gpu::paddedLinearLayout(memTy), regLayout);
  } else {
    LinearLayout sharedLayout = triton::gpu::toLinearLayout(memTy);
    if (regLayout.isModular()) {
      auto allocShape = triton::gpu::getAllocationShapePerCTA(memTy);
      sharedLayout =
          triton::gpu::toLinearLayout(allocShape, memTy.getEncoding());
      SmallVector<std::pair<StringAttr, int32_t>> paddedOutDims;
      for (StringAttr dim : regLayout.getOutDimNames())
        paddedOutDims.push_back({dim, sharedLayout.getOutDimSize(dim)});
      regLayout = LinearLayout(regLayout.getBases(), paddedOutDims,
                               /*requireSurjective=*/false);
    }
    cvt = triton::invertAndComposeBlockLocal(sharedLayout, regLayout);
  }
  cvt = triton::actionRemoveBroadcastedRegs(cvt).apply(cvt);
  std::optional<int> maxVec;
  if (triton::gpu::isPaddedEncoding(memTy.getEncoding()))
    maxVec = triton::gpu::getMinInterval(memTy.getEncoding());
  unsigned bitWidth = memTy.getElementType().getIntOrFloatBitWidth();
  auto [elemsPerVec, permutation] =
      triton::largestVectorisation(op->getContext(), cvt, bitWidth, maxVec);
  (void)permutation;
  auto kReg = StringAttr::get(op->getContext(), "register");
  return std::max(1, cvt.getInDimSize(kReg) / elemsPerVec);
}

static unsigned clampVectorSize(unsigned vec, RankedTensorType tensorTy) {
  if (vec <= 1)
    return vec;
  if (!llvm::isPowerOf2_32(vec))
    vec = 1u << llvm::Log2_32(vec);
  if (auto blocked =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(tensorTy.getEncoding())) {
    auto order = triton::gpu::getOrder(tensorTy);
    unsigned sizePerThread = blocked.getSizePerThread()[order[0]];
    if (sizePerThread && !llvm::isPowerOf2_32(sizePerThread))
      vec = std::min(vec, sizePerThread & (0u - sizePerThread));
  }
  return vec;
}

// Match BufferLoadOpConversion / BufferLoadToLocalOpConversion: axis
// contiguity and base-pointer alignment determine the actual buffer_load width.
struct VMEMPrice {
  unsigned count;
  unsigned accessBytes;
};

static VMEMPrice
priceBufferAccess(Value ptr, Value offset, unsigned contiguityHint,
                  triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  auto offsetTy = dyn_cast<RankedTensorType>(offset.getType());
  if (!offsetTy)
    return {/*count=*/1, /*accessBytes=*/16};
  unsigned elemBits = triton::getPointeeBitWidth(ptr.getType());
  unsigned elemBytes = std::max(1u, elemBits / 8);
  unsigned contiguity = axisInfo.getContiguity(offset, elemBits);
  if (auto *info = axisInfo.getAxisInfo(ptr))
    contiguity = std::min(
        contiguity, std::max(1u, static_cast<unsigned>(
                                     info->getDivisibility(0) / elemBytes)));

  auto linear = triton::gpu::toLinearLayout(offsetTy);
  auto linearAttr = triton::gpu::LinearEncodingAttr::get(offsetTy.getContext(),
                                                         std::move(linear));
  auto order = triton::gpu::getOrder(offsetTy);
  auto perThread = linearAttr.getContigPerThread();
  contiguity = std::min(contiguity, perThread[order[0]]);
  unsigned vec = std::min(128u / elemBits, contiguity);
  vec = clampVectorSize(vec, offsetTy);
  vec = std::max(vec, contiguityHint);
  unsigned elems = triton::gpu::getTotalElemsPerThread(offsetTy);
  return {/*count=*/std::max(1u, (elems + vec - 1) / vec),
          /*accessBytes=*/std::max(1u, vec * elemBytes)};
}
static VMEMPrice priceVMEM(Operation *op,
                           triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  if (auto load = dyn_cast<triton::amdgpu::BufferLoadOp>(op))
    return priceBufferAccess(load.getPtr(), load.getOffsets(),
                             load.getContiguity(), axisInfo);
  if (auto load = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
    VMEMPrice price = priceBufferAccess(load.getPtr(), load.getOffsets(),
                                        load.getContiguity(), axisInfo);
    auto dstTy = load.getDest().getType();
    auto padded =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    auto target = triton::amdgpu::TargetFeatures::fromModuleOp(
        op->getParentOfType<ModuleOp>());
    if (padded && !target.supportsDirectToLdsScatter()) {
      // Direct-to-LDS lowering clamps a padded destination's vector width so
      // padding is inserted only at wave boundaries. Mirror that clamp here;
      // otherwise one TTGIR copy may be priced as one dwordx4 even though it
      // lowers to four buffer_load_dword instructions, and the requested MFMA
      // cover cannot be materialized by the machine scheduler.
      unsigned elemBits = triton::getPointeeBitWidth(load.getPtr().getType());
      unsigned elemBytes = std::max(1u, elemBits / 8);
      unsigned requestedVec = std::max(1u, price.accessBytes / elemBytes);
      unsigned paddedVec = padded.getMinInterval() / target.getWarpSize();
      unsigned loweredVec = std::max(1u, std::min(requestedVec, paddedVec));
      auto offsetTy = cast<RankedTensorType>(load.getOffsets().getType());
      unsigned elems = triton::gpu::getTotalElemsPerThread(offsetTy);
      price.count = std::max(1u, (elems + loweredVec - 1) / loweredVec);
      price.accessBytes = loweredVec * elemBytes;
    }
    return price;
  }
  // The tile is the DESTINATION memdesc for buffer_load_to_local; operand(0)
  // is the base pointer.
  for (Value r : op->getResults()) {
    if (auto md = dyn_cast<triton::gpu::MemDescType>(r.getType())) {
      unsigned eb = std::max<unsigned>(
          1, md.getElementType().getIntOrFloatBitWidth() / 8);
      int64_t e = 1;
      for (int64_t d : md.getShape())
        e *= d;
      return {/*count=*/std::max<unsigned>(1, (unsigned)((e * eb) / 256) / 16),
              /*accessBytes=*/16};
    }
    if (isa<RankedTensorType>(r.getType()))
      return {/*count=*/accessCount(r.getType(), 16), /*accessBytes=*/16};
  }
  for (Value v : op->getOperands())
    if (auto md = dyn_cast<triton::gpu::MemDescType>(v.getType())) {
      unsigned eb = std::max<unsigned>(
          1, md.getElementType().getIntOrFloatBitWidth() / 8);
      int64_t e = 1;
      for (int64_t d : md.getShape())
        e *= d;
      return {/*count=*/std::max<unsigned>(1, (unsigned)((e * eb) / 256) / 16),
              /*accessBytes=*/16};
    }
  return {/*count=*/1, /*accessBytes=*/16};
}

struct IntraWaveStage {
  int64_t pair;
  StringAttr label;
  bool autoInterleave;
  bool proportionalCover;
  Operation *begin;
  Operation *end;
  SmallVector<Operation *> ops;
};

struct IntraWaveStageCounts {
  unsigned mfma = 0;
  unsigned memory = 0;
  SmallVector<std::pair<int32_t, unsigned>> memoryRuns;
};

struct IntraWaveOpChunk {
  SmallVector<Operation *> ops;
  int32_t machineMask = 0;
  unsigned machineCount = 0;
  unsigned serviceCycles = 0;
};

static SmallVector<unsigned> distributeUniformly(unsigned itemCount,
                                                 unsigned budget) {
  assert(itemCount && "cannot distribute a budget over no items");
  SmallVector<unsigned> shares;
  shares.reserve(itemCount);
  for (unsigned index = 0; index < itemCount; ++index) {
    uint64_t begin = static_cast<uint64_t>(index) * budget;
    uint64_t end = static_cast<uint64_t>(index + 1) * budget;
    shares.push_back(end / itemCount - begin / itemCount);
  }
  return shares;
}

static void materializeIntraWaveMachinePair(OpBuilder &builder, Location loc,
                                            int32_t memoryMask,
                                            unsigned computeCount, int64_t pair,
                                            unsigned syncId) {
  auto memoryGroup = ROCDL::SchedGroupBarrier::create(
      builder, loc, static_cast<ROCDL::SchedGroupMask>(memoryMask), 1, syncId);
  memoryGroup->setAttr("triton.intra_wave_pipeline.pair",
                       builder.getI32IntegerAttr(pair));
  if (!computeCount)
    return;
  auto computeGroup = ROCDL::SchedGroupBarrier::create(
      builder, loc, ROCDL::SchedGroupMask::mfma_wmma, computeCount, syncId);
  computeGroup->setAttr("triton.intra_wave_pipeline.pair",
                        builder.getI32IntegerAttr(pair));
}

static void materializeIntraWaveChunkPair(OpBuilder &builder, Location loc,
                                          const IntraWaveOpChunk &memoryChunk,
                                          unsigned computeCount, int64_t pair,
                                          unsigned syncId) {
  assert(memoryChunk.machineCount && "memory chunk has no machine operation");
  unsigned quotient = computeCount / memoryChunk.machineCount;
  unsigned remainder = computeCount % memoryChunk.machineCount;
  for (unsigned index = 0; index < memoryChunk.machineCount; ++index)
    materializeIntraWaveMachinePair(builder, loc, memoryChunk.machineMask,
                                    quotient + (index < remainder ? 1 : 0),
                                    pair, syncId);
}

// Estimate issue-pipeline occupancy, not end-to-end result latency.  The
// resulting values are used only to divide a fixed, user-selected MFMA budget
// inside one already-proven-independent window.
//
// The ratios mirror the useful part of the gfx950 Gluon LLIR scheduler:
//   * MI16 and MI32 MFMAs occupy 16 and 32 cycles respectively;
//   * LDS traffic consumes cycles proportional to bytes on the shared LDS
//     issue path;
//   * a 128-bit VMEM instruction receives two MI16-equivalent issue slots,
//     scaled down with its actual vector width. This intentionally models
//     issue occupancy rather than full memory latency: assigning the latter
//     here starts every future load too early and raises register pressure.
// Exact dependencies and the window boundary remain hard constraints, so a
// cost-model error changes only the distribution inside the window.
static unsigned
intraWaveServiceCycles(Operation *op, int32_t mask, unsigned machineCount,
                       triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  if (mask == kMaskMFMA) {
    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy)
      return 0;
    auto mma = dyn_cast_or_null<triton::gpu::AMDMfmaEncodingAttr>(
        resultTy.getEncoding());
    if (!mma || mma.getInstrShape().empty())
      return 0;
    unsigned instrM = mma.getInstrShape()[0];
    unsigned cycles = instrM == 32 ? 32 : instrM == 16 ? 16 : 0;
    return cycles * machineCount;
  }

  if (mask == kMaskDSRead)
    return bytesPerThread(op->getResult(0).getType());
  if (mask == kMaskDSWrite) {
    auto store = cast<triton::gpu::LocalStoreOp>(op);
    return bytesPerThread(store.getSrc().getType());
  }
  if (mask == kMaskVMEMRead || mask == kMaskLDSDMA) {
    VMEMPrice price = priceVMEM(op, axisInfo);
    // A dwordx4 load occupies two MI16-equivalent issue slots, with narrower
    // accesses scaled proportionally. Preserve that ratio for MI32 by
    // normalizing below.
    return price.count * std::max(4u, price.accessBytes) * 2;
  }
  return 0;
}

// Paired stages describe independent work that may be interleaved.  Reject a
// dataflow edge in either direction instead of relying on the machine
// scheduler to discover that the requested group order is impossible.  This
// direct-edge check is sufficient because every producer and consumer inside
// either region is present in its stage set: a transitive path must cross the
// stage boundary through one such edge. Control flow is already rejected.
static bool intraWaveStagesAreIndependent(const IntraWaveStage &first,
                                          const IntraWaveStage &second) {
  llvm::DenseSet<Operation *> firstOps(first.ops.begin(), first.ops.end());
  llvm::DenseSet<Operation *> secondOps(second.ops.begin(), second.ops.end());
  auto dependsOn = [](ArrayRef<Operation *> consumers,
                      const llvm::DenseSet<Operation *> &producers) {
    return llvm::any_of(consumers, [&](Operation *consumer) {
      return llvm::any_of(consumer->getOperands(), [&](Value operand) {
        Operation *producer = operand.getDefiningOp();
        return producer && producers.contains(producer);
      });
    });
  };
  return !dependsOn(second.ops, firstOps) && !dependsOn(first.ops, secondOps);
}

static bool isIntraWaveMarker(Operation *op) {
  return op->hasAttr(kIntraWaveMarker);
}

static void clearIntraWaveMarker(Operation *op) {
  op->removeAttr(kIntraWaveMarker);
  op->removeAttr(kIntraWaveLabel);
  op->removeAttr(kIntraWavePair);
  op->removeAttr(kIntraWaveAutoInterleave);
  op->removeAttr(kIntraWaveCoverPolicy);
}

static std::optional<std::pair<int32_t, unsigned>>
priceIntraWaveOp(Operation *op, triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  if (isa<triton::DotOp, triton::DotScaledOp, triton::amdgpu::ScheduledMfmaOp>(
          op))
    return std::pair<int32_t, unsigned>{kMaskMFMA, mfmaCountOf(op)};
  if (isa<triton::gpu::LocalLoadOp>(op))
    return std::pair<int32_t, unsigned>{kMaskDSRead, dsReadCountOf(op)};
  if (isa<triton::gpu::LocalStoreOp>(op))
    return std::pair<int32_t, unsigned>{kMaskDSWrite, dsWriteCountOf(op)};
  if (isa<triton::gpu::AsyncCopyGlobalToLocalOp,
          triton::amdgpu::BufferLoadToLocalOp>(op)) {
    VMEMPrice price = priceVMEM(op, axisInfo);
    return std::pair<int32_t, unsigned>{kMaskLDSDMA, price.count};
  }
  if (isa<triton::amdgpu::BufferLoadOp, triton::LoadOp>(op)) {
    VMEMPrice price = priceVMEM(op, axisInfo);
    return std::pair<int32_t, unsigned>{kMaskVMEMRead, price.count};
  }
  return std::nullopt;
}

static LogicalResult
collectIntraWaveStages(Block *block, SmallVectorImpl<IntraWaveStage> &stages) {
  std::optional<IntraWaveStage> active;
  for (Operation &op : *block) {
    auto marker = op.getAttrOfType<StringAttr>(kIntraWaveMarker);
    if (!marker) {
      if (active)
        active->ops.push_back(&op);
      continue;
    }

    auto label = op.getAttrOfType<StringAttr>(kIntraWaveLabel);
    auto pair = op.getAttrOfType<IntegerAttr>(kIntraWavePair);
    if (!label || !pair) {
      op.emitError("malformed intra-wave pipeline marker");
      return failure();
    }
    auto coverPolicy = op.getAttrOfType<StringAttr>(kIntraWaveCoverPolicy);
    bool proportionalCover = false;
    if (coverPolicy) {
      if (coverPolicy.getValue() != "proportional") {
        op.emitError("unknown intra-wave cover policy");
        return failure();
      }
      proportionalCover = true;
    }
    bool autoInterleave = op.hasAttr(kIntraWaveAutoInterleave);
    if (proportionalCover && !autoInterleave) {
      op.emitError(
          "a non-default cover policy requires automatic interleaving");
      return failure();
    }
    if (marker.getValue() == "begin") {
      if (active) {
        op.emitError("nested intra-wave pipeline stages are not supported");
        return failure();
      }
      active.emplace(IntraWaveStage{pair.getInt(),
                                    label,
                                    autoInterleave,
                                    proportionalCover,
                                    &op,
                                    nullptr,
                                    {}});
      continue;
    }
    if (marker.getValue() != "end") {
      op.emitError("unknown intra-wave pipeline marker kind");
      return failure();
    }
    if (!active || active->pair != pair.getInt() || active->label != label) {
      op.emitError("intra-wave pipeline end does not match its begin marker");
      return failure();
    }
    if (active->autoInterleave != autoInterleave) {
      op.emitError("intra-wave stage begin/end markers disagree on automatic "
                   "interleaving");
      return failure();
    }
    if (active->proportionalCover != proportionalCover) {
      op.emitError(
          "intra-wave stage begin/end markers disagree on cover policy");
      return failure();
    }
    active->end = &op;
    stages.push_back(std::move(*active));
    active.reset();
  }
  if (active) {
    active->begin->emitError("unterminated intra-wave pipeline stage");
    return failure();
  }
  return success();
}

static LogicalResult
countIntraWaveStage(const IntraWaveStage &stage,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfo,
                    IntraWaveStageCounts &counts) {
  for (Operation *op : stage.ops) {
    if (op->getNumRegions() != 0) {
      op->emitError("control flow inside an intra-wave pipeline stage is not "
                    "supported");
      return failure();
    }
    auto priced = priceIntraWaveOp(op, axisInfo);
    if (!priced) {
      if (!isMemoryEffectFree(op) &&
          !isa<triton::gpu::AsyncCommitGroupOp>(op)) {
        op->emitError("unsupported side effect inside an intra-wave pipeline "
                      "stage");
        return failure();
      }
      continue;
    }
    auto [mask, count] = *priced;
    if (mask == kMaskMFMA) {
      counts.mfma += count;
      continue;
    }
    counts.memory += count;
    if (!counts.memoryRuns.empty() && counts.memoryRuns.back().first == mask)
      counts.memoryRuns.back().second += count;
    else
      counts.memoryRuns.push_back({mask, count});
  }
  if (counts.mfma && counts.memory && !stage.autoInterleave) {
    stage.begin->emitError(
        "one intra-wave stage can mix MFMA and memory operations only with "
        "automatic interleaving");
    return failure();
  }
  return success();
}

// Partition one mixed automatic window into independent memory and compute
// streams. Priced operations are the anchors. Pure operations that consume an
// anchor follow that stream; pure setup that feeds exactly one stream moves
// with it. Setup shared by both streams remains at the source boundary. Any
// cross-stream dataflow fails closed instead of turning a scheduling hint into
// an illegal reorder.
static LogicalResult
partitionMixedIntraWaveStage(const IntraWaveStage &mixed,
                             triton::AMD::ModuleAxisInfoAnalysis &axisInfo,
                             IntraWaveStage &memory, IntraWaveStage &compute) {
  constexpr unsigned kMemoryStream = 1;
  constexpr unsigned kComputeStream = 2;
  llvm::DenseMap<Operation *, unsigned> positions;
  for (auto [index, op] : llvm::enumerate(mixed.ops)) {
    // Writes and asynchronous commits carry ordering through memory rather
    // than SSA. Keep those in the explicit two-region API until alias and
    // async-token dependence can be proven; accepting them here can move an
    // LDS overwrite ahead of the MFMA that consumes the old stage.
    if (isa<triton::gpu::LocalStoreOp, triton::gpu::AsyncCopyGlobalToLocalOp,
            triton::amdgpu::BufferLoadToLocalOp,
            triton::gpu::AsyncCommitGroupOp>(op)) {
      op->emitError("one-region automatic intra-wave interleaving supports "
                    "only read-only memory operations");
      return failure();
    }
    if (auto load = dyn_cast<triton::LoadOp>(op);
        load && load.getIsVolatile()) {
      op->emitError("one-region automatic intra-wave interleaving does not "
                    "support volatile loads");
      return failure();
    }
    positions[op] = index;
  }

  SmallVector<unsigned> upstream(mixed.ops.size(), 0);
  SmallVector<unsigned> downstream(mixed.ops.size(), 0);
  for (auto [index, op] : llvm::enumerate(mixed.ops)) {
    if (auto priced = priceIntraWaveOp(op, axisInfo)) {
      unsigned stream =
          priced->first == kMaskMFMA ? kComputeStream : kMemoryStream;
      upstream[index] = stream;
      downstream[index] = stream;
    }
  }

  // Propagate anchor reachability backward to pure address/layout setup.
  for (int64_t index = mixed.ops.size() - 1; index >= 0; --index) {
    Operation *op = mixed.ops[index];
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        auto position = positions.find(use.getOwner());
        if (position != positions.end())
          upstream[index] |= upstream[position->second];
      }
    }
  }

  // Propagate anchor reachability forward to result transforms and users.
  for (auto [index, op] : llvm::enumerate(mixed.ops)) {
    for (Value operand : op->getOperands()) {
      auto position = positions.find(operand.getDefiningOp());
      if (position != positions.end())
        downstream[index] |= downstream[position->second];
    }
  }

  memory = mixed;
  compute = mixed;
  memory.ops.clear();
  compute.ops.clear();
  for (auto [index, op] : llvm::enumerate(mixed.ops)) {
    unsigned stream = upstream[index] | downstream[index];
    bool sharedSetup = upstream[index] == (kMemoryStream | kComputeStream) &&
                       downstream[index] == 0;
    if (stream == (kMemoryStream | kComputeStream) && !sharedSetup) {
      op->emitError("automatic intra-wave window contains dataflow between "
                    "its memory and MFMA streams");
      return failure();
    }
    if (stream == kMemoryStream)
      memory.ops.push_back(op);
    else if (stream == kComputeStream)
      compute.ops.push_back(op);
    // Pure setup feeding both streams has upstream==3 and downstream==0. It
    // is intentionally kept in place rather than cloned or assigned above.
  }
  return success();
}

// Split a stage at the operations that lower to the requested machine class.
// Assign pure setup/transform operations by SSA reachability rather than only
// by source adjacency. Earlier canonicalization may legally hoist the address
// calculations for every local_load ahead of all loads; attaching that whole
// prefix to the first load would recreate a long-lived address burst. An op
// with an anchor predecessor stays with that producer; otherwise it follows
// the earliest anchor it feeds. Preserving order inside each resulting chunk
// makes the later merge a stable topological reorder.
static SmallVector<IntraWaveOpChunk>
chunkIntraWaveStage(const IntraWaveStage &stage, bool computeStage,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  llvm::DenseMap<Operation *, unsigned> positions;
  for (auto [index, op] : llvm::enumerate(stage.ops))
    positions[op] = index;

  SmallVector<IntraWaveOpChunk> chunks;
  SmallVector<std::optional<unsigned>> anchors(stage.ops.size());
  for (auto [index, op] : llvm::enumerate(stage.ops)) {
    auto priced = priceIntraWaveOp(op, axisInfo);
    if (!priced)
      continue;
    auto [mask, count] = *priced;
    if ((mask == kMaskMFMA) != computeStage)
      continue;

    IntraWaveOpChunk chunk;
    chunk.machineMask = mask;
    chunk.machineCount = count;
    chunk.serviceCycles = intraWaveServiceCycles(op, mask, count, axisInfo);
    anchors[index] = chunks.size();
    chunks.push_back(std::move(chunk));
  }

  assert(!chunks.empty() && "a validated stage must have a priced anchor");
  SmallVector<std::optional<unsigned>> lowerBounds(stage.ops.size());
  SmallVector<std::optional<unsigned>> upperBounds(stage.ops.size());

  // The latest anchor in an operation's transitive operand slice is its
  // earliest legal chunk.
  for (auto [index, op] : llvm::enumerate(stage.ops)) {
    if (anchors[index])
      lowerBounds[index] = anchors[index];
    for (Value operand : op->getOperands()) {
      auto position = positions.find(operand.getDefiningOp());
      if (position == positions.end() || !lowerBounds[position->second])
        continue;
      unsigned lower = *lowerBounds[position->second];
      if (!lowerBounds[index] || lower > *lowerBounds[index])
        lowerBounds[index] = lower;
    }
  }

  // The earliest anchor transitively consuming an operation is its latest
  // useful chunk. This recovers per-load address setup even when that setup
  // was hoisted ahead of every load in the coarse source region.
  for (int64_t index = stage.ops.size() - 1; index >= 0; --index) {
    Operation *op = stage.ops[index];
    if (anchors[index])
      upperBounds[index] = anchors[index];
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        auto position = positions.find(use.getOwner());
        if (position == positions.end() || !upperBounds[position->second])
          continue;
        unsigned upper = *upperBounds[position->second];
        if (!upperBounds[index] || upper < *upperBounds[index])
          upperBounds[index] = upper;
      }
    }
  }

  unsigned precedingAnchor = 0;
  bool hasPrecedingAnchor = false;
  for (auto [index, op] : llvm::enumerate(stage.ops)) {
    unsigned chunkIndex;
    if (anchors[index]) {
      chunkIndex = *anchors[index];
      precedingAnchor = chunkIndex;
      hasPrecedingAnchor = true;
    } else if (lowerBounds[index]) {
      chunkIndex = *lowerBounds[index];
      assert((!upperBounds[index] || chunkIndex <= *upperBounds[index]) &&
             "intra-wave stage is not topologically chunkable");
    } else if (upperBounds[index]) {
      chunkIndex = *upperBounds[index];
    } else {
      // Dead or region-independent pure setup is kept near its source
      // position. Such operations do not constrain the cross-stage merge.
      chunkIndex = hasPrecedingAnchor ? precedingAnchor : 0;
    }
    chunks[chunkIndex].ops.push_back(op);
  }
  return chunks;
}

// Preserve source-level operand granularity first, then let each source anchor
// subdivide its cover according to the number of machine instructions it
// lowers to. This avoids giving a B operand twice the register-lifetime budget
// of an A operand merely because the B read uses two narrower DS instructions.
static SmallVector<unsigned>
computeIntraWaveChunkCovers(ArrayRef<IntraWaveOpChunk> memoryChunks,
                            ArrayRef<IntraWaveOpChunk> computeChunks,
                            unsigned mfmaCount, bool proportionalCover) {
  SmallVector<unsigned> covers;
  covers.reserve(memoryChunks.size());

  // A homogeneous source window retains the original equal-per-source policy.
  // Besides preserving existing schedules, this is the right abstraction for
  // repeated fragments of one operand: a source local_load lowering to two
  // narrow DS reads should not receive twice the lifetime budget of a source
  // local_load lowering to one wide read.
  bool mixedPipelines = llvm::any_of(memoryChunks, [&](const auto &chunk) {
    return chunk.machineMask != memoryChunks.front().machineMask;
  });
  if (!mixedPipelines)
    return distributeUniformly(memoryChunks.size(), mfmaCount);

  unsigned mfmaCycles = 0;
  for (const IntraWaveOpChunk &chunk : computeChunks) {
    if (!chunk.machineCount || !chunk.serviceCycles)
      continue;
    mfmaCycles = chunk.serviceCycles / chunk.machineCount;
    break;
  }
  if (!mfmaCycles) {
    // Unknown instruction shape: fail back to the previously validated equal
    // distribution instead of applying an uncalibrated target model.
    return distributeUniformly(memoryChunks.size(), mfmaCount);
  }

  // Convert each memory anchor's issue cost into an MFMA demand. Reads need
  // distance before their result is consumed, while writes also need issue
  // bandwidth and must not be collapsed into the following global load. For
  // MI16 this gives a one-MFMA slot to a 16-byte LDS write and about four to a
  // buffer_load_dwordx4 before proportional apportionment.
  //
  // When demand exceeds the available compute, reserve one cover per read
  // whenever the budget permits, then prefix-apportion the remainder. When
  // demand fits, fill the complete window as well: clustering that compute at
  // the tail would issue every future load early and lengthen all of their live
  // ranges at once. The default distributes surplus evenly; proportional mode
  // instead scales the target issue-cost ratios across the full window.
  SmallVector<unsigned> demands;
  demands.reserve(memoryChunks.size());
  uint64_t totalDemand = 0;
  for (const IntraWaveOpChunk &chunk : memoryChunks) {
    unsigned demand = llvm::divideCeil(chunk.serviceCycles, mfmaCycles);
    demands.push_back(demand);
    totalDemand += demand;
  }
  if (!totalDemand) {
    return distributeUniformly(memoryChunks.size(), mfmaCount);
  }

  unsigned activeAnchors =
      llvm::count_if(demands, [](unsigned demand) { return demand != 0; });
  if (mfmaCount < activeAnchors) {
    // With fewer compute instructions than memory anchors, latency-weighted
    // apportionment starves the cheaper anchors (usually LDS writes) and
    // clusters them ahead of the longer-latency VMEM stream. That increases
    // the number of simultaneously live prefetched operands. Spread the
    // scarce cover across source anchors first; zero-cover anchors are then
    // bundled with their nearest covered neighbor below. Latency weighting
    // becomes useful only after every active source anchor can receive one
    // cover instruction.
    SmallVector<unsigned> activeShares =
        distributeUniformly(activeAnchors, mfmaCount);
    covers.assign(memoryChunks.size(), 0);
    unsigned activeIndex = 0;
    for (unsigned index = 0; index < demands.size(); ++index)
      if (demands[index])
        covers[index] = activeShares[activeIndex++];
    return covers;
  }
  if (totalDemand <= mfmaCount) {
    if (proportionalCover) {
      uint64_t prefixDemand = 0;
      for (unsigned demand : demands) {
        uint64_t begin = prefixDemand * mfmaCount / totalDemand;
        prefixDemand += demand;
        uint64_t end = prefixDemand * mfmaCount / totalDemand;
        covers.push_back(end - begin);
      }
      return covers;
    }
    covers = demands;
    SmallVector<unsigned> surplus =
        distributeUniformly(activeAnchors, mfmaCount - totalDemand);
    unsigned activeIndex = 0;
    for (unsigned index = 0; index < demands.size(); ++index)
      if (demands[index])
        covers[index] += surplus[activeIndex++];
    return covers;
  }

  uint64_t budget = mfmaCount;
  if (budget >= activeAnchors) {
    // Reserve one cover for every active memory operation before distributing
    // the remaining latency budget. Pure proportional apportionment can round
    // a short operation down to zero beside a wider one. Reserving future
    // shares also prevents a greedy walk from starving the tail of the window.
    covers.assign(memoryChunks.size(), 0);
    for (auto [index, demand] : llvm::enumerate(demands))
      covers[index] = demand != 0;

    uint64_t remainingBudget = budget - activeAnchors;
    uint64_t remainingDemand = totalDemand - activeAnchors;
    if (!remainingBudget || !remainingDemand)
      return covers;

    uint64_t prefixDemand = 0;
    for (unsigned index = 0; index < demands.size(); ++index) {
      uint64_t demand = demands[index] - (demands[index] != 0);
      uint64_t begin = prefixDemand * remainingBudget / remainingDemand;
      prefixDemand += demand;
      uint64_t end = prefixDemand * remainingBudget / remainingDemand;
      covers[index] += end - begin;
    }
    return covers;
  }

  uint64_t prefixDemand = 0;
  for (unsigned index = 0; index < memoryChunks.size(); ++index) {
    uint64_t begin = prefixDemand * budget / totalDemand;
    prefixDemand += demands[index];
    uint64_t end = prefixDemand * budget / totalDemand;
    covers.push_back(end - begin);
  }
  return covers;
}

// Materialize the same fine-grained order that users previously had to spell
// with one source pair per load.  The two stage orders remain unchanged; only
// their independent chunks are merged.  Reordering before LLVM lowering is
// important because scheduling-group constraints alone do not shorten the
// virtual-register live ranges seen by register allocation.
static void interleaveIntraWaveStages(ArrayRef<IntraWaveOpChunk> memoryChunks,
                                      ArrayRef<IntraWaveOpChunk> computeChunks,
                                      ArrayRef<unsigned> chunkCovers,
                                      Operation *insertBefore, int64_t pair,
                                      unsigned &nextSyncId) {
  unsigned desiredComputeCount = 0;
  unsigned scheduledComputeCount = 0;
  unsigned computeChunkIndex = 0;
  SmallVector<const IntraWaveOpChunk *> pendingMemory;

  auto moveChunk = [&](const IntraWaveOpChunk &chunk) {
    for (Operation *op : chunk.ops)
      op->moveBefore(insertBefore);
  };
  // All chunks belong to one user-selected, dependency-proven scheduling
  // window. Keep them in one scheduling-group pipeline so LLVM can realize
  // the complete memory/compute cadence without turning every source anchor
  // into a separate hard completion boundary. The outer source markers remain
  // the hard boundary of the window.
  unsigned syncId = nextSyncId++;
  for (unsigned memoryIndex = 0; memoryIndex < memoryChunks.size();
       ++memoryIndex) {
    const IntraWaveOpChunk &memoryChunk = memoryChunks[memoryIndex];
    unsigned cover = chunkCovers[memoryIndex];
    moveChunk(memoryChunk);
    desiredComputeCount += cover;
    while (computeChunkIndex < computeChunks.size() &&
           scheduledComputeCount < desiredComputeCount) {
      const IntraWaveOpChunk &computeChunk = computeChunks[computeChunkIndex++];
      moveChunk(computeChunk);
      scheduledComputeCount += computeChunk.machineCount;
    }
    // A zero-cover memory anchor belongs to the next covered anchor. Do not
    // insert an artificial scheduling boundary between them: when there are
    // fewer MFMAs than source loads this reconstructs a multi-load/one-MFMA
    // window, and for publication pipelines it can pair an LDS write with the
    // following global read.
    bool hasFollowingMemory = memoryIndex + 1 < memoryChunks.size();
    if (!cover && hasFollowingMemory) {
      pendingMemory.push_back(&memoryChunk);
      continue;
    }

    // Materialize each reconstructed memory/compute bundle where its
    // operations now live. A shared sync ID preserves their ordered pipeline;
    // the bundles do not need an additional full scheduling barrier between
    // adjacent anchors inside the same proven-independent window.
    OpBuilder builder(insertBefore);
    for (const IntraWaveOpChunk *memory : pendingMemory)
      materializeIntraWaveChunkPair(builder, insertBefore->getLoc(), *memory,
                                    /*computeCount=*/0, pair, syncId);
    pendingMemory.clear();
    materializeIntraWaveChunkPair(builder, insertBefore->getLoc(), memoryChunk,
                                  cover, pair, syncId);
  }
  assert(pendingMemory.empty() && "unterminated intra-wave memory bundle");
  while (computeChunkIndex < computeChunks.size())
    moveChunk(computeChunks[computeChunkIndex++]);
}

static void materializeAutomaticIntraWaveWindow(
    const IntraWaveStage &memoryStage, const IntraWaveStage &computeStage,
    const IntraWaveStageCounts &computeCounts, Operation *insertBefore,
    int64_t pair, unsigned &nextSyncId,
    triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  SmallVector<IntraWaveOpChunk> memoryChunks =
      chunkIntraWaveStage(memoryStage, /*computeStage=*/false, axisInfo);
  SmallVector<IntraWaveOpChunk> computeChunks =
      chunkIntraWaveStage(computeStage, /*computeStage=*/true, axisInfo);
  SmallVector<unsigned> chunkCovers = computeIntraWaveChunkCovers(
      memoryChunks, computeChunks, computeCounts.mfma,
      memoryStage.proportionalCover);

  // A one-anchor window is already at source granularity. Larger windows are
  // reconstructed as fine memory/compute chunks before register allocation.
  if (memoryChunks.size() > 1 && computeChunks.size() > 1) {
    interleaveIntraWaveStages(memoryChunks, computeChunks, chunkCovers,
                              insertBefore, pair, nextSyncId);
    return;
  }

  OpBuilder builder(insertBefore);
  for (auto [memoryChunk, chunkCover] : llvm::zip(memoryChunks, chunkCovers)) {
    unsigned syncId = nextSyncId++;
    materializeIntraWaveChunkPair(builder, insertBefore->getLoc(), memoryChunk,
                                  chunkCover, pair, syncId);
  }
}

static LogicalResult
materializeIntraWaveWindows(ModuleOp mod,
                            triton::AMD::ModuleAxisInfoAnalysis &axisInfo) {
  SmallVector<Block *> markedBlocks;
  mod.walk([&](Operation *op) {
    if (!isIntraWaveMarker(op))
      return;
    Block *block = op->getBlock();
    if (!llvm::is_contained(markedBlocks, block))
      markedBlocks.push_back(block);
  });

  unsigned nextSyncId = 1;
  for (Block *block : markedBlocks) {
    SmallVector<IntraWaveStage> stages;
    if (failed(collectIntraWaveStages(block, stages)))
      return failure();

    for (unsigned i = 0; i < stages.size();) {
      IntraWaveStage &first = stages[i];
      IntraWaveStageCounts firstCounts;
      if (failed(countIntraWaveStage(first, axisInfo, firstCounts)))
        return failure();

      // A single automatic region may carry both independent streams. This
      // removes the source-level requirement to spell two artificial paired
      // regions while preserving the same hard outer scheduling boundary.
      if (first.autoInterleave && firstCounts.mfma && firstCounts.memory) {
        IntraWaveStage memoryStage;
        IntraWaveStage computeStage;
        if (failed(partitionMixedIntraWaveStage(first, axisInfo, memoryStage,
                                                computeStage)))
          return failure();
        IntraWaveStageCounts memoryCounts;
        IntraWaveStageCounts computeCounts;
        if (failed(countIntraWaveStage(memoryStage, axisInfo, memoryCounts)) ||
            failed(countIntraWaveStage(computeStage, axisInfo, computeCounts)))
          return failure();
        if (!memoryCounts.memory || memoryCounts.mfma || !computeCounts.mfma ||
            computeCounts.memory ||
            !intraWaveStagesAreIndependent(memoryStage, computeStage)) {
          first.begin->emitError(
              "automatic intra-wave window requires independent memory and "
              "MFMA streams");
          return failure();
        }
        materializeAutomaticIntraWaveWindow(memoryStage, computeStage,
                                            computeCounts, first.end,
                                            first.pair, nextSyncId, axisInfo);
        clearIntraWaveMarker(first.begin);
        clearIntraWaveMarker(first.end);
        ++i;
        continue;
      }

      if (i + 1 == stages.size()) {
        first.begin->emitError(
            "an intra-wave pipeline pair must contain exactly two stages");
        return failure();
      }
      if (first.pair != stages[i + 1].pair) {
        first.begin->emitError(
            "adjacent intra-wave pipeline stages must use the same pair");
        return failure();
      }
      if (first.autoInterleave != stages[i + 1].autoInterleave) {
        first.begin->emitError(
            "paired intra-wave stages must agree on automatic interleaving");
        return failure();
      }
      if (first.proportionalCover != stages[i + 1].proportionalCover) {
        first.begin->emitError(
            "paired intra-wave stages must agree on cover policy");
        return failure();
      }

      IntraWaveStage &second = stages[i + 1];
      IntraWaveStageCounts secondCounts;
      if (failed(countIntraWaveStage(second, axisInfo, secondCounts)))
        return failure();
      if (!intraWaveStagesAreIndependent(first, second)) {
        first.begin->emitError(
            "intra-wave pipeline stages must be dataflow-independent");
        return failure();
      }

      IntraWaveStageCounts *compute = nullptr;
      IntraWaveStageCounts *memory = nullptr;
      if (firstCounts.mfma && !firstCounts.memory && secondCounts.memory &&
          !secondCounts.mfma) {
        compute = &firstCounts;
        memory = &secondCounts;
      } else if (secondCounts.mfma && !secondCounts.memory &&
                 firstCounts.memory && !firstCounts.mfma) {
        compute = &secondCounts;
        memory = &firstCounts;
      } else {
        first.begin->emitError(
            "an intra-wave pipeline pair requires one MFMA-only stage and "
            "one memory-only stage");
        return failure();
      }

      if (first.autoInterleave) {
        IntraWaveStage *computeStage =
            compute == &firstCounts ? &first : &second;
        IntraWaveStage *memoryStage = memory == &firstCounts ? &first : &second;
        materializeAutomaticIntraWaveWindow(*memoryStage, *computeStage,
                                            *compute, second.end, first.pair,
                                            nextSyncId, axisInfo);
      }

      // Retain the outer markers as hard scheduling boundaries, but erase the
      // two inner markers so LLVM may interleave the paired regions. The group
      // sizes are the complete user-selected regions; no operations outside
      // the pair are borrowed to satisfy a latency heuristic.
      clearIntraWaveMarker(first.begin);
      OpBuilder builder(second.end);
      // A source memory operation may lower to several machine loads. Keep
      // every one as its own scheduling anchor, and divide exactly the MFMA
      // instructions selected by the user across those anchors. This turns a
      // source pair lowering to 2 DS reads and 4 MFMAs into two 1:2 machine
      // groups; it never borrows MFMAs from outside the explicit pair.
      if (!first.autoInterleave) {
        unsigned syncId = nextSyncId++;
        // Preserve the original machine-count distribution exactly for every
        // existing fine-grained source pair. Coarse source-anchor accounting
        // is an explicit opt-in, not a silent scheduler policy change.
        unsigned quotient = compute->mfma / memory->memory;
        unsigned remainder = compute->mfma % memory->memory;
        unsigned memoryIndex = 0;
        for (auto [memoryMask, runCount] : memory->memoryRuns) {
          for (unsigned index = 0; index < runCount; ++index, ++memoryIndex) {
            unsigned cover = quotient + (memoryIndex < remainder ? 1 : 0);
            materializeIntraWaveMachinePair(builder, second.end->getLoc(),
                                            memoryMask, cover, first.pair,
                                            syncId);
          }
        }
      }
      clearIntraWaveMarker(second.end);

      // The stage boundary markers inside the pair must disappear: retaining
      // either would forbid precisely the cross-region scheduling requested by
      // the source. The outer two markers remain as ordinary sched barriers.
      first.end->erase();
      second.begin->erase();
      i += 2;
    }
  }
  return success();
}

struct TritonAMDGPUIntraWavePipelinePass
    : public impl::TritonAMDGPUIntraWavePipelineBase<
          TritonAMDGPUIntraWavePipelinePass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    triton::AMD::ModuleAxisInfoAnalysis axisInfo(mod);
    if (failed(materializeIntraWaveWindows(mod, axisInfo)))
      signalPassFailure();
  }
};

struct TritonAMDGPUSchedGroupBarrierSchedulerPass
    : public impl::TritonAMDGPUSchedGroupBarrierSchedulerBase<
          TritonAMDGPUSchedGroupBarrierSchedulerPass> {
  using impl::TritonAMDGPUSchedGroupBarrierSchedulerBase<
      TritonAMDGPUSchedGroupBarrierSchedulerPass>::
      TritonAMDGPUSchedGroupBarrierSchedulerBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    triton::AMD::ModuleAxisInfoAnalysis axisInfo(mod);

    auto annotate = [&](Operation *op, int32_t mask, unsigned count,
                        unsigned mfmaCover = 0) {
      Builder b(op->getContext());
      op->setAttr("ttg.amd.sched_group_barrier.machine_mask",
                  b.getI32IntegerAttr(mask));
      op->setAttr("ttg.amd.sched_group_barrier.machine_count",
                  b.getI32IntegerAttr(count));
      if (mfmaCover)
        op->setAttr("ttg.amd.sched_group_barrier.mfma_cover",
                    b.getI32IntegerAttr(mfmaCover));
    };
    mod.walk([&](Operation *op) {
      if (isa<triton::DotOp, triton::DotScaledOp>(op))
        annotate(op, kMaskMFMA, mfmaCountOf(op));
      else if (isa<triton::gpu::LocalLoadOp>(op))
        annotate(op, kMaskDSRead, dsReadCountOf(op));
      else if (isa<triton::gpu::LocalStoreOp>(op))
        annotate(op, kMaskDSWrite, dsWriteCountOf(op));
      else if (isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                   triton::amdgpu::BufferLoadToLocalOp>(op)) {
        VMEMPrice price = priceVMEM(op, axisInfo);
        unsigned bytes = std::min(price.accessBytes, 16u);
        unsigned cover = std::max(
            1u, (static_cast<unsigned>(mfmaPerDwordx4) * bytes + 15u) / 16u);
        annotate(op, kMaskLDSDMA, price.count, cover);
      } else if (isa<triton::amdgpu::BufferLoadOp, triton::LoadOp>(op)) {
        VMEMPrice price = priceVMEM(op, axisInfo);
        // Keep the measured dwordx4 schedule as the calibration point, then
        // scale the MFMA cover with the actual lowering width. This avoids
        // over-covering a dword/dwordx2 stream merely because it contains more
        // machine loads for the same TTGIR operation.
        unsigned bytes = std::min(price.accessBytes, 16u);
        unsigned cover = std::max(
            1u, (static_cast<unsigned>(mfmaPerDwordx4) * bytes + 15u) / 16u);
        annotate(op, kMaskVMEMRead, price.count, cover);
      }
    });

    Builder b(&getContext());
    mod->setAttr("ttg.amd.sched_group_barrier.enabled", b.getUnitAttr());
    mod->setAttr(
        "ttg.amd.sched_group_barrier.required_region_count",
        b.getI32IntegerAttr(static_cast<unsigned>(requiredRegionCount)));
  }
};

} // namespace
