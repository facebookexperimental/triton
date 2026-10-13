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
#include "mlir/IR/Builders.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUSCHEDGROUPBARRIERSCHEDULER
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
};

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
  if (auto load = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op))
    return priceBufferAccess(load.getPtr(), load.getOffsets(),
                             load.getContiguity(), axisInfo);
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
                   triton::amdgpu::BufferLoadToLocalOp,
                   triton::amdgpu::BufferLoadOp, triton::LoadOp>(op)) {
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
