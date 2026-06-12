// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "LatencyModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "modulo-scheduling-latency"

namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {

llvm::StringRef getPipelineName(HWPipeline pipeline) {
  switch (pipeline) {
  case HWPipeline::TMA:
    return "TMA";
  case HWPipeline::TC:
    return "TC";
  case HWPipeline::CUDA:
    return "CUDA";
  case HWPipeline::SFU:
    return "SFU";
  case HWPipeline::NONE:
    return "NONE";
  }
  llvm_unreachable("unknown pipeline");
}

// Estimate total elements in the tensor an op produces or consumes.
// Tries result types first; for store ops (whose result is a memdesc handle)
// scans operand types and picks the largest tensor/memdesc among them.
int64_t LatencyModel::getTensorElements(Operation *op) const {
  auto countShape = [](auto shape) {
    int64_t e = 1;
    for (auto d : shape)
      e *= d;
    return e;
  };
  for (auto r : op->getResults()) {
    if (auto tt = dyn_cast<RankedTensorType>(r.getType()))
      return countShape(tt.getShape());
  }
  // Fall back to operand-derived size — needed for store-style ops
  // (ttng.tmem_store, ttg.local_store) whose result is a memdesc handle.
  int64_t best = 0;
  for (auto v : op->getOperands()) {
    if (auto tt = dyn_cast<RankedTensorType>(v.getType()))
      best = std::max(best, countShape(tt.getShape()));
    else if (auto md = dyn_cast<triton::gpu::MemDescType>(v.getType()))
      best = std::max(best, countShape(md.getShape()));
  }
  return best;
}

// TMA load full latencies (cycles): time from cp.async.bulk issue to data
// available in SMEM for a consumer to read. Single number per size — there is
// no separate "pipeline occupancy" vs "DRAM overhead" component in the model.
//
// Refit from B200 cycle-accurate microbenchmarks (Phase 0b, May 2026)
// using k_tma_load_chained in users/wl/wlei/latency_table/ncu_bench.py:
// barrier_expect → async_descriptor_load → barrier_wait, repeated.
struct TMALatencyEntry {
  int64_t bytes;
  int cycles;
};
static constexpr TMALatencyEntry kTMALoadTable[] = {
    {128 * 64 * 2, 530},  // 128x64 or 64x128 bf16/fp16 = 16KB (was 640)
    {128 * 128 * 2, 663}, // 128x128 bf16/fp16 = 32KB (was 808)
    {256 * 64 * 2, 654},  // 256x64 bf16 = 32KB (was 807)
    {256 * 128 * 2, 925}, // 256x128 bf16 = 64KB (was 1134)
};

// Issue latency for async TMA operations. The SM spends this many cycles
// programming the TMA descriptor and triggering the copy, then the TMA engine
// runs independently. This is the MEM pipeline occupancy (selfLatency), NOT
// the full transfer time — the transfer time only affects edge weights (when
// data becomes available to consumers).
constexpr int kTMAIssueLatency = 30;

// Issue latency for async MMA operations (tcgen05.mma on Blackwell).
// The SM issues the MMA instruction to the tensor cores asynchronously,
// then the TC hardware executes independently. The SM can issue subsequent
// instructions (including more MMAs) after the issue cost.
constexpr int kMMAIssueLatency = 30;

/// Look up TMA load latency (cycles) by total bytes. Table lookup first, then
/// linear interpolation from 128x64 baseline as fallback.
static int lookupTMALoadOccupancy(int64_t totalBytes) {
  for (const auto &entry : kTMALoadTable) {
    if (entry.bytes == totalBytes)
      return entry.cycles;
  }
  // Fallback: linear interpolation from 128x64 baseline.
  constexpr int64_t kBaseBytes = 128 * 64 * 2;
  constexpr int kBaseCycles = 530;
  return static_cast<int>(kBaseCycles * static_cast<double>(totalBytes) /
                          kBaseBytes);
}

// Tile geometry of a TMA op: total bytes and inner (contiguous, last-dim) bytes.
// Reads the result tensor/memdesc first; for store-style ops (memdesc result)
// falls back to the largest tensor/memdesc operand. innerBytes drives the
// hardware instruction count (see getTMANumInsts).
static void getTMATile(Operation *op, int64_t &totalBytes, int64_t &innerBytes) {
  totalBytes = 0;
  innerBytes = 0;
  auto fromShape = [&](ArrayRef<int64_t> shape, int64_t esize) {
    if (shape.empty())
      return;
    int64_t elems = 1;
    for (auto d : shape)
      elems *= d;
    totalBytes = elems * esize;
    innerBytes = shape.back() * esize;
  };
  for (auto r : op->getResults()) {
    if (auto tt = dyn_cast<RankedTensorType>(r.getType()))
      return fromShape(tt.getShape(), tt.getElementTypeBitWidth() / 8);
    if (auto md = dyn_cast<triton::gpu::MemDescType>(r.getType()))
      return fromShape(md.getShape(),
                       md.getElementType().getIntOrFloatBitWidth() / 8);
  }
  int64_t best = 0;
  for (auto v : op->getOperands()) {
    if (auto tt = dyn_cast<RankedTensorType>(v.getType())) {
      int64_t e = 1;
      for (auto d : tt.getShape())
        e *= d;
      if (e * (tt.getElementTypeBitWidth() / 8) > best) {
        best = e * (tt.getElementTypeBitWidth() / 8);
        fromShape(tt.getShape(), tt.getElementTypeBitWidth() / 8);
      }
    } else if (auto md = dyn_cast<triton::gpu::MemDescType>(v.getType())) {
      int64_t e = 1;
      for (auto d : md.getShape())
        e *= d;
      int64_t es = md.getElementType().getIntOrFloatBitWidth() / 8;
      if (e * es > best) {
        best = e * es;
        fromShape(md.getShape(), es);
      }
    }
  }
}

// A source TMA op lowers to ceil(innerBytes/512) hardware cp.async.bulk.tensor
// instructions (the box inner dim caps at 512 bytes); the outer (M) dim is not
// split. Verified by PTX instruction counting on B200 (latency_model study).
static int getTMANumInsts(Operation *op) {
  int64_t totalBytes, innerBytes;
  getTMATile(op, totalBytes, innerBytes);
  if (innerBytes <= 0)
    return 1;
  return static_cast<int>((innerBytes + 511) / 512);
}

// TMA LOAD round-trip latency (cycles), B200-calibrated to the num_insts law:
//   lat = base + perKB·KB + perInst·min(num_insts−1, cap)
// Latency is driven by the instruction count (inner dim), ~independent of M, and
// saturates by ~num_insts=3 (per-load instructions' DRAM drains overlap). Fits
// the measured grid within ~10% (e.g. [8,512] N=2 ≈ 748, [128,64] N=1 ≈ 556).
int LatencyModel::getTMALoadLatency(Operation *op) const {
  constexpr int kBase = 460, kPerKB = 6, kPerInst = 240, kInstCap = 2;
  int64_t totalBytes, innerBytes;
  getTMATile(op, totalBytes, innerBytes);
  if (totalBytes <= 0)
    return lookupTMALoadOccupancy(128 * 64 * 2);
  int nInsts = getTMANumInsts(op);
  int kb = static_cast<int>(totalBytes / 1024);
  return kBase + kPerKB * kb + kPerInst * std::min(nInsts - 1, kInstCap);
}

// TMA STORE latency (cycles): bandwidth-bound, ~linear in bytes (B200: 52 cyc/KB
// + ~130 fixed). The store engine does not pipeline (occupancy ≈ latency).
int LatencyModel::getTMAStoreLatency(Operation *op) const {
  constexpr int kBase = 130, kPerKB = 52;
  int64_t totalBytes, innerBytes;
  getTMATile(op, totalBytes, innerBytes);
  if (totalBytes <= 0)
    return lookupTMALoadOccupancy(128 * 64 * 2);
  return kBase + kPerKB * static_cast<int>(totalBytes / 1024);
}

// MMA latencies from design doc microbenchmarks (Blackwell tcgen05.mma).
// Scales with the product M*N*K.
constexpr int kMMALatency128x128x128 = 900;
constexpr int kMMALatency128x128x64 = 559;

int LatencyModel::getMMALatency(Operation *op) const {
  // Extract the K dimension of the MMA. tcgen05_mma operands A and B are
  // SMEM/TMEM memdescs (NOT RankedTensorType), so dyn_cast against tensor
  // alone misses them — accept either.
  auto getShape = [](Value v) -> ArrayRef<int64_t> {
    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
      return t.getShape();
    if (auto md = dyn_cast<triton::gpu::MemDescType>(v.getType()))
      return md.getShape();
    return {};
  };
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto aShape = getShape(mma->getOperand(0)); // [M, K]
    if (aShape.size() >= 2) {
      int64_t K = aShape[1];
      return K >= 128 ? kMMALatency128x128x128 : kMMALatency128x128x64;
    }
  }
  return kMMALatency128x128x128; // conservative default
}

// Latency values refit from B200 cycle-accurate microbenchmarks
// (users/wl/wlei/latency_table/cycle_bench.py, measured via tlx.clock64()).
// These reflect SERIAL chain cost, which is what the modulo scheduler
// uses for both ResMII (resource pressure) and RecMII (dep recurrence).
//
// All shape-aware: latency scales with `elements` (= prod of tensor shape)
// using a per-element cost. Baseline measurements at 128x128 = 16384 elems:
//   acc_update (mulf):  138 cyc → 0.0084 cyc/elem (treat as fixed-overhead)
//   scale_sub (subf×2): 143 cyc → similar
//   rowmax:             461 cyc / 128 rows = ~3.6 cyc/row + log2(N) tree
//   rowsum:           1,691 cyc / 128 rows = ~13 cyc/row (multi-stage)
//   exp2:             1,140 cyc → 0.07 cyc/elem
//   log2:            11,268 cyc → 0.69 cyc/elem (slow!)
//
// For shapes outside 128x128, scale linearly with elements (verified up to
// 256x128: rowsum 10455 ≈ 0.32 cyc/elem × 32768 elems ≈ 10485, matches).
constexpr int kBaseElems = 128 * 128;

static int scaleByElements(int baseCycles, int64_t elements) {
  if (elements <= 0)
    return baseCycles;
  // Linear scaling from the 128x128 baseline. Rounded.
  return static_cast<int>(static_cast<int64_t>(baseCycles) * elements /
                          kBaseElems);
}

int LatencyModel::getCUDALatency(Operation *op) const {
  // Ops that don't produce tensor results but have real latency.
  // Check these before the scalar early-return.
  if (isa<ttng::WaitBarrierOp>(op))
    return 30;
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op))
    return 20;

  int64_t elements = getTensorElements(op);

  // TMEM load: NCU bench shows 179 cyc at 128x64 (8192 elems) and
  // 532 cyc at 128x128 (16384 elems). Non-linear — use scaled base 532
  // at 16384, which matches at 128x128 and over-counts modestly at 128x64.
  if (isa<ttng::TMEMLoadOp>(op)) {
    if (elements == 0)
      return 105;
    return scaleByElements(532, elements);
  }
  // TMEM store: 64 cyc at 128x64, 96 cyc at 128x128. Roughly linear
  // with elements at base ~96 / 16384.
  if (isa<ttng::TMEMStoreOp>(op)) {
    if (elements == 0)
      return 105;
    return scaleByElements(96, elements);
  }

  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return 105;

  if (elements == 0)
    return 1; // scalar arith — 1 cycle on the SM CUDA cores

  // Reductions — refit to NCU chained measurements:
  //   rowmax 425 cyc at 128x64, 461 cyc at 128x128, 1407 cyc at 256x128
  //   rowsum 574 cyc at 128x64, 1691 cyc at 128x128, 10460 at 256x128
  // Take base at 128x128 (16384 elems) and scale linearly. This under-counts
  // 128x64 (model says 230 vs real 425, 287 vs real 574) — but the cleanest
  // simple rule. Reductions should ideally key off the reduce-axis size, not
  // total elements; left as future work.
  if (auto reduceOp = dyn_cast<tt::ReduceOp>(op)) {
    bool isSum = false;
    reduceOp.getBody()->walk([&](Operation *inner) {
      if (isa<arith::AddFOp>(inner))
        isSum = true;
    });
    return scaleByElements(isSum ? 1691 : 461, elements);
  }

  // Type conversions (truncf, extf, etc.). NCU benchmark reports the chain
  // is folded by the compiler (~0–4 cyc), so the real cost is dominated by
  // surrounding ops in production kernels. Keep modest 105-baseline as a
  // reasonable upper bound until a forced-RAW microbench is added.
  if (isa<arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp, arith::SIToFPOp,
          tt::FpToFpOp, tt::BitcastOp>(op))
    return scaleByElements(105, elements);

  // Multiply: keeping 138 baseline because that matches the broadcast
  // pattern (data * alpha[:, None]) used in FA softmax. Pure element-wise
  // mulf measures 254 at 128x128 — different cost profile, not refit here.
  if (isa<arith::MulFOp>(op))
    return scaleByElements(138, elements);

  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp,
          triton::gpu::ConvertLayoutOp>(op))
    return scaleByElements(105, elements);

  // Integer type conversions: same as float conversions.
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp>(
          op))
    return scaleByElements(105, elements);

  // Default elementwise (addf, subf, maxnumf, etc.): broadcast pattern at
  // 128x64 measures ~66 cyc, scales to 130 at 128x128.
  return scaleByElements(130, elements);
}

// CUDA per-op pipe-occupancy cost (selfLat) from NCU
// `sm__pipe_fma_cycles_active.sum / N_op / 8 FMA-units-per-SM`.
// Materially smaller than `latency` for ops where the chain is RAW-stall-
// dominated (notably reductions). Used for ResMII (resource conflict),
// while `latency` is used for RecMII (dependency recurrence).
int LatencyModel::getCUDASelfLat(Operation *op) const {
  if (isa<ttng::WaitBarrierOp>(op))
    return 30;
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op))
    return 20;

  int64_t elements = getTensorElements(op);

  // TMEM ops — pipe-active matches latency for load (RAW stall is small);
  // for store the FMA pipe shows ~0 active (folded into surrounding ops).
  if (isa<ttng::TMEMLoadOp>(op))
    return elements == 0 ? 105 : scaleByElements(256, elements);
  if (isa<ttng::TMEMStoreOp>(op))
    return elements == 0 ? 105 : scaleByElements(48, elements);
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return 105;

  if (elements == 0)
    return 1; // scalar arith — 1 cycle of pipe occupancy on the SM

  // Reductions: NCU pipe_fma_active per unit per op
  //   rowmax 128:304:520 at 8192/16384/32768 elems  → base 304 at 16384
  //   rowsum 304:639:1280                           → base 639 at 16384
  if (auto reduceOp = dyn_cast<tt::ReduceOp>(op)) {
    bool isSum = false;
    reduceOp.getBody()->walk([&](Operation *inner) {
      if (isa<arith::AddFOp>(inner))
        isSum = true;
    });
    return scaleByElements(isSum ? 639 : 304, elements);
  }

  // Conversions: compiler-folded in microbench, so use small selfLat.
  if (isa<arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp, arith::SIToFPOp,
          tt::FpToFpOp, tt::BitcastOp, arith::ExtUIOp, arith::ExtSIOp,
          arith::TruncIOp, arith::IndexCastOp>(op))
    return std::max(1, static_cast<int>(elements / 256));

  if (isa<triton::gpu::ConvertLayoutOp>(op))
    return scaleByElements(105, elements);

  // Element-wise (mulf, addf, subf, maxnumf, ...): NCU pipe_fma_active
  // per unit per op ~ 64 at 128x64, 128 at 128x128 — linear with elems.
  // Base 128 at 16384 elems → 0.0078 cyc/elem.
  return scaleByElements(128, elements);
}

int LatencyModel::getSFULatency(Operation *op) const {
  int64_t elements = getTensorElements(op);
  if (elements == 0)
    return 43; // scalar exp2 (Alpha = Exp2(scalar))

  // SFU ops scale steeply with shape (16x at 256x128 vs 128x128).
  // Differentiate by op kind: log2 is ~10x slower than exp2.
  if (isa<math::Log2Op, math::LogOp>(op))
    return scaleByElements(11268, elements); // log2(128x128) = 11,268 cyc
  // exp2, exp, sqrt, rsqrt, etc.
  return scaleByElements(1140, elements); // exp2(128x128) = 1,140 cyc
}

// SFU per-op pipe-occupancy. Blackwell exposes no `pipe_xu_cycles_active`
// counter, only `inst_executed_pipe_xu`. Estimate as warp-insts ÷ 4
// subpartitions (XU is one unit per subpartition, ~1 inst/cycle/unit):
//   exp2 128x64  : 256 insts → 64 cyc/subp self-occupancy
//   exp2 128x128 : 512 insts → 128 cyc/subp
//   exp2 1D 128  :   4 insts →   1 cyc/subp
// log2 has same instruction count; same selfLat (latency differs, not
// pipe occupancy).
int LatencyModel::getSFUSelfLat(Operation *op) const {
  int64_t elements = getTensorElements(op);
  if (elements == 0)
    return 1;
  // 128x128 (16384 elems) → 128 cyc per subpartition; linear scaling.
  return scaleByElements(128, elements);
}

HWPipeline LatencyModel::classifyPipeline(Operation *op) const {
  // MEM: TMA loads, regular loads, and stores
  if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
    return HWPipeline::TMA;
  // MEM: Lowered TMA loads (TLX kernels use async_tma_copy instead of
  // descriptor_load)
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return HWPipeline::TMA;
  if (isa<tt::LoadOp>(op)) {
    // Regular tt.load (before TMA lowering) — classify as MEM if tensor
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::TMA;
  }
  if (isa<tt::DescriptorStoreOp>(op))
    return HWPipeline::TMA;
  // MEM: Lowered TMA stores (TLX path)
  if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return HWPipeline::TMA;

  // TC: Tensor Core MMA operations
  if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
    return HWPipeline::TC;
  if (isa<ttng::WarpGroupDotOp>(op))
    return HWPipeline::TC;
  // TC: tt.dot (before lowering to TCGen5MMAOp / WarpGroupDotOp)
  if (isa<tt::DotOp>(op))
    return HWPipeline::TC;

  // CUDA: TMEM load/store (data movement between registers and TMEM)
  if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
    return HWPipeline::CUDA;

  // CUDA: SMEM load/store (data movement between registers and SMEM)
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Layout conversions on tensors (may involve SMEM round-trips)
  if (isa<triton::gpu::ConvertLayoutOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Barrier operations (synchronization between warp groups).
  // These carry timing dependencies between producers and consumers
  // in warp-specialized kernels.
  if (isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(
          op))
    return HWPipeline::CUDA;

  // MEM: Regular tensor stores to global memory
  if (isa<tt::StoreOp>(op)) {
    if (op->getNumOperands() > 1) {
      auto valOperand = op->getOperand(1);
      if (isa<RankedTensorType>(valOperand.getType()))
        return HWPipeline::TMA;
    }
  }

  // SFU: Transcendental math operations on tensors
  if (isa<math::Exp2Op, math::ExpOp, math::Log2Op, math::LogOp, math::SqrtOp,
          math::RsqrtOp, math::TanhOp>(op)) {
    // Only classify as SFU if operating on tensors
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::SFU;
    return HWPipeline::NONE; // scalar math is free
  }

  // CUDA: Reductions
  if (isa<tt::ReduceOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Float arithmetic (scalar AND tensor — both run on SM CUDA cores).
  // Scalar ops cost ~1 cycle (handled in getCUDALatency); tensor ops scale
  // with element count. Classifying scalars as CUDA (not NONE) gives the
  // scheduler a realistic non-zero latency for index/offset chains.
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
          arith::MinNumFOp, arith::NegFOp, arith::CmpFOp, arith::CmpIOp,
          arith::SelectOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Integer arithmetic (index computation, masking) — scalar AND tensor.
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
          arith::DivSIOp, arith::RemUIOp, arith::RemSIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRUIOp,
          arith::ShRSIOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Integer type conversions — scalar AND tensor.
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp>(
          op))
    return HWPipeline::CUDA;

  // CUDA: Float type conversions on tensors
  if (isa<arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp, arith::SIToFPOp,
          tt::FpToFpOp, tt::BitcastOp>(op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::CUDA;
  }

  // NONE: Scalar ops, index arithmetic, control flow, barriers, and
  // local_alloc (which is a pure IR-level rename — TMA wrote bytes directly
  // to SMEM, local_alloc just wraps the buffer in a memdesc and does not
  // consume any pipeline resource).
  return HWPipeline::NONE;
}

// Per-op minimum warp count to achieve the modeled `selfLatency`. If the
// containing WG actually has fewer warps, `selfLatency` should scale up
// roughly linearly: effective ≈ selfLat × min_warps / actual_warps.
//
// Rationale per category:
//  - Async producers (TMA, MMA, barriers): 1 warp issues the instruction;
//    the rest of the work is done by the TMA engine / tensor core / mbarrier
//    hardware. Adding warps doesn't help.
//  - TMEM load/store: TLX/Blackwell hardware requires numWarps == 4 or 8.
//  - SMEM load/store, layout conversions: parallel across the 4
//    subpartitions' LSUs.
//  - SFU on tensors (math.exp2, log2, etc.): 1 SFU per subpartition × 4
//    subpartitions per SM. With 1 warp on 1 subpartition, only 1 SFU pipe
//    is used; with 4 warps spread across subpartitions, all 4 pipes run
//    in parallel.
//  - CUDA tile arith (mulf/addf/subf/maxnumf on tensors), reductions: same
//    parallel-across-subpartitions argument as SFU.
//  - Scalar ops, NONE pipeline: 1 warp suffices.
int LatencyModel::getMinWarps(Operation *op) const {
  // Async producers — 1 warp issues, hardware does the rest.
  if (isa<tt::DescriptorLoadOp, tt::DescriptorStoreOp, tt::DescriptorGatherOp,
          ttng::AsyncTMACopyGlobalToLocalOp, ttng::AsyncTMACopyLocalToGlobalOp>(
          op))
    return 1;
  if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp, ttng::WarpGroupDotOp,
          tt::DotOp>(op))
    return 1;
  // Synchronization primitives.
  if (isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(
          op))
    return 1;
  // TMEM ops — TLX/Blackwell hardware constraint requires 4 or 8 warps.
  if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
    return 4;
  // SMEM load/store and layout conversions on tensors — parallel access.
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp,
          triton::gpu::ConvertLayoutOp>(op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return 4;
    return 1;
  }
  // Tile-parallel work uses all 4 subpartitions; detect by tensor result.
  if (op->getNumResults() > 0 &&
      isa<RankedTensorType>(op->getResult(0).getType()))
    return 4;
  return 1;
}

OpLatencyInfo LatencyModel::getLatency(Operation *op) const {
  auto pipeline = classifyPipeline(op);

  int latency = 0;
  int selfLatency = 0;
  switch (pipeline) {
  case HWPipeline::TMA: {
    // local_alloc fed by a TMA load is a pure IR-level rename: the TMA already
    // delivered the bytes to SMEM. No pipeline occupancy, no extra wait.
    if (isa<triton::gpu::LocalAllocOp>(op))
      return OpLatencyInfo{pipeline, /*latency=*/0, /*selfLatency=*/0,
                           /*minWarps=*/1, /*occupancy=*/0};
    // selfLatency = issue cost (SM blocked dispatching). latency = round-trip.
    // occupancy = cycles the TMA engine is held for ResMII: a store is
    // bandwidth-bound (occ ≈ latency), but a load is multi-outstanding so it
    // occupies the engine only ~bytes/bandwidth (≈ 6 cyc/KB), NOT its latency.
    bool isStore = isa<tt::DescriptorStoreOp, ttng::AsyncTMACopyLocalToGlobalOp>(op);
    int64_t totalBytes, innerBytes;
    getTMATile(op, totalBytes, innerBytes);
    selfLatency = kTMAIssueLatency;
    int occupancy;
    if (isStore) {
      latency = getTMAStoreLatency(op);
      occupancy = latency; // bandwidth-bound: engine held for the full transfer
    } else {
      latency = getTMALoadLatency(op);
      occupancy = std::max(kTMAIssueLatency,
                           static_cast<int>(6 * (totalBytes / 1024)));
    }
    return OpLatencyInfo{pipeline, latency, selfLatency, /*minWarps=*/1,
                         occupancy};
  }
  case HWPipeline::TC:
    latency = getMMALatency(op);
    // selfLatency = issue cost (SM dispatch pipeline occupancy).
    // Design doc: 30 cycles for tcgen05.mma. occupancy = latency: the tensor
    // core genuinely serializes one MMA at a time (keeps GEMM/FA ResMII right).
    selfLatency = kMMAIssueLatency;
    return OpLatencyInfo{pipeline, latency, selfLatency, getMinWarps(op),
                         /*occupancy=*/latency};
  case HWPipeline::CUDA:
    // latency  = full RAW-dep cost per op (clock64-chained microbench).
    //            Used by RecMII to size dependency recurrences.
    // selfLat  = per-op pipe-occupancy (NCU pipe_fma_active per FMA-unit
    //            per op). Used by ResMII to size resource conflicts.
    // For RAW-stall-dominated ops (notably reductions) selfLat is
    // materially smaller than latency; treating them as equal (Phase 0a)
    // over-counted CUDA pipe pressure ~1.5–3× and led the WG partitioner
    // to over-aggressively split softmax across warp groups.
    latency = getCUDALatency(op);
    selfLatency = getCUDASelfLat(op);
    break;
  case HWPipeline::SFU:
    // Same split as CUDA. SFU has no `cycles_active` counter on Blackwell
    // so selfLat is estimated from `inst_executed_pipe_xu` (insts ÷ 4
    // subpartitions × ~1 cyc/inst), which gives ~64 at 128x64, ~128 at
    // 128x128. Latency includes the ~570/1140 cyc transcendental wait.
    latency = getSFULatency(op);
    selfLatency = getSFUSelfLat(op);
    break;
  case HWPipeline::NONE:
    latency = 0;
    selfLatency = 0;
    break;
  }

  // CUDA/SFU are pipelined synchronous units: occupancy = selfLatency (the
  // per-op pipe slot count). NONE → 0 (no resource).
  return OpLatencyInfo{pipeline, latency, selfLatency, getMinWarps(op),
                       /*occupancy=*/selfLatency};
}

} // namespace mlir::triton::gpu
