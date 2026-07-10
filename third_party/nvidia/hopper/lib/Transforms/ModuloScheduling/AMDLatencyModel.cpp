// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// AMD CDNA (gfx9) latency model. See AMDLatencyModel.h. Cycle counts are
// PLACEHOLDERS pending an AMD latency microbenchmark (refactor plan Phase C).

#include "AMDLatencyModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <limits>

namespace mlir::triton::gpu {

HWPipeline AMDLatencyModel::classifyPipeline(Operation *op) const {
  // Matrix engine: a tt.dot whose result carries an AMD MFMA encoding.
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    if (auto rtt = dyn_cast<RankedTensorType>(dot.getType()))
      if (isa_and_nonnull<AMDMfmaEncodingAttr>(rtt.getEncoding()))
        return HWPipeline::MFMA;
  }
  // LDS unit: ds_read / ds_write (shared-memory load/store).
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return HWPipeline::LDS;
  // Async global memory: a plain tt.load, or — once the loop is lowered — the
  // staged global->LDS copy (ttg.async_copy_global_to_local). Both carry the
  // long global round-trip latency, so modulo prefetches them ahead of the
  // consuming local_load. (ttg op only — no AMD-dialect dep in this core lib.)
  if (isa<triton::LoadOp, triton::gpu::AsyncCopyGlobalToLocalOp>(op))
    return HWPipeline::GLOBAL;
  // Vector ALU: elementwise arith / conversions / reductions.
  if (op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::TruncFOp,
          arith::ExtFOp, triton::ReduceOp>(op))
    return HWPipeline::VALU;
  return HWPipeline::NONE;
}

OpLatencyInfo AMDLatencyModel::getLatency(Operation *op) const {
  // PLACEHOLDER cycle counts — calibrate via an AMD microbench before using for
  // real scheduling. `latency` = result-ready delay (drives RecMII);
  // `selfLatency`/`occupancy` = pipe-hold (drives ResMII).
  HWPipeline pipe = classifyPipeline(op);
  switch (pipe) {
  case HWPipeline::MFMA: {
    // A block-level tt.dot lowers to MANY hardware MFMAs; its cost scales with
    // the per-wave MFMA count = (M/(instrM*warpsM)) * (N/(instrN*warpsN)) *
    // (K/instrK). Treating the whole dot as a single 16-cyc MFMA grossly
    // underestimates the loop's compute time -> II too small -> the scheduler
    // over-stages the prefetch (e.g. 52 stages to hide a 790-cyc load). Scaling
    // by the MFMA count makes II reflect real compute, so a long global load
    // overlaps within ~1-2 iterations (correct double-buffering).
    int64_t nMfma = 1;
    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      auto rt = dyn_cast<RankedTensorType>(dot.getType());
      auto aT = dyn_cast<RankedTensorType>(dot.getA().getType());
      auto mma = rt ? dyn_cast<AMDMfmaEncodingAttr>(rt.getEncoding()) : nullptr;
      if (rt && aT && mma && rt.getRank() == 2 && aT.getRank() >= 2) {
        auto instr = mma.getInstrShape();  // [instrM, instrN, instrK]
        auto warps = mma.getWarpsPerCTA(); // [warpsM, warpsN]
        if (instr.size() >= 3 && warps.size() >= 2) {
          int64_t tileM = std::max<int64_t>(1, (int64_t)instr[0] * warps[0]);
          int64_t tileN = std::max<int64_t>(1, (int64_t)instr[1] * warps[1]);
          int64_t iK = std::max<int64_t>(1, (int64_t)instr[2]);
          int64_t M = rt.getShape()[0], N = rt.getShape()[1];
          int64_t K = aT.getShape()[1];
          nMfma = ((M + tileM - 1) / tileM) * ((N + tileN - 1) / tileN) *
                  ((K + iK - 1) / iK);
          if (nMfma < 1)
            nMfma = 1;
        }
      }
    }
    // ~16 cyc per 16x16x32 MFMA (LLIR anchor). Clamp before narrowing to the
    // int `latency` field: nMfma is int64 (product of M/N/K tile counts), so a
    // very large dot could overflow int and wrap to a tiny/negative value,
    // silently defeating the MFMA-count scaling.
    int lat =
        (int)std::min<int64_t>(nMfma * 16, std::numeric_limits<int>::max());
    return OpLatencyInfo{pipe, /*latency=*/lat, /*selfLatency=*/lat,
                         /*minWarps=*/1, /*occupancy=*/lat};
  }
  case HWPipeline::LDS:
    return OpLatencyInfo{pipe, /*latency=*/30, /*selfLatency=*/4,
                         /*minWarps=*/1, /*occupancy=*/4};
  case HWPipeline::GLOBAL:
    // Multi-outstanding async global (HBM) load: long round-trip, short
    // occupancy. latency=790 measured on gfx950 by
    // claude/amd_latency_microbench.py (pointer-chase: ~360 ns/load x 2.2 GHz,
    // stable across runs).
    return OpLatencyInfo{pipe, /*latency=*/790, /*selfLatency=*/8,
                         /*minWarps=*/1, /*occupancy=*/8};
  case HWPipeline::VALU:
    return OpLatencyInfo{pipe, /*latency=*/16, /*selfLatency=*/4,
                         /*minWarps=*/4, /*occupancy=*/4};
  default:
    return OpLatencyInfo{HWPipeline::NONE, /*latency=*/0, /*selfLatency=*/0,
                         /*minWarps=*/1, /*occupancy=*/0};
  }
}

} // namespace mlir::triton::gpu
