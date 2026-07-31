// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// AMD CDNA (gfx9) latency model. See AMDLatencyModel.h.

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
  // `latency` = result-ready delay (drives RecMII);
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
          auto saturatingMul = [](int64_t lhs, int64_t rhs) {
            if (lhs == 0 || rhs == 0)
              return int64_t{0};
            if (lhs > std::numeric_limits<int64_t>::max() / rhs)
              return std::numeric_limits<int64_t>::max();
            return lhs * rhs;
          };
          int64_t mCount = (M + tileM - 1) / tileM;
          int64_t nCount = (N + tileN - 1) / tileN;
          int64_t kCount = (K + iK - 1) / iK;
          nMfma = saturatingMul(saturatingMul(mCount, nCount), kCount);
          if (nMfma < 1)
            nMfma = 1;
        }
      }
    }
    // gfx950 measurement using tlx.clock64/s_memtime and a runtime loop:
    //   dependent v_mfma_f32_16x16x32_f16: 18.25 ticks ~= 18 cycles
    //   four independent streams:          16.62 ticks/MFMA ~= 17 cycles
    // s_memtime measured at 2.183 GHz while the gfx950 shader clock reached
    // 2.2 GHz (1.008 cycles/tick). Scale the block-level dot by its per-wave
    // hardware MFMA count and keep dependency latency distinct from occupancy.
    int64_t intMax = std::numeric_limits<int>::max();
    int latency = (int)std::min<int64_t>(nMfma, intMax / 18) * 18;
    int occupancy = (int)std::min<int64_t>(nMfma, intMax / 17) * 17;
    return OpLatencyInfo{pipe, /*latency=*/latency,
                         /*selfLatency=*/occupancy, /*minWarps=*/1,
                         /*occupancy=*/occupancy};
  }
  case HWPipeline::LDS:
    // gfx950 pure LDS read measurement using tlx.local_gather over a
    // preinitialized 1D i32 LDS table:
    //   dependent idx_{i+1}=table[idx_i]: 68.16 ticks ~= 69 cycles
    //   four independent streams:         28.17 ticks/gather ~= 28 cycles
    // A CTA barrier separates table initialization from timing. The dependent
    // case isolates result-ready ds_read latency: the timed s_memtime region
    // contains ds_read_b32 and no ds_write. Keep occupancy at the prior
    // conservative value because x4 does not establish the issue-rate floor.
    return OpLatencyInfo{pipe, /*latency=*/69, /*selfLatency=*/4,
                         /*minWarps=*/1, /*occupancy=*/4};
  case HWPipeline::GLOBAL:
    // Multi-outstanding async global (HBM) load: long round-trip, short
    // occupancy. Keep the conservative 790-cycle estimate: the current
    // microbenchmark measures a warm self-pointer load, not DRAM latency.
    return OpLatencyInfo{pipe, /*latency=*/790, /*selfLatency=*/8,
                         /*minWarps=*/1, /*occupancy=*/8};
  case HWPipeline::VALU:
    // A dependent v_fma_f32 chain measures 8.30 s_memtime ticks ~= 8 cycles.
    // The existing four-cycle occupancy remains conservative until a dedicated
    // instruction-counted independent-stream benchmark is available.
    return OpLatencyInfo{pipe, /*latency=*/8, /*selfLatency=*/4,
                         /*minWarps=*/2, /*occupancy=*/4};
  default:
    return OpLatencyInfo{HWPipeline::NONE, /*latency=*/0, /*selfLatency=*/0,
                         /*minWarps=*/1, /*occupancy=*/0};
  }
}

} // namespace mlir::triton::gpu
