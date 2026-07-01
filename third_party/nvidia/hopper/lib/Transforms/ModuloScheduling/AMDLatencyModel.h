// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TRITON_GPU_MODULO_SCHEDULING_AMD_LATENCY_MODEL_H
#define TRITON_GPU_MODULO_SCHEDULING_AMD_LATENCY_MODEL_H

#include "LatencyModel.h"

namespace mlir::triton::gpu {

/// AMD CDNA (gfx9) latency model — concrete LatencyModel for AMD MFMA kernels.
///
/// Interim placement in the (otherwise backend-neutral) core library: it
/// classifies using only TritonGPU IR ops (tt.dot + AMDMfmaEncodingAttr, ttg
/// local_load/store, tt.load) — NO AMD backend dialect — so it compiles here
/// with no extra dependency. When AMD-dialect-specific ops (buffer_load_to_lds)
/// are needed, this should move into the AMD backend once the core is relocated
/// to a neutral path.
///
/// Cycle counts (gfx950): GLOBAL latency=790 is MEASURED
/// (claude/amd_latency_microbench.py, pointer-chase ~360 ns x 2.2 GHz). MFMA =
/// 16 cyc PER hardware MFMA (LLIR 16x16x32 anchor) SCALED by the dot's per-wave
/// MFMA count, so a block-level tt.dot's cost reflects real compute (critical:
/// the un-scaled single-MFMA cost made II tiny -> the scheduler over-staged the
/// prefetch). LDS=30 / VALU=16 are still coarse CDNA estimates (refine later).
/// The accumulator hooks intentionally use the base no-op defaults: AMD
/// accumulates MFMA results in registers, so there is no TMEM-style cross-loop
/// hazard.
class AMDLatencyModel : public LatencyModel {
public:
  OpLatencyInfo getLatency(Operation *op) const override;

  /// Classify which hardware pipeline an operation uses.
  HWPipeline classifyPipeline(Operation *op) const;
};

} // namespace mlir::triton::gpu

#endif // TRITON_GPU_MODULO_SCHEDULING_AMD_LATENCY_MODEL_H
