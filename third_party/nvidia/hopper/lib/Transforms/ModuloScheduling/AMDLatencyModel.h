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
/// Cycle counts (gfx950): `third_party/tlx/tools/microbench/amd_latency.py`
/// measures s_memtime at 2.183 GHz versus a 2.2 GHz shader clock (1.008
/// cycles/tick). A dependent 16x16x32 fp16 MFMA measures ~18 cycles and four
/// independent streams measure ~17 cycles/MFMA. Those values model one dot
/// node's result latency/self latency; the dot's total MFMA count only scales
/// resource occupancy. A dependent fp32 VALU FMA measures ~8 cycles. A
/// dependent LDS `tlx.local_gather` pointer chase measures ~69 cycles with no
/// timed ds_write. GLOBAL latency=790 remains the conservative DRAM estimate; the
/// new pointer-chase result is a warm/self-pointer load and is not substituted.
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
