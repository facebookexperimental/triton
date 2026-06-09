#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu {

/// Hardware pipeline classification for Blackwell SM100.
/// Each op executes on exactly one pipeline; distinct pipelines overlap.
enum class HWPipeline {
  TMA,  // Async TMA engine: descriptor_load/store/gather. Named TMA (not
        // MEM) because a regular ld.global has CUDA-core-like self latency;
        // this pipeline specifically models the async TMA unit.
  TC,   // Tensor Core (tc_gen05_mma, warp_group_dot)
  CUDA, // General CUDA cores (arith.*, tt.reduce, type conversions)
  SFU,  // Special Function Unit (math.exp2, math.log2, math.rsqrt)
  NONE  // Scalar/index ops, control flow — zero latency, no resource
};

/// Return a human-readable name for a pipeline.
llvm::StringRef getPipelineName(HWPipeline pipeline);

/// Latency info for a single operation.
///
/// The two fields play DISTINCT roles in modulo scheduling. Always use the
/// definitions below — confusing them inflates II and breaks scheduling.
///
/// `latency` = dependency edge weight. Cycles from this op's ISSUE to when
///   a *dependent consumer* can issue. (Equivalently: result-available
///   delay.) Drives RecMII = max over SCCs of Σ(latency)/Σ(distance).
///
/// `selfLatency` = resource reservation. Cycles this op OCCUPIES its pipe —
///   how soon another *unrelated* op on the same pipe can issue. Drives
///   ResMII = for each pipe, Σ(selfLat × freq).
///
/// Invariant: 0 ≤ selfLatency ≤ latency.
///   - selfLat == latency: op fully serializes its pipe (no overlap).
///   - selfLat == 1: "fire-and-forget" — next pipe op issues 1 cyc later
///     even if this op's result isn't ready.
///
/// Examples on Blackwell (FA fwd inner loop):
///   tcgen05.mma 128×128×128 : selfLat=30,  latency=900,  min_warps=1
///   descriptor_load 128×64  : selfLat=30,  latency=530,  min_warps=1
///   math.exp2 128×64 (SFU)  : selfLat=64,  latency=570,  min_warps=4
///   arith.mulf 128×64 (FMA) : selfLat=64,  latency=67,   min_warps=4
///
/// `min_warps` = the warp count assumed by the modeled `selfLat`. If the
/// containing WG actually has fewer warps, `selfLat` scales up roughly
/// linearly: `effective = selfLat × min_warps / actual_warps`. Async ops
/// (TMA, MMA) only need 1 warp to issue, so `min_warps=1`. Tile-parallel
/// CUDA/SFU ops use all 4 subpartitions, so `min_warps=4`. Used by the
/// partitioner to size each WG's warp count and avoid mis-charging the
/// pipe-occupancy cost.
struct OpLatencyInfo {
  HWPipeline pipeline{HWPipeline::NONE};
  int latency{0};
  int selfLatency{0};
  int minWarps{1};
};

/// Hardware latency model for Blackwell SM100.
///
/// Classifies TTGIR operations into hardware pipelines and assigns
/// cycle-accurate latencies from microbenchmark data. Initially hardcoded
/// for Blackwell; designed to be subclassed for other architectures.
///
/// Latency values are from the WS Global Instruction Scheduling design doc
/// (D95269626) and validated by the latency microbenchmark harness.
class LatencyModel {
public:
  virtual ~LatencyModel() = default;

  /// Classify an operation and return its pipeline + latency.
  virtual OpLatencyInfo getLatency(Operation *op) const;

  /// Classify which hardware pipeline an operation uses.
  HWPipeline classifyPipeline(Operation *op) const;

private:
  int getTMALoadLatency(Operation *op) const;
  int getTMAStoreLatency(Operation *op) const;
  int getMMALatency(Operation *op) const;
  int getCUDALatency(Operation *op) const;
  int getCUDASelfLat(Operation *op) const;
  int getSFULatency(Operation *op) const;
  int getSFUSelfLat(Operation *op) const;
  int getMinWarps(Operation *op) const;

  /// Estimate tensor size in elements from an op's result type.
  int64_t getTensorElements(Operation *op) const;
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H
