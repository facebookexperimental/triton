#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu {

/// Hardware pipeline (resource) classification. Each op executes on exactly one
/// pipeline; distinct pipelines overlap. The set spans both backends — a given
/// LatencyModel only ever emits its own target's subset.
enum class HWPipeline {
  // NVIDIA (Hopper/Blackwell):
  TMA,  // Async TMA engine: descriptor_load/store/gather. Named TMA (not
        // MEM) because a regular ld.global has CUDA-core-like self latency;
        // this pipeline specifically models the async TMA unit.
  TC,   // Tensor Core (tc_gen05_mma, warp_group_dot)
  CUDA, // General CUDA cores (arith.*, tt.reduce, type conversions)
  SFU,  // Special Function Unit (math.exp2, math.log2, math.rsqrt)
  // AMD (CDNA):
  MFMA,   // Matrix engine (v_mfma) — AMD analog of TC
  LDS,    // LDS unit (ds_read / ds_write)
  GLOBAL, // Async global memory (buffer_load, incl. buffer_load_to_lds)
  VALU,   // Vector ALU (arith.*, conversions) — AMD analog of CUDA
  NONE    // Scalar/index ops, control flow — zero latency, no resource
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
  // Cycles this op holds its hardware pipeline (for ResMII / reservation
  // table). Distinct from `latency` for async TMA *loads*: the TMA engine is
  // multi-outstanding, so a load occupies the engine only ~bytes/bandwidth, not
  // its full round-trip latency (validated on B200, see latency_model study).
  // TMA stores are bandwidth-bound (occupancy ≈ latency); TC = latency;
  // CUDA/SFU = selfLatency. 0 means "unset" (fall back to the old rule).
  int occupancy{0};
  // True HARDWARE floor on the containing WG's warp count (TLX/Blackwell
  // TMEM load/store requires 4). Distinct from `minWarps`, which is only the
  // throughput anchor the calibration assumed — a WG may be narrowed below
  // `minWarps` (paying the modeled issue-cost scaling) but never below
  // `hardMinWarps`.
  int hardMinWarps{1};
  // Cross-warp reduce facts (tt.reduce only; 0 otherwise). `reduceAxisWarps`
  // = warpsPerCTA[reduce axis] at the op's actual encoding: > 1 means the
  // lowering (ReduceOpToLLVM) stages partials through SMEM between two
  // WG-wide bar.syncs EVERY iteration; == 1 means the reduce is
  // warp-synchronous (shuffles only, no SMEM, no bar.sync).
  int reduceAxisWarps{0};
  // Modeled warp-issue cost of the WARP-SYNCHRONOUS form of this reduce at
  // 1 warp (~ inputElems / 32 lanes × 1 cyc/elem combine). Used when a
  // width search narrows the WG enough that the reduce goes warp-synchronous
  // — the anchor-width `selfLatency` (measured on the cross-warp form)
  // scaled by minWarps/actual would mis-price that regime. 0 = not a reduce.
  int reduceSyncSelfLat1w{0};
};

/// Abstract latency-model interface consumed by the (hardware-agnostic)
/// scheduling core (DataDependenceGraph, reservation table, schedulers). A
/// backend supplies a concrete subclass — NVLatencyModel below, and a future
/// AMDLatencyModel in the AMD backend. Only getLatency is required; the
/// accumulator hooks default to "no hazard / not an accumulator alloc", which
/// is correct for targets (e.g. AMD) that accumulate in registers.
class LatencyModel {
public:
  virtual ~LatencyModel() = default;

  /// Classify an operation and return its pipeline + latency.
  virtual OpLatencyInfo getLatency(Operation *op) const = 0;

  /// Cross-region accumulator-hazard hooks. The DDG uses these to add an edge
  /// from a super-node (inner loop) to a later op that reads an accumulator the
  /// loop's MMA wrote, so the reader is not scheduled before accumulation
  /// finishes. Defaults: no hazard / not an accumulator alloc.
  ///   getAccumulatorWrite: accumulator buffer this op writes, or null.
  ///   getAccumulatorRead:  accumulator buffer this op reads, or null.
  ///   getAccumulatorAllocCols: column count of a target-specific accumulator
  ///     alloc (e.g. Blackwell TMEM) for buffer feasibility; 0 otherwise.
  virtual Value getAccumulatorWrite(Operation *op) const { return {}; }
  virtual Value getAccumulatorRead(Operation *op) const { return {}; }
  virtual int64_t getAccumulatorAllocCols(Operation *op) const { return 0; }
};

/// NVIDIA Hopper/Blackwell latency model.
///
/// Classifies TTGIR operations into hardware pipelines and assigns
/// cycle-accurate latencies from microbenchmark data. Latency values are from
/// the WS Global Instruction Scheduling design doc (D95269626) and validated by
/// the latency microbenchmark harness.
class NVLatencyModel : public LatencyModel {
public:
  OpLatencyInfo getLatency(Operation *op) const override;
  Value getAccumulatorWrite(Operation *op) const override;
  Value getAccumulatorRead(Operation *op) const override;
  int64_t getAccumulatorAllocCols(Operation *op) const override;

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
