#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu {

/// Hardware pipeline classification for Blackwell SM100.
/// Each op executes on exactly one pipeline; distinct pipelines overlap.
enum class HWPipeline {
  MEM,  // TMA loads/stores (descriptor_load, descriptor_store, descriptor_gather)
  TC,   // Tensor Core (tc_gen05_mma, warp_group_dot)
  CUDA, // General CUDA cores (arith.*, tt.reduce, type conversions)
  SFU,  // Special Function Unit (math.exp2, math.log2, math.rsqrt)
  NONE  // Scalar/index ops, control flow — zero latency, no resource
};

/// Return a human-readable name for a pipeline.
llvm::StringRef getPipelineName(HWPipeline pipeline);

/// Latency info for a single operation.
struct OpLatencyInfo {
  HWPipeline pipeline{HWPipeline::NONE};
  int latency{0};     // Total latency: cycles from op start to result available.
                      // Used for dependency analysis (RecMII — how long a
                      // consumer must wait for the result).
  int selfLatency{0}; // Pipeline occupancy: cycles this op blocks its pipeline.
                      // Used for resource conflict analysis (ResMII — how much
                      // pipeline bandwidth is consumed).
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
  int getSFULatency(Operation *op) const;

  /// Estimate tensor size in elements from an op's result type.
  int64_t getTensorElements(Operation *op) const;
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_LATENCY_MODEL_H
