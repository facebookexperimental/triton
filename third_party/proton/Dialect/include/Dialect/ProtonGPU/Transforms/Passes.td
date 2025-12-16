#ifndef PROTONGPU_TRANSFORMS_PASSES
#define PROTONGPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ScheduleBufferStorePass: Pass<"proton-schedule-buffer-store", "mlir::ModuleOp"> {
  let summary = "Pass to move all Proton buffer stores to the end of the function";

  let description = "This pass makes the measurement more accurate by moving the expensive "
                    "shared memory stores to the end of the measured region after the measurements.";

  let dependentDialects = ["gpu::ProtonGPUDialect"];
}

def MppStoreBarrierInfoPass: Pass<"proton-mpp-store-barrier-info", "mlir::ModuleOp"> {
  let summary = "Replace ReadCounterOp with barrier allocOpId and index for barrier record ops";

  let description = [{
    This pass finds RecordOp pairs that track barrier operations and replaces
    the generated ReadCounterOp with barrier allocation IDs (for start records)
    and barrier indices (for end records).

    The pass is gated by the PROTON_ENABLE_MPP_STORE_BARRIER_INFO_PASS environment variable.
    When enabled, it tracks barrier info (allocOpId, index) through value propagation
    and replaces the counter values in CircularStoreOp with the computed values.
  }];

  let dependentDialects = ["gpu::ProtonGPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

#endif  // PROTONGPU_TRANSFORMS_PASSES
