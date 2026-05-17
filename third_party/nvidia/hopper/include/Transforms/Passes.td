#ifndef NV_TRANSFORMS_PASSES
#define NV_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def NVGPUWarpSpecialization : Pass<"nvgpu-warp-specialization", "mlir::ModuleOp"> {
  let summary = "Automatic Warp specialization for NVIDIA GPU";

  let description = [{
    This pass automatically partitions user-defined kernels into
    warp-specialized kernels, enabling finer-grained scheduling
    and improved utilization of hardware resources.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::nvws::NVWSDialect"];
  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of buffers for warp specialization">,
    Option<"capability", "capability",
           "int32_t", /*default*/"100",
           "NVIDIA compute capability">,
    Option<"pingpongAutoWS", "pingpong-auto-ws",
           "bool", /*default*/"false",
           "Enable ping pong barrier insertion around critical regions">,
    Option<"dumpIntermediateSteps", "dump-intermediate-steps",
             "bool", /*default*/"false",
             "Dump intermediate steps">,
    Option<"smemBudget", "smem-budget",
             "int32_t", /*default*/"0",
             "SMEM budget in bytes (0 = auto-detect from target)">,
    Option<"generateSubtiledRegion", "generate-subtiled-region",
             "bool", /*default*/"false",
             "Generate SubtiledRegionOp from epilogue split patterns">
    ];
}

def NVGPUTestWSTaskPartition : Pass<"nvgpu-test-ws-task-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization task partition";

  let description = "This pass computes a warp schedule partition by annoating anchor operations with async task ids";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGPUTestWSMemoryPlanner : Pass<"nvgpu-test-ws-memory-planner", "mlir::ModuleOp"> {
  let summary = "test warp specialization memory planner";

  let description = "This pass computes a memory configuration for autoWS";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numBuffers", "num-buffers",
           "int32_t", /*default*/"0",
           "number of buffering for warp specialization">,
    Option<"smemAllocAlgo", "smem-alloc-algo",
           "int32_t", /*default*/"0",
           "SMEM allocation algorithm: 0 = original, 1 = WSBuffer-based">,
    Option<"smemBudget", "smem-budget",
           "int32_t", /*default*/"0",
           "SMEM budget in bytes (0 = auto-detect from target)">,
    Option<"smemCircularReuse", "smem-circular-reuse",
           "bool", /*default*/"false",
           "Enable circular buffer reuse for SMEM allocation">,
    Option<"readDecisionFile", "read-decision-file",
           "std::string", /*default*/"\"\"",
           "path to JSON file containing buffer decisions to apply">,
    Option<"writeDecisionFile", "write-decision-file",
           "std::string", /*default*/"\"\"",
           "path to JSON file to write buffer decisions to">
  ];
}

def NVGPUTestWSTaskIdPropagate : Pass<"nvgpu-test-taskid-propagate", "mlir::ModuleOp"> {
  let summary = "test warp specialization task id propagation";

  let description = [{
    This pass propagates the `async_task_id` annotation to the dependencies
    of any op that has it set.  This has the functional effect of partitioning
    the graph into multiple async tasks, based on the initial annotation.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGPUWSDataPartition : Pass<"nvgpu-ws-data-partition", "mlir::ModuleOp"> {
  let summary = "warp specialization data partition";

  let description = "This pass partitions operations into multiple suboperations which operate on smaller data shapes";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGPUTestWSCodePartition: Pass<"nvgpu-test-ws-code-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization code partition";

  let description = "This pass generates warp specialized code baed on task id attributes.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::nvws::NVWSDialect"];
  let options = [
    Option<"numBuffers", "num-buffers",
           "int32_t", /*default*/"0",
           "number of buffering for producer-consumer">,
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">,
    Option<"requestedRegisters", "requested-registers",
           "int32_t", /*default*/"232",
           "number of register requested for computation group">,
    Option<"postChannelCreation", "post-channel-creation",
           "int32_t", /*default*/"0",
           "running post channel creation">
  ];
}

def NVGPUTestPingPongSync : Pass<"nvgpu-test-ping-pong-sync", "mlir::ModuleOp"> {
  let summary = "test ping pong sync";

  let description = "This pass inserts named barriers to enforce ping pong around critical resources";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">,
    Option<"capability", "capability",
           "int32_t", /*default*/"10",
           "NVIDIA compute capability">
  ];
}

def NVGPUTest1DTMEMAlloc : Pass<"nvgpu-test-1D-tmem-alloc", "mlir::ModuleOp"> {
  let summary = "test allocating tmem for a 1D tensor that should be passed across partitions.";

  let description = "This pass takes producers with tmem.start and establishes a TMEM allocation for communication with other partitions.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def NVGPUTestWSBufferAllocation : Pass<"nvgpu-test-ws-buffer-allocation", "mlir::ModuleOp"> {
  let summary = "test buffer allocation";

  let description = "This pass creates buffers for each async task channel.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def NVGPUTestWSHoistTMEMStore : Pass<"nvgpu-test-ws-hoist-tmem-store", "mlir::ModuleOp"> {
  let summary = "test hoisting loop-invariant TMEM stores";

  let description = "This pass hoists loop-invariant TMEM stores out of outer loops.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def NVGPUTestPingPongPrep : Pass<"nvgpu-test-ping-pong-prep", "mlir::ModuleOp"> {
  let summary = "test ping pong preprocessing";

  let description = "This pass groups expensive operations into ping-pong regions.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">,
    Option<"capability", "capability",
           "int32_t", /*default*/"10",
           "NVIDIA compute capability">,
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of stages for software pipelining">,
  ];
}

def NVGPUWSTMAStoreLowering : Pass<"nvgpu-ws-tma-store-lowering", "mlir::ModuleOp"> {
  let summary = "Lower descriptor stores to async TMA copies via shared memory";

  let description = [{
    This pass lowers `tt.descriptor_store` ops into an SMEM local_alloc +
    local_store + async TMA copy sequence.  Running it as a standalone pass
    (before partition scheduling) ensures the created `local_alloc` is visible
    to the scheduler and can later be hoisted by buffer allocation.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def NVGPUTestAnnotateTMAStoreWaits : Pass<"nvgpu-test-annotate-tma-store-waits", "mlir::ModuleOp"> {
  let summary = "Annotate TMA store waits with can_rotate_by_buffer_count";

  let description = [{
    This pass walks `scf.for` loops to find `ttng.async_tma_store_token_wait`
    ops whose SMEM buffer has a `buffer.copy` attribute (set by the memory
    planner).  For each such wait, it sets `can_rotate_by_buffer_count = K`
    where K = buffer.copy - 1, indicating that the wait can be delayed by
    up to K iterations because K+1 buffer slots are available.
  }];

  let dependentDialects = ["mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def NVGPUTestTMAStoreTokenWaitReorder : Pass<"nvgpu-test-tma-store-token-wait-reorder", "mlir::ModuleOp"> {
  let summary = "Reschedule TMA store waits using the SWP CoarseSchedule";

  let description = [{
    When a `ttng.async_tma_store_token_wait` op carries the
    `can_rotate_by_buffer_count` attribute (an integer K representing the
    number of SMEM buffer copies), this pass uses the software pipeliner's
    CoarseSchedule to reschedule the wait K positions forward in the
    linearized pipeline order.

    The pass deserializes the CoarseSchedule from the `scf.for` loop,
    walks the linearized schedule from the defining TMA store to find the
    K-th `local_store` to the same buffer, then assigns the wait to a new
    cluster just before that K-th write's cluster.  This ensures the wait
    is placed at the correct pipeline stage for buffer reuse without
    physically moving ops in the IR.

    If the loop has no SWP schedule (no stage/cluster attributes), the
    pass creates a basic single-stage schedule for the entire loop before
    attempting the reorder.
  }];

  let dependentDialects = ["mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def NVGPUTMAStoreTokenWaitLowering : Pass<"nvgpu-tma-store-token-wait-lowering", "mlir::ModuleOp"> {
  let summary = "Lower TMAStoreTokenWaitOp with barriers into TMAStoreWaitOp + ArriveBarrierOp";

  let description = [{
    This pass splits `ttng.async_tma_store_token_wait` ops that have attached
    barriers into a `ttng.async_tma_store_wait` followed by one
    `ttng.arrive_barrier` per barrier.  Running this before the LLVM lowering
    pass allows the membar analysis to insert CTA-level barriers (bar.sync 0)
    between the wait and the arrive, ensuring all warps complete the wait
    before any thread signals the mbarrier.
  }];

  let dependentDialects = ["mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def NVGPUPartitionSchedulingMeta : Pass<"nvgpu-partition-scheduling-meta", "mlir::ModuleOp"> {
  let summary = "Meta warp specialization partitioning pass";

  let description = [{
    The `nvgpu-partition-scheduling-meta` is Meta's version of the partition
    scheduling pass. It analyzes the loads, MMAs, and other operations in a loop
    that is meant to be warp specialized and determines which partitions to
    assign to each operation.
  }];

  let options = [
    Option<"mergeEpilogue", "merge-epilogue",
           "bool", /*default*/"false",
           "If true, merge epilogue ops into the correction/reduction partition "
           "(or computation partition if neither exists)">,
    Option<"mergeEpilogueToComputation", "merge-epilogue-to-computation",
           "bool", /*default*/"false",
           "If true, merge epilogue ops directly into computation[dpId] "
           "partitions, even if correction/reduction exists">,
    Option<"mergeCorrection", "merge-correction",
           "bool", /*default*/"false",
           "If true, merge correction ops into computation[dpId] partitions">,
    Option<"mergeReduction", "merge-reduction",
           "bool", /*default*/"false",
           "If true, merge reduction ops into computation[dpId] partitions">,
    Option<"separateEpilogueStore", "separate-epilogue-store",
           "bool", /*default*/"false",
           "If true, place epilogue store ops in a dedicated 1-warp partition">
  ];
}

def NVGPUMultiCTAReduction : Pass<"nvgpu-multi-cta-reduction", "mlir::ModuleOp"> {
  let summary = "Multi-CTA reduction for NVIDIA GPU";
  let description = [{
    Detects scf.for loops with tt.multi_cta attribute and partitions loop
    iterations across CTAs in a cluster. Post-loop tt.reduce ops are
    transformed into cross-CTA reduction using DSM.
  }];
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}


def NVGPUInsert2CTASync : Pass<"nvgpu-insert-2cta-sync", "mlir::ModuleOp"> {
  let summary = "Insert cross-CTA synchronization for 2-CTA MMA operations";

  let description = [{
    This pass inserts cross-CTA synchronization barriers for TCGen5MMAOp
    operations that use 2-CTA mode (tcgen05.mma.cta_group::2).

    It implements the "arrive remote, wait local" pattern:
    - Both CTAs arrive on the leader CTA's barrier
    - Only the leader CTA (even-ranked) waits
    - Then both CTAs issue the 2-CTA MMA instruction

    This pattern is compatible with warp specialization because the sync
    ops are inserted after the pipeline pass, and the MMA hardware handles
    the actual cross-CTA execution synchronization.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::triton::nvgpu::NVGPUDialect"
  ];
}

def NVGPU2CTATransformLoads : Pass<"nvgpu-2cta-transform-loads", "mlir::ModuleOp"> {
  let summary = "Transform B matrix loads for 2-CTA MMA operations";

  let description = [{
    This pass transforms B matrix descriptor loads for 2-CTA mode. When a
    TCGen5MMAOp has two_ctas=true (and is not async/TLX-managed), each CTA
    should load only half of B (BLOCK_N/2 columns). This pass:

    1. Traces the B operand from the MMA back to its DescriptorLoadOp
    2. Clones the MakeTensorDescOp with half-width block shape
    3. Inserts ClusterCTAIdOp to compute per-CTA offset
    4. Creates a new DescriptorLoadOp loading BLOCK_K x BLOCK_N/2
    5. Updates the LocalAllocOp and MemDesc shapes to match

    This pass is required for the ctas_per_cga=(2,1,1) approach where
    num_ctas=1 (bypasses PlanCTA). It must run after accelerate_matmul
    and before insert_2cta_sync.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::triton::nvgpu::NVGPUDialect"
  ];
}

#endif // NV_TRANSFORMS_PASSES
