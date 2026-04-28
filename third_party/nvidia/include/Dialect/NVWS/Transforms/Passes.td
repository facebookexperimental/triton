// Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NVWS_PASSES
#define NVWS_PASSES

include "mlir/Pass/PassBase.td"

def NVWSLowerWarpGroup : Pass<"nvws-lower-warp-group", "mlir::ModuleOp"> {
  let summary = "Convert nvws.warp_group to ttg.warp_specialize.";

  let description = [{
    Convert nvws.warp_group to ttg.warp_specialize.

    If the first group of nvws.warp_group matches the global
    ttg.num_warps, it will be come the default region of ttg.warp_specialize.
    If not, the ttg.warp_specialize default region will be empty, and all
    warp groups will become isolated regions.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def NVWSAssignStagePhase : Pass<"nvws-assign-stage-phase", "mlir::ModuleOp"> {
  // See sema-docs/assign-stage-phase-and-lower-semaphores.md.
  let summary = "Assign buffer stage to nvws.semaphore.*.";

  let description = [{
    Assign buffer stage & phase to nvws.semaphore.*

    The pass will assign buffer stage to each semaphore op, and phase for
    acquire/release ops.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

}

def NVWSLowerSemaphore : Pass<"nvws-lower-semaphore", "mlir::ModuleOp"> {
  // See sema-docs/assign-stage-phase-and-lower-semaphores.md.
  let summary = "Convert nvws.semaphore.* to ttng.*barrier* ops.";

  let description = [{
    Convert nvws.semaphore.* to ttng.*barrier* ops.

    The pass will convert each semaphore to a matched barrier set,
    and will determined appropriate waits/signalling for values being
    "empty" or "full" from the use/def chain of semaphore acquire/release.

    This lowering may yield non-ideal parallelism in certain cases,
    which will be optimized by follow up peephole passes.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def NVWSInsertAllocas: Pass<"nvws-insert-allocas", "mlir::ModuleOp"> {
  // See sema-docs/insert-allocas.md.
  let summary = "Materialize NVWS communication allocations and accesses.";

  let description = [{
    Materializes explicit local/TMEM communication buffers and memory accesses
    for cross-partition SSA values without inserting semaphore operations.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

}

def NVWSMetaToNVWSConvert
    : Pass<"nvws-meta-to-nvws-convert", "mlir::ModuleOp"> {
  let summary = "Convert Meta warp-specialization planning metadata to NVWS IR";

  let description = [{
    Mechanically converts the output of the Meta warp-specialization planning
    prefix into the representation consumed by the NVWS mechanism passes. The
    pass does not choose partitions, schedules, buffers, or synchronization.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSOrderBufferGroups
    : Pass<"nvws-order-buffer-groups", "mlir::ModuleOp"> {
  let summary = "Order NVWS buffer groups by predicted producer readiness";

  let description = [{
    Orders eligible source-free entry-block TMEM allocation representatives so
    that InsertSemas processes groups which are predicted to become ready later
    before groups predicted to become ready earlier.  This makes the order of
    co-located blocking waits an explicit pre-InsertSemas policy instead of an
    accidental consequence of memory-planner allocation order.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSPackEpilogueSlices
    : Pass<"nvws-pack-epilogue-slices", "mlir::ModuleOp"> {
  let summary = "Pack data-partitioned epilogue producer/store slices";

  let description = [{
    Shortens register live ranges in data-partitioned merged epilogues by
    placing each explicit local_store immediately after its producer slice.
    This is a scheduling-only optimization over buffers already materialized
    by Meta buffer allocation and must run before memory planning.
  }];

  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSInsertSemas: Pass<"nvws-insert-semas", "mlir::ModuleOp"> {
  // See sema-docs/insert-semas/overview.md.
  let summary = "Insert semaphore ops for ownership transfer.";

  let description = [{
    Access-DAG-driven SMEM/TMEM semaphore insertion for ownership transfer
    between warp-specialized partitions.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"useMetaPartitioner", "use-meta-partitioner",
           "bool", /*default*/"false",
           "If true, InsertSemas is running after the NVWS meta partitioner">,
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "pipeline depth used by LowerSemaphore auto-multibuffering">
  ];
}

def NVWSHoistTmemStore: Pass<"nvws-hoist-tmem-store", "mlir::ModuleOp"> {
  let summary = "Hoist tmem store before the inner loop to the top level if possible.";

  let description = [{
    The HoistTMEMAlloc pass in TritonGPU, when applied to nested loops, puts the hoisted alloc and store inside the outer loop.
    Given such input IR, this pass tries to hoist alloc and store across all loop nests, while threading the token variable appropriately.

    For example, this IR

    scf.for ... {
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>}
      %16 = ttng.tmem_store %zero, %result[%token], %true {ttg.partition = array<i32: 0>}
      scf.for ... iter_args(%useD = %false, %arg9 = %16){
        ...
        %28 = ttng.tc_gen5_mma %lhs, %rhs, %result[%arg9], %useD, %true {ttg.partition = array<i32: 1>}
        ...
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %28
      }
    }{tt.warp_specialize, ...}

    is transformed into

    %result, %token = ttng.tmem_alloc %zero {ttg.partition = array<i32: 0>}
    scf.for ... iter_args(%token_arg = %token) { // The token variable is threaded across loops
      %res = scf.for ... iter_args(%useD = %false, %arg9 = %token_arg){
        ...
        %28 = ttng.tc_gen5_mma %lhs, %rhs, %result[%arg9], %useD, %true {ttg.partition = array<i32: 1>}
        ...
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %28
      }
      yield %res#0 // Note there is now an explicit yield op
    }{tt.warp_specialize, ...}

    This is valid, since the useD flag initialized to false means that the zero clear of the accumulator can be skipped.
    If the inner loop does not execute at all, we would be returning the accumulator filled with zeros for all output tiles.

    This transformation is strictly an optimization. Note that the tmem_store before the inner loop is assigned to the partition 0, while the accumulator
    is used by the MMA op in partition 1. This would result in a semaphore being created for this use of TMEM, along with acquire/release in
    the two partitions, meaning an additional synchronization before the inner loop just to clear the accumulator. When the useD flag is intialized to false,
    hoisting the tmem_store to the top level eliminates such unnecessary synchronization.

    Cares must be taken in such hoisting across loop nests. This transformation is valid as long as all instances of the inner loop execute
    the same number of times - either at least once or none. This does not hold when the number of iterations of the inner loop depends on an outer-loop
    iterator. But even in the presece of a variable iteration count, hoisting is still valid if we can statically prove that the inner loop executes
    at least once. A Triton kernel can use tl.assume op to assert a certain bound on a variable. Given an inner loop with a variable iteration count,
    this pass checks if there is an assumption on the bounds of the loop which allows us to prove that the loop executes at least once.
    Hoisting is enabled in such cases.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSStripPartitionAttrsOutsideWS
    : Pass<"nvws-strip-partition-attrs-outside-ws", "mlir::triton::FuncOp"> {
  let summary = "Strip partition attrs outside tt.warp_specialize loops.";

  let description = [{
    Removes internal warp-specialization partition attributes from operations
    outside tt.warp_specialize loops while preserving the attributes on the
    loop itself and operations nested inside it.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // NVWS_PASSES
