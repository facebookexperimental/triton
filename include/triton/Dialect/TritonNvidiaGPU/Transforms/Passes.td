// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef TRITONNVIDIAGPU_PASSES
#define TRITONNVIDIAGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPlanCTAPass : Pass<"triton-nvidia-gpu-plan-cta", "mlir::ModuleOp"> {
  let summary = "plan CTA";

  let description = [{
    This pass computes and applies "optimized" CTA tilings to DotOp, ReduceOp
    and StoreLikeOps operations.
  }];

  let constructor = "mlir::triton::nvidia_gpu::createTritonNvidiaGPUPlanCTAPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonGPUFenceInsertion : Pass<"triton-nvidia-gpu-fence-insertion", "mlir::ModuleOp"> {
  let summary = "Insert fences across generic and async proxy.";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
    This pass inserts fences at optimized location.
    There is a pass later to handle all the functional requirements
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"90",
           "device compute capability">
  ];
}

def TritonGPUProxyFenceInsertion : Pass<"triton-nvidia-gpu-proxy-fence-insertion", "mlir::ModuleOp"> {
  let summary = "Insert fences across generic and async proxy";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"90",
           "device compute capability">
  ];
}

def TritonNvidiaGPUFuseTMEMLoadReducePass : Pass<"triton-nvidia-tmem-load-reduce", "mlir::ModuleOp"> {
  let summary = "Fuse tmem_load and reduce";

  let description = [{
    Fuse ttng.tmem_load followed by a row reduction into ttng.tmem_load with a
    reduction op, to generate tcgen05.ld.red.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUTMALoweringPass : Pass<"triton-nvidia-tma-lowering", "mlir::ModuleOp"> {
  let summary = "lower to TMA load/store operations";

  let description = [{
    Lower Triton descriptor load to TMA load/store operations in TritonNvidiaGPUDialect.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUTMAStoreBufferReusePass
    : Pass<"triton-nvidia-tma-store-buffer-reuse", "mlir::ModuleOp"> {
  let summary = "Reuse SMEM buffers across sequential TMA stores";
  let description = [{
    After TMA lowering, sequential descriptor stores each allocate their own
    shared memory buffer. When a tma_store_wait with pendings=0 guarantees
    the buffer is safe to reuse, this pass merges compatible allocations
    into a single mutable buffer with local_store writes.
  }];
  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonTensorMemoryAllocationPass : Pass<"triton-tensor-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign tensor memory allocation";

  let description = [{
    Decide on tensor memory allocation and assign attributes to each allocation.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUMMALoweringPass : Pass<"triton-nvidia-mma-lowering", "mlir::ModuleOp"> {
  let summary = "lower mma operations if needed";

  let description = [{
    Lower MMA ops to prepare for conversion to LLVM.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUPromoteLHSToTMemPass : Pass<"tritongpu-promote-lhs-to-tmem", "mlir::ModuleOp"> {
  let summary = "Promote LHS operand of MMAv5 op to Tensor Memory";

  let description = [{
    Promote LHS operand of MMAv5 op to Tensor Memory.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUOptimizeDescriptorEncodingPass : Pass<"triton-nvidia-optimize-descriptor-encoding", "mlir::ModuleOp"> {
  let summary = "Set encodings on tensor descriptor types";

  let description = [{
    Set shared memory encoding on tensor descriptors, which decides the swizzling mode and message size of the tma descriptor.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUOptimizeTMemLayoutsPass : Pass<"triton-nvidia-optimize-tmem-layouts", "mlir::ModuleOp"> {
  let summary = "Optimize TMEM layouts.";

  let description = [{
    Optimize TMEM layouts by selecting a layouts to enable better subtiling,
    reduction performance, etc.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUInterleaveTMemPass : Pass<"triton-nvidia-interleave-tmem", "mlir::ModuleOp"> {
  let summary = "Interleave TMEM loads/stores.";

  let description = [{
    The `triton-nvidia-interleave-tmem` pass attempts to sink TMEM loads and
    hoist TMEM stores, and potentially interleave them, to reduce register
    pressure.
  }];
}

def TritonNvidiaGPUTestGenerateSubtiledRegionPass
    : Pass<"triton-nvidia-gpu-test-generate-subtiled-region", "mlir::ModuleOp"> {
  let summary = "Test pass: generate subtiled_region ops from split patterns";

  let description = [{
    This pass finds the GEMM epilogue subtiling pattern:
      tmem_load -> reshape -> trans{[0,2,1]} -> split
    followed by per-tile code (truncf, convert_layout, TMA store), and wraps
    it in a `ttng.subtiled_region` op.

    The pass runs after the memory planner and before code partition in the WS
    pipeline. It captures the setup chain (tmem_load through split) in the
    setup region and the per-tile code in the tile region body.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::arith::ArithDialect"
  ];
}



def TritonNvidiaGPURemoveTMEMTokensPass : Pass<"triton-nvidia-gpu-remove-tmem-tokens", "mlir::ModuleOp"> {
  let summary = "remove TMEM tokens";

  let description = [{
    The `triton-nvidia-gpu-remove-tmem-tokens` pass removes TMEM memory
    dependency tokens from the IR, after they are no longer needed.
  }];
}

def TritonNvidiaGPUPruneUnusedBarriersPass
    : Pass<"triton-nvidia-gpu-prune-unused-barriers", "mlir::ModuleOp"> {
  let summary = "Prune barriers with no wait uses after warp specialization";

  let description = [{
    After warp specialization materializes barriers for producer-consumer
    communication channels, some barriers may have no corresponding wait ops.
    This pass finds and removes such unused barriers and their associated
    init/arrive/expect/commit ops.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUCheckMatmulTwoCTAPass : Pass<"triton-nvidia-check-matmul-two-cta", "mlir::ModuleOp"> {
  let summary = "Verify legal and consistent two_ctas usage across matmuls";

  let description = [{
    Inspect all matmul operations and ensure they agree on the `two_ctas`
    setting. Propagate the chosen value to the module so later lowering steps
    can access it. Compilation fails if mixed configurations are detected, or if
    one 2-CTA `tt.dot` consumes a value derived from another 2-CTA `tt.dot`.
  }];
}

def TritonNvidiaGPUCLCSplitPass : Pass<"triton-nvidia-gpu-clc-split", "mlir::ModuleOp"> {
  let summary = "Split ttng.clc_advance into the token form (clc_try_cancel_async + clc_read)";

  let description = [{
    Stage 1 of the CLC tile-scheduler lowering. Rewrites each `ttng.clc_advance`
    into `ttng.clc_try_cancel_async` (async issue -> !ttg.async.token) followed by
    `ttng.clc_read` (await the token -> decoded tile). No mbarrier/buffer is
    introduced; the token models the async dependency.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUCLCHoistPass : Pass<"triton-nvidia-gpu-clc-hoist", "mlir::ModuleOp"> {
  let summary = "Hoist ttng.clc_try_cancel_async to overlap CLC latency with compute";

  let description = [{
    Stage 2 of the CLC tile-scheduler lowering. Promotes each
    `ttng.clc_try_cancel_async` to the top of its block so the CLC round-trip
    overlaps the tile's compute before the paired `ttng.clc_read`.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUCLCMaterializePass : Pass<"triton-nvidia-gpu-clc-materialize", "mlir::ModuleOp"> {
  let summary = "Materialize the CLC token form into response and reuse barriers";

  let description = [{
    Stage 4 of the CLC tile-scheduler lowering. Materializes each
    `clc_try_cancel_async` / `clc_read` pair into a response buffer + completion
    mbarrier. Explicit clusters also get an empty-barrier rendezvous that
    prevents reuse until every CTA consumes the multicast. The issue becomes
    `barrier_expect` + `clc_try_cancel`, and the read
    becomes `wait_barrier` + `clc_load_result` + `clc_is_canceled` /
    `clc_get_program_id`, with a loop-carried, per-iteration-toggled phase.
    Supports single-CTA and explicit ctas_per_cga program-id semantics;
    Triton's num-ctas > 1 program-id model is rejected.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::triton::nvgpu::NVGPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TritonNvidiaGPUPromoteLoadToTMAPass : Pass<"triton-nvidia-promote-load-to-tma", "mlir::ModuleOp"> {
  let summary = "Auto-promote eligible tt.load to a real TMA descriptor_load";

  let description = [{
    Analyze pointer-arithmetic patterns in tt.load operations. When a masked,
    block-structured load is TMA-eligible (contiguous innermost dimension, rank
    1-5, supported element type, >= 16 byte inner dim, zero OOB fill), rewrite it
    into the explicit-TMA form -- tt.make_tensor_descriptor + tt.descriptor_load
    -- so the standard nvidia TMA lowering turns it into an asynchronous TMA copy
    on sm_90+. Non-eligible loads are left unchanged.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif
