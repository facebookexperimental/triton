// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect "-tritongpu-schedule-loops=num-stages=3 use-meta-ws=true" | FileCheck %s

// A descriptor load whose buffer feeds both MMA and scalar work is limited to
// two software-pipeline stages.  This prevents the next TMA acquire from being
// scheduled before the forwarding copy for the current MMA operand.

// CHECK-LABEL: @mixed_descriptor_consumers
// CHECK: tt.descriptor_load {{.*}}loop.stage = 0 : i32
// CHECK: ttng.tc_gen5_mma {{.*}}loop.stage = 0 : i32
// CHECK: tt.scheduled_max_stage = 1 : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mixed_descriptor_consumers(%desc: !tt.tensordesc<128x64xf16, #shared>, %b: !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc_tok: !ttg.async.token, %out: !tt.ptr<f16>, %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {tt.latency = 2 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a = ttg.local_alloc %tile : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %mma_tok = ttng.tc_gen5_mma %a, %b, %acc[%tok], %false, %true {tt.latency = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %loaded = ttg.local_load %a : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
      tt.store %ptrs, %loaded : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// An ordinary descriptor-to-MMA path still uses all three requested stages.

// CHECK-LABEL: @mma_only_descriptor_consumer
// CHECK: tt.descriptor_load {{.*}}loop.stage = 0 : i32
// CHECK: ttng.tc_gen5_mma {{.*}}loop.stage = 2 : i32
// CHECK: tt.scheduled_max_stage = 2 : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mma_only_descriptor_consumer(%desc: !tt.tensordesc<128x64xf16, #shared>, %b: !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc_tok: !ttg.async.token, %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {tt.latency = 2 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a = ttg.local_alloc %tile : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %mma_tok = ttng.tc_gen5_mma %a, %b, %acc[%tok], %false, %true {tt.latency = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}
