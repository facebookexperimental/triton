// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -tritongpu-promote-lhs-to-tmem | FileCheck %s

// An NPOT K A goes to tensor memory in blocks of its pow2 alloc shape.
#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#bl = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$A:.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
  // CHECK-LABEL: @promote_k96
  // CHECK: %[[A:.*]] = ttng.tmem_alloc %{{.*}} : (tensor<128x96xf16, #{{.*}}>) -> !ttg.memdesc<128x96xf16, #[[$A]], #ttng.tensor_memory, 128x128>
  // CHECK: ttng.tc_gen5_mma %[[A]],
  tt.func public @promote_k96(%b: !ttg.memdesc<96x64xf16, #shared, #ttg.shared_memory>,
                              %c: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %a = arith.constant dense<1.0> : tensor<128x96xf16, #tl>
    %a_sh = ttg.local_alloc %a : (tensor<128x96xf16, #tl>) -> !ttg.memdesc<128x96xf16, #shared, #ttg.shared_memory>
    ttng.tc_gen5_mma %a_sh, %b, %c, %true, %true : !ttg.memdesc<128x96xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<96x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-promote-lhs-to-tmem -tritongpu-assign-latencies=num-stages=2 -tritongpu-schedule-loops=num-stages=2 -tritongpu-pipeline=num-stages=2 | FileCheck %s --check-prefix=PIPE
  // FA fwd h64, BLOCK_M 128, BLOCK_N 96, 2 stages: the pipeliner hoists the
  // P x V operand's TMEM buffer out of the loop, and it keeps its pow2 alloc
  // shape.
  // PIPE-LABEL: @hoist_k96
  // PIPE: %[[P:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x96xf16, #{{[^,]*}}, #ttng.tensor_memory, mutable, 128x128>
  // PIPE: scf.for
  // PIPE: ttng.tmem_store %{{.*}}, %[[P]],
  // PIPE: ttng.tc_gen5_mma %[[P]],
  tt.func public @hoist_k96(%v: !ttg.memdesc<96x64xf16, #shared, #ttg.shared_memory>, %n: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %o, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      %f = arith.sitofp %i : i32 to f16
      %p = tt.splat %f : f16 -> tensor<128x96xf16, #tl>
      %p_sh = ttg.local_alloc %p : (tensor<128x96xf16, #tl>) -> !ttg.memdesc<128x96xf16, #shared, #ttg.shared_memory>
      %t1 = ttng.tc_gen5_mma %p_sh, %v, %o[%t], %true, %true : !ttg.memdesc<128x96xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<96x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %t1 : !ttg.async.token
    }
    tt.return
  }

  // PIPE-LABEL: @hoist_k128
  // PIPE: %[[P:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #{{[^,]*}}, #ttng.tensor_memory, mutable>
  // PIPE: scf.for
  // PIPE: ttng.tmem_store %{{.*}}, %[[P]],
  // PIPE: ttng.tc_gen5_mma %[[P]],
  tt.func public @hoist_k128(%v: !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, %n: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %o, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      %f = arith.sitofp %i : i32 to f16
      %p = tt.splat %f : f16 -> tensor<128x128xf16, #bl>
      %p_sh = ttg.local_alloc %p : (tensor<128x128xf16, #bl>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %t1 = ttng.tc_gen5_mma %p_sh, %v, %o[%t], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %t1 : !ttg.async.token
    }
    tt.return
  }
}
