// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=65536" | FileCheck %s
// XFAIL: *

// Regression test for B-18-F2 / T273497318.
// The only non-innermost SMEM buffer is larger than every distinct allocated
// target, so the planner must not satisfy the budget by making it reuse itself.

// CHECK-LABEL: @smem_no_self_reuse
// CHECK-NOT: allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @smem_no_self_reuse(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<64x256xf16, #shared>>,
      %c_desc: !tt.tensordesc<tensor<128x256xf16, #shared>>) {
    %A_smem = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %B_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %C_smem = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>

    %outer = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg0 = %c0) -> (i32) : i32 {
      %init = ttng.tmem_store %cst, %result[%token], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %inner:2 = scf.for %kv = %c0 to %c10 step %c1 iter_args(%acc_flag = %false, %acc_tok = %init) -> (i1, !ttg.async.token) : i32 {
        %a = tt.descriptor_load %a_desc[%c0, %c0] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %a, %A_smem : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %b = tt.descriptor_load %b_desc[%c0, %c0] : !tt.tensordesc<tensor<64x256xf16, #shared>> -> tensor<64x256xf16, #blocked2>
        ttg.local_store %b, %B_smem : tensor<64x256xf16, #blocked2> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
        %mma = ttng.tc_gen5_mma %A_smem, %B_smem, %result[%acc_tok], %acc_flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %mma : i1, !ttg.async.token
      }

      %res, %res_tok = ttng.tmem_load %result[%inner#1] {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %res_f16 = arith.truncf %res {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>
      %res_cvt = ttg.convert_layout %res_f16 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #blocked2>
      ttg.local_store %res_cvt, %C_smem {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x256xf16, #blocked2> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %c_val = ttg.local_load %C_smem {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked2>
      tt.descriptor_store %c_desc[%c0, %c0], %c_val {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x256xf16, #shared>>, tensor<128x256xf16, #blocked2>
      scf.yield %arg0 : i32
    } {tt.warp_specialize}

    tt.return
  }
}
