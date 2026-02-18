// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Test swapTransposedLocalAllocs: when a local_alloc stores into a transposed
// nvmma_shared layout and its sole use is a memdesc_trans feeding into
// operand A of a tc_gen5_mma, swap the layouts so the alloc uses the
// non-transposed layout. This enables buffer sharing with other allocs of the
// same source value that already use non-transposed layout.

// CHECK-LABEL: @swap_transposed_alloc
//
// After buffer allocation, both dsT and dq allocs should have non-transposed
// #shared layout. The two separated local_alloc ops (hoisted above the loop)
// should both be non-transposed:
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
//
// Inside the loop, memdesc_trans goes from #shared (non-transposed) to #shared1
// (transposed), confirming the swap happened:
// CHECK: ttg.memdesc_trans {{.*}} !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swap_transposed_alloc(%desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c4_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 4 : i32
    %dk, %dk_token = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dq, %dq_token = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %k = tt.descriptor_load %desc_k[%c0_i32, %c0_i32] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked>
    %k_smem = ttg.local_alloc %k {async_task_id = array<i32: 1>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %q = tt.descriptor_load %desc_q[%c0_i32, %c0_i32] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked>
    %q_smem = ttg.local_alloc %q {async_task_id = array<i32: 1>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %loop:4 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%use_d = %false, %dk_dep = %dk_token, %dq_dep = %dq_token, %prev = %true) -> (i1, !ttg.async.token, !ttg.async.token, i1) : i32 {
      %dsT_val = tt.descriptor_load %desc_k[%c0_i32, %c0_i32] {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked>
      // dsT alloc: non-transposed layout, feeds dk MMA operand A directly.
      %dsT = ttg.local_alloc %dsT_val {async_task_id = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %dk_tok = ttng.tc_gen5_mma %dsT, %q_smem, %dk[%dk_dep], %use_d, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // dq alloc: TRANSPOSED layout, then memdesc_trans back to non-transposed.
      // This is the pattern that should be swapped.
      %dq_alloc = ttg.local_alloc %dsT_val {async_task_id = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared_T, #smem>
      %dq_trans = ttg.memdesc_trans %dq_alloc {async_task_id = array<i32: 0>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared_T, #smem> -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %dq_tok = ttng.tc_gen5_mma %dq_trans, %k_smem, %dq[%dq_dep], %use_d, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %dk_tok, %dq_tok, %prev : i1, !ttg.async.token, !ttg.async.token, i1
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.warp_specialize}
    tt.return
  }
}

// -----

// Negative test: memdesc_trans feeds into operand B (not A) of tc_gen5_mma.
// The swap should NOT apply.

// CHECK-LABEL: @no_swap_operand_b
// The transposed alloc should remain transposed (no swap).
// Note: #shared1 is the transposed layout alias in the output.
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>

#blocked_2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T_2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_2 = #ttg.shared_memory
#tmem_2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_swap_operand_b(%desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared_2>>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c4_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 4 : i32
    %acc, %acc_token = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem_2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %a_val = tt.descriptor_load %desc_k[%c0_i32, %c0_i32] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared_2>> -> tensor<128x128xbf16, #blocked_2>
    %a_smem = ttg.local_alloc %a_val {async_task_id = array<i32: 1>} : (tensor<128x128xbf16, #blocked_2>) -> !ttg.memdesc<128x128xbf16, #shared_2, #smem_2>
    %loop:2 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%use_d = %false, %dep = %acc_token) -> (i1, !ttg.async.token) : i32 {
      %b_val = tt.descriptor_load %desc_k[%c0_i32, %c0_i32] {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xbf16, #shared_2>> -> tensor<128x128xbf16, #blocked_2>
      // Transposed alloc whose memdesc_trans feeds operand B, not A.
      %b_alloc = ttg.local_alloc %b_val {async_task_id = array<i32: 3>} : (tensor<128x128xbf16, #blocked_2>) -> !ttg.memdesc<128x128xbf16, #shared_T_2, #smem_2>
      %b_trans = ttg.memdesc_trans %b_alloc {async_task_id = array<i32: 0>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared_T_2, #smem_2> -> !ttg.memdesc<128x128xbf16, #shared_2, #smem_2>
      // Note: %b_trans is operand B (second operand), not A.
      %tok = ttng.tc_gen5_mma %a_smem, %b_trans, %acc[%dep], %use_d, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xbf16, #shared_2, #smem_2>, !ttg.memdesc<128x128xbf16, #shared_2, #smem_2>, !ttg.memdesc<128x128xf32, #tmem_2, #ttng.tensor_memory, mutable>
      scf.yield %true, %tok : i1, !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.warp_specialize}
    tt.return
  }
}
