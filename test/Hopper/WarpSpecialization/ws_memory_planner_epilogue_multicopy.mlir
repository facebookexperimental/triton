// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=220000" | FileCheck %s --check-prefix=LARGE
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=200000" | FileCheck %s --check-prefix=TIGHT

// Test: Phase 4.5 multi-copy for fused epilogue buffers.
// Two epilogue SMEM buffers (128x128xf16 = 32768 bytes each) are fused into
// the same buffer.id by Phase 3.5. Phase 4 gives innermost-loop buffers
// (A: 128x64xf16 = 16384, B: 64x256xf16 = 32768) up to 3 copies.
//
// With a large budget (220000):
//   Innermost: (16384 + 32768) * 3 = 147456
//   Epilogue fused (2 copies): 32768 * 2 = 65536
//   Total: 212992 ≤ 220000 → epilogue gets buffer.copy=2.
//
// With a tight budget (200000):
//   Innermost: 147456
//   Epilogue fused (1 copy): 32768
//   Total: 180224 ≤ 200000, but 2 copies → 212992 > 200000
//   → epilogue stays at buffer.copy=1.

// LARGE-LABEL: @epilogue_multicopy
// LARGE: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = [[ID:[0-9]+]] : i32}
// LARGE-SAME: 128x128xf16
// LARGE: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = [[ID]] : i32}
// LARGE-SAME: 128x128xf16

// TIGHT-LABEL: @epilogue_multicopy
// TIGHT: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[ID:[0-9]+]] : i32}
// TIGHT-SAME: 128x128xf16
// TIGHT: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[ID]] : i32}
// TIGHT-SAME: 128x128xf16

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @epilogue_multicopy(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<64x256xf16, #shared>>,
      %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>) {
    // Innermost-loop SMEM buffers (for A and B operands).
    %A_smem = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %B_smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    // Epilogue SMEM buffers — both fed from the same tmem_load via split.
    %C0_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C1_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {async_task_id = array<i32: 0>} false
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c10 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 10 : i32
    %c64 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c128 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // Outer persistent loop.
    %0 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg0 = %c0) -> (i32) : i32 {
      %init = ttng.tmem_store %cst, %result[%token], %true {async_task_id = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // Inner k-loop (innermost loop).
      %1:2 = scf.for %kv = %c0 to %c10 step %c1 iter_args(%acc_flag = %false, %acc_tok = %init) -> (i1, !ttg.async.token) : i32 {
        %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %a, %A_smem {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<64x256xf16, #shared>> -> tensor<64x256xf16, #blocked2>
        ttg.local_store %b, %B_smem {async_task_id = array<i32: 1>} : tensor<64x256xf16, #blocked2> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
        %mma = ttng.tc_gen5_mma %A_smem, %B_smem, %result[%acc_tok], %acc_flag, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %mma : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1>}
      // Epilogue: tmem_load → reshape → trans → split → truncf → local_store.
      %res, %res_tok = ttng.tmem_load %result[%1#1] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %reshaped = tt.reshape %res {async_task_id = array<i32: 2>} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked3>
      %transposed = tt.trans %reshaped {async_task_id = array<i32: 2>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3> -> tensor<128x128x2xf32, #blocked4>
      %lhs, %rhs = tt.split %transposed {async_task_id = array<i32: 2>} : tensor<128x128x2xf32, #blocked4> -> tensor<128x128xf32, #blocked5>
      // First sub-tile: truncf → convert_layout → local_store to C0_smem.
      %lhs_f16 = arith.truncf %lhs {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %lhs_cvt = ttg.convert_layout %lhs_f16 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %lhs_cvt, %C0_smem {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Consumer of C0_smem: TMA store.
      %c0_val = ttg.local_load %C0_smem {async_task_id = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %c_desc[%c0, %c0], %c0_val {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      // Second sub-tile: truncf → convert_layout → local_store to C1_smem.
      %rhs_f16 = arith.truncf %rhs {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %rhs_cvt = ttg.convert_layout %rhs_f16 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %rhs_cvt, %C1_smem {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Consumer of C1_smem: TMA store.
      %c1_val = ttg.local_load %C1_smem {async_task_id = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %c_desc[%c0, %c128], %c1_val {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %arg0 : i32
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize}
    tt.return
  }
}
