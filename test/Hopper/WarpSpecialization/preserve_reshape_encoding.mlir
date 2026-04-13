// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Test that doBufferAllocation preserves the encoding of memdesc_reshape ops.
// When a local_alloc with shared_linear encoding feeds into a memdesc_reshape
// that produces nvmma_shared encoding, the buffer allocation should preserve
// the nvmma_shared encoding on the reshape output, not re-infer it (which
// would incorrectly produce shared_linear).

// Note: #shared = shared_linear (3D), #shared1 = nvmma_shared (2D) in output.

// CHECK-LABEL: @preserve_reshape_nvmma_shared
//
// The local_alloc is hoisted and made mutable with shared_linear encoding:
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<128x2x32xbf16, #shared, #smem, mutable>
// CHECK: scf.for
// CHECK:   ttg.local_store
// The reshape output must preserve nvmma_shared (#shared1), not shared_linear:
// CHECK:   ttg.memdesc_reshape {{.*}} -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
// CHECK:   ttng.tc_gen5_mma

#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sl3d = #ttg.shared_linear<{offset = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0], [1, 0, 8], [2, 0, 16], [4, 1, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]]}, alignment = 1024>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_reshape_nvmma_shared(%src_3d: tensor<128x2x32xbf16, #blocked3d>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c4_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 4 : i32
    %acc, %acc_token = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // B operand
    %b_smem = ttg.local_alloc {async_task_id = array<i32: 1>} : () -> !ttg.memdesc<64x128xbf16, #nvmma, #smem, mutable>
    %loop:2 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%use_d = %false, %dep = %acc_token) -> (i1, !ttg.async.token) : i32 {
      // Producer (task 3): alloc A with shared_linear 3D encoding
      %a_alloc = ttg.local_alloc %src_3d {async_task_id = array<i32: 3>} : (tensor<128x2x32xbf16, #blocked3d>) -> !ttg.memdesc<128x2x32xbf16, #sl3d, #smem>
      // Consumer (task 0): reshape to nvmma_shared 2D encoding, then MMA
      %a_reshaped = ttg.memdesc_reshape %a_alloc {async_task_id = array<i32: 0>} : !ttg.memdesc<128x2x32xbf16, #sl3d, #smem> -> !ttg.memdesc<128x64xbf16, #nvmma, #smem>
      %tok = ttng.tc_gen5_mma %a_reshaped, %b_smem, %acc[%dep], %use_d, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xbf16, #nvmma, #smem>, !ttg.memdesc<64x128xbf16, #nvmma, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %tok : i1, !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.warp_specialize}
    tt.return
  }
}
