// RUN: triton-opt %s -split-input-file --nvgpu-insert-2cta-sync | FileCheck %s

// Test that the pass inserts cross-CTA sync before 2-CTA MMA ops.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_insert_2cta_sync_in_loop
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_insert_2cta_sync_in_loop(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: ttg.local_alloc
    // CHECK: ttng.init_barrier
    // CHECK: scf.for
    // CHECK:   nvg.cluster_id
    // CHECK:   ttng.map_to_remote_buffer
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier
    // CHECK:   ttng.tc_gen5_mma
    scf.for %iv = %c0 to %c4 step %c1 {
      %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
        !ttg.memdesc<128x64xf16, #shared, #smem>,
        !ttg.memdesc<64x128xf16, #shared1, #smem>,
        !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    tt.return
  }
}

// -----

// Test that the pass skips when no cluster (cluster_dim < 2).

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_no_sync_without_cluster
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_no_sync_without_cluster(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %true = arith.constant true
    // CHECK-NOT: nvg.cluster_id
    // CHECK-NOT: ttng.arrive_barrier
    // CHECK: ttng.tc_gen5_mma
    %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test that multiple 2-CTA MMA ops in the same loop each get their own
// cross-CTA barrier slot.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_insert_2cta_sync_multiple_mmas_in_loop
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_insert_2cta_sync_multiple_mmas_in_loop(
      %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok0: !ttg.async.token,
      %acc_tok1: !ttg.async.token) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK: ttng.init_barrier
    // CHECK: ttng.init_barrier
    // CHECK: scf.for
    // CHECK:   ttng.map_to_remote_buffer
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier
    // CHECK:   ttng.tc_gen5_mma
    // CHECK:   ttng.map_to_remote_buffer
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier
    // CHECK:   ttng.tc_gen5_mma
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    scf.for %iv = %c0 to %c4 step %c1 {
      %tok0 = ttng.tc_gen5_mma %a0, %b, %acc0[%acc_tok0], %true, %true {two_ctas} :
        !ttg.memdesc<128x64xf16, #shared, #smem>,
        !ttg.memdesc<64x128xf16, #shared1, #smem>,
        !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tok1 = ttng.tc_gen5_mma %a1, %b, %acc1[%acc_tok1], %true, %true {two_ctas} :
        !ttg.memdesc<128x64xf16, #shared, #smem>,
        !ttg.memdesc<64x128xf16, #shared1, #smem>,
        !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    tt.return
  }
}
