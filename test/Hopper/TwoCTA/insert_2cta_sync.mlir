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

// A software-pipelined persistent kernel can leave MMA copies directly in a
// warp-specialized scf.while body. The cross-CTA barrier lives outside the
// warp-specialize op, so its phase must rotate with the while iteration.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_persistent_while_phase
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_persistent_while_phase(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    ttg.warp_specialize(%a, %b, %acc, %acc_tok)
    default {
      ttg.warp_yield
    }
    partition0(%part_a: !ttg.memdesc<128x64xf16, #shared, #smem>,
               %part_b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
               %part_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %part_acc_tok: !ttg.async.token) num_warps(4) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32 = arith.constant 0 : i32
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      // CHECK: scf.while
      // CHECK: ^bb0(%{{.*}}: i32, %[[WHILE_ITER:.*]]: i64):
      // CHECK: %[[PHASE_STRIDE:.*]] = arith.constant 1 : i64
      // CHECK: %[[LINEAR_ITER:.*]] = arith.muli %[[WHILE_ITER]], %[[PHASE_STRIDE]] : i64
      // CHECK: %[[TWO:.*]] = arith.constant 2 : i64
      // CHECK: %[[REM:.*]] = arith.remui %[[LINEAR_ITER]], %[[TWO]] : i64
      // CHECK: %[[PHASE:.*]] = arith.trunci %[[REM]] : i64 to i32
      // CHECK: ttng.wait_barrier %{{.*}}, %[[PHASE]]
      // CHECK: ttng.tc_gen5_mma
      %result:2 = scf.while (%keep_going = %true, %tile = %c0_i32, %iter = %c0_i64) : (i1, i32, i64) -> (i32, i64) {
        scf.condition(%keep_going) %tile, %iter : i32, i64
      } do {
      ^bb0(%tile_arg: i32, %iter_arg: i64):
        %tok = ttng.tc_gen5_mma %part_a, %part_b, %part_acc[%part_acc_tok], %true, %true {two_ctas} :
          !ttg.memdesc<128x64xf16, #shared, #smem>,
          !ttg.memdesc<64x128xf16, #shared1, #smem>,
          !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %next_iter = arith.addi %iter_arg, %c1_i64 : i64
        scf.yield %false, %tile_arg, %next_iter : i1, i32, i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<128x64xf16, #shared, #smem>,
         !ttg.memdesc<64x128xf16, #shared1, #smem>,
         !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
         !ttg.async.token) -> ()
    tt.return
  }

  // CHECK-LABEL: @test_persistent_while_inner_loop_phase
  tt.func @test_persistent_while_inner_loop_phase(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    ttg.warp_specialize(%a, %b, %acc, %acc_tok)
    default {
      ttg.warp_yield
    }
    partition0(%part_a: !ttg.memdesc<128x64xf16, #shared, #smem>,
               %part_b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
               %part_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %part_acc_tok: !ttg.async.token) num_warps(4) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      // CHECK: scf.while
      // CHECK: ^bb0(%{{.*}}: i32, %[[OUTER_ITER:.*]]: i64):
      // CHECK: scf.for
      // CHECK: %[[OUTER_CONTRIB:.*]] = arith.muli %[[OUTER_ITER]], %[[INNER_TRIP:.*]] : i64
      // CHECK: %[[LINEAR_ITER:.*]] = arith.addi %[[OUTER_CONTRIB]], %{{.*}} : i64
      // CHECK: %[[REM:.*]] = arith.remui %[[LINEAR_ITER]], %{{.*}} : i64
      // CHECK: %[[PHASE:.*]] = arith.trunci %[[REM]] : i64 to i32
      // CHECK: ttng.wait_barrier %{{.*}}, %[[PHASE]]
      // CHECK: ttng.tc_gen5_mma
      %result:2 = scf.while (%keep_going = %true, %tile = %c0_i32, %iter = %c0_i64) : (i1, i32, i64) -> (i32, i64) {
        scf.condition(%keep_going) %tile, %iter : i32, i64
      } do {
      ^bb0(%tile_arg: i32, %iter_arg: i64):
        scf.for %inner = %c0 to %c3 step %c1 {
          %tok = ttng.tc_gen5_mma %part_a, %part_b, %part_acc[%part_acc_tok], %true, %true {two_ctas} :
            !ttg.memdesc<128x64xf16, #shared, #smem>,
            !ttg.memdesc<64x128xf16, #shared1, #smem>,
            !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        }
        %next_iter = arith.addi %iter_arg, %c1_i64 : i64
        scf.yield %false, %tile_arg, %next_iter : i1, i32, i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<128x64xf16, #shared, #smem>,
         !ttg.memdesc<64x128xf16, #shared1, #smem>,
         !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
         !ttg.async.token) -> ()
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
    // CHECK: %[[BARS:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK: %[[INIT_C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[INIT_BAR0:.*]] = ttg.memdesc_index %[[BARS]][%[[INIT_C0]]]
    // CHECK: ttng.init_barrier %[[INIT_BAR0]], 2
    // CHECK: %[[INIT_C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[INIT_BAR1:.*]] = ttg.memdesc_index %[[BARS]][%[[INIT_C1]]]
    // CHECK: ttng.init_barrier %[[INIT_BAR1]], 2
    // CHECK: scf.for
    // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
    // CHECK:   %[[BAR0:.*]] = ttg.memdesc_index %[[BARS]][%[[C0]]]
    // CHECK:   ttng.map_to_remote_buffer %[[BAR0]]
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier %[[BAR0]]
    // CHECK:   ttng.tc_gen5_mma
    // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
    // CHECK:   %[[BAR1:.*]] = ttg.memdesc_index %[[BARS]][%[[C1]]]
    // CHECK:   ttng.map_to_remote_buffer %[[BAR1]]
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier %[[BAR1]]
    // CHECK:   ttng.tc_gen5_mma
    // CHECK: ttng.inval_barrier %{{.*}}
    // CHECK: ttng.inval_barrier %{{.*}}
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
