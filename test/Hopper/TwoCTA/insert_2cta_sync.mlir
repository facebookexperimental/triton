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
    // CHECK: %[[BARS:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK: %[[INIT_C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[INIT_BAR0:.*]] = ttg.memdesc_index %[[BARS]][%[[INIT_C0]]]
    // CHECK: ttng.init_barrier %[[INIT_BAR0]], 2
    // CHECK: %[[INIT_C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[INIT_BAR1:.*]] = ttg.memdesc_index %[[BARS]][%[[INIT_C1]]]
    // CHECK: ttng.init_barrier %[[INIT_BAR1]], 2
    // Each MMA owns its own base index (0 and 1 here); the slot add is emitted
    // unconditionally and folds to the bare base at depth 1 once the
    // canonicalizer runs.
    // CHECK: scf.for
    // CHECK:   %[[C0:.*]] = arith.constant 0 : i32
    // CHECK:   %[[IDX0:.*]] = arith.addi %[[C0]], %{{.*}} : i32
    // CHECK:   %[[BAR0:.*]] = ttg.memdesc_index %[[BARS]][%[[IDX0]]]
    // CHECK:   ttng.map_to_remote_buffer %[[BAR0]]
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier %[[BAR0]]
    // CHECK:   ttng.tc_gen5_mma
    // CHECK:   %[[C1:.*]] = arith.constant 1 : i32
    // CHECK:   %[[IDX1:.*]] = arith.addi %[[C1]], %{{.*}} : i32
    // CHECK:   %[[BAR1:.*]] = ttg.memdesc_index %[[BARS]][%[[IDX1]]]
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

// -----

// Multi-buffered operands: the cross-CTA barrier must be allocated at the same
// depth as the operand buffers and indexed by the same rotating slot/phase.
// With a single slot and a bare parity phase, a follower CTA that runs ahead
// within the operand pipeline laps the phase bit and the CTA pair deadlocks.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_2cta_sync_multibuffered_operands
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_2cta_sync_multibuffered_operands(
      %abuf: !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>,
      %bbuf: !ttg.memdesc<3x64x128xf16, #shared1, #smem, mutable>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    // Barrier array is 3 deep (matching the operand buffers), not 1.
    // CHECK: %[[BARS:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK-COUNT-3: ttng.init_barrier %{{.*}}, 2
    // CHECK: scf.for
    // Slot = iter % 3, phase = (iter / 3) & 1 -- not a bare parity.
    // CHECK:   %[[D:.*]] = arith.constant 3 : i64
    // CHECK:   %[[SLOT:.*]] = arith.remui %{{.*}}, %[[D]]
    // CHECK:   %[[GEN:.*]] = arith.divui %{{.*}}, %[[D]]
    // CHECK:   arith.andi %[[GEN]]
    // CHECK:   %[[IDX:.*]] = arith.addi
    // CHECK:   %[[BAR:.*]] = ttg.memdesc_index %[[BARS]][%[[IDX]]]
    // CHECK:   ttng.map_to_remote_buffer %[[BAR]]
    // CHECK:   ttng.arrive_barrier
    // CHECK:   ttng.wait_barrier %[[BAR]]
    // CHECK:   ttng.tc_gen5_mma
    scf.for %iv = %c0 to %c4 step %c1 {
      %a = ttg.memdesc_index %abuf[%c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b = ttg.memdesc_index %bbuf[%c0_i32] : !ttg.memdesc<3x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      %tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %true, %true {two_ctas} :
        !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>,
        !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    tt.return
  }
}

// -----

// B is multi-buffered but reaches the MMA through a memdesc_trans, and A is
// single-buffered so nothing else supplies the depth. The view walk has to
// follow the trans or the barrier collapses to one slot and the pair can lap
// the phase bit -- the deadlock this pass prevents.
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_tb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_t = #ttg.shared_memory
#tmem_t = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: @test_2cta_sync_transposed_b_depth
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_2cta_sync_transposed_b_depth(
      %a: !ttg.memdesc<128x64xf16, #shared_t, #smem_t, mutable>,
      %bbuf: !ttg.memdesc<4x128x64xf16, #shared_t, #smem_t, mutable>,
      %acc: !ttg.memdesc<128x128xf32, #tmem_t, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    // Depth must come from B through the trans: 4 slots, not 1.
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<4x1xi64
    // CHECK: %[[D:.*]] = arith.constant 4 : i64
    // CHECK: arith.remui %{{.*}}, %[[D]]
    // CHECK: arith.divui %{{.*}}, %[[D]]
    // CHECK: ttng.tc_gen5_mma
    scf.for %iv = %c0 to %c8 step %c1 {
      %bslot = ttg.memdesc_index %bbuf[%c0_i32] : !ttg.memdesc<4x128x64xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<128x64xf16, #shared_t, #smem_t, mutable>
      %bt = ttg.memdesc_trans %bslot {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<64x128xf16, #shared_tb, #smem_t, mutable>
      %tok = ttng.tc_gen5_mma %a, %bt, %acc[%acc_tok], %true, %true {two_ctas} :
        !ttg.memdesc<128x64xf16, #shared_t, #smem_t, mutable>,
        !ttg.memdesc<64x128xf16, #shared_tb, #smem_t, mutable>,
        !ttg.memdesc<128x128xf32, #tmem_t, #ttng.tensor_memory, mutable>
    }
    tt.return
  }
}
