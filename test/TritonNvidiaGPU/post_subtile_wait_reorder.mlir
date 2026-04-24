// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-post-subtile-wait-reorder | FileCheck %s

// Test: four TMA store + wait pairs (subtile=4) with can_rotate_by_buffer_count=2.
// wait2 and wait3 are pushed to stage 1 (cross-iteration); wait0 and wait1
// stay at stage 0 but are repositioned after the 2nd subsequent copy.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @reorder_four_waits
  // Schedule has 2 stages: stage 0 for most ops, stage 1 for cross-iter waits.
  // CHECK: scf.for
  // wait0 at stage 0, repositioned after copy1:
  // CHECK:   ttng.async_tma_store_token_wait {{.*}} loop.stage = 0
  // wait1 at stage 0, repositioned after copy2:
  // CHECK:   ttng.async_tma_store_token_wait {{.*}} loop.stage = 0
  // wait2 crosses to stage 1 (the K-th copy from copy2 wraps around):
  // CHECK:   ttng.async_tma_store_token_wait {{.*}} loop.stage = 1
  // wait3 crosses to stage 1:
  // CHECK:   ttng.async_tma_store_token_wait {{.*}} loop.stage = 1
  // Annotations are stripped:
  // CHECK-NOT: can_rotate_by_buffer_count
  // Schedule is serialized:
  // CHECK: tt.scheduled_max_stage = 1
  tt.func @reorder_four_waits(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off_m: i32, %off0: i32, %off1: i32, %off2: i32, %off3: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %smem0 = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem2 = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem3 = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %dummy_data = arith.constant dense<0.0> : tensor<128x64xf16, #blocked>

    scf.for %iv = %c0 to %c10 step %c1 {
      ttg.local_store %dummy_data, %smem0 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%off_m, %off0] %smem0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token

      ttg.local_store %dummy_data, %smem1 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%off_m, %off1] %smem1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token

      ttg.local_store %dummy_data, %smem2 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%off_m, %off2] %smem2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok2 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token

      ttg.local_store %dummy_data, %smem3 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok3 = ttng.async_tma_copy_local_to_global %desc[%off_m, %off3] %smem3 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok3 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
    }

    tt.return
  }
}

// -----

// Test: no annotation → pass is a no-op.

#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @no_annotation_no_reorder
  // No schedule attributes should be added:
  // CHECK: scf.for
  // CHECK-NOT: loop.stage
  // CHECK:   ttng.async_tma_copy_local_to_global
  // CHECK:   ttng.async_tma_store_token_wait
  // CHECK:   ttng.async_tma_copy_local_to_global
  // CHECK:   ttng.async_tma_store_token_wait
  tt.func @no_annotation_no_reorder(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared2>>,
      %off0: i32, %off1: i32, %off2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>

    %dummy_data = arith.constant dense<0.0> : tensor<128x64xf16, #blocked2>

    scf.for %iv = %c0 to %c10 step %c1 {
      ttg.local_store %dummy_data, %smem0 : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 : !ttg.async.token

      ttg.local_store %dummy_data, %smem1 : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 : !ttg.async.token
    }

    tt.return
  }
}

// -----

// Test: buffer.copy mismatch → safety check fails → no reorder.

#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem3 = #ttg.shared_memory
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @buffer_copy_mismatch_no_reorder
  // No schedule attributes:
  // CHECK: scf.for
  // CHECK-NOT: loop.stage
  // CHECK:   ttng.async_tma_copy_local_to_global
  // CHECK:   ttng.async_tma_store_token_wait
  // CHECK:   ttng.async_tma_copy_local_to_global
  // CHECK:   ttng.async_tma_store_token_wait
  tt.func @buffer_copy_mismatch_no_reorder(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared3>>,
      %off0: i32, %off1: i32, %off2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %smem0 = ttg.local_alloc {buffer.copy = 3 : i32} : () -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 3 : i32} : () -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>

    %dummy_data = arith.constant dense<0.0> : tensor<128x64xf16, #blocked3>

    scf.for %iv = %c0 to %c10 step %c1 {
      ttg.local_store %dummy_data, %smem0 : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 : !tt.tensordesc<tensor<128x64xf16, #shared3>>, !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token

      ttg.local_store %dummy_data, %smem1 : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 : !tt.tensordesc<tensor<128x64xf16, #shared3>>, !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 {can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
    }

    tt.return
  }
}
