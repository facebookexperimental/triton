// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner=num-buffers=3 | FileCheck %s

// Test: When a SMEM buffer requires TMA split copies (inner dim exceeds the
// swizzle byte width), the memory planner assigns it a unique buffer.id.
// This prevents it from sharing a barrier with other buffers, since each split
// copy emits a separate barrier_expect/arrive, and sharing would cause barrier
// over-arrival (UB per CUDA PTX spec).
//
// A_smem (128x64xf16, swizzle=128): inner dim = 64 × 2B = 128B = swizzle → no split
// B_smem (64x128xf16, swizzle=128): inner dim = 128 × 2B = 256B > swizzle → split needed
//
// Without the fix, both allocs would share buffer.id = 0 (same innermost loop).
// With the fix, B_smem gets a separate buffer.id to use its own barrier.

// CHECK-LABEL: @tma_split_copy_separate_buffer_id
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 1 : i32}
// CHECK-SAME: 64x128xf16

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_split_copy_separate_buffer_id(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    // A: inner dim fits swizzle (64 elems × 2B = 128B = swizzle) → no split
    %A_smem = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // B: inner dim exceeds swizzle (128 elems × 2B = 256B > 128B swizzle) → split
    %B_smem = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      // Producer task 1: TMA loads into SMEM
      %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      ttg.local_store %a, %A_smem {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked>
      ttg.local_store %b, %B_smem {async_task_id = array<i32: 1>} : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      // Consumer task 0: reads from SMEM
      %a_val = ttg.local_load %A_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %b_val = ttg.local_load %B_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #blocked>
      scf.yield
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// Test: When ALL innermost-loop buffers require TMA split copies (same split
// pattern), they CAN share a buffer.id. The barrier arrive count is symmetric
// across all buffers in the group, so the accounting remains correct.
// This is the flash-attention case: v and k are both 128x128xbf16 with
// swizzle=128, so both need split copies (inner dim 128 × 2B = 256B > 128B).
// They should share a buffer.id to avoid wasting SMEM.

// CHECK-LABEL: @tma_split_copy_both_split_share
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[SPLIT_ID:[0-9]+]] : i32}
// CHECK-SAME: 128x128xbf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[SPLIT_ID]] : i32}
// CHECK-SAME: 128x128xbf16

#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_split_copy_both_split_share(
      %v_desc: !tt.tensordesc<tensor<128x128xbf16, #shared2>>,
      %k_desc: !tt.tensordesc<tensor<128x128xbf16, #shared2>>) {
    %v_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable>
    %k_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %v = tt.descriptor_load %v_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared2>> -> tensor<128x128xbf16, #blocked2>
      ttg.local_store %v, %v_smem {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable>
      %k = tt.descriptor_load %k_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared2>> -> tensor<128x128xbf16, #blocked2>
      ttg.local_store %k, %k_smem {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable>
      %v_val = ttg.local_load %v_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable> -> tensor<128x128xbf16, #blocked2>
      %k_val = ttg.local_load %k_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xbf16, #shared2, #smem2, mutable> -> tensor<128x128xbf16, #blocked2>
      scf.yield
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// Test: Split buffers with different TMA block shapes must NOT share a
// buffer.id. Both C (64x128) and D (128x128) split along the contiguous dim
// (128 → 64), but their outer dims differ (64 vs 128), producing different
// TMA block shapes ([64, 64] vs [128, 64]). Conservatively separated.

// CHECK-LABEL: @tma_split_copy_different_split_factors
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[ID1:[0-9]+]] : i32}
// CHECK-SAME: 64x128xf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[ID2:[0-9]+]] : i32}
// CHECK-SAME: 128x128xf16

#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem3 = #ttg.shared_memory
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_split_copy_different_split_factors(
      %c_desc: !tt.tensordesc<tensor<64x128xf16, #shared3>>,
      %d_desc: !tt.tensordesc<tensor<128x128xf16, #shared3>>) {
    // C: 64x128, inner dim = 128 × 2B = 256B > 128B swizzle → block [64, 64], split into 2
    %C_smem = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared3, #smem3, mutable>
    // D: 128x128, inner dim = 128 × 2B = 256B > 128B swizzle → block [128, 64], split into 2
    // Different outer dim means different TMA block shape → separate group
    %D_smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared3, #smem3, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for %iv = %c0 to %c10 step %c1 : i32 {
      %c = tt.descriptor_load %c_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<64x128xf16, #shared3>> -> tensor<64x128xf16, #blocked3>
      ttg.local_store %c, %C_smem {async_task_id = array<i32: 1>} : tensor<64x128xf16, #blocked3> -> !ttg.memdesc<64x128xf16, #shared3, #smem3, mutable>
      %d = tt.descriptor_load %d_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared3>> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %d, %D_smem {async_task_id = array<i32: 1>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared3, #smem3, mutable>
      %c_val = ttg.local_load %C_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf16, #shared3, #smem3, mutable> -> tensor<64x128xf16, #blocked3>
      %d_val = ttg.local_load %D_smem {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared3, #smem3, mutable> -> tensor<128x128xf16, #blocked3>
      scf.yield
    } {tt.warp_specialize}
    tt.return
  }
}
