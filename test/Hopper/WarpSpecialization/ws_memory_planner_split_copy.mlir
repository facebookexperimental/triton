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
