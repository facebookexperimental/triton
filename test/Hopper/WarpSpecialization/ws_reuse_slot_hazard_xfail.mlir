// RUN: triton-opt %s --nvgpu-verify-reuse-slot-hazard --verify-diagnostics
// XFAIL: *
//
// Executable spec for a late-pass barrier-aware / aliasing analysis that
// detects the FA-bwd 3-buffer TMEM reuse slot hazard.
// See docs/BwdTmemReuseSlotHazard.md.
//
// One TMEM slot (buffer.id = 5) is shared by the dpT/dq MMA results and dsT
// (dk's operand A). dsT is materialized into the slot by the *computation*
// partition (async_task_id = 3) and read by the dk MMA (async_task_id = 1);
// the dq MMA (async_task_id = 1) writes the same slot. dk MUST read dsT before
// dq overwrites the slot. Here dq is emitted BEFORE dk in the gemm partition
// with no barrier that can order dk first -> dq clobbers dsT -> NaN at runtime
// (the bug fixed by emitting dk before dq in the kernel).
//
// This module is shaped like post-insertAsyncComm / pre-specializeRegion IR:
// a single region whose ops carry async_task_id and whose cross-partition
// dependencies are materialized as mbarriers. The proposed pass
// (--nvgpu-verify-reuse-slot-hazard, not yet implemented) must:
//   1. group accesses by physical slot (resolve tmem_subslice / reinterpret /
//      memdesc_index aliasing to the same alloc + overlapping columns),
//   2. build a happens-before graph from arrive_barrier/wait_barrier (+ gen5
//      completion barriers) and intra-partition program order,
//   3. flag a writer of the slot that can run between the reader's intended
//      producer and the reader without a happens-before edge ordering it after
//      the read.
// XFAIL until that pass exists; the expected-error below is the target
// diagnostic.

#smem = #ttg.shared_memory
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func public @bwd_reuse_slot_hazard(
      %dsT_val: tensor<128x128xf16, #linear2>,
      %a_dq: !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>,
      %b_smem: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32

    // Shared TMEM slot (buffer.id = 5): dpT/dq results and dsT alias this slot.
    %slot = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // dk accumulator (buffer.id = 10), not reused.
    %dk_acc = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // dsT producer->consumer barrier (computation arrives, dk waits).
    %dsT_full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_full, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>

    // dsT aliases slot[N=0] reinterpreted f32->f16 (same physical columns).
    %sub = ttng.tmem_subslice %slot {N = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %re = ttg.memdesc_reinterpret %sub : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem3, #ttng.tensor_memory, mutable>

    // --- computation partition (task 3): write dsT into the slot, then signal.
    %dsT_slot = ttg.memdesc_index %re[%c0] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %dsT_val, %dsT_slot, %true {async_task_id = array<i32: 3>} : tensor<128x128xf16, #linear2> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.arrive_barrier %dsT_full, 1 {async_task_id = array<i32: 3>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>

    // --- gemm partition (task 1) ---
    // HAZARD: dq writes the shared slot (opndD, buffer.id=5) before dk reads
    // dsT from it, in the same partition, with no barrier ordering dk first.
    %dq_d = ttg.memdesc_index %slot[%c0] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error@+1 {{reuse group slot hazard}}
    ttng.tc_gen5_mma %a_dq, %b_smem, %dq_d, %false, %true {async_task_id = array<i32: 1>, is_async} : !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // dk waits for dsT then reads it (operand A, buffer.id=5) -- too late, dq
    // already clobbered the slot above.
    ttng.wait_barrier %dsT_full, %c0 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_d = ttg.memdesc_index %dk_acc[%c0] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dsT_read = ttg.memdesc_index %re[%c0] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %dsT_read, %b_smem, %dk_d, %false, %true {async_task_id = array<i32: 1>, is_async} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
