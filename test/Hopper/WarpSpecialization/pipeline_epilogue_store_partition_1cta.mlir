// RUN: TRITON_USE_META_WS=1 triton-opt %s '--tritongpu-pipeline=num-stages=3' | FileCheck %s

// Regression for the 1-CTA AutoWS path: the partition-type filter in
// expandLoops must not run when the kernel is not 2-CTA.
//
// That filter only expands "gemm" partitions containing an MMA plus "load"
// partitions. It exists to prune the loops that the extra ScheduleLoops re-run
// re-staged, and that re-run is itself gated on max(cluster_dims) >= 2 in
// CUDABackend.make_ttgir. Here every cluster dim is 1, so no re-scheduling
// happened and every partition loop still carries a valid post-WS schedule.
// Filtering anyway drops the "epilogue" and "epilogue_store" loops and
// silently miscompiles the kernel (T286353056).
//
// partition1 is the epilogue_store worker. Once expanded, the pipeliner peels
// its four subtile TMA stores into a prologue ahead of the loop and the loop
// picks up the pipelined iter_args. If the filter skips it, the loop is left
// untouched: the four stores appear only inside the body and nothing precedes
// the scf.for.
// CHECK-LABEL: partition1(
// CHECK-COUNT-4: ttng.async_tma_copy_local_to_global
// CHECK:         scf.for
// CHECK-COUNT-4: ttng.async_tma_copy_local_to_global
// CHECK-LABEL: partition2(

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i64 = arith.constant {async_task_id = array<i32: 0>} 2 : i64
    %true = arith.constant true
    %c3_i64 = arith.constant {async_task_id = array<i32: 0>} 3 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c4_i64 = arith.constant {async_task_id = array<i32: 0>} 4 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0>} 148 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0>} 127 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %3 = ttg.memdesc_index %0[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %4 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %6 = ttg.memdesc_index %4[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %7 = ttg.memdesc_index %4[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %8 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %9 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %10 = ttg.memdesc_index %8[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %11 = ttg.memdesc_index %8[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %12 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %13 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %13, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %14 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %15 = ttg.memdesc_index %12[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %16 = ttg.local_alloc {alignment = 8 : i32, ttg.ws_generated_barrier} : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %17 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %17, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %18 = ttg.memdesc_index %16[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %19 = ttg.memdesc_index %16[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %20 = ttg.memdesc_index %16[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %21 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %24, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %25 = ttg.memdesc_index %21[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %26 = ttg.memdesc_index %22[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %27 = ttg.memdesc_index %21[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %28 = ttg.memdesc_index %22[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %29 = ttg.memdesc_index %21[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %30 = ttg.memdesc_index %22[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %31 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %32 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %33 = ttg.memdesc_index %31[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %34 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %35 = ttg.memdesc_index %31[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %36 = ttg.memdesc_index %32[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %36, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %37 = ttg.memdesc_index %31[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %38 = ttg.memdesc_index %32[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.barrier local
    %result, %token = ttng.tmem_alloc {buffer.copy = 4 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %39 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 3 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %40 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %41 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    ttg.warp_specialize(%arg15, %arg16, %arg17, %41, %4, %result, %40, %8, %12, %0, %16, %39, %arg10, %arg0, %arg5, %21, %22, %31, %32) attributes {ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"]}
    default {
      %72 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %73 = arith.addi %arg15, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %74 = arith.divsi %73, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %75 = arith.addi %arg16, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %76 = arith.divsi %75, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %77 = arith.muli %74, %76 {async_task_id = array<i32: 0>} : i32
      %78:2 = scf.for %arg18 = %72 to %77 step %c148_i32 iter_args(%arg19 = %c0_i64, %arg20 = %c0_i64) -> (i64, i64)  : i32 {
        %79 = arith.divui %arg19, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %80 = arith.muli %79, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %81 = arith.subi %arg19, %80 {async_task_id = array<i32: 0>} : i64
        %82 = arith.trunci %81 {async_task_id = array<i32: 0>} : i64 to i32
        %83 = arith.andi %79, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %84 = arith.trunci %83 {async_task_id = array<i32: 0>} : i64 to i1
        %85 = arith.divui %arg19, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %86 = arith.muli %85, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %87 = arith.subi %arg19, %86 {async_task_id = array<i32: 0>} : i64
        %88 = arith.trunci %87 {async_task_id = array<i32: 0>} : i64 to i32
        %89 = ttg.memdesc_index %result[%88] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>
        %90 = ttg.memdesc_index %16[%82] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %91 = arith.extui %84 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %90, %91 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %result_0, %token_1 = ttng.tmem_load %89[] {async_task_id = array<i32: 0>, tmem.end = array<i32: 0>} : !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %92 = ttg.memdesc_index %22[%82] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.arrive_barrier %92, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %93 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x2x64xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %94 = tt.trans %93 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x64x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %outLHS, %outRHS = tt.split %94 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %95 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x2x32xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %96 = tt.trans %95 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x32x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %outLHS_2, %outRHS_3 = tt.split %96 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %97 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x2x32xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %98 = tt.trans %97 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x32x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>>
        %outLHS_4, %outRHS_5 = tt.split %98 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>> -> tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %99 = arith.truncf %outLHS_2 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> to tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %100 = ttg.convert_layout %99 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %101 = arith.divui %arg20, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %102 = arith.muli %101, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %103 = arith.subi %arg20, %102 {async_task_id = array<i32: 0>} : i64
        %104 = arith.trunci %103 {async_task_id = array<i32: 0>} : i64 to i32
        %105 = ttg.memdesc_index %39[%104] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %106 = arith.divui %arg20, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %107 = arith.muli %106, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %108 = arith.subi %arg20, %107 {async_task_id = array<i32: 0>} : i64
        %109 = arith.trunci %108 {async_task_id = array<i32: 0>} : i64 to i32
        %110 = arith.andi %106, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %111 = arith.trunci %110 {async_task_id = array<i32: 0>} : i64 to i1
        %112 = ttg.memdesc_index %32[%109] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %113 = arith.xori %111, %true : i1
        %114 = arith.extui %113 : i1 to i32
        ttng.wait_barrier %112, %114 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttg.local_store %100, %105 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %115 = ttg.memdesc_index %31[%109] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.arrive_barrier %115, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %116 = arith.truncf %outRHS_3 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> to tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %117 = ttg.convert_layout %116 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %118 = arith.addi %arg20, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %119 = arith.divui %118, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %120 = arith.muli %119, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %121 = arith.subi %118, %120 {async_task_id = array<i32: 0>} : i64
        %122 = arith.trunci %121 {async_task_id = array<i32: 0>} : i64 to i32
        %123 = ttg.memdesc_index %39[%122] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %124 = arith.addi %arg20, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %125 = arith.divui %124, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %126 = arith.muli %125, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %127 = arith.subi %124, %126 {async_task_id = array<i32: 0>} : i64
        %128 = arith.trunci %127 {async_task_id = array<i32: 0>} : i64 to i32
        %129 = arith.andi %125, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %130 = arith.trunci %129 {async_task_id = array<i32: 0>} : i64 to i1
        %131 = ttg.memdesc_index %32[%128] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %132 = arith.xori %130, %true : i1
        %133 = arith.extui %132 : i1 to i32
        ttng.wait_barrier %131, %133 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttg.local_store %117, %123 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %134 = ttg.memdesc_index %31[%128] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.arrive_barrier %134, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %135 = arith.truncf %outLHS_4 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> to tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %136 = ttg.convert_layout %135 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %137 = arith.addi %arg20, %c2_i64 {async_task_id = array<i32: 0>} : i64
        %138 = arith.divui %137, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %139 = arith.muli %138, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %140 = arith.subi %137, %139 {async_task_id = array<i32: 0>} : i64
        %141 = arith.trunci %140 {async_task_id = array<i32: 0>} : i64 to i32
        %142 = ttg.memdesc_index %39[%141] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %143 = arith.addi %arg20, %c2_i64 {async_task_id = array<i32: 0>} : i64
        %144 = arith.divui %143, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %145 = arith.muli %144, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %146 = arith.subi %143, %145 {async_task_id = array<i32: 0>} : i64
        %147 = arith.trunci %146 {async_task_id = array<i32: 0>} : i64 to i32
        %148 = arith.andi %144, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %149 = arith.trunci %148 {async_task_id = array<i32: 0>} : i64 to i1
        %150 = ttg.memdesc_index %32[%147] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %151 = arith.xori %149, %true : i1
        %152 = arith.extui %151 : i1 to i32
        ttng.wait_barrier %150, %152 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttg.local_store %136, %142 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %153 = ttg.memdesc_index %31[%147] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.arrive_barrier %153, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %154 = arith.truncf %outRHS_5 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> to tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>>
        %155 = ttg.convert_layout %154 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>> -> tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %156 = arith.addi %arg20, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %157 = arith.divui %156, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %158 = arith.muli %157, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %159 = arith.subi %156, %158 {async_task_id = array<i32: 0>} : i64
        %160 = arith.trunci %159 {async_task_id = array<i32: 0>} : i64 to i32
        %161 = ttg.memdesc_index %39[%160] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %162 = arith.addi %arg20, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %163 = arith.divui %162, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %164 = arith.muli %163, %c3_i64 {async_task_id = array<i32: 0>} : i64
        %165 = arith.subi %162, %164 {async_task_id = array<i32: 0>} : i64
        %166 = arith.trunci %165 {async_task_id = array<i32: 0>} : i64 to i32
        %167 = arith.andi %163, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %168 = arith.trunci %167 {async_task_id = array<i32: 0>} : i64 to i1
        %169 = ttg.memdesc_index %32[%166] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %170 = arith.xori %168, %true : i1
        %171 = arith.extui %170 : i1 to i32
        ttng.wait_barrier %169, %171 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttg.local_store %155, %161 {async_task_id = array<i32: 0>} : tensor<128x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %172 = ttg.memdesc_index %31[%166] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.arrive_barrier %172, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %173 = arith.addi %arg20, %c4_i64 {async_task_id = array<i32: 0>} : i64
        %174 = arith.addi %arg19, %c1_i64 {async_task_id = array<i32: 0>} : i64
        scf.yield {async_task_id = array<i32: 0>} %174, %173 : i64, i64
      } {async_task_id = array<i32: 0>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg22: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg23: !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, %arg24: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg25: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg26: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg27: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg28: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg29: !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg30: !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, %arg31: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg32: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg33: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg34: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg35: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg36: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) num_warps(4) {
      %c3_i64_0 = arith.constant {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} 3 : i64
      %c1_i64_1 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c4_i64_2 = arith.constant {async_task_id = array<i32: 1>} 4 : i64
      %c0_i64_3 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %c63_i32 = arith.constant {async_task_id = array<i32: 1>} 63 : i32
      %c127_i32_4 = arith.constant {async_task_id = array<i32: 1>} 127 : i32
      %c1_i32_5 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c0_i32_6 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %c148_i32_7 = arith.constant {async_task_id = array<i32: 1>} 148 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %c128_i32_8 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %true_9 = arith.constant {async_task_id = array<i32: 1>} true
      %false = arith.constant {async_task_id = array<i32: 1>} false
      %72 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %73 = arith.addi %arg18, %c127_i32_4 {async_task_id = array<i32: 1>} : i32
      %74 = arith.divsi %73, %c128_i32_8 {async_task_id = array<i32: 1>} : i32
      %75 = arith.addi %arg19, %c127_i32_4 {async_task_id = array<i32: 1>} : i32
      %76 = arith.divsi %75, %c128_i32_8 {async_task_id = array<i32: 1>} : i32
      %77 = arith.addi %arg20, %c63_i32 {async_task_id = array<i32: 1>} : i32
      %78 = arith.divsi %77, %c64_i32 {async_task_id = array<i32: 1>} : i32
      %79 = arith.muli %74, %76 {async_task_id = array<i32: 1>} : i32
      %80:2 = scf.for %arg37 = %72 to %79 step %c148_i32_7 iter_args(%arg38 = %c0_i64_3, %arg39 = %c0_i64_3) -> (i64, i64)  : i32 {
        %81 = arith.divui %arg38, %c4_i64_2 {async_task_id = array<i32: 1>} : i64
        %82 = arith.muli %81, %c4_i64_2 {async_task_id = array<i32: 1>} : i64
        %83 = arith.subi %arg38, %82 {async_task_id = array<i32: 1>} : i64
        %84 = arith.trunci %83 {async_task_id = array<i32: 1>} : i64 to i32
        %85 = arith.andi %81, %c1_i64_1 {async_task_id = array<i32: 1>} : i64
        %86 = arith.trunci %85 {async_task_id = array<i32: 1>} : i64 to i1
        %87 = ttg.memdesc_index %arg34[%84] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %88 = arith.xori %86, %true_9 : i1
        %89 = arith.extui %88 : i1 to i32
        ttng.wait_barrier %87, %89 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %90:2 = scf.for %arg40 = %c0_i32_6 to %78 step %c1_i32_5 iter_args(%arg41 = %false, %arg42 = %arg39) -> (i1, i64)  : i32 {
          %93 = arith.divui %arg42, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %94 = arith.muli %93, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %95 = arith.subi %arg42, %94 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %96 = arith.trunci %95 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %97 = arith.andi %93, %c1_i64_1 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %98 = arith.trunci %97 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i1
          %99 = arith.divui %arg42, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %100 = arith.muli %99, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %101 = arith.subi %arg42, %100 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %102 = arith.trunci %101 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %103 = arith.andi %99, %c1_i64_1 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %104 = arith.trunci %103 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i1
          %105 = arith.divui %arg42, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %106 = arith.muli %105, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %107 = arith.subi %arg42, %106 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %108 = arith.trunci %107 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %109 = ttg.memdesc_index %arg21[%108] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          %110 = ttg.memdesc_index %arg22[%102] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %111 = arith.extui %104 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1 to i32
          ttng.wait_barrier %110, %111, %true_9 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %112 = ttg.memdesc_trans %109 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          %113 = arith.divui %arg38, %c4_i64_2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %114 = arith.muli %113, %c4_i64_2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %115 = arith.subi %arg38, %114 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %116 = arith.trunci %115 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %117 = ttg.memdesc_index %arg23[%116] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>
          %118 = arith.divui %arg42, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %119 = arith.muli %118, %c3_i64_0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %120 = arith.subi %arg42, %119 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %121 = arith.trunci %120 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %122 = ttg.memdesc_index %arg24[%121] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          %123 = ttg.memdesc_index %arg25[%96] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %124 = ttg.memdesc_index %arg26[%96] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %125 = arith.extui %98 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1 to i32
          ttng.wait_barrier %124, %125, %true_9 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %126 = ttg.memdesc_index %arg27[%102] {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %127 = ttng.tc_gen5_mma %122, %112, %117[], %arg41, %true_9, %123[%true_9], %126[%true_9] {async_task_id = array<i32: 1>, is_async, loop.cluster = 0 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 0>} : !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %128 = arith.addi %arg42, %c1_i64_1 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          scf.yield {async_task_id = array<i32: 1>} %true_9, %128 : i1, i64
        } {async_task_id = array<i32: 1>, tt.scheduled_max_stage = 0 : i32, tt.warp_specialize}
        %91 = ttg.memdesc_index %arg28[%84] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.tc_gen5_commit %91 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %92 = arith.addi %arg38, %c1_i64_1 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %92, %90#1 : i64, i64
      } {async_task_id = array<i32: 1>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg22: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg23: !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, %arg24: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg25: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg26: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg27: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg28: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg29: !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg30: !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, %arg31: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg32: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg33: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg34: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg35: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg36: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) num_warps(4) {
      %c4_i64_0 = arith.constant {async_task_id = array<i32: 2>} 4 : i64
      %c2_i64_1 = arith.constant {async_task_id = array<i32: 2>} 2 : i64
      %true_2 = arith.constant true
      %c3_i64_3 = arith.constant {async_task_id = array<i32: 2>} 3 : i64
      %c1_i64_4 = arith.constant {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} 1 : i64
      %c0_i64_5 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %c127_i32_6 = arith.constant {async_task_id = array<i32: 2>} 127 : i32
      %c148_i32_7 = arith.constant {async_task_id = array<i32: 2>} 148 : i32
      %c96_i32 = arith.constant {async_task_id = array<i32: 2>} 96 : i32
      %c32_i32 = arith.constant {async_task_id = array<i32: 2>} 32 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
      %c128_i32_8 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c8_i32 = arith.constant {async_task_id = array<i32: 2>} 8 : i32
      %72 = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %73 = arith.addi %arg18, %c127_i32_6 {async_task_id = array<i32: 2>} : i32
      %74 = arith.divsi %73, %c128_i32_8 {async_task_id = array<i32: 2>} : i32
      %75 = arith.addi %arg19, %c127_i32_6 {async_task_id = array<i32: 2>} : i32
      %76 = arith.divsi %75, %c128_i32_8 {async_task_id = array<i32: 2>} : i32
      %77 = arith.muli %74, %76 {async_task_id = array<i32: 2>} : i32
      %78 = arith.muli %76, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %79 = scf.for %arg37 = %72 to %77 step %c148_i32_7 iter_args(%arg38 = %c0_i64_5) -> (i64)  : i32 {
        %80 = arith.divsi %arg37, %78 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %81 = arith.muli %80, %c8_i32 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %82 = arith.subi %74, %81 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %83 = arith.minsi %82, %c8_i32 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %84 = arith.remsi %arg37, %83 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %85 = arith.addi %81, %84 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %86 = arith.remsi %arg37, %78 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %87 = arith.divsi %86, %83 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %88 = arith.muli %85, %c128_i32_8 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %89 = arith.muli %87, %c128_i32_8 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %90 = arith.divui %arg38, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %91 = arith.muli %90, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %92 = arith.subi %arg38, %91 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %93 = arith.trunci %92 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
        %94 = arith.andi %90, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %95 = arith.trunci %94 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i1
        %96 = arith.divui %arg38, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %97 = arith.muli %96, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %98 = arith.subi %arg38, %97 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %99 = arith.trunci %98 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
        %100 = ttg.memdesc_index %arg29[%99] {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %101 = ttg.memdesc_index %arg35[%93] {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %102 = arith.extui %95 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1 to i32
        ttng.wait_barrier %101, %102 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %103 = ttng.async_tma_copy_local_to_global %arg30[%88, %89] %100 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.async.token
        %104 = ttg.memdesc_index %arg36[%93] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.async_tma_store_token_wait %103 , %104[%true_2]  {async_task_id = array<i32: 2>, loop.cluster = 7 : i32, loop.stage = 0 : i32, planned_pending_count = 2 : i32} : !ttg.async.token, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %105 = arith.addi %89, %c32_i32 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %106 = arith.addi %arg38, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %107 = arith.divui %106, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %108 = arith.muli %107, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %109 = arith.subi %106, %108 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %110 = arith.trunci %109 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
        %111 = arith.andi %107, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %112 = arith.trunci %111 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i1
        %113 = arith.addi %arg38, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %114 = arith.divui %113, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %115 = arith.muli %114, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %116 = arith.subi %113, %115 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %117 = arith.trunci %116 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
        %118 = ttg.memdesc_index %arg29[%117] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %119 = ttg.memdesc_index %arg35[%110] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %120 = arith.extui %112 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i1 to i32
        ttng.wait_barrier %119, %120 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %121 = ttng.async_tma_copy_local_to_global %arg30[%88, %105] %118 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.async.token
        %122 = ttg.memdesc_index %arg36[%110] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.async_tma_store_token_wait %121 , %122[%true_2]  {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 1 : i32, planned_pending_count = 2 : i32} : !ttg.async.token, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %123 = arith.addi %89, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
        %124 = arith.addi %arg38, %c2_i64_1 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %125 = arith.divui %124, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %126 = arith.muli %125, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %127 = arith.subi %124, %126 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %128 = arith.trunci %127 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64 to i32
        %129 = arith.andi %125, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %130 = arith.trunci %129 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64 to i1
        %131 = arith.addi %arg38, %c2_i64_1 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %132 = arith.divui %131, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %133 = arith.muli %132, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %134 = arith.subi %131, %133 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64
        %135 = arith.trunci %134 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64 to i32
        %136 = ttg.memdesc_index %arg29[%135] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %137 = ttg.memdesc_index %arg35[%128] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %138 = arith.extui %130 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i1 to i32
        ttng.wait_barrier %137, %138 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %139 = ttng.async_tma_copy_local_to_global %arg30[%88, %123] %136 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.async.token
        %140 = ttg.memdesc_index %arg36[%128] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.async_tma_store_token_wait %139 , %140[%true_2]  {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 1 : i32, planned_pending_count = 2 : i32} : !ttg.async.token, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %141 = arith.addi %89, %c96_i32 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i32
        %142 = arith.addi %arg38, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %143 = arith.divui %142, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %144 = arith.muli %143, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %145 = arith.subi %142, %144 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %146 = arith.trunci %145 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64 to i32
        %147 = arith.andi %143, %c1_i64_4 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %148 = arith.trunci %147 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64 to i1
        %149 = arith.addi %arg38, %c4_i64_0 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %150 = arith.addi %arg38, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %151 = arith.divui %150, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %152 = arith.muli %151, %c3_i64_3 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %153 = arith.subi %150, %152 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64
        %154 = arith.trunci %153 {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : i64 to i32
        %155 = ttg.memdesc_index %arg29[%154] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %156 = ttg.memdesc_index %arg35[%146] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %157 = arith.extui %148 {loop.cluster = 6 : i32, loop.stage = 0 : i32} : i1 to i32
        ttng.wait_barrier %156, %157 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}, loop.cluster = 8 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %158 = ttng.async_tma_copy_local_to_global %arg30[%88, %141] %155 {async_task_id = array<i32: 2>, loop.cluster = 8 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.async.token
        %159 = ttg.memdesc_index %arg36[%146] {async_task_id = array<i32: 2>, loop.cluster = 8 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        ttng.async_tma_store_token_wait %158 , %159[%true_2]  {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 1 : i32, planned_pending_count = 2 : i32} : !ttg.async.token, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        scf.yield {async_task_id = array<i32: 2>} %149 : i64
      } {async_task_id = array<i32: 2>, tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition2(%arg18: i32, %arg19: i32, %arg20: i32, %arg21: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg22: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg23: !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, %arg24: !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg25: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg26: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg27: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg28: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg29: !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg30: !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, %arg31: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg32: !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, %arg33: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg34: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg35: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg36: !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) num_warps(4) {
      %true_0 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} true
      %c1_i64_1 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} 1 : i64
      %c3_i64_2 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} 3 : i64
      %c0_i64_3 = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %c63_i32 = arith.constant {async_task_id = array<i32: 3>} 63 : i32
      %c127_i32_4 = arith.constant {async_task_id = array<i32: 3>} 127 : i32
      %c1_i32_5 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %c0_i32_6 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %c148_i32_7 = arith.constant {async_task_id = array<i32: 3>} 148 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
      %c128_i32_8 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %c8_i32 = arith.constant {async_task_id = array<i32: 3>} 8 : i32
      %72 = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %73 = arith.addi %arg18, %c127_i32_4 {async_task_id = array<i32: 3>} : i32
      %74 = arith.divsi %73, %c128_i32_8 {async_task_id = array<i32: 3>} : i32
      %75 = arith.addi %arg19, %c127_i32_4 {async_task_id = array<i32: 3>} : i32
      %76 = arith.divsi %75, %c128_i32_8 {async_task_id = array<i32: 3>} : i32
      %77 = arith.addi %arg20, %c63_i32 {async_task_id = array<i32: 3>} : i32
      %78 = arith.divsi %77, %c64_i32 {async_task_id = array<i32: 3>} : i32
      %79 = arith.muli %74, %76 {async_task_id = array<i32: 3>} : i32
      %80 = arith.muli %76, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %81 = scf.for %arg37 = %72 to %79 step %c148_i32_7 iter_args(%arg38 = %c0_i64_3) -> (i64)  : i32 {
        %82 = arith.divsi %arg37, %80 {async_task_id = array<i32: 3>} : i32
        %83 = arith.muli %82, %c8_i32 {async_task_id = array<i32: 3>} : i32
        %84 = arith.subi %74, %83 {async_task_id = array<i32: 3>} : i32
        %85 = arith.minsi %84, %c8_i32 {async_task_id = array<i32: 3>} : i32
        %86 = arith.remsi %arg37, %85 {async_task_id = array<i32: 3>} : i32
        %87 = arith.addi %83, %86 {async_task_id = array<i32: 3>} : i32
        %88 = arith.remsi %arg37, %80 {async_task_id = array<i32: 3>} : i32
        %89 = arith.divsi %88, %85 {async_task_id = array<i32: 3>} : i32
        %90 = arith.muli %87, %c128_i32_8 {async_task_id = array<i32: 3>} : i32
        %91 = arith.muli %89, %c128_i32_8 {async_task_id = array<i32: 3>} : i32
        %92 = scf.for %arg39 = %c0_i32_6 to %78 step %c1_i32_5 iter_args(%arg40 = %arg38) -> (i64)  : i32 {
          %93 = arith.muli %arg39, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
          %94 = arith.divui %arg40, %c3_i64_2 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %95 = arith.muli %94, %c3_i64_2 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %96 = arith.subi %arg40, %95 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %97 = arith.trunci %96 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %98 = arith.andi %94, %c1_i64_1 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %99 = arith.trunci %98 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i1
          %100 = ttg.memdesc_index %arg25[%97] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %101 = arith.xori %99, %true_0 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1
          %102 = arith.extui %101 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1 to i32
          ttng.wait_barrier %100, %102 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %103 = ttg.memdesc_index %arg26[%97] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          ttng.barrier_expect %103, 16384 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32}, %true_0 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %104 = ttg.memdesc_index %arg24[%97] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          ttng.async_tma_copy_global_to_local %arg31[%90, %93] %104, %103, %true_0 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          %105 = arith.divui %arg40, %c3_i64_2 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %106 = arith.muli %105, %c3_i64_2 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %107 = arith.subi %arg40, %106 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %108 = arith.trunci %107 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
          %109 = arith.andi %105, %c1_i64_1 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          %110 = arith.trunci %109 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i1
          %111 = ttg.memdesc_index %arg27[%108] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %112 = arith.xori %110, %true_0 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1
          %113 = arith.extui %112 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i1 to i32
          ttng.wait_barrier %111, %113 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %114 = ttg.memdesc_index %arg22[%108] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          ttng.barrier_expect %114, 16384 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32}, %true_0 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
          %115 = ttg.memdesc_index %arg21[%108] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          ttng.async_tma_copy_global_to_local %arg32[%91, %93] %115, %114, %true_0 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
          %116 = arith.addi %arg40, %c1_i64_1 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
          scf.yield {async_task_id = array<i32: 3>} %116 : i64
        } {async_task_id = array<i32: 3>, tt.scheduled_max_stage = 0 : i32, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 3>} %92 : i64
      } {async_task_id = array<i32: 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<3x128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !tt.tensordesc<128x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>>, !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, !tt.tensordesc<128x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) -> ()
    %42 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %42 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %43 = ttg.memdesc_index %4[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %44 = ttg.memdesc_index %4[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %45 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %45 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %46 = ttg.memdesc_index %8[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %46 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %47 = ttg.memdesc_index %8[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %48 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %48 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %49 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %50 = ttg.memdesc_index %12[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %50 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %51 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %51 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %52 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %52 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %53 = ttg.memdesc_index %0[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %53 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %54 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %54 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %55 = ttg.memdesc_index %16[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %56 = ttg.memdesc_index %16[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %56 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %57 = ttg.memdesc_index %16[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %57 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %58 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %58 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %59 = ttg.memdesc_index %21[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %59 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %60 = ttg.memdesc_index %21[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %60 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %61 = ttg.memdesc_index %21[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %61 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %62 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %62 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %63 = ttg.memdesc_index %22[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %63 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %64 = ttg.memdesc_index %22[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %64 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %65 = ttg.memdesc_index %22[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %65 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %66 = ttg.memdesc_index %31[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %66 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %67 = ttg.memdesc_index %31[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %67 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %68 = ttg.memdesc_index %31[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %68 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %69 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %69 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %70 = ttg.memdesc_index %32[%c1_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %70 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %71 = ttg.memdesc_index %32[%c2_i32] : !ttg.memdesc<3x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
}
