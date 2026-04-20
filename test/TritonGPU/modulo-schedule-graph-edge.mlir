// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect "-nvgpu-modulo-schedule=print-schedule-graph=true" 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Edge case 0: Single-stage schedule (maxStage=0).
// MMA-only loop: no TMA copy, no result use. The MMA self-latency (900) is
// the only thing on the TC pipeline, so II = 900 and the MMA lands at
// cycle 0, stage 0 — max_stage = 0.
//
// Regression test for Devmate review: tt.num_stages must be set even when
// maxStage = 0 so downstream pipelining recognises the loop as scheduled.
//===----------------------------------------------------------------------===//

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// Verify the maxStage=0 dump and the loop's tt.num_stages=1 attribute.
// CHECK: ii = 900, max_stage = 0
// CHECK: @maxstage_0_mma_only
// CHECK: tt.modulo_ii = 900 : i32
// CHECK-SAME: tt.num_stages = 1 : i32
// CHECK-SAME: tt.scheduled_max_stage = 0 : i32
tt.func @maxstage_0_mma_only(
  %a: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
  %b: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
  %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %k_tiles = arith.constant 4 : i32
  %true = arith.constant true

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 : i32 {
    ttng.tc_gen5_mma %a, %b, %c, %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  }

  tt.return
}

}

// -----

//===----------------------------------------------------------------------===//
// Edge case 1: Loop with no schedulable ops (no TMA load, no MMA).
// The pass selection filter (`hasTMALoad || hasMMAv5`) must skip this loop
// cleanly — no schedule attrs emitted, no ScheduleGraph dump.
//===----------------------------------------------------------------------===//

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @no_schedulable_ops
// CHECK: scf.for
// CHECK-NOT: tt.modulo_ii
// CHECK-NOT: tt.scheduled_max_stage
tt.func @no_schedulable_ops(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %k_tiles = arith.constant 4 : i32

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 : i32 {
    %0 = arith.muli %k, %arg0 : i32
    "test.use"(%0) : (i32) -> ()
  }

  tt.return
}

}

// -----

//===----------------------------------------------------------------------===//
// Edge case 2: Outer loop containing an inner loop with no schedulable ops.
// The outer loop qualifies for scheduling (has TMA load), but the inner has
// only scalar ops. The pass must not crash on the empty inner DDG when
// building the child ScheduleLoop — exercises the
// `if (innerDDG.getNumNodes() == 0) return loopId;` guard in
// buildChildScheduleLoop.
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @outer_loop_with_empty_inner
// CHECK: tt.return
tt.func @outer_loop_with_empty_inner(
  %a_desc: !tt.tensordesc<tensor<128x64xf16>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %tiles = arith.constant 4 : i32

  scf.for %t = %c0_i32 to %tiles step %c1_i32 : i32 {
    %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    "test.use"(%a_shared) : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()

    // Inner loop with no schedulable ops — exercises empty-DDG guard.
    scf.for %k = %c0_i32 to %tiles step %c1_i32 : i32 {
      %0 = arith.addi %k, %t : i32
      "test.use"(%0) : (i32) -> ()
    }
  }

  tt.return
}

}
