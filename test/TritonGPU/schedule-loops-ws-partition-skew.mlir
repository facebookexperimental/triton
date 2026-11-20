// RUN: triton-opt %s "-tritongpu-schedule-loops=num-stages=4 use-meta-ws=true" | FileCheck %s

// A loop partitioned out of a warp-specialized loop keeps the stages of the
// original loop. Here the producer partition of a persistent GEMM whose K loop
// folded away holds the operand load at stage 0 and an epilogue (bias) load at
// stage 3. The operand ring has 2 slots, so pipelining the partition with a
// load skew of 3 can deadlock: the operand load for tile t waits on a release that
// can need the bias of tile t-2, which the skewed loop has not issued yet.
// ScheduleLoops must not pipeline it.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1d = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, rank = 1}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @producer_skew_exceeds_ring_depth
// CHECK: scf.for
// CHECK-NOT: loop.stage = 3
// CHECK: tt.scheduled_max_stage = 0
tt.func public @producer_skew_exceeds_ring_depth(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// The same partition with a skew of 1, within the 2-slot ring, stays pipelined.

// CHECK-LABEL: @producer_skew_within_ring_depth
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}}loop.stage = 0
// CHECK: ttng.wait_barrier {{.*}}loop.stage = 1
// CHECK: tt.scheduled_max_stage = 1
tt.func public @producer_skew_within_ring_depth(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
  tt.return
}

// A skew equal to the ring depth stays pipelined: the stage-2 op runs before
// the wait in the loop body, so it has been issued for the tile the wait needs.

// CHECK-LABEL: @producer_skew_equals_ring_depth
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}}loop.stage = 0
// CHECK: ttng.wait_barrier {{.*}}loop.stage = 2
// CHECK: tt.scheduled_max_stage = 2
tt.func public @producer_skew_equals_ring_depth(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize}
  tt.return
}

// A constant index into an array of barriers (e.g. one per data partition) is
// a single-slot ring, so a skew of 2 is not pipelined.

// CHECK-LABEL: @producer_constant_index_barrier
// CHECK: scf.for
// CHECK-NOT: loop.stage = 2
// CHECK: tt.scheduled_max_stage = 0
tt.func public @producer_constant_index_barrier(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%c1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize}
  tt.return
}

// Only late loads are checked: a late op that consumes data already loaded
// (here an arrive) stays pipelined even past the ring depth.

// CHECK-LABEL: @late_non_load_stays_pipelined
// CHECK: scf.for
// CHECK: ttng.arrive_barrier {{.*}}loop.stage = 3
// CHECK: tt.scheduled_max_stage = 3
tt.func public @late_non_load_stays_pipelined(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.arrive_barrier %bfull, 1 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// The slot comes from a loop-carried counter, as in partitioned IR. A load
// skew of 3 over 2 slots is not pipelined.

// CHECK-LABEL: @iter_arg_slot_skew_exceeds_ring
// CHECK: scf.for
// CHECK-NOT: loop.stage = 3
// CHECK: tt.scheduled_max_stage = 0
tt.func public @iter_arg_slot_skew_exceeds_ring(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %b_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %b_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %b_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%cnt = %c0) -> (i32) : i32 {
    %slot = arith.remsi %cnt, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %cnt, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    %next = arith.addi %cnt, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    scf.yield %next : i32
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// Two waits on the same slot Value (e.g. the A and B loads sharing one empty
// barrier) take one slot per iteration, not two: the 4-slot ring is reused
// after 4 iterations, so a load skew of 3 stays pipelined.

// CHECK-LABEL: @same_slot_waited_twice
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}}loop.stage = 3
// CHECK: tt.scheduled_max_stage = 3
tt.func public @same_slot_waited_twice(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %s0 = arith.remsi %i, %c4 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty0 = ttg.memdesc_index %a_bars[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty0, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty0, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full0 = ttg.memdesc_index %a_full[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf0 = ttg.memdesc_index %a_bufs[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf0, %full0, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// Two data partitions share one 4-slot ring, taking 2 slots per iteration, so
// a slot is reused after 2 iterations: a load skew of 3 is not pipelined.

// CHECK-LABEL: @shared_ring_two_slots_per_iteration
// CHECK: scf.for
// CHECK-NOT: loop.stage = 3
// CHECK: tt.scheduled_max_stage = 0
tt.func public @shared_ring_two_slots_per_iteration(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>,
    %b_bars: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %b_full: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %b_bufs: !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %i2 = arith.muli %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s0 = arith.remsi %i2, %c4 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %i21 = arith.addi %i2, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s1 = arith.remsi %i21, %c4 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty0 = ttg.memdesc_index %a_bars[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty0, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full0 = ttg.memdesc_index %a_full[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf0 = ttg.memdesc_index %a_bufs[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf0, %full0, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %empty1 = ttg.memdesc_index %a_bars[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty1, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full1 = ttg.memdesc_index %a_full[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf1 = ttg.memdesc_index %a_bufs[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf1, %full1, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// The loop takes 4 slots per iteration from a 2-slot ring, so a slot is
// reused within the iteration (reuse distance 0): even a load skew of 1 is not
// pipelined.

// CHECK-LABEL: @more_slots_per_iteration_than_ring
// CHECK: scf.for
// CHECK-NOT: loop.stage = 1
// CHECK: tt.scheduled_max_stage = 0
tt.func public @more_slots_per_iteration_than_ring(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %i4 = arith.muli %i, %c4 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s0 = arith.remsi %i4, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty0 = ttg.memdesc_index %a_bars[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty0, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full0 = ttg.memdesc_index %a_full[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf0 = ttg.memdesc_index %a_bufs[%s0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf0, %full0, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %o1 = arith.addi %i4, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s1 = arith.remsi %o1, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty1 = ttg.memdesc_index %a_bars[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty1, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full1 = ttg.memdesc_index %a_full[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf1 = ttg.memdesc_index %a_bufs[%s1] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf1, %full1, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %o2 = arith.addi %i4, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s2 = arith.remsi %o2, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty2 = ttg.memdesc_index %a_bars[%s2] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty2, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full2 = ttg.memdesc_index %a_full[%s2] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf2 = ttg.memdesc_index %a_bufs[%s2] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf2, %full2, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %o3 = arith.addi %i4, %c3 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %s3 = arith.remsi %o3, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty3 = ttg.memdesc_index %a_bars[%s3] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty3, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full3 = ttg.memdesc_index %a_full[%s3] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf3 = ttg.memdesc_index %a_bufs[%s3] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf3, %full3, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
  tt.return
}

// Two data partitions on separate 4-slot rings reuse a slot after 4
// iterations: a load skew of 3 stays pipelined.

// CHECK-LABEL: @separate_rings_one_slot_per_iteration
// CHECK: scf.for
// CHECK: async_tma_copy_global_to_local {{.*}}loop.stage = 3
// CHECK: tt.scheduled_max_stage = 3
tt.func public @separate_rings_one_slot_per_iteration(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>,
    %b_bars: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %b_full: !ttg.memdesc<4x1xi64, #bar, #smem, mutable>,
    %b_bufs: !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c4 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty0 = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty0, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full0 = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf0 = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf0, %full0, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %empty1 = ttg.memdesc_index %b_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty1, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full1 = ttg.memdesc_index %b_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf1 = ttg.memdesc_index %b_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf1, %full1, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize}
  tt.return
}

// A load skew equal to the ring depth with the late load after the wait in
// the loop body is left pipelined. It deadlocks only if the slot's release
// depends on that load; FA backward pipelines this shape without that
// dependency, and flattening it would cost FA backward its pipelining.

// CHECK-LABEL: @equal_skew_load_after_wait
// CHECK: scf.for
// CHECK: async_tma_copy_global_to_local {{.*}}loop.stage = 2
// CHECK: tt.scheduled_max_stage = 2
tt.func public @equal_skew_load_after_wait(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %b_bars: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %b_full: !ttg.memdesc<2x1xi64, #bar, #smem, mutable>,
    %b_bufs: !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    %slot = arith.remsi %i, %c2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %phase = arith.andi %i, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a_desc[%i, %c0] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bempty = ttg.memdesc_index %bias_bars[%c0] {loop.cluster = 2 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase {loop.cluster = 2 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] {loop.cluster = 2 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] {loop.cluster = 2 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32} : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize}
  tt.return
}

// A K loop that did not fold away: the tile loop is an outer loop and is not
// pipelined, and the K loop's loads are within their 3-slot ring.

// CHECK-LABEL: @non_folded_k_loop
// CHECK: scf.for
// CHECK: scf.for
// CHECK: async_tma_copy_global_to_local {{.*}}loop.stage = 1
// CHECK: tt.scheduled_max_stage = 1
// CHECK: async_tma_copy_global_to_local
// CHECK-NOT: loop.stage
// CHECK: } {tt.warp_specialize}
tt.func public @non_folded_k_loop(
    %a_desc: !tt.tensordesc<128x64xbf16, #shared>,
    %bias_desc: !tt.tensordesc<32xbf16, #shared1d>,
    %a_bars: !ttg.memdesc<3x1xi64, #bar, #smem, mutable>,
    %a_full: !ttg.memdesc<3x1xi64, #bar, #smem, mutable>,
    %a_bufs: !ttg.memdesc<3x128x64xbf16, #shared, #smem, mutable>,
    %bias_bars: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_full: !ttg.memdesc<1x1xi64, #bar, #smem, mutable>,
    %bias_bufs: !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable>,
    %n: i32, %k_tiles: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    scf.for %k = %c0 to %k_tiles step %c1 : i32 {
      %slot = arith.remsi %k, %c3 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %phase = arith.andi %k, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %empty = ttg.memdesc_index %a_bars[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
      ttng.wait_barrier %empty, %phase {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
      %full = ttg.memdesc_index %a_full[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
      %buf = ttg.memdesc_index %a_bufs[%slot] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<3x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %a_desc[%i, %k] %buf, %full, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      %slot1 = arith.addi %slot, %c1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %slot2 = arith.remsi %slot1, %c3 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %full2 = ttg.memdesc_index %a_full[%slot2] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<3x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
      %buf2 = ttg.memdesc_index %a_bufs[%slot2] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<3x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %a_desc[%k, %i] %buf2, %full2, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    %phase = arith.andi %i, %c1 : i32
    %bempty = ttg.memdesc_index %bias_bars[%c0] : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.wait_barrier %bempty, %phase : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bfull = ttg.memdesc_index %bias_full[%c0] : !ttg.memdesc<1x1xi64, #bar, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %bbuf = ttg.memdesc_index %bias_bufs[%c0] : !ttg.memdesc<1x32xbf16, #shared1d, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
    ttng.async_tma_copy_global_to_local %bias_desc[%i] %bbuf, %bfull, %true : !tt.tensordesc<32xbf16, #shared1d>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32xbf16, #shared1d, #smem, mutable>
  } {tt.warp_specialize}
  tt.return
}

}
