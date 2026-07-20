// RUN: triton-opt %s --nvgpu-verify-ws-barriers="check=deadlock" -split-input-file -verify-diagnostics
//
// Phase-1 deadlock check, condition (i) of the design doc: a FULL (forward)
// wait_barrier whose barrier alloc has NO producing arrive anywhere is a genuine
// deadlock -- the phase it polls is never produced. Backward (reuse) waits with
// no arrive are NOT flagged here (they may be satisfied by an init-time pre-arm;
// handled by the later pre-arm-aware SCC phase).
// See lib/Transforms/WarpSpecialization/docs/WSAliasingCoverage.proposal.md.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A FULL/forward wait whose barrier is never arrived -> deadlock.
  tt.func public @full_wait_no_producer() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %b = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v, 1 {buffer.id = 0 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @below {{FULL wait_barrier has no producing arrive}}
    ttng.wait_barrier %v, %c1_i32 {constraints = {WSBarrier = {direction = "forward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A FULL/forward wait WITH a producing arrive -> no deadlock (no diagnostic).
  tt.func public @full_wait_with_producer() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %b = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v, 1 {buffer.id = 0 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %v, 1 {constraints = {WSBarrier = {dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %v, %c1_i32 {constraints = {WSBarrier = {direction = "forward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A BACKWARD (reuse) wait with no arrive is NOT flagged (may be pre-armed).
  tt.func public @backward_wait_no_producer_ok() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %b = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v, 1 {buffer.id = 0 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %v, %c1_i32 {constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
