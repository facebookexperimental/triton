// RUN: triton-opt %s --nvgpu-verify-ws-barriers="dump-buffer-id=0" 2>&1 1>/dev/null | FileCheck %s
//
// --dump-buffer-id shows how each barrier op's slot index and phase relate to
// the loop accumulation counter. Here a 2-buffered barrier is driven by a loop
// iter-arg `acc`: slot = acc % 2, phase = (acc / 2) & 1 -- the mbarrier phase
// model expressed in terms of the loop counter. (On IR from the real pipeline
// the iter-arg is tagged NameLoc("accum_cnt") and prints as "accumCnt" plus an
// "[accumCnt] loop ..." summary line; a hand-written iter-arg prints as %argN.)

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @accum_demo(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %bar = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %v0 = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v0, 1 {buffer.id = 0 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step iter_args(%acc = %c0_i32) -> (i32) : i32 {
      %idx = arith.remui %acc, %c2_i32 : i32
      %d = arith.divui %acc, %c2_i32 : i32
      %ph = arith.andi %d, %c1_i32 : i32
      %bv = ttg.memdesc_index %bar[%idx] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %bv, 1 {constraints = {WSBarrier = {dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.wait_barrier %bv, %ph {constraints = {WSBarrier = {direction = "forward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %accn = arith.addi %acc, %c1_i32 : i32
      scf.yield %accn : i32
    }
    tt.return
  }
}

// CHECK: ops for reuse group buffer.id=0
// The loop body is control region r#0 (a for). The arrive/wait live in it and
// their slot+phase are shown relative to the loop counter (%arg1 here;
// "accumCnt" on real-pipeline IR). Being in-loop, the phase is accumCnt-derived
// (toggles), not a constant; both ops reference the same guarding barrier bar#0.
// CHECK: regions: r#0=for
// CHECK: [arrive] r#0 {{.*}}bar#0 idx=(%arg{{[0-9]+}} % 2){{.*}}arrive_barrier
// CHECK: [wait(fwd)] r#0 {{.*}}bar#0 idx=(%arg{{[0-9]+}} % 2) phase=((%arg{{[0-9]+}} / 2) & 1){{.*}}wait_barrier
