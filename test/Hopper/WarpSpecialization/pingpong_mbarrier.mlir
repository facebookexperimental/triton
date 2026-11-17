// RUN: triton-opt %s --nvgpu-test-ping-pong-sync="capability=100 num-warp-groups=3" | FileCheck %s --check-prefix=MBAR
// RUN: triton-opt %s --nvgpu-test-ping-pong-sync="capability=100 num-warp-groups=3" --triton-nvidia-gpu-promote-mbarrier-to-named-barrier | FileCheck %s --check-prefix=NAMED

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_avoids_user_and_warp_specialize_ids
  // MBAR: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // MBAR: ttng.init_barrier
  // MBAR: ttng.init_barrier
  // MBAR: partition0
  // MBAR: ttng.wait_barrier
  // MBAR: ttng.arrive_barrier
  // MBAR: partition1
  // MBAR: ttng.arrive_barrier
  // MBAR: ttng.wait_barrier
  // MBAR: ttng.arrive_barrier
  // MBAR: ttng.inval_barrier
  // MBAR: ttng.inval_barrier
  // MBAR: ttg.local_dealloc
  // MBAR-NOT: barrier_named

  // NAMED: module attributes
  // NAMED-SAME: ttng.warp_specialize_barrier_ids = array<i32: 2, 5>
  // NAMED-LABEL: @pingpong_avoids_user_and_warp_specialize_ids
  // NAMED-NOT: ttg.local_alloc
  // NAMED: partition0
  // NAMED: arith.constant 7 : i32
  // NAMED: ttng.wait_barrier_named
  // NAMED: arith.constant 6 : i32
  // NAMED: ttng.arrive_barrier_named
  // NAMED: partition1
  // NAMED: arith.constant 7 : i32
  // NAMED: ttng.arrive_barrier_named
  // NAMED: arith.constant 6 : i32
  // NAMED: ttng.wait_barrier_named
  // NAMED: arith.constant 7 : i32
  // NAMED: ttng.arrive_barrier_named
  // NAMED-NOT: ttng.init_barrier
  // NAMED-NOT: ttng.inval_barrier
  tt.func @pingpong_avoids_user_and_warp_specialize_ids(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %user3 = ttng.user_named_barrier_id %c3 : i32
    %user4 = ttng.user_named_barrier_id %c4 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c2) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        tt.store %arg0, %iv {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        tt.store %arg1, %iv {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// A while around one side restarts its inner counter on every outer
// iteration, while the other side counts straight. The loop bounds alone
// match, but the two sides cannot rendezvous reliably.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_skips_while_enclosed_nest
  // NAMED-LABEL: @pingpong_skips_while_enclosed_nest
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: scf.while
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: tt.return
  tt.func @pingpong_skips_while_enclosed_nest(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c8) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      %w = scf.while (%i = %lb) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %ub : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          scf.for %j = %lb to %ub step %step : i32 {
            tt.store %arg0, %j {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
            scf.yield
          }
          scf.yield %ia : i32
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %j = %lb to %ub step %step : i32 {
        tt.store %arg1, %j {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// A guard around one side's nest can skip all of its iterations while the
// other side still waits, even when their inner for bounds match.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_skips_if_guarded_nest
  // NAMED-LABEL: @pingpong_skips_if_guarded_nest
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: scf.if
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: tt.return
  tt.func @pingpong_skips_if_guarded_nest(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c8) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      %guard = arith.cmpi slt, %lb, %ub : i32
      scf.if %guard {
        scf.for %j = %lb to %ub step %step : i32 {
          tt.store %arg0, %j {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %j = %lb to %ub step %step : i32 {
        tt.store %arg1, %j {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// An exact counter on each side does not make a rendezvous safe when the two
// loops have different trip counts: ping's first wait has no pong arrival.
// Skip the entire ping-pong region before allocating any barriers.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_mismatched_loop_bounds
  // MBAR-NOT: ttg.local_alloc
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: ttng.wait_barrier
  // MBAR-NOT: ttng.arrive_barrier
  // MBAR: tt.return
  // NAMED-LABEL: @pingpong_mismatched_loop_bounds
  // NAMED-NOT: ttg.local_alloc
  // NAMED-NOT: ttng.wait_barrier
  // NAMED-NOT: ttng.arrive_barrier
  // NAMED: tt.return
  tt.func @pingpong_mismatched_loop_bounds(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 2 : i32
    %step = arith.constant 1 : i32
    ttg.warp_specialize(%ping_ptr, %pong_ptr, %lb, %ub, %step) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %start: i32,
               %end: i32, %stride: i32) num_warps(4) {
      scf.for %iv = %start to %end step %stride : i32 {
        tt.store %arg0, %iv {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %start: i32,
               %end: i32, %stride: i32) num_warps(4) {
      scf.for %iv = %start to %start step %stride : i32 {
        tt.store %arg1, %iv {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// A negative step, which `scf.for` permits. The phase is the parity of a
// loop-carried counter that advances once per iteration, so it does not depend
// on the sign of the step or on the loop bounds -- the region synchronizes
// normally. Deriving the phase from `(iv - lb) / step` instead would wrap the
// subtraction under an unsigned division and mis-phase the wait.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_negative_step
  // NAMED-LABEL: @pingpong_negative_step
  // MBAR: ttng.init_barrier
  // MBAR: partition0
  // MBAR: arith.andi
  // MBAR: ttng.wait_barrier
  // MBAR: partition1
  // MBAR: arith.andi
  // MBAR: ttng.wait_barrier
  tt.func @pingpong_negative_step(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %cneg1 = arith.constant -1 : i32
    %c8 = arith.constant 8 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c8, %cneg1, %c0) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        tt.store %arg0, %iv {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        tt.store %arg1, %iv {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// A triangular nest: the inner trip count depends on the outer induction
// variable, as in a causal mask. Linearizing `outer * inner_trip + inner`
// reads that trip count once and treats it as fixed, so the phase drifts from
// the first inner iteration onward. The carried counter advances once per
// inner iteration regardless, so the nest synchronizes normally and the
// counter is threaded through both loops.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_triangular_nest
  // NAMED-LABEL: @pingpong_triangular_nest
  // MBAR: ttng.init_barrier
  // MBAR: partition0
  // The inner loop resumes from the counter the outer loop is carrying, and
  // the phase and the increment both come off the inner one -- using the outer
  // counter would restart the parity on every outer iteration.
  // MBAR: scf.for {{.*}} iter_args(%[[OUTER:.+]] = %{{.+}}) -> (i32)
  // MBAR: scf.for {{.*}} iter_args(%[[INNER:.+]] = %[[OUTER]]) -> (i32)
  // MBAR: arith.andi %[[INNER]]
  // MBAR: ttng.wait_barrier
  // MBAR: arith.addi %[[INNER]]
  tt.func @pingpong_triangular_nest(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c8) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %i = %lb to %ub step %step : i32 {
        scf.for %j = %lb to %i step %step : i32 {
          tt.store %arg0, %j {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %i = %lb to %ub step %step : i32 {
        scf.for %j = %lb to %i step %step : i32 {
          tt.store %arg1, %j {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// An inner loop guarded by an `scf.if` is not directly nested in its parent
// loop: threading a counter through the guard would yield the inner loop's
// result -- defined inside the `scf.if` region -- from the outer loop's
// terminator, which does not dominate it. The region is skipped instead of
// producing invalid IR.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_guarded_inner_loop_skipped
  // NAMED-LABEL: @pingpong_guarded_inner_loop_skipped
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: scf.if
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: iter_args
  // MBAR: tt.return
  tt.func @pingpong_guarded_inner_loop_skipped(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c8) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %i = %lb to %ub step %step : i32 {
        %guard = arith.cmpi slt, %i, %ub : i32
        scf.if %guard {
          scf.for %j = %lb to %ub step %step : i32 {
            tt.store %arg0, %j {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
            scf.yield
          }
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %i = %lb to %ub step %step : i32 {
        %guard = arith.cmpi slt, %i, %ub : i32
        scf.if %guard {
          scf.for %j = %lb to %ub step %step : i32 {
            tt.store %arg1, %j {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
            scf.yield
          }
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}
