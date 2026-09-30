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

// The phase is `(iv - lb) / step` under an unsigned division, so a negative
// step would wrap into a huge value and mis-phase the wait. `scf.for` allows
// one, so the region is skipped rather than synchronized incorrectly.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_skips_negative_step
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: ttng.wait_barrier
  // MBAR-NOT: ttng.arrive_barrier
  tt.func @pingpong_skips_negative_step(
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

// Both sides derive their wait phase from their own loop nest. If the nests
// disagree, one side's waits silently observe stale phases instead of pairing
// with the other side's arrivals, so the region is skipped.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_skips_mismatched_nests
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: ttng.wait_barrier
  // MBAR-NOT: ttng.arrive_barrier
  tt.func @pingpong_skips_mismatched_nests(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c8, %c16) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %pub: i32, %oub: i32) num_warps(4) {
      scf.for %iv = %lb to %pub step %step : i32 {
        tt.store %arg0, %iv {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %pub: i32, %oub: i32) num_warps(4) {
      scf.for %iv = %lb to %oub step %step : i32 {
        tt.store %arg1, %iv {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// The phase linearizes nested iterations, which equals the true arrival
// ordinal only when inner trip counts are outer-invariant. Here the inner ub
// is the outer iv (triangular nest), so the computed parity would drift from
// the real arrivals and the region is skipped.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // MBAR-LABEL: @pingpong_skips_triangular_nest
  // MBAR-NOT: ttng.init_barrier
  // MBAR-NOT: ttng.wait_barrier
  // MBAR-NOT: ttng.arrive_barrier
  tt.func @pingpong_skips_triangular_nest(
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
      scf.for %oiv = %lb to %ub step %step : i32 {
        scf.for %i = %lb to %oiv step %step : i32 {
          tt.store %arg0, %i {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %oiv = %lb to %ub step %step : i32 {
        scf.for %i = %lb to %oiv step %step : i32 {
          tt.store %arg1, %i {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
          scf.yield
        }
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}
