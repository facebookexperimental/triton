// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-verify-named-barrier-counts --verify-diagnostics -o /dev/null
// RUN: not triton-opt %s -split-input-file --triton-nvidia-gpu-verify-named-barrier-counts 2>&1 | FileCheck %s --check-prefix=FAIL

// Counts covering every participating task pass silently.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @verify_ok() {
    ttg.warp_specialize()
    default {
      %c4 = arith.constant 4 : i32
      %c160 = arith.constant 160 : i32
      %id = ttng.compiler_named_barrier_id %c4 : i32
      ttng.arrive_barrier_named %id, %c160 : !ttng.named_barrier_id, i32
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      %c4_0 = arith.constant 4 : i32
      %c160_0 = arith.constant 160 : i32
      %id_0 = ttng.compiler_named_barrier_id %c4_0 : i32
      ttng.wait_barrier_named %id_0, %c160_0 : !ttng.named_barrier_id, i32
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// A count baked from pre-shrink task sizes fails compilation instead of
// deadlocking at bar.sync. Default has 4 warps, partition0 has 1 warp, so
// the barrier below requires (4 + 1) * 32 = 160 threads, not 256.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @verify_stale_count() {
    ttg.warp_specialize()
    default {
      %c4 = arith.constant 4 : i32
      %c256 = arith.constant 256 : i32
      %id = ttng.compiler_named_barrier_id %c4 : i32
      // expected-error@+1 {{named barrier participant count mismatch: baked 256 threads but participating tasks now require 160}}
      ttng.arrive_barrier_named %id, %c256 : !ttng.named_barrier_id, i32
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      %c4_0 = arith.constant 4 : i32
      %c256_0 = arith.constant 256 : i32
      %id_0 = ttng.compiler_named_barrier_id %c4_0 : i32
      ttng.wait_barrier_named %id_0, %c256_0 : !ttng.named_barrier_id, i32
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// FAIL: named barrier participant count mismatch: baked 256 threads but participating tasks now require 160

// -----

// User-allocated barriers carry hand-written counts and are skipped.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @user_barrier_skipped() {
    ttg.warp_specialize()
    default {
      %c3 = arith.constant 3 : i32
      %c999 = arith.constant 999 : i32
      %id = ttng.user_named_barrier_id %c3 : i32
      ttng.arrive_barrier_named %id, %c999 : !ttng.named_barrier_id, i32
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
