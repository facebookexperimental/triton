// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

// Meta treats a marked loop with no schedulable load or MMA as a no-op. The
// NVWS transactional wrapper must preserve that success behavior.

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @no_schedule
  // CHECK: scf.for
  // CHECK: } {tt.warp_specialize}
  // CHECK-NOT: ttg.partition
  tt.func @no_schedule(%lb: i32, %ub: i32, %step: i32) {
    scf.for %iv = %lb to %ub step %step : i32 {
      %next = arith.addi %iv, %step : i32
      "use"(%next) : (i32) -> ()
    } {tt.warp_specialize}
    tt.return
  }
}
