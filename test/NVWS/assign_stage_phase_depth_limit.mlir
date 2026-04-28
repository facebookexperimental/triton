// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-assign-stage-phase 2>&1 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: multiphase bitmask supports at most 32 buffer stages, got 33
  tt.func @reject_phase_mask_depth_above_32() {
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<33x1xi32, #shared, #smem, mutable>
    %semaphore = nvws.semaphore.create %buffer released = -1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<33x1xi32, #shared, #smem, mutable>]>
    tt.return
  }
}
