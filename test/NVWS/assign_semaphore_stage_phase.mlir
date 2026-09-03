// RUN: triton-opt %s --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  tt.func @assign_stage_basic(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: {{%.*}}:2 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[ONE]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[VIEW:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %val = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      ttg.local_store %val, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
