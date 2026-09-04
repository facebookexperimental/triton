// RUN: triton-opt %s --split-input-file --allow-unregistered-dialect --nvws-lower-semaphore --verify-diagnostics | FileCheck %s --implicit-check-not=nvws.semaphore

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @lower_only
  tt.func @lower_only() {
    // CHECK: [[STAGE:%.*]] = arith.constant 1 : i32
    %stage = arith.constant 1 : i32
    // CHECK: [[PHASE:%.*]] = arith.constant 1 : i32
    %phase = arith.constant 1 : i32
    // CHECK: [[BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK-COUNT-2: ttng.init_barrier {{%.*}}, 1
    %sem = nvws.semaphore.create %buf released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[WAIT_VIEW:%.*]] = ttg.memdesc_index [[MBAR]][[[STAGE]]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttng.wait_barrier [[WAIT_VIEW]], [[PHASE]] {ttg.partition = array<i32: 0>}
    %tok = nvws.semaphore.acquire %sem[%stage, %phase] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[DATA_VIEW:%.*]] = ttg.memdesc_index [[BUF]][[[STAGE]]] {ttg.partition = array<i32: 0>}
    %view = nvws.semaphore.buffer %sem[%stage], %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: "use"([[DATA_VIEW]])
    "use"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    // CHECK: [[ARRIVE_VIEW:%.*]] = ttg.memdesc_index [[MBAR]][[[STAGE]]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttng.arrive_barrier [[ARRIVE_VIEW]], 1 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem[%stage], %tok [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK-COUNT-2: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR]]
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @requires_assignment() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // expected-error @below {{requires stage and phase operands; run nvws-assign-semaphore-stage-phase first}}
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{requires a stage operand; run nvws-assign-semaphore-stage-phase first}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // expected-error @below {{requires a stage operand; run nvws-assign-semaphore-stage-phase first}}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    "use"(%view) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
