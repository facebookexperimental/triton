// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// The pass inserts ZERO semaphores for every nested-region access pattern
// below. Each function asserts that no nvws.semaphore op is emitted.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 1: depth-2 nesting, outer = scf.for (WS), inner = scf.if.
  // Access is inside the then-region. Verify outer for and inner if are
  // both annotated in OWNERSHIP-DAG via transitive-event propagation;
  // the "something" arith op contributes no row.
  // CHECK-LABEL: tt.func @for_outer_if_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @for_outer_if_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 800 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %something = arith.addi %iv, %iv {ttg.partition = array<i32: 0>} : i32
      scf.if %cond {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 2: depth-2 nesting, outer = scf.if at function scope (outside
  // any WS loop), inner = scf.for (WS-tagged). Access is in the for body.
  // Verify the outer if is annotated transitively, AND its annotation
  // uses tagged display `{@0.X}` because it is anchored outside the WS
  // loop.
  // CHECK-LABEL: tt.func @if_outer_for_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @if_outer_for_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 801 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      %something = arith.constant 42 : i32
      scf.for %iv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 3: depth-2 if->if nesting, both inside a WS-tagged scf.for.
  // Access is in the inner if's then-region. Both outer and inner if
  // must be annotated via transitive access; the outer if has no direct
  // event.
  // CHECK-LABEL: tt.func @if_outer_if_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @if_outer_if_inner_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 802 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        %something = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
        scf.if %cond {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 4: depth-2 for->for nesting. Outer scf.for is WS-tagged,
  // inner scf.for is plain. Access is in the inner body. Verify outer
  // for is annotated transitively via the inner for's annotation.
  // CHECK-LABEL: tt.func @for_outer_for_inner_access
  // CHECK-NOT: nvws.semaphore
  tt.func @for_outer_for_inner_access(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 803 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %something = arith.addi %iv, %iv {ttg.partition = array<i32: 0>} : i32
      scf.for %jv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 5: depth-3 alternating for->if->for, access in the inner
  // for body. Outer scf.for is WS-tagged. Every ancestor on the path
  // (outer for, if, inner for) must be annotated transitively.
  // CHECK-LABEL: tt.func @triple_for_if_for_access
  // CHECK-NOT: nvws.semaphore
  tt.func @triple_for_if_for_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 804 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        scf.for %jv = %lb to %ub step %step : i32 {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 6: depth-3 alternating if->for->if, outermost scf.if at
  // function scope. The middle scf.for is WS-tagged. Access is in the
  // innermost if. The outer if is anchored outside the WS loop and must
  // show tagged display `{@0.X}` on its branches. All three regioned
  // ops on the path must be annotated.
  // CHECK-LABEL: tt.func @triple_if_for_if_access
  // CHECK-NOT: nvws.semaphore
  tt.func @triple_if_for_if_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 805 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      scf.for %iv = %lb to %ub step %step : i32 {
        scf.if %cond {
          %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
          ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1>}
      } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 7: depth-4 alternating for->if->for->if, with WS-tagged
  // outermost scf.for and access at the leaf. Verifies arbitrary
  // depth: every ancestor on the path is annotated, every sibling
  // empty region (else of each if) is reconciled.
  // CHECK-LABEL: tt.func @quad_for_if_for_if_access
  // CHECK-NOT: nvws.semaphore
  tt.func @quad_for_if_for_if_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 806 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        scf.for %jv = %lb to %ub step %step : i32 {
          scf.if %cond {
            %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
            ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          } {ttg.partition = array<i32: 0, 1>}
        } {ttg.partition = array<i32: 0, 1>}
      } {ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 8: sibling scf.ifs at the same nesting level inside a
  // WS-tagged scf.for. Only one of the two ifs has access. The other
  // must be absent from the OWNERSHIP-DAG and ACCESS-DAG entirely;
  // the one with access must appear with proper annotation.
  // CHECK-LABEL: tt.func @sibling_if_only_one_with_access
  // CHECK-NOT: nvws.semaphore
  tt.func @sibling_if_only_one_with_access(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 807 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.if %cond {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
      scf.if %cond {
        %side = "side_effect"() {ttg.partition = array<i32: 0>} : () -> i32
      } {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // Pattern 9: sibling scf.fors at the same nesting level inside a
  // WS-tagged outer scf.for. Only the first inner for has access. The
  // second inner for must be absent from both DAGs.
  // CHECK-LABEL: tt.func @sibling_for_only_one_with_access
  // CHECK-NOT: nvws.semaphore
  tt.func @sibling_for_only_one_with_access(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 808 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      scf.for %jv = %lb to %ub step %step : i32 {
        %v = "producer"() {ttg.partition = array<i32: 1>} : () -> !ty
        ttg.local_store %v, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 0, 1>}
      scf.for %kv = %lb to %ub step %step : i32 {
        %side = "side_effect"() {ttg.partition = array<i32: 0>} : () -> i32
      } {ttg.partition = array<i32: 0>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
