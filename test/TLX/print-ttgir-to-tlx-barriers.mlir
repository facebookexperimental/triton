// RUN: triton-opt --tlx-print-ttgir-to-tlx -split-input-file --verify-diagnostics %s | FileCheck %s

// Test that barrier allocations round-trip with the arrive count they were
// initialized with.
//
// A barrier allocation is recovered from the ttg.local_alloc that backs it,
// whether a slot is selected out of it with ttg.memdesc_index or it is used
// directly, and ttng.init_barrier's `count` selects the emitted form: a count
// that is a positive multiple of the warp size is a warp barrier, any other
// non-unit count is passed through, and the default count of 1 is left implicit.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // A single-slot barrier is used directly, with no memdesc_index selecting a
  // slot. It is still a barrier allocation, not a plain buffer.
  // CHECK-LABEL: def single_slot_barrier(
  // CHECK: tlx.alloc_barriers(1)
  // CHECK-NOT: tlx.local_alloc(
  tt.func public @single_slot_barrier() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // The count is read off init_barrier on the direct path too, not just when a
  // slot is selected with memdesc_index.
  // CHECK-LABEL: def single_slot_arrive_count(
  // CHECK: tlx.alloc_barriers(1, 2)
  tt.func public @single_slot_arrive_count() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A non-unit arrive count is emitted as the second argument.
  // CHECK-LABEL: def non_unit_arrive_count(
  // CHECK: tlx.alloc_barriers(1, 2)
  tt.func public @non_unit_arrive_count() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %view, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %view, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // An arrive count that is a multiple of the warp size is a warp barrier,
  // carrying the number of warps rather than the raw thread count.
  // CHECK-LABEL: def warp_barrier(
  // CHECK: tlx.alloc_warp_barrier(2, 4)
  tt.func public @warp_barrier() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %view, 128 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %view, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // The default arrive count of 1 stays implicit.
  // CHECK-LABEL: def default_arrive_count(
  // CHECK: tlx.alloc_barriers(2)
  // CHECK-NOT: tlx.alloc_barriers(2, 1)
  tt.func public @default_arrive_count() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %view, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %view, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // An arrive count is an attribute, not an operand, and is emitted positionally.
  // CHECK-LABEL: def arrive_with_count(
  // CHECK: tlx.barrier_arrive({{.*}}, 2)
  tt.func public @arrive_with_count() attributes {noinline = false} {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A predicate is an optional operand and must be named, not passed
  // positionally into the arrive_count slot.
  // CHECK-LABEL: def arrive_with_predicate(
  // CHECK: tlx.barrier_arrive({{.*}}, pred=
  tt.func public @arrive_with_predicate() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %pred = arith.cmpi sgt, %pid, %c0_i32 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A perThread arrive needs no keyword: barrier_arrive picks the per-thread
  // lowering off the barrier, which alloc_warp_barrier restores.
  // CHECK-LABEL: def per_thread_arrive(
  // CHECK: tlx.alloc_warp_barrier(1, 4)
  // CHECK: tlx.barrier_arrive(
  // CHECK-NOT: per_thread=
  tt.func public @per_thread_arrive() attributes {noinline = false} {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 128 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 {perThread} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Warp size comes from the module, not a hardcoded 32: with 64-wide warps the
// same count of 128 is two warps rather than four.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: def warp_barrier_64_wide(
  // CHECK: tlx.alloc_warp_barrier(1, 2)
  tt.func public @warp_barrier_64_wide() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 128 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// One allocation emits one alloc call, so slots that disagree on arrive count
// cannot be represented: either count would be wrong for the other.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def conflicting_arrive_counts(
  // CHECK: # unsupported: barrier slots with differing arrive counts
  // CHECK-NOT: tlx.alloc_barriers(
  tt.func public @conflicting_arrive_counts() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // expected-error @+1 {{barrier slots with differing arrive counts do not round-trip to TLX}}
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %v0 = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %v1 = ttg.memdesc_index %bar[%c1_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %v1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A multicast arrive signals every CTA in its equivalence class; emitting it
// as a local arrive would silently drop the multicast. It needs more than one
// CTA and the identity CGA layout, hence a separate module.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {

  // CHECK-LABEL: def multicast_arrive(
  // CHECK: # unsupported: multicast arrive_barrier (ctaMask=1)
  // CHECK-NOT: tlx.barrier_arrive(
  tt.func public @multicast_arrive() attributes {noinline = false} {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #shared, #smem, mutable>
    // expected-error @+1 {{multicast arrive_barrier does not round-trip to TLX}}
    ttng.arrive_barrier %bar, 1 {ctaMask = 1 : i32} : !ttg.memdesc<2xi64, #shared, #smem, mutable>
    tt.return
  }
}
