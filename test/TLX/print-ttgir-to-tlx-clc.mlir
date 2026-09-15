// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that Cluster Launch Control ops round-trip to their TLX spellings.
//
// A persistent CLC kernel issues a cancel request into a response buffer,
// waits on an mbarrier, then decodes the stolen tile out of the response.
// None of those ops have a generic mapping: the response buffer's ui128
// element type is unnameable in TLX, and the issue and query ops take
// different operand orders and result counts than the printer's default.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // The ui128 response buffer gets its own allocator rather than a local_alloc
  // of an element type TLX cannot name.
  // CHECK-LABEL: def clc_response_alloc(
  // CHECK: tlx._alloc_clc_responses(1)
  // CHECK-NOT: tlx.local_alloc(
  tt.func public @clc_response_alloc() attributes {noinline = false} {
    %resp = ttg.local_alloc : () -> !ttg.memdesc<1xui128, #shared, #smem, mutable>
    tt.return
  }

  // The issue op takes (mbarrier, response) in TTGIR but (response, barrier)
  // in TLX, so the operands are swapped rather than printed in order.
  // Binding each name where it is defined is what makes the swap observable:
  // the barrier is allocated first, so a call emitted in source order would
  // read _clc_issue([[BAR]], [[RESP]]) and fail here.
  // CHECK-LABEL: def clc_issue(
  // CHECK: [[BAR:[a-zA-Z_0-9]+]] = tlx.alloc_barriers(1)
  // CHECK: [[RESP:[a-zA-Z_0-9]+]] = tlx._alloc_clc_responses(1)
  // CHECK: tlx._clc_issue([[RESP]], [[BAR]])
  tt.func public @clc_issue() attributes {noinline = false} {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %resp = ttg.local_alloc : () -> !ttg.memdesc<1xui128, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_clc_try_cancel %bar, %resp : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xui128, #shared, #smem, mutable>
    tt.return
  }

  // The query decodes three results, which are bound as one tuple.
  // CHECK-LABEL: def clc_query(
  // CHECK: {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}} = tlx._clc_query(
  tt.func public @clc_query() attributes {noinline = false} {
    %resp = ttg.local_alloc : () -> !ttg.memdesc<1xui128, #shared, #smem, mutable>
    %x, %y, %z = ttng.clc_query_cancel %resp : (!ttg.memdesc<1xui128, #shared, #smem, mutable>) -> (i32, i32, i32)
    tt.return
  }

  // Despite its name, nvg.cluster_id is the CTA's rank within its cluster, in
  // [0, num_ctas): it lowers to %cluster_ctarank, not to a cluster index.
  // CHECK-LABEL: def cluster_id(
  // CHECK: tlx.cluster_cta_rank()
  // CHECK-NOT: tl.program_id
  tt.func public @cluster_id() attributes {noinline = false} {
    %cid = "nvg.cluster_id"() : () -> i32
    tt.return
  }
}
