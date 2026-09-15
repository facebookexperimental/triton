// RUN: triton-opt --tlx-print-ttgir-to-tlx -split-input-file --verify-diagnostics %s | FileCheck %s

// Test that every emitted async_task has a well-formed body.
//
// A warp_specialize partition can hold only ops the printer skips, or only an
// `# unsupported` marker. Python needs a body after `with`, so either case has to
// be spelled `pass` -- otherwise the whole generated file fails to parse, not just
// the offending block.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // The default region carries no printable op, so it needs `pass`; the
  // partition after it does have one and must be unaffected.
  // CHECK-LABEL: def empty_default_task(
  // CHECK: with tlx.async_tasks():
  // CHECK-NEXT: with tlx.async_task("default"):
  // CHECK-NEXT: pass
  // CHECK: with tlx.async_task(num_warps=4
  // CHECK-NEXT: {{[a-zA-Z_0-9]+}} = tl.full
  tt.func public @empty_default_task() attributes {noinline = false} {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// A body holding only an `# unsupported` marker is as unparseable as an empty
// one, so the marker is kept and `pass` is added after it.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def comment_only_task(
  // CHECK: with tlx.async_task("default"):
  // CHECK-NEXT: # unsupported: barrier slots with differing arrive counts
  // CHECK-NEXT: pass
  tt.func public @comment_only_task() attributes {noinline = false} {
    ttg.warp_specialize()
    default {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      // expected-error @+1 {{barrier slots with differing arrive counts do not round-trip to TLX}}
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
      %v0 = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %v1 = ttg.memdesc_index %bar[%c1_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.init_barrier %v0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.init_barrier %v1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
