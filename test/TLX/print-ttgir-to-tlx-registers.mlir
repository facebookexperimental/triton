// RUN: triton-opt --tlx-print-ttgir-to-tlx -split-input-file %s | FileCheck %s

// Register counts come from actualRegisters, which AllocateWarpGroups fills in
// before this printer runs: the default region's count first, then one per
// partition. A requested -1 never reaches the output, and the padding
// partition AllocateWarpGroups appends is empty, so it is skipped.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def actual_registers(
  // CHECK: with tlx.async_task("default", registers=160):
  // CHECK: with tlx.async_task(num_warps=4, registers=24):
  // CHECK: with tlx.async_task(num_warps=4, registers=160):
  // CHECK-NOT: with tlx.async_task(
  // CHECK-NOT: registers=-1
  tt.func public @actual_registers() attributes {noinline = false} {
    ttg.warp_specialize() attributes {actualRegisters = array<i32: 160, 24, 160, 24>, requestedRegisters = array<i32: 24, -1, 24>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %cst = arith.constant dense<2.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    }
    partition2() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// Without actualRegisters, concrete requests print as they are.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def requested_registers(
  // CHECK: with tlx.async_task("default", registers=80):
  // CHECK: with tlx.async_task(num_warps=4, registers=48):
  tt.func public @requested_registers() attributes {noinline = false} {
    ttg.warp_specialize() attributes {defaultRequestedRegisters = 80 : i32, requestedRegisters = array<i32: 48>}
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

// Without actualRegisters, a request still holding -1 prints no counts: the
// frontend has no spelling for it and takes counts for all tasks or none.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def unresolved_registers(
  // CHECK: with tlx.async_task("default"):
  // CHECK: with tlx.async_task(num_warps=4):
  // CHECK: with tlx.async_task(num_warps=4):
  tt.func public @unresolved_registers() attributes {noinline = false} {
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24, -1>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %cst = arith.constant dense<2.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// A default share above 256 is what the default task gets without asking, and
// tlx.async_task accepts at most 256, so it is left unprinted; a partition's
// count is clamped to 256, as lowering does.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def registers_above_limit(
  // CHECK: with tlx.async_task("default"):
  // CHECK: with tlx.async_task(num_warps=4, registers=24):
  // CHECK: with tlx.async_task(num_warps=4, registers=256):
  tt.func public @registers_above_limit() attributes {noinline = false} {
    ttg.warp_specialize() attributes {actualRegisters = array<i32: 464, 24, 280>, requestedRegisters = array<i32: 24, -1>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %cst = arith.constant dense<2.000000e+00> : tensor<128xf32, #blocked>
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// An empty actualRegisters carries no counts, so the concrete request prints.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def empty_actual_registers(
  // CHECK: with tlx.async_task("default"):
  // CHECK: with tlx.async_task(num_warps=4, registers=48):
  tt.func public @empty_actual_registers() attributes {noinline = false} {
    ttg.warp_specialize() attributes {actualRegisters = array<i32>, requestedRegisters = array<i32: 48>}
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
