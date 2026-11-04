// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck --enable-var-scope %s

// Shared-memory pipelining needs a radix-2 layout, so an NPOT Product load
// stays a plain load in the loop; the power-of-two control still pipelines.

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @product_load_not_pipelined
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // CHECK-NEXT: tt.store
  tt.func public @product_load_not_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pow2_load_pipelined
  // CHECK: ttg.local_alloc
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  tt.func public @pow2_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %b = tt.addptr %t, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %b, %v : tensor<128x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// A user latency on the Product load is dropped as well.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @product_load_user_latency
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // CHECK-NEXT: tt.store
  tt.func public @product_load_user_latency(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a {tt.latency = 2 : i32} : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}
