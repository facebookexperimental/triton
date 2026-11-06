// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck --enable-var-scope %s

// A mixed-radix load pipelines through shared memory when its async copy and
// local_load lower on the target; otherwise it stays a plain load.

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_pipelined
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x96xf32
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load {{.*}} -> tensor<96xf32, #linear>
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %step = arith.constant dense<96> : tensor<96xi32, #p96>
    %x = scf.for %i = %c0 to %n step %c1 iter_args(%q0 = %a) -> (tensor<96x!tt.ptr<f32>, #p96>) : i32 {
      %v = tt.load %q0 : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
      %q1 = tt.addptr %q0, %step : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
      scf.yield %q1 : tensor<96x!tt.ptr<f32>, #p96>
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

// hip can't copy a mixed-radix layout asynchronously, so even a user latency on
// the load is dropped.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_user_latency
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_user_latency(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
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

// -----

// sm80 reads the stage back through a mixed-radix local_load, which takes the
// convert's layout.
#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_convert_sm80
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load {{.*}} -> tensor<192xf32, #linear1>
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_convert_sm80(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #src>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<f32>, #src>, tensor<192xi32, #src>
    %rd = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #dst>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #dst>
    %b = tt.addptr %t, %rd : tensor<192x!tt.ptr<f32>, #dst>, tensor<192xi32, #dst>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<192x!tt.ptr<f32>, #src>
      %w = ttg.convert_layout %v : tensor<192xf32, #src> -> tensor<192xf32, #dst>
      tt.store %b, %w : tensor<192x!tt.ptr<f32>, #dst>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// A non-zero `other` is selected after the mixed-radix local_load.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_other
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load
  // CHECK-NEXT: arith.select
  tt.func public @mixed_radix_load_other(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32, %m: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<96xf32, #p96>
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %ms = tt.splat %m : i32 -> tensor<96xi32, #p96>
    %k = arith.cmpi slt, %r, %ms : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a, %k, %one : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}
