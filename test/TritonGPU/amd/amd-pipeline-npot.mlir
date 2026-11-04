// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritonamdgpu-schedule-loops=num_stages=2 -tritonamdgpu-pipeline -canonicalize | FileCheck --enable-var-scope %s

// The stream pipeliner shares the load filter: an NPOT Product load is not
// scheduled, while the power-of-two control is.

#p192 = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @product_load_not_pipelined
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // CHECK-NEXT: tt.store
  tt.func public @product_load_not_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p192>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p192>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<f32>, #p192>, tensor<192xi32, #p192>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p192>
    %b = tt.addptr %t, %r : tensor<192x!tt.ptr<f32>, #p192>, tensor<192xi32, #p192>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<192x!tt.ptr<f32>, #p192>
      tt.store %b, %v : tensor<192x!tt.ptr<f32>, #p192>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @pow2_load_pipelined
  // CHECK: tt.load
  // CHECK: scf.for {{.*}} iter_args
  tt.func public @pow2_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %a = tt.addptr %s, %r : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %b = tt.addptr %t, %r : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<256x!tt.ptr<f32>, #blocked>
      tt.store %b, %v : tensor<256x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}
