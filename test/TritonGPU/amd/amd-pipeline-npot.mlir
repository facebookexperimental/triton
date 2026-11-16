// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritonamdgpu-schedule-loops=num_stages=2 -tritonamdgpu-pipeline -canonicalize | FileCheck --enable-var-scope %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritonamdgpu-schedule-loops=num_stages=2 -tritonamdgpu-pipeline=use_async_copy=1 -canonicalize | FileCheck --enable-var-scope %s --check-prefix=ASYNC

// The stream pipeliner shares the load filter, so an NPOT mixed-radix load is
// scheduled like the power-of-two control.

#p192 = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_radix_load_pipelined
  // CHECK: tt.load
  // CHECK: scf.for {{.*}} iter_args
  tt.func public @mixed_radix_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
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

// -----

// An NPOT stage in shared memory is filled by async copy. Without it the load
// stays in registers, since local_store can't take the layout.
#p192 = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_radix_load_lds
  // CHECK-NOT: ttg.local_store
  // CHECK: scf.for {{.*}} iter_args({{.*}}tensor<192xf32
  // CHECK-NOT: ttg.local_store
  // CHECK: tt.return
  // ASYNC-LABEL: @mixed_radix_load_lds
  // ASYNC: ttg.async_copy_global_to_local
  // ASYNC: scf.for
  // ASYNC: ttg.async_wait
  // ASYNC: ttg.async_copy_global_to_local
  // ASYNC: ttg.local_load
  tt.func public @mixed_radix_load_lds(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p192>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p192>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<f32>, #p192>, tensor<192xi32, #p192>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p192>
    %b = tt.addptr %t, %r : tensor<192x!tt.ptr<f32>, #p192>, tensor<192xi32, #p192>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<192x!tt.ptr<f32>, #p192>
      %m = ttg.local_alloc %v : (tensor<192xf32, #p192>) -> !ttg.memdesc<192xf32, #s, #smem>
      %w = ttg.local_load %m : !ttg.memdesc<192xf32, #s, #smem> -> tensor<192xf32, #p192>
      tt.store %b, %w : tensor<192x!tt.ptr<f32>, #p192>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}
