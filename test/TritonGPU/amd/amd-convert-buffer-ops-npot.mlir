// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950" | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritonamdgpu-canonicalize-pointers | FileCheck %s --check-prefix=CANON
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950" --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=LLVM

// An exact mixed-radix load/store becomes a buffer load/store in its layout, as
// a power-of-two one does. A mixed-radix atomic stays global, as buffer atomics
// reject its layout.
#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#b = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @mixed_radix
  // CHECK: amdg.buffer_load {{.*}} -> tensor<192xf32, #linear>
  // CHECK: amdg.buffer_store {{.*}} -> tensor<192xf32, #linear>
  // CHECK: tt.atomic_rmw {{.*}} -> tensor<192xi32, #linear>
  // CHECK: tt.atomic_cas {{.*}} -> tensor<192xi32, #linear>
  // LLVM-LABEL: llvm.func @mixed_radix
  // LLVM-COUNT-3: llvm.atomicrmw add {{.*}} : !llvm.ptr<1>, i32
  // LLVM-COUNT-3: llvm.cmpxchg {{.*}} : !llvm.ptr<1>, i32
  // CANON-LABEL: tt.func @mixed_radix
  // CANON: tt.load {{.*}} : tensor<192x!tt.ptr<f32>, #linear>
  tt.func @mixed_radix(%x: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %y: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %z: !tt.ptr<i32> {tt.pointer_range = 32 : i32}) {
    %pid = tt.get_program_id x : i32
    %c192 = arith.constant 192 : i32
    %base = arith.muli %pid, %c192 : i32
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p>
    %s = tt.splat %base : i32 -> tensor<192xi32, #p>
    %o = arith.addi %s, %r : tensor<192xi32, #p>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p>
    %xp = tt.addptr %xs, %o : tensor<192x!tt.ptr<f32>, #p>, tensor<192xi32, #p>
    %v = tt.load %xp : tensor<192x!tt.ptr<f32>, #p>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p>
    %yp = tt.addptr %ys, %o : tensor<192x!tt.ptr<f32>, #p>, tensor<192xi32, #p>
    tt.store %yp, %v : tensor<192x!tt.ptr<f32>, #p>
    %zs = tt.splat %z : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #p>
    %zp = tt.addptr %zs, %o : tensor<192x!tt.ptr<i32>, #p>, tensor<192xi32, #p>
    %a = tt.atomic_rmw add, relaxed, gpu, %zp, %o : (tensor<192x!tt.ptr<i32>, #p>, tensor<192xi32, #p>) -> tensor<192xi32, #p>
    %c = tt.atomic_cas relaxed, gpu, %zp, %o, %o : (tensor<192x!tt.ptr<i32>, #p>, tensor<192xi32, #p>, tensor<192xi32, #p>) -> tensor<192xi32, #p>
    tt.return
  }

  // CHECK-LABEL: tt.func @pow2
  // CHECK: amdg.buffer_load {{.*}} -> tensor<256xf32, #blocked>
  // CHECK: amdg.buffer_store {{.*}} -> tensor<256xf32, #blocked>
  // CHECK: amdg.buffer_atomic_rmw
  // CHECK: amdg.buffer_atomic_cas
  tt.func @pow2(%x: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %y: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %z: !tt.ptr<i32> {tt.pointer_range = 32 : i32}) {
    %pid = tt.get_program_id x : i32
    %c256 = arith.constant 256 : i32
    %base = arith.muli %pid, %c256 : i32
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #b>
    %s = tt.splat %base : i32 -> tensor<256xi32, #b>
    %o = arith.addi %s, %r : tensor<256xi32, #b>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #b>
    %xp = tt.addptr %xs, %o : tensor<256x!tt.ptr<f32>, #b>, tensor<256xi32, #b>
    %v = tt.load %xp : tensor<256x!tt.ptr<f32>, #b>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #b>
    %yp = tt.addptr %ys, %o : tensor<256x!tt.ptr<f32>, #b>, tensor<256xi32, #b>
    tt.store %yp, %v : tensor<256x!tt.ptr<f32>, #b>
    %zs = tt.splat %z : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #b>
    %zp = tt.addptr %zs, %o : tensor<256x!tt.ptr<i32>, #b>, tensor<256xi32, #b>
    %a = tt.atomic_rmw add, relaxed, gpu, %zp, %o : (tensor<256x!tt.ptr<i32>, #b>, tensor<256xi32, #b>) -> tensor<256xi32, #b>
    %c = tt.atomic_cas relaxed, gpu, %zp, %o, %o : (tensor<256x!tt.ptr<i32>, #b>, tensor<256xi32, #b>, tensor<256xi32, #b>) -> tensor<256xi32, #b>
    tt.return
  }
}
