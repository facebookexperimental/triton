// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s --check-prefix=WAIT

// A mixed-radix copy lowers as one radix-2 copy per odd register value, each
// starting at its slab of the shared layout.
#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#s1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_radix_hip
  // CHECK-COUNT-3: rocdl.global.load.async.lds
  // CHECK-NOT: rocdl.global.load.async.lds
  tt.func @mixed_radix_hip(%p: !tt.ptr<f32>, %d: !ttg.memdesc<192xf32, #s1, #smem, mutable>) {
    %a = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #p>
    %t = ttg.async_copy_global_to_local %a, %d : tensor<192x!tt.ptr<f32>, #p> -> <192xf32, #s1, #smem, mutable>
    tt.return
  }
}

// -----

// K96 bf16: three 32-row slabs, two 16-byte loads per slab.
#k96 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], warp = [[8, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [96, 64]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @k96_buffer_load
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK: %[[S1:.*]] = llvm.mlir.constant(4096 : i32) : i32
  // CHECK-NEXT: llvm.getelementptr %{{.*}}[%[[S1]]]
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK: %[[S2:.*]] = llvm.mlir.constant(8192 : i32) : i32
  // CHECK-NEXT: llvm.getelementptr %{{.*}}[%[[S2]]]
  // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK-NOT: rocdl.raw.ptr.buffer.load.async.lds
  // WAIT-LABEL: @k96_buffer_load
  // WAIT: amdg.async_wait {{.*}} {num_inst = 6 : i32
  tt.func @k96_buffer_load(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %d0: !ttg.memdesc<96x64xbf16, #s, #smem, mutable>,
      %d1: !ttg.memdesc<96x64xbf16, #s, #smem, mutable>) {
    %offsets = arith.constant dense<0> : tensor<96x64xi32, #k96>
    %0 = amdg.buffer_load_to_local %ptr[%offsets] into %d0 {contiguity = 8 : i32} : !tt.ptr<bf16>[tensor<96x64xi32, #k96>] -> <96x64xbf16, #s, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = amdg.buffer_load_to_local %ptr[%offsets] into %d1 {contiguity = 8 : i32} : !tt.ptr<bf16>[tensor<96x64xi32, #k96>] -> <96x64xbf16, #s, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_wait %1 {num = 1 : i32}
    tt.return
  }
}

// -----

// K-contiguous K96 bf16: the dense tile stores three 64x32 panels, so each
// slab starts a panel and its lanes write contiguous LDS.
#k96c = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [64, 96]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @k96_panels
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK: %[[P1:.*]] = llvm.mlir.constant(4096 : i32) : i32
  // CHECK-NEXT: llvm.getelementptr %{{.*}}[%[[P1]]]
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK: %[[P2:.*]] = llvm.mlir.constant(8192 : i32) : i32
  // CHECK-NEXT: llvm.getelementptr %{{.*}}[%[[P2]]]
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK-NOT: rocdl.raw.ptr.buffer.load.async.lds
  tt.func @k96_panels(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %d: !ttg.memdesc<64x96xbf16, #s, #smem, mutable>) {
    %offsets = arith.constant dense<0> : tensor<64x96xi32, #k96c>
    %0 = amdg.buffer_load_to_local %ptr[%offsets] into %d {contiguity = 8 : i32} : !tt.ptr<bf16>[tensor<64x96xi32, #k96c>] -> <64x96xbf16, #s, #smem, mutable>
    tt.return
  }
}

// -----

// One load writes M0 + lane * size, so a slab must start 16-byte aligned.
#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#pad = #ttg.padded_shared<[64:+1] {order = [0], shape = [256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @misaligned_slab(%ptr: !tt.ptr<f32>, %d: !ttg.memdesc<192xf32, #pad, #smem, mutable, 256>) {
    %offsets = arith.constant dense<0> : tensor<192xi32, #p>
    // expected-error @+1 {{NPOT async copy is not supported on hip:gfx950: a slab starts at LDS byte 260, which is not 16-byte aligned}}
    %t = amdg.buffer_load_to_local %ptr[%offsets] into %d : !tt.ptr<f32>[tensor<192xi32, #p>] -> <192xf32, #pad, #smem, mutable, 256>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#s1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @pow2_control
  // CHECK-COUNT-4: rocdl.global.load.async.lds
  // CHECK-NOT: rocdl.global.load.async.lds
  tt.func @pow2_control(%p: !tt.ptr<f32>, %d: !ttg.memdesc<256xf32, #s1, #smem, mutable>) {
    %a = tt.splat %p : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %t = ttg.async_copy_global_to_local %a, %d : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #s1, #smem, mutable>
    tt.return
  }
}
