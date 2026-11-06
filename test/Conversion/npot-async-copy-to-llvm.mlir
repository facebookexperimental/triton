// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

// Each thread holds 8 contiguous f16 on dim 1, rows +32/+64, and an odd digit
// (columns +32/+64): 96 registers.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#s4 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_16b
  tt.func @mixed_radix_16b(%p: !tt.ptr<f16>, %c: i1) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #p>
    %m = tt.splat %c : i1 -> tensor<128x96xi1, #p>
    %d = ttg.local_alloc : () -> !ttg.memdesc<128x96xf16, #s8, #smem, mutable>
    // CHECK-COUNT-12: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, ${{[0-9]+}};
    // CHECK-NOT: cp.async.c
    %t = ttg.async_copy_global_to_local %a, %d mask %m {contiguity = 8 : i32} : tensor<128x96x!tt.ptr<f16>, #p> -> <128x96xf16, #s8, #smem, mutable>
    tt.return
  }

  // Row 1 starts 4 elements into a 4-wide swizzle chunk, so runs halve.
  // CHECK-LABEL: @mixed_radix_run_split
  tt.func @mixed_radix_run_split(%p: !tt.ptr<f16>) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #p>
    %d = ttg.local_alloc : () -> !ttg.memdesc<128x96xf16, #s4, #smem, mutable>
    // CHECK-COUNT-24: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8;
    // CHECK-NOT: cp.async.c
    %t = ttg.async_copy_global_to_local %a, %d {contiguity = 8 : i32} : tensor<128x96x!tt.ptr<f16>, #p> -> <128x96xf16, #s4, #smem, mutable>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pow2_control
  tt.func @pow2_control(%p: !tt.ptr<f16>) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #b>
    %d = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #s8, #smem, mutable>
    // CHECK-COUNT-16: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10;
    // CHECK-NOT: cp.async.c
    %t = ttg.async_copy_global_to_local %a, %d {contiguity = 8 : i32} : tensor<128x128x!tt.ptr<f16>, #b> -> <128x128xf16, #s8, #smem, mutable>
    tt.return
  }
}

// -----

#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_pow2_alloc_view(%p: !tt.ptr<f16>, %d: !ttg.memdesc<128x96xf16, #s8, #smem, mutable, 128x128>) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #p>
    // expected-error @+1 {{NPOT mixed-radix access to a padded, partitioned or partial buffer is not yet supported}}
    %t = ttg.async_copy_global_to_local %a, %d : tensor<128x96x!tt.ptr<f16>, #p> -> <128x96xf16, #s8, #smem, mutable, 128x128>
    tt.return
  }
}

// -----

// No target answers for a mixed-radix copy, so it stays illegal.
#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#s1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mixed_radix_no_target(%p: !tt.ptr<f16>, %d: !ttg.memdesc<192xf16, #s1, #smem, mutable>) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<192x!tt.ptr<f16>, #p>
    // expected-error @+2 {{NPOT async copy is not yet supported without a target}}
    // expected-error @+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %t = ttg.async_copy_global_to_local %a, %d : tensor<192x!tt.ptr<f16>, #p> -> <192xf16, #s1, #smem, mutable>
    tt.return
  }
}

// -----

// B's mixed-radix registers copy asynchronously in 16-byte runs.
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 96]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @async_copy_64x96
  // CHECK-COUNT-6: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10;
  // CHECK-NOT: cp.async.c
  tt.func @async_copy_64x96(%pb: !tt.ptr<f16>) {
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>, #linear>
    %d = ttg.local_alloc : () -> !ttg.memdesc<64x96xf16, #shared1, #smem, mutable>
    %t = ttg.async_copy_global_to_local %sb, %d {contiguity = 8 : i32} : tensor<64x96x!tt.ptr<f16>, #linear> -> <64x96xf16, #shared1, #smem, mutable>
    tt.return
  }
}
