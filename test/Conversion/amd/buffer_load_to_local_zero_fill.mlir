// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --implicit-check-not=llvm.cond_br --implicit-check-not=llvm.store
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s --implicit-check-not=llvm.cond_br --implicit-check-not=llvm.store

// Explicit other=0 must use the hardware buffer OOB zero-fill path, not a
// conditional asynchronous load followed by an LDS zero store. The latter
// fragments the grouped-GEMM pipeline and introduces register spills.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @buffer_load_to_local_zero_fill
  tt.func public @buffer_load_to_local_zero_fill(
      %src: !tt.ptr<f32>, %limit: i32,
      %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
    %limits = tt.splat %limit : i32 -> tensor<256xi32, #blocked>
    %mask = arith.cmpi slt, %offset, %limits : tensor<256xi32, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    // CHECK: %[[OFFSET:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.load.async.lds {{.*}}%[[OFFSET]]
    amdg.buffer_load_to_local %src[%offset] mask=%mask other=%zero into %dst : <f32>[tensor<256xi32, #blocked>] tensor<256xf32, #blocked> -> <256xf32, #shared, #smem, mutable>
    // CHECK: llvm.return
    tt.return
  }
}
