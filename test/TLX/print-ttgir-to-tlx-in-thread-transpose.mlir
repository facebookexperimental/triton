// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that amdg.in_thread_transpose is erased rather than printed.
//
// The AMD backend inserts it on the gfx942 register-staged path (global -> VGPR
// -> LDS) between the load and the local_store. It is a CDNA variant of
// ttg.convert_layout, so recompiling the generated TLX re-creates it; printing
// it instead emits a name no TLX module defines. The local_store must therefore
// name the load directly.

#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#shared = #ttg.amd_rotating_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: def in_thread_transpose_is_erased(
  // CHECK-NOT: in_thread_transpose
  // CHECK: [[B:[a-zA-Z_0-9]+]] = tl.load(
  // CHECK: tlx.local_store({{[a-zA-Z_0-9]+}}, [[B]])
  tt.func public @in_thread_transpose_is_erased(%b_ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) attributes {noinline = false} {
    %b = tt.load %b_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
    %t = amdg.in_thread_transpose %b : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #linear>
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %t, %view : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}
