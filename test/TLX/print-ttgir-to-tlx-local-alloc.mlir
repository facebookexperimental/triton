// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that a ttg.local_alloc's buffer count comes from whether it is indexed,
// not from its rank. TLX allocations always carry a leading slot dim, but
// Triton's pipeliner and TMA lowering do not: a ring of 1-D rows is rank 2,
// and a single tile loaded once is used directly with no memdesc_index.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // An unindexed tile is a single buffer: every use (here the TMA load that
  // fills it) sees its 2-D shape, through a view taken once at the allocation.
  // CHECK-LABEL: def single_tile(
  // CHECK: [[Q:[a-zA-Z_0-9]+]] = tlx.local_view(tlx.local_alloc((128, 128), tl.float16, 1), 0)
  // CHECK: tlx.async_descriptor_load({{[a-zA-Z_0-9]+}}, [[Q]], [{{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}], {{[a-zA-Z_0-9]+}})
  tt.func public @single_tile(%desc: !tt.tensordesc<128x128xf16, #shared>, %y: i32) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%y, %c0] %q, %bar, %true : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }

  // A pipelined ring of 1-D rows (softmax's row load) is indexed, so its
  // leading dim is the slot count even though the allocation is only rank 2.
  // CHECK-LABEL: def ring_of_rows(
  // CHECK: [[R:[a-zA-Z_0-9]+]] = tlx.local_alloc((4096,), tl.float32, 1)
  // CHECK: tlx.local_view([[R]], 0)
  tt.func public @ring_of_rows() -> tensor<4096xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %row = ttg.local_alloc : () -> !ttg.memdesc<1x4096xf32, #shared1, #smem, mutable>
    %slot = ttg.memdesc_index %row[%c0] : !ttg.memdesc<1x4096xf32, #shared1, #smem, mutable> -> !ttg.memdesc<4096xf32, #shared1, #smem, mutable>
    %v = ttg.local_load %slot : !ttg.memdesc<4096xf32, #shared1, #smem, mutable> -> tensor<4096xf32, #blocked>
    tt.return %v : tensor<4096xf32, #blocked>
  }

  // The base of a memdesc_reinterpret alias keeps the rank rule, so a 2-D one
  // is a bare one-slot allocation the alias can name in `reuse=`. An op that
  // uses that base directly as a tile still has to take slot 0, or it sees a
  // 1x128x64 buffer.
  // CHECK-LABEL: def aliased_tile_used_directly(
  // CHECK: [[B:[a-zA-Z_0-9]+]] = tlx.local_alloc((128, 64), tl.float16, 1)
  // CHECK: tlx.local_alloc((128, 64), tl.bfloat16, 1, reuse=[[B]])
  // CHECK: tlx.async_descriptor_load({{[a-zA-Z_0-9]+}}, tlx.local_view([[B]], 0), [{{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}], {{[a-zA-Z_0-9]+}})
  // CHECK: tlx.local_store(tlx.local_view([[B]], 0), {{[a-zA-Z_0-9]+}})
  // CHECK: tlx.async_descriptor_store({{[a-zA-Z_0-9]+}}, tlx.local_view([[B]], 0), [{{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}}])
  tt.func public @aliased_tile_used_directly(%desc: !tt.tensordesc<128x64xf16, #shared>, %v: tensor<128x64xf16, #blocked2>, %y: i32) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %alias = ttg.memdesc_reinterpret %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%y, %c0] %buf, %bar, %true : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.local_store %v, %buf : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%y, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }

  // A multi-slot ring of 2-D tiles keeps its count.
  // CHECK-LABEL: def ring_of_tiles(
  // CHECK: [[K:[a-zA-Z_0-9]+]] = tlx.local_alloc((128, 64), tl.float16, 3)
  // CHECK: tlx.local_view([[K]], {{[a-zA-Z_0-9]+}})
  tt.func public @ring_of_tiles(%i: i32) -> tensor<128x64xf16, #blocked2> {
    %k = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %k[%i] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %v = ttg.local_load %slot : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked2>
    tt.return %v : tensor<128x64xf16, #blocked2>
  }
}
