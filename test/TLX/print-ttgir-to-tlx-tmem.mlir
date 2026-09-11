// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that tensor-memory allocations and subslices round-trip.
//
// local_alloc takes one buffer's shape plus a count. A buffer is up to 2-D, so only
// a rank above that is multi-buffering: a rank-2 TMEM memdesc is one 2-D tile, not a
// count and a 1-D tile.

#tmem_wide = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_tile = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_half = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // A rank-2 TMEM memdesc is one tile: both dimensions belong to the shape and
  // the count is 1.
  // CHECK-LABEL: def tmem_rank2(
  // CHECK: tlx.local_alloc((128, 256), tl.int32, 1, tlx.storage_kind.tmem)
  tt.func public @tmem_rank2() attributes {noinline = false} {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xi32, #tmem_wide, #ttng.tensor_memory, mutable>
    tt.return
  }

  // A rank-3 TMEM memdesc keeps the existing split: the leading dimension is the
  // buffer count and the trailing two are the tile.
  // CHECK-LABEL: def tmem_rank3(
  // CHECK: tlx.local_alloc((128, 128), tl.float32, 2, tlx.storage_kind.tmem)
  tt.func public @tmem_rank3() attributes {noinline = false} {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable>
    tt.return
  }

  // The storage-alias path (memdesc_reinterpret) applies the same rule, and is
  // where the rank-2 base allocation of an aliased buffer shows up.
  // CHECK-LABEL: def tmem_alias(
  // CHECK: [[BASE:[a-zA-Z_0-9]+]] = tlx.local_alloc((128, 256), tl.int32, 1, tlx.storage_kind.tmem)
  // CHECK: tlx.local_alloc((128, 128), tl.float32, 2, tlx.storage_kind.tmem, reuse=[[BASE]])
  tt.func public @tmem_alias() attributes {noinline = false} {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xi32, #tmem_wide, #ttng.tensor_memory, mutable>
    %1 = ttg.memdesc_reinterpret %0 : !ttg.memdesc<128x256xi32, #tmem_wide, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable>
    tt.return
  }

  // memdesc_reinterpret is legal on shared memory too. The alias must use the
  // same buffer-count rule as the base allocation, or the two describe different
  // buffer counts over the same storage, and the storage kind is the default.
  // CHECK-LABEL: def smem_alias(
  // CHECK: [[SBASE:[a-zA-Z_0-9]+]] = tlx.local_alloc((2, 256), tl.int32, 1)
  // CHECK: tlx.local_alloc((2, 128), tl.float32, 1, reuse=[[SBASE]])
  // CHECK-NOT: storage_kind.tmem
  tt.func public @smem_alias() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x256xi32, #shared, #smem, mutable>
    %1 = ttg.memdesc_reinterpret %0 : !ttg.memdesc<2x256xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: def subslice_low(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.subslice({{[a-zA-Z_0-9]+}}, 0, 64)
  tt.func public @subslice_low() attributes {noinline = false} {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_subslice %0 {offset = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem_half, #ttng.tensor_memory, mutable>
    tt.return
  }

  // The offset is what distinguishes the two halves; dropping it aliased them.
  // CHECK-LABEL: def subslice_high(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.subslice({{[a-zA-Z_0-9]+}}, 64, 64)
  tt.func public @subslice_high() attributes {noinline = false} {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_subslice %0 {offset = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem_tile, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem_half, #ttng.tensor_memory, mutable>
    tt.return
  }
}
