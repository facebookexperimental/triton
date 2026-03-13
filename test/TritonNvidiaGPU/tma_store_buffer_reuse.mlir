// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-store-buffer-reuse | FileCheck %s

// Test 1: basic_two_stores — Two sequential TMA stores with compatible types
// should be merged into a single mutable buffer.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @basic_two_stores
//       CHECK: %[[BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_store %arg2, %[[BUF]]
//       CHECK: ttng.fence_async_shared
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_store %arg3, %[[BUF]]
//       CHECK: ttng.fence_async_shared
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//  CHECK-NOT: ttg.local_alloc %
  tt.func @basic_two_stores(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 2: three_stores — Three sequential TMA stores should all share one buffer.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @three_stores
//       CHECK: %[[BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_store %arg2, %[[BUF]]
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_store %arg3, %[[BUF]]
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_store %arg4, %[[BUF]]
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg1, %arg1] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//  CHECK-NOT: ttg.local_alloc %
  tt.func @three_stores(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>,
      %src2: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %alloc2 = ttg.local_alloc %src2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 3: scatter_op — Two sequential TMA scatter ops should also be merged.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @scatter_op
//       CHECK: %[[BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>
//       CHECK: ttg.local_store %arg3, %[[BUF]]
//       CHECK: ttng.fence_async_shared
//       CHECK: ttng.async_tma_scatter %arg0[%arg1, %arg2] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_store %arg4, %[[BUF]]
//       CHECK: ttng.fence_async_shared
//       CHECK: ttng.async_tma_scatter %arg0[%arg1, %arg2] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//  CHECK-NOT: ttg.local_alloc %
  tt.func @scatter_op(
      %desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>,
      %x_offsets: tensor<32xi32, #blocked>,
      %y_offset: i32,
      %src0: tensor<32x128xbf16, #blocked1>,
      %src1: tensor<32x128xbf16, #blocked1>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<32x128xbf16, #blocked1>) -> !ttg.memdesc<32x128xbf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %alloc0 : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %alloc1 = ttg.local_alloc %src1 : (tensor<32x128xbf16, #blocked1>) -> !ttg.memdesc<32x128xbf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %alloc1 : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 4: interleaved_incompatible — A (128x64xf16), B (64x128xf32), C (128x64xf16).
// A and C have compatible types and should merge; B keeps its original alloc.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @interleaved_incompatible
//       CHECK: %[[BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_store %arg3, %[[BUF]]
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg2, %arg2] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_alloc %arg4 : (tensor<64x128xf32, #blocked>) -> !ttg.memdesc<64x128xf32, #shared1, #smem>
//       CHECK: ttng.async_tma_copy_local_to_global %arg1[%arg2, %arg2]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
//       CHECK: ttg.local_store %arg5, %[[BUF]]
//       CHECK: ttng.async_tma_copy_local_to_global %arg0[%arg2, %arg2] %[[BUF]]
//       CHECK: ttng.async_tma_store_wait {pendings = 0 : i32}
  tt.func @interleaved_incompatible(
      %desc_f16: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %desc_f32: !tt.tensordesc<tensor<64x128xf32, #shared1>>,
      %x: i32,
      %srcA: tensor<128x64xf16, #blocked>,
      %srcB: tensor<64x128xf32, #blocked>,
      %srcC: tensor<128x64xf16, #blocked>) {
    // Store A (f16)
    %allocA = ttg.local_alloc %srcA : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc_f16[%x, %x] %allocA : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // Store B (f32) — incompatible type
    %allocB = ttg.local_alloc %srcB : (tensor<64x128xf32, #blocked>) -> !ttg.memdesc<64x128xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc_f32[%x, %x] %allocB : !tt.tensordesc<tensor<64x128xf32, #shared1>>, !ttg.memdesc<64x128xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // Store C (f16) — compatible with A
    %allocC = ttg.local_alloc %srcC : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc_f16[%x, %x] %allocC : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 5: incompatible_types — Two stores with different buffer shapes/element types.
// Each type group has only one candidate, so no transformation occurs.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @incompatible_types
//       CHECK: ttg.local_alloc %arg3 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//       CHECK: ttg.local_alloc %arg4 : (tensor<64x128xf32, #blocked>) -> !ttg.memdesc<64x128xf32, #shared1, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @incompatible_types(
      %desc_f16: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %desc_f32: !tt.tensordesc<tensor<64x128xf32, #shared1>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<64x128xf32, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc_f16[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %alloc1 = ttg.local_alloc %src1 : (tensor<64x128xf32, #blocked>) -> !ttg.memdesc<64x128xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc_f32[%x, %x] %alloc1 : !tt.tensordesc<tensor<64x128xf32, #shared1>>, !ttg.memdesc<64x128xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 6: no_wait — Two stores with no tma_store_wait at all.
// No transformation (findDonePoint returns nullptr for both).

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @no_wait
//       CHECK: ttg.local_alloc %arg2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//       CHECK: ttg.local_alloc %arg3 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @no_wait(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    tt.return
  }
}

// -----

// Test 7: nonzero_pendings — Two stores separated by tma_store_wait {pendings = 1}.
// No transformation (only pendings = 0 qualifies as a done point).

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @nonzero_pendings
//       CHECK: ttg.local_alloc %arg2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//       CHECK: ttg.local_alloc %arg3 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @nonzero_pendings(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 1 : i32}
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    tt.return
  }
}

// -----

// Test 8: single_store — Only one TMA store in the block.
// No transformation (need >= 2 candidates).

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @single_store
//       CHECK: ttg.local_alloc %arg2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @single_store(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

// Test 9: multiple_users — First alloc is used by both a TMA copy and a local_load.
// The alloc has two users so it is not a candidate. Only one valid candidate remains.
// No transformation.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @multiple_users
//       CHECK: ttg.local_alloc %arg2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//       CHECK: ttg.local_load
//       CHECK: ttg.local_alloc %arg3 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @multiple_users(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #blocked> {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %loaded = ttg.local_load %alloc0 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return %loaded : tensor<128x64xf16, #blocked>
  }
}

// -----

// Test 10: ordering_violation — Two compatible stores where the second alloc
// appears before the first's tma_store_wait. The chain condition
// (prev.donePoint < curr.alloc) fails, so no transformation.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @ordering_violation
//       CHECK: ttg.local_alloc %arg2 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//       CHECK: ttg.local_alloc %arg3 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
//   CHECK-NOT: ttg.local_store
  tt.func @ordering_violation(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %x: i32,
      %src0: tensor<128x64xf16, #blocked>,
      %src1: tensor<128x64xf16, #blocked>) {
    %alloc0 = ttg.local_alloc %src0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.fence_async_shared {bCluster = false}
    // Second alloc appears BEFORE first's store_wait
    %alloc1 = ttg.local_alloc %src1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
