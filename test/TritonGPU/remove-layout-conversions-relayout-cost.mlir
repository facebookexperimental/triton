// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// A single-candidate neighbour is not automatically a fixed one. `%res0`/`%res1`
// reach `arith.extf` with one encoding only because that is the one propagation
// carried to them, not because extf needs it -- extf has no layout preference at
// all and will follow whatever the chain settles on. Charging a relayout against
// it walls the chain off and drags it down to the neighbour's incidental layout.
//
// Both choices move the same 8192 bytes here: staying in #linear converts the two
// bf16 operands (2 x 4096), dropping to #blocked converts the f32 tmem_load result
// (1 x 8192). Equal cost, so the vectorization score decides, and #linear is twice
// as wide. Note the convert *count* goes up while the answer gets better -- count
// is not the objective.

// CHECK-LABEL: @transparent_neighbour_is_not_a_wall
// CHECK:         %[[T:.*]] = ttng.tmem_load
// CHECK-NOT:     ttg.convert_layout %[[T]]
// CHECK:         arith.mulf {{.*}} tensor<64x32xf32, #linear>
// CHECK:         arith.addf {{.*}} tensor<64x32xf32, #linear>
// CHECK:         arith.truncf {{.*}} tensor<64x32xf32, #linear> to tensor<64x32xbf16, #linear>
// CHECK:         ttg.local_store

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @transparent_neighbour_is_not_a_wall(
      %acc: !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x128>,
      %res0: tensor<64x32xbf16, #blocked>,
      %res1: tensor<64x32xbf16, #blocked>,
      %out: !ttg.memdesc<64x32xbf16, #shared, #smem, mutable>) {
    %t = ttng.tmem_load %acc : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x128> -> tensor<64x32xf32, #linear>
    %tb = ttg.convert_layout %t : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked>
    %e0 = arith.extf %res0 : tensor<64x32xbf16, #blocked> to tensor<64x32xf32, #blocked>
    %e1 = arith.extf %res1 : tensor<64x32xbf16, #blocked> to tensor<64x32xf32, #blocked>
    %a1 = arith.mulf %tb, %e0 : tensor<64x32xf32, #blocked>
    %a2 = arith.addf %a1, %e1 : tensor<64x32xf32, #blocked>
    %r = arith.truncf %a2 : tensor<64x32xf32, #blocked> to tensor<64x32xbf16, #blocked>
    ttg.local_store %r, %out : tensor<64x32xbf16, #blocked> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}
