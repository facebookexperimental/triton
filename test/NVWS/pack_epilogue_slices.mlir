// RUN: triton-opt %s --nvws-pack-epilogue-slices | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @pack_two_slices
  tt.func @pack_two_slices(
      %desc: !tt.tensordesc<tensor<64x64xf32, #shared>>,
      %lb: i32, %ub: i32, %step: i32,
      %x0: tensor<64x64xf32, #blocked>,
      %x1: tensor<64x64xf32, #blocked>,
      %tmem_value: tensor<64x64xf32, #linear>) {
    %true = arith.constant true
    %tmem, %tmem_token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK:   scf.for
      scf.for %j = %lb to %ub step %step : i32 {
        scf.yield
      }
      // CHECK:   ttng.tmem_store
      %prerequisite = ttng.tmem_store %tmem_value, %tmem[%tmem_token], %true : tensor<64x64xf32, #linear> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %a0 = arith.addf %x0, %x0 : tensor<64x64xf32, #blocked>
      %a1 = arith.addf %x1, %x1 : tensor<64x64xf32, #blocked>
      %b0 = arith.mulf %a0, %x0 : tensor<64x64xf32, #blocked>
      %b1 = arith.mulf %a1, %x1 : tensor<64x64xf32, #blocked>
      // CHECK:   %[[A0:.*]] = arith.addf %arg4, %arg4
      // CHECK-NEXT:   %[[B0:.*]] = arith.mulf %[[A0]], %arg4
      // CHECK-NEXT:   ttg.local_store %[[B0]], %[[BUF0:.*]]
      // CHECK-NEXT:   %[[A1:.*]] = arith.addf %arg5, %arg5
      // CHECK-NEXT:   %[[B1:.*]] = arith.mulf %[[A1]], %arg5
      // CHECK-NEXT:   ttg.local_store %[[B1]], %[[BUF1:.*]]
      ttg.local_store %b0, %buf0 : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      ttg.local_store %b1, %buf1 : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf0 : !tt.tensordesc<tensor<64x64xf32, #shared>>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf1 : !tt.tensordesc<tensor<64x64xf32, #shared>>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 : !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 : !ttg.async.token
      scf.yield
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true,
       tt.warp_specialize}
    tt.return
  }
}
