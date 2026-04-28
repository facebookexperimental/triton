// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --verify-each=false --nvws-insert-semas --nvws-lower-semaphore --triton-nvidia-tma-lowering -cse | FileCheck %s --check-prefix=LOWER

// TMA lowering does not propagate partition attrs to its new helper ops, so
// this synthetic pre-partition fixture disables per-pass verification only
// for the cross-pass order check. The production pipeline partitions first.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @direct_descriptor_store_completion
  // SEMA: ttg.local_load
  // SEMA-NEXT: tt.descriptor_store
  // SEMA-NEXT: nvws.semaphore.release
  // LOWER-LABEL: sym_name = "direct_descriptor_store_completion"
  // LOWER: ttng.async_tma_copy_local_to_global
  // LOWER-NEXT: ttng.async_tma_store_wait
  // LOWER: ttng.arrive_barrier
  tt.func @direct_descriptor_store_completion(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 600 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      tt.descriptor_store %desc[%i, %i], %loaded {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // SEMA-LABEL: @converted_descriptor_store_completion
  // SEMA: ttg.local_load
  // SEMA-NEXT: ttg.convert_layout
  // SEMA-NEXT: tt.descriptor_store
  // SEMA-NEXT: nvws.semaphore.release
  tt.func @converted_descriptor_store_completion(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 601 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %first = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      ttg.local_store %first, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #linear>
      %converted = ttg.convert_layout %loaded {ttg.partition = array<i32: 1>} : tensor<128x64xf16, #linear> -> tensor<128x64xf16, #blocked>
      tt.descriptor_store %desc[%i, %i], %converted {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
      %next = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x64xf16, #linear>
      ttg.local_store %next, %alloc {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
