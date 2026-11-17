// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" 2>/dev/null | FileCheck %s
// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// A register-MMA GEMM (mma v2 tt.dot, no TMEM) whose K-loop accumulator feeds
// a reduction epilogue in a separate partition. Buffer allocation stores the
// K-loop result to SMEM with the task ids of the loop, which carries every
// partition, so several partitions would write the buffer the epilogue reads.
// createAllocChannel cannot express that; release builds used to keep one
// arbitrary producer and miscompile, debug builds asserted. Warp
// specialization must now be rejected before the descriptor loads are
// rewritten, leaving a plain kernel.

// Only the first function is rejected.
// WARN-COUNT-1: warning: warp specialization disabled: this value would be written to shared memory by more than one partition
// WARN-NOT: warning: warp specialization disabled
// CHECK-LABEL: tt.func public @gemm_reduction_epilogue_register_mma(
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: nvws.descriptor_load
// CHECK-NOT: async_task_id
// CHECK: tt.descriptor_load
// CHECK: tt.descriptor_load
// CHECK: tt.dot
// CHECK: tt.reduce
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK-NOT: ttg.warp_specialize
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#slice = #ttg.slice<{dim = 1, parent = #mma}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func public @gemm_reduction_epilogue_register_mma(
  %a_desc: !tt.tensordesc<32x32xbf16, #shared>,
  %b_desc: !tt.tensordesc<32x64xbf16, #shared>,
  %y_desc: !tt.tensordesc<32x64xf32, #shared1>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %zero = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c32_i32 : i32
    %acc = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_k = %zero) -> (tensor<32x64xf32, #mma>) : i32 {
      %offs_k = arith.muli %k, %c32_i32 : i32
      %a = tt.descriptor_load %a_desc[%offs, %offs_k] : !tt.tensordesc<32x32xbf16, #shared> -> tensor<32x32xbf16, #blocked>
      %b = tt.descriptor_load %b_desc[%offs_k, %c0_i32] : !tt.tensordesc<32x64xbf16, #shared> -> tensor<32x64xbf16, #blocked>
      %a_op = ttg.convert_layout %a : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %b_op = ttg.convert_layout %b : tensor<32x64xbf16, #blocked> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a_op, %b_op, %acc_k : tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      scf.yield %d : tensor<32x64xf32, #mma>
    }
    %sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<32x64xf32, #mma>) -> tensor<32xf32, #slice>
    %sum_e = tt.expand_dims %sum {axis = 1 : i32} : tensor<32xf32, #slice> -> tensor<32x1xf32, #mma>
    %sum_b = tt.broadcast %sum_e : tensor<32x1xf32, #mma> -> tensor<32x64xf32, #mma>
    %centered = arith.subf %acc, %sum_b : tensor<32x64xf32, #mma>
    %stage = ttg.local_alloc %centered : (tensor<32x64xf32, #mma>) -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<32x64xf32, #shared1>, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true}
  tt.return
}
}

// -----

// The same kernel with tt.merge_epilogue keeps the dot, its K loop and the
// epilogue in one partition, so no buffer has more than one producer and warp
// specialization must still apply.

// CHECK-LABEL: tt.func public @gemm_reduction_epilogue_register_mma_merged
// CHECK: ttg.warp_specialize

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#slice = #ttg.slice<{dim = 1, parent = #mma}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func public @gemm_reduction_epilogue_register_mma_merged(
  %a_desc: !tt.tensordesc<32x32xbf16, #shared>,
  %b_desc: !tt.tensordesc<32x64xbf16, #shared>,
  %y_desc: !tt.tensordesc<32x64xf32, #shared1>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %zero = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c32_i32 : i32
    %acc = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_k = %zero) -> (tensor<32x64xf32, #mma>) : i32 {
      %offs_k = arith.muli %k, %c32_i32 : i32
      %a = tt.descriptor_load %a_desc[%offs, %offs_k] : !tt.tensordesc<32x32xbf16, #shared> -> tensor<32x32xbf16, #blocked>
      %b = tt.descriptor_load %b_desc[%offs_k, %c0_i32] : !tt.tensordesc<32x64xbf16, #shared> -> tensor<32x64xbf16, #blocked>
      %a_op = ttg.convert_layout %a : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %b_op = ttg.convert_layout %b : tensor<32x64xbf16, #blocked> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a_op, %b_op, %acc_k : tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      scf.yield %d : tensor<32x64xf32, #mma>
    }
    %sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<32x64xf32, #mma>) -> tensor<32xf32, #slice>
    %sum_e = tt.expand_dims %sum {axis = 1 : i32} : tensor<32xf32, #slice> -> tensor<32x1xf32, #mma>
    %sum_b = tt.broadcast %sum_e : tensor<32x1xf32, #mma> -> tensor<32x64xf32, #mma>
    %centered = arith.subf %acc, %sum_b : tensor<32x64xf32, #mma>
    %stage = ttg.local_alloc %centered : (tensor<32x64xf32, #mma>) -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<32x64xf32, #shared1>, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true, tt.merge_epilogue = true}
  tt.return
}
}
