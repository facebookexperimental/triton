// RUN: triton-opt %s --nvgpu-test-add-subtile-regions --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test that mismatched async_task_id between setup ops (task 0) and
  // per-tile ops (task 1) causes an error.
  tt.func @mismatched_async_partitions(
      %acc_memdesc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token) {

    %acc, %token2 = ttng.tmem_load %acc_memdesc[%token] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Split: 128x128 -> 2 x 128x64 (setup ops — task 0)
    %reshaped = tt.reshape %acc {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>, async_task_id = array<i32: 0>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    // expected-error @+1 {{ops in subtile region have inconsistent async_task_id partitions}}
    %outLHS, %outRHS = tt.split %transposed {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>

    // Per-subtile ops — all consistently task 1 (mismatched with setup task 0)
    %c0 = arith.truncf %outLHS {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked5> to tensor<128x64xf16, #blocked5>
    %c0_cvt = ttg.convert_layout %c0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked5> -> tensor<128x64xf16, #blocked9>
    %c0_alloc = ttg.local_alloc %c0_cvt {async_task_id = array<i32: 1>} : (tensor<128x64xf16, #blocked9>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>

    %c1 = arith.truncf %outRHS {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked5> to tensor<128x64xf16, #blocked5>
    %c1_cvt = ttg.convert_layout %c1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked5> -> tensor<128x64xf16, #blocked9>
    %c1_alloc = ttg.local_alloc %c1_cvt {async_task_id = array<i32: 1>} : (tensor<128x64xf16, #blocked9>) -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>

    tt.return
  }
}
