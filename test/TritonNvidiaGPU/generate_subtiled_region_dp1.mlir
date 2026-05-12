// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-test-generate-subtiled-region | FileCheck %s

// Test: DP=1 epilogue subtiling with convert_layout in chain.
// The split feeds into truncf → convert_layout → local_store for each tile.
// The subtile operator should create a SubtiledRegionOp.

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @dp1_epilogue_subtile
  // CHECK: ttng.subtiled_region
  tt.func @dp1_epilogue_subtile(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %smem0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %smem1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %off0: i32, %off1: i32, %off2: i32) {

    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] {async_task_id = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %reshaped = tt.reshape %loaded#0 {async_task_id = array<i32: 1>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
    %transposed = tt.trans %reshaped {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
    %lhs, %rhs = tt.split %transposed {async_task_id = array<i32: 1>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>

    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
    %cvt0 = ttg.convert_layout %trunc0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked1>
    ttg.local_store %cvt0, %smem0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
    %cvt1 = ttg.convert_layout %trunc1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked1>
    ttg.local_store %cvt1, %smem1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    tt.return
  }
}

// -----

// Test: DP=1 epilogue subtiling inside scf.for loop body.
// This is the pattern from real persistent matmul kernels where the split
// lives inside the outer tile loop.

#blocked1b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linearb = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1b = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2b = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3b = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#sharedb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smemb = #ttg.shared_memory
#tmemb = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @dp1_epilogue_subtile_in_loop
  // CHECK: scf.for
  // CHECK:   ttng.subtiled_region
  tt.func @dp1_epilogue_subtile_in_loop(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmemb, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #sharedb>>,
      %smem0: !ttg.memdesc<128x64xf16, #sharedb, #smemb, mutable>,
      %smem1: !ttg.memdesc<128x64xf16, #sharedb, #smemb, mutable>,
      %off0: i32, %off1: i32, %off2: i32,
      %lb: i32, %ub: i32, %step: i32) {

    scf.for %iv = %lb to %ub step %step  : i32 {
      %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] {async_task_id = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmemb, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linearb>
      %reshaped = tt.reshape %loaded#0 {async_task_id = array<i32: 1>} : tensor<128x128xf32, #linearb> -> tensor<128x2x64xf32, #linear1b>
      %transposed = tt.trans %reshaped {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1b> -> tensor<128x64x2xf32, #linear2b>
      %lhs, %rhs = tt.split %transposed {async_task_id = array<i32: 1>} : tensor<128x64x2xf32, #linear2b> -> tensor<128x64xf32, #linear3b>

      %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #linear3b> to tensor<128x64xf16, #linear3b>
      %cvt0 = ttg.convert_layout %trunc0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #linear3b> -> tensor<128x64xf16, #blocked1b>
      ttg.local_store %cvt0, %smem0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked1b> -> !ttg.memdesc<128x64xf16, #sharedb, #smemb, mutable>

      %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #linear3b> to tensor<128x64xf16, #linear3b>
      %cvt1 = ttg.convert_layout %trunc1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #linear3b> -> tensor<128x64xf16, #blocked1b>
      ttg.local_store %cvt1, %smem1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked1b> -> !ttg.memdesc<128x64xf16, #sharedb, #smemb, mutable>
    }

    tt.return
  }
}
