// Test file demonstrating named locations (loc("...")) in MLIR for channel graph visualization.
// Named locations in edge labels are extracted by getNamedLoc() in CodePartitionUtility.cpp.
// To test, this file needs to be integrated into a working warp specialization pipeline.
// Example named locations: loc("load_Q"), loc("alloc_Q_smem"), loc("compute_QK_mma"), etc.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem_mem = #ttng.tensor_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_named_loc_channels(%arg0: !tt.tensordesc<tensor<64x128xbf16>>, %arg1: !tt.tensordesc<tensor<128x128xbf16>>, %arg2: !tt.tensordesc<tensor<64x128xbf16>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %false = arith.constant false

    // Load Q matrix (producer partition 0)
    %q_loaded = tt.descriptor_load %arg0[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x128xbf16>> -> tensor<64x128xbf16, #blocked> loc("load_Q")
    %q_smem = ttg.local_alloc %q_loaded {async_task_id = array<i32: 0>} : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem> loc("alloc_Q_smem")

    // Load K matrix (producer partition 0)
    %k_loaded = tt.descriptor_load %arg1[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked> loc("load_K")
    %k_smem = ttg.local_alloc %k_loaded {async_task_id = array<i32: 0>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc("alloc_K_smem")
    %k_trans = ttg.memdesc_trans %k_smem {order = array<i32: 1, 0>, async_task_id = array<i32: 1>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared_trans, #smem> loc("transpose_K")

    // Allocate TMEM for QK result (will create OperandD channel)
    %qk_tmem, %qk_token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #tmem_mem, mutable>, !ttg.async.token) loc("alloc_QK_tmem")

    // Initialize QK with zeros
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %init_token = ttng.tmem_store %cst_zero, %qk_tmem[%qk_token], %true : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #tmem, #tmem_mem, mutable> loc("init_QK_zero")

    // Compute QK = Q @ K^T (consumer partition 1, producer of TMEM channel)
    %mma_token = ttng.tc_gen5_mma %q_smem, %k_trans, %qk_tmem[%init_token], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared_trans, #smem>, !ttg.memdesc<64x128xf32, #tmem, #tmem_mem, mutable> loc("compute_QK_mma")

    // Load QK result for softmax (consumer partition 2)
    %qk_result, %load_token = ttng.tmem_load %qk_tmem[%mma_token] {async_task_id = array<i32: 2>} : !ttg.memdesc<64x128xf32, #tmem, #tmem_mem, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> loc("load_QK_for_softmax")

    // Simple reduction (simulating softmax)
    %max_val = "tt.reduce"(%qk_result) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %max = arith.maxnumf %arg3, %arg4 : f32
      tt.reduce.return %max : f32
    }) {async_task_id = array<i32: 2>} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> loc("softmax_reduce_max")

    tt.return
  }
}
