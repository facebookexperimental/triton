// RUN: triton-opt %s --nvgpu-partition-scheduling-meta --verify-each=false | FileCheck %s

// Tests that on Hopper (cuda:90) with DATA_PARTITION_FACTOR=2 and
// WarpGroupDotOp, the partition scheduler correctly creates per-dpId
// computation partitions using the WarpGroupDotOp fallback (since
// WSDataPartition already split the dots, leaving no DataPartition-
// categorized ops in backward slices). Epilogue is merged into
// computation partitions so each MMA's truncf + TMA store lives
// alongside it.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: hopper_data_partitioned_gemm
//
// --- Inner k-loop: descriptor_loads and local_allocs → load partition ---
// CHECK: descriptor_load{{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: descriptor_load{{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: descriptor_load{{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: local_alloc{{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: local_alloc{{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: local_alloc{{.*}}ttg.partition = array<i32: [[LOAD]]>
//
// --- Inner k-loop: each warp_group_dot in its own computation partition ---
// CHECK: warp_group_dot{{.*}}ttg.partition = array<i32: [[COMP_A:[0-9]+]]>
// CHECK: warp_group_dot{{.*}}ttg.partition = array<i32: [[COMP_B:[0-9]+]]>
//
// --- Epilogue: each half's truncf + TMA store in same partition as its MMA ---
// CHECK: truncf{{.*}}ttg.partition = array<i32: [[COMP_A]]>
// CHECK: truncf{{.*}}ttg.partition = array<i32: [[COMP_B]]>
// CHECK: async_tma_copy_local_to_global{{.*}}ttg.partition = array<i32: [[COMP_A]]>
// CHECK: async_tma_copy_local_to_global{{.*}}ttg.partition = array<i32: [[COMP_B]]>
//
// --- Partition types: computation partitions before load ---
// CHECK: partition.types = ["computation", "computation", "load"
tt.func public @hopper_data_partitioned_gemm(
    %a_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
    %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %c_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>,
    %M: i32 {tt.divisibility = 16 : i32},
    %N: i32 {tt.divisibility = 16 : i32},
    %K: i32 {tt.divisibility = 16 : i32}
) {
  %c132_i32 = arith.constant 132 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c127_i32 = arith.constant 127 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>

  %start_pid = tt.get_program_id x : i32
  %num_pid_m = arith.addi %M, %c127_i32 : i32
  %num_pid_m_div = arith.divsi %num_pid_m, %c128_i32 : i32
  %num_pid_n = arith.addi %N, %c127_i32 : i32
  %num_pid_n_div = arith.divsi %num_pid_n, %c128_i32 : i32
  %k_tiles = arith.addi %K, %c64_i32 : i32
  %k_tiles_div = arith.divsi %k_tiles, %c64_i32 : i32
  %num_tiles = arith.muli %num_pid_m_div, %num_pid_n_div : i32
  %tile_id_c_init = arith.subi %start_pid, %c132_i32 : i32
  %num_pid_in_group = arith.muli %num_pid_n_div, %c8_i32 : i32

  %tile_id_c_out = scf.for %tile_id = %start_pid to %num_tiles step %c132_i32
      iter_args(%tile_id_c = %tile_id_c_init) -> (i32) : i32 {
    %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
    %first_pid_m = arith.muli %group_id, %c8_i32 : i32
    %group_size_m = arith.subi %num_pid_m_div, %first_pid_m : i32
    %group_size_m_clamped = arith.minsi %group_size_m, %c8_i32 : i32
    %pid_m = arith.remsi %tile_id, %group_size_m_clamped : i32
    %pid_m_final = arith.addi %first_pid_m, %pid_m : i32
    %pid_n_tmp = arith.remsi %tile_id, %num_pid_in_group : i32
    %pid_n = arith.divsi %pid_n_tmp, %group_size_m_clamped : i32
    %offs_am = arith.muli %pid_m_final, %c128_i32 : i32
    %offs_am_1 = arith.addi %offs_am, %c64_i32 : i32
    %offs_bn = arith.muli %pid_n, %c128_i32 : i32

    // Inner k-loop with two WarpGroupDotOps (data-partitioned)
    %acc:2 = scf.for %ki = %c0_i32 to %k_tiles_div step %c1_i32
        iter_args(%acc0 = %cst, %acc1 = %cst) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32

      %a0 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
      %a1 = tt.descriptor_load %a_desc[%offs_am_1, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>

      %a0_smem = ttg.local_alloc %a0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %a1_smem = ttg.local_alloc %a1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>

      %dot0 = ttng.warp_group_dot %a0_smem, %b_trans, %acc0 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<64x128xf32, #mma>
      %dot1 = ttng.warp_group_dot %a1_smem, %b_trans, %acc1 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<64x128xf32, #mma>

      scf.yield %dot0, %dot1 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}

    // Epilogue
    %tile_id_c_next = arith.addi %tile_id_c, %c132_i32 : i32
    %group_id_c = arith.divsi %tile_id_c_next, %num_pid_in_group : i32
    %first_pid_m_c = arith.muli %group_id_c, %c8_i32 : i32
    %group_size_m_c = arith.subi %num_pid_m_div, %first_pid_m_c : i32
    %group_size_m_c_clamped = arith.minsi %group_size_m_c, %c8_i32 : i32
    %pid_m_c = arith.remsi %tile_id_c_next, %group_size_m_c_clamped : i32
    %pid_m_c_final = arith.addi %first_pid_m_c, %pid_m_c : i32
    %pid_n_c_tmp = arith.remsi %tile_id_c_next, %num_pid_in_group : i32
    %pid_n_c = arith.divsi %pid_n_c_tmp, %group_size_m_c_clamped : i32
    %offs_am_c = arith.muli %pid_m_c_final, %c128_i32 : i32
    %offs_am_c_1 = arith.addi %offs_am_c, %c64_i32 : i32
    %offs_bn_c = arith.muli %pid_n_c, %c128_i32 : i32

    %c0_f16 = arith.truncf %acc#0 : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %c1_f16 = arith.truncf %acc#1 : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %c0_cvt = ttg.convert_layout %c0_f16 : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    %c1_cvt = ttg.convert_layout %c1_f16 : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    %c0_smem = ttg.local_alloc %c0_cvt : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %store_tok0 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c] %c0_smem : !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok0 : !ttg.async.token
    %c1_smem = ttg.local_alloc %c1_cvt : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %store_tok1 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c_1, %offs_bn_c] %c1_smem : !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok1 : !ttg.async.token

    scf.yield %tile_id_c_next : i32
  } {tt.data_partition_factor = 2 : i32, tt.smem_alloc_algo = 0 : i32, tt.warp_specialize}
  tt.return
}

} // module
