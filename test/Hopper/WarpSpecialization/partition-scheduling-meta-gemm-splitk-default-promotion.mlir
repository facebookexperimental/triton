// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// Tests that partition scheduling promotes the epilogue partition (which
// contains tmem_load, requiring 4 warps) to index 0 so it becomes the
// default warp group in the final warp_specialize lowering.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @persistent_splitk_gemm_default_promotion
//
// Epilogue partition (tmem_load + truncf + descriptor_store) should be
// promoted to index 0 because tmem_load requires 4 warps.
//
// --- In-loop: loads → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- In-loop: memdesc_trans and MMA → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
//
// --- Epilogue: tmem_load, truncf, descriptor_store → epilogue partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[EPIL:[0-9]+]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[EPIL]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[EPIL]]>
//
// --- Partition types: epilogue is first (index 0 = default warp group) ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["epilogue", "gemm", "load"
tt.func public @persistent_splitk_gemm_default_promotion(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %ws_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
  %M: i32 {tt.divisibility = 16 : i32},
  %N: i32 {tt.divisibility = 16 : i32},
  %K: i32 {tt.divisibility = 16 : i32}
) {
  %false = arith.constant false
  %true = arith.constant true
  %c148_i32 = arith.constant 148 : i32
  %c128_i32 = arith.constant 128 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32

  %start_pid = tt.get_program_id x : i32
  %num_pid_m = arith.addi %M, %c128_i32 : i32
  %num_pid_m_div = arith.divsi %num_pid_m, %c128_i32 : i32
  %num_pid_n = arith.addi %N, %c128_i32 : i32
  %num_pid_n_div = arith.divsi %num_pid_n, %c128_i32 : i32
  %k_tiles = arith.addi %K, %c64_i32 : i32
  %k_tiles_div = arith.divsi %k_tiles, %c64_i32 : i32
  %num_mn_tiles = arith.muli %num_pid_m_div, %num_pid_n_div : i32
  %num_tiles = arith.muli %num_mn_tiles, %c2_i32 : i32
  %k_per_split = arith.addi %k_tiles_div, %c1_i32 : i32
  %k_per_split_div = arith.divsi %k_per_split, %c2_i32 : i32

  %tile_id_c_out = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32
      iter_args(%tile_id_c = %c0_i32) -> (i32) : i32 {
    %split_id = arith.divsi %tile_id, %num_mn_tiles : i32
    %k_start = arith.muli %split_id, %k_per_split_div : i32
    %k_end = arith.addi %k_start, %k_per_split_div : i32
    %k_end_clamped = arith.minsi %k_end, %k_tiles_div : i32
    %pid_m = arith.remsi %tile_id, %num_pid_m_div : i32
    %pid_n = arith.divsi %tile_id, %num_pid_m_div : i32
    %offs_am = arith.muli %pid_m, %c128_i32 : i32
    %offs_bn = arith.muli %pid_n, %c128_i32 : i32

    // Accumulator init
    %acc_mem, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // Inner k-loop
    %loop_out:2 = scf.for %ki = %k_start to %k_end_clamped step %c1_i32
        iter_args(%use_acc = %false, %loop_tok = %acc_tok) -> (i1, !ttg.async.token) : i32 {
      %offs_k = arith.muli %ki, %c64_i32 : i32
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_trans = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_trans, %acc_mem[%loop_tok], %use_acc, %true {tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %mma_tok : i1, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}

    // Epilogue: tmem_load + truncf + TMA store to workspace
    %result, %result_tok = ttng.tmem_load %acc_mem[%loop_out#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %c = arith.truncf %result : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %row_base = arith.muli %split_id, %M : i32
    %ws_row = arith.addi %row_base, %offs_am : i32
    tt.descriptor_store %ws_desc[%ws_row, %offs_bn], %c : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>

    %tile_id_c_next = arith.addi %tile_id_c, %c1_i32 : i32
    scf.yield %tile_id_c_next : i32
  } {tt.disallow_acc_multi_buffer, tt.flatten, tt.warp_specialize}

  tt.return
}

}
