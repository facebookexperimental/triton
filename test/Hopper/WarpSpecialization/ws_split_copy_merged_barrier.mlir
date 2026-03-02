// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=3 capability=100" | FileCheck %s

// Test: When two SMEM buffers share a reuse group (same buffer.id) and one
// requires TMA split copies, the code partition pass merges their consumer
// groups so a single barrier_expect + wait is emitted. Without the merge,
// each channel's separate insertAsyncComm call would create its own
// BarrierExpectOp, causing barrier over-arrival (UB).
//
// A (128x64xf16): inner dim = 64 * 2B = 128B = swizzle -> no split
// B (64x256xf16): inner dim = 256 * 2B = 512B > 128B swizzle -> split copies
//
// Both buffers share buffer.id = 0 (same reuse group), and the merged
// barrier_expect has size 49152 = 128*64*2 + 64*256*2.

// CHECK-LABEL: @matmul_kernel_tma_persistent
// Both SMEM allocations share buffer.id = 0:
// CHECK-DAG: ttg.local_alloc {{{.*}}buffer.id = 0{{.*}}} : () -> !ttg.memdesc<3x64x256xf16
// CHECK-DAG: ttg.local_alloc {{{.*}}buffer.id = 0{{.*}}} : () -> !ttg.memdesc<3x128x64xf16
// CHECK: ttg.warp_specialize
// Default group: MMA consumer
// CHECK: default
// CHECK: ttng.tc_gen5_mma
// Producer partition: single barrier_expect for merged consumer group
// CHECK: partition0
// CHECK: ttng.barrier_expect {{.*}}, 49152
// CHECK-NOT: ttng.barrier_expect
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// Epilogue partition: load from TMEM and store results
// CHECK: partition1
// CHECK: ttng.tmem_load
// CHECK: tt.descriptor_store
// CHECK: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<64x256xf16, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc_or_ptr: !tt.tensordesc<tensor<128x128xf16, #shared>>, %c_desc_or_ptr_8: i32, %c_desc_or_ptr_9: i32, %c_desc_or_ptr_10: i64, %c_desc_or_ptr_11: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %num_pid_m = arith.constant 127 : i32
    %num_pid_n = arith.constant 255 : i32
    %k_tiles = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m_12 = arith.addi %M, %num_pid_m : i32
    %num_pid_m_13 = arith.divsi %num_pid_m_12, %c128_i32 : i32
    %num_pid_n_14 = arith.addi %N, %num_pid_n : i32
    %num_pid_n_15 = arith.divsi %num_pid_n_14, %c256_i32 : i32
    %k_tiles_16 = arith.addi %K, %k_tiles : i32
    %k_tiles_17 = arith.divsi %k_tiles_16, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m_13, %num_pid_n_15 : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_15, %c8_i32 : i32
    %tile_id_c_18 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_19 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_13, %first_pid_m : i32
      %group_size_m_20 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_20 : i32
      %pid_m_21 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_22 = arith.divsi %pid_n, %group_size_m_20 : i32
      %offs_am = arith.muli %pid_m_21, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_22, %c256_i32 : i32
      %accumulator, %accumulator_23 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_24 = ttng.tmem_store %cst, %accumulator[%accumulator_23], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_25:2 = scf.for %accumulator_40 = %c0_i32 to %k_tiles_17 step %c1_i32 iter_args(%arg22 = %false, %accumulator_41 = %accumulator_24) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_40, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %a_42 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x256xf16, #shared>> -> tensor<64x256xf16, #blocked2>
        %b_43 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<64x256xf16, #blocked2>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %accumulator_44 = ttng.tc_gen5_mma %a_42, %b_43, %accumulator[%accumulator_41], %arg22, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_44 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %tile_id_c_26 = arith.addi %tile_id_c_19, %c148_i32 : i32
      %group_id_27 = arith.divsi %tile_id_c_26, %num_pid_in_group : i32
      %first_pid_m_28 = arith.muli %group_id_27, %c8_i32 : i32
      %group_size_m_29 = arith.subi %num_pid_m_13, %first_pid_m_28 : i32
      %group_size_m_30 = arith.minsi %group_size_m_29, %c8_i32 : i32
      %pid_m_31 = arith.remsi %tile_id_c_26, %group_size_m_30 : i32
      %pid_m_32 = arith.addi %first_pid_m_28, %pid_m_31 : i32
      %pid_n_33 = arith.remsi %tile_id_c_26, %num_pid_in_group : i32
      %pid_n_34 = arith.divsi %pid_n_33, %group_size_m_30 : i32
      %offs_am_c = arith.muli %pid_m_32, %c128_i32 : i32
      %offs_bn_c = arith.muli %pid_n_34, %c256_i32 : i32
      %accumulator_35, %accumulator_36 = ttng.tmem_load %accumulator[%accumulator_25#1] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %acc = tt.reshape %accumulator_35 : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked3>
      %acc_37 = tt.trans %acc {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3> -> tensor<128x128x2xf32, #blocked4>
      %outLHS, %outRHS = tt.split %acc_37 : tensor<128x128x2xf32, #blocked4> -> tensor<128x128xf32, #blocked5>
      %c0 = arith.truncf %outLHS : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %c0_38 = ttg.convert_layout %c0 : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked6>
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %offs_bn_c], %c0_38 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked6>
      %c1 = arith.truncf %outRHS : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %c1_39 = ttg.convert_layout %c1 : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked6>
      %0 = arith.addi %offs_bn_c, %c128_i32 : i32
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %0], %c1_39 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked6>
      scf.yield %tile_id_c_26 : i32
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
