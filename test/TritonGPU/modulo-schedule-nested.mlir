// REQUIRES: asserts
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -debug-only=nvgpu-modulo-schedule 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Nested loop (persistent GEMM) — outer tile loop + inner K-loop
//   Verify that both loops are scheduled and the kernel-wide SMEM budget
//   check accounts for outer + inner buffers simultaneously.
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: [PASS-A] === Loop ScheduleGraph ===
// CHECK: modulo.schedule @loop0 {
//
// CHECK: [PASS-A] === Loop ScheduleGraph ===
// CHECK: modulo.schedule @loop0 {
//
// Inner loop gets tt.num_stages (no loop.stage — uses emitMMAAnnotations).
// Outer loop gets loop.stage attrs via emitScheduleAttributes.
// CHECK-LABEL: @persistent_gemm_nested
// Inner loop has tt.num_stages:
// CHECK: scf.for
// CHECK: tt.num_stages
// Outer loop has schedule attrs:
// CHECK: tt.modulo_ii
  tt.func public @persistent_gemm_nested(
      %a_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %c_desc: !tt.tensordesc<tensor<256x256xf16, #shared>>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}
  ) {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %k_tiles = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #linear>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c255_i32 : i32
    %num_pid_m_12 = arith.divsi %num_pid_m, %c256_i32 : i32
    %num_pid_n = arith.addi %N, %c255_i32 : i32
    %num_pid_n_13 = arith.divsi %num_pid_n, %c256_i32 : i32
    %k_tiles_14 = arith.addi %K, %k_tiles : i32
    %k_tiles_15 = arith.divsi %k_tiles_14, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m_12, %num_pid_n_13 : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %tile_id_c_16 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_17 = %tile_id_c) -> (i32) : i32 {
      %pid_m = arith.divsi %tile_id, %num_pid_n_13 : i32
      %pid_n = arith.remsi %tile_id, %num_pid_n_13 : i32
      %offs_am = arith.muli %pid_m, %c256_i32 : i32
      %offs_bn = arith.muli %pid_n, %c256_i32 : i32
      %accumulator, %accumulator_18 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_19 = ttng.tmem_store %cst, %accumulator[%accumulator_18], %true : tensor<256x256xf32, #linear> -> !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_20:2 = scf.for %k = %c0_i32 to %k_tiles_15 step %c1_i32 iter_args(%arg21 = %false, %accumulator_25 = %accumulator_19) -> (i1, !ttg.async.token) : i32 {
        %offs_k = arith.muli %k, %c64_i32 : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked>
        %a_26 = ttg.local_alloc %a : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked>
        %arg2 = ttg.local_alloc %b : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        %arg2_27 = ttg.memdesc_trans %arg2 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
        %accumulator_28 = ttng.tc_gen5_mma %a_26, %arg2_27, %accumulator[%accumulator_25], %arg21, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared1, #smem>, !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_28 : i1, !ttg.async.token
      }
      %tile_id_c_21 = arith.addi %tile_id_c_17, %c148_i32 : i32
      %pid_m_c = arith.divsi %tile_id_c_21, %num_pid_n_13 : i32
      %pid_n_c = arith.remsi %tile_id_c_21, %num_pid_n_13 : i32
      %accumulator_22, %accumulator_23 = ttng.tmem_load %accumulator[%accumulator_20#1] : !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x256xf32, #linear>
      %c = arith.truncf %accumulator_22 : tensor<256x256xf32, #linear> to tensor<256x256xf16, #linear>
      %0 = arith.muli %pid_m_c, %c256_i32 : i32
      %1 = arith.muli %pid_n_c, %c256_i32 : i32
      %2 = ttg.convert_layout %c : tensor<256x256xf16, #linear> -> tensor<256x256xf16, #blocked1>
      tt.descriptor_store %c_desc[%0, %1], %2 : !tt.tensordesc<tensor<256x256xf16, #shared>>, tensor<256x256xf16, #blocked1>
      scf.yield %tile_id_c_21 : i32
    } {tt.flatten, tt.warp_specialize}
    tt.return
  }
}
