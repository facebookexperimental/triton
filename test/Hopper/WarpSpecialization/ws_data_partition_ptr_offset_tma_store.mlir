// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

// Test that data partition shifts a descriptor coordinate only on the sliced
// descriptor op. Here the TMA store coordinate pid_m * 256 is the same SSA
// value as the base of the pointer-loaded A rows. The A rows of partition 1
// are already shifted by the sliced tt.make_range {128, 256}, so their splat
// must use the unshifted base. Shifting it as well would load rows 256..383
// of the tile into the second half of the accumulator.

// CHECK-LABEL: @ptr_a_dp2_tma_store
// CHECK-DAG: [[R0:%.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: [[R1:%.*]] = tt.make_range {end = 256 : i32, start = 128 : i32}
// CHECK: scf.for
// CHECK: [[BASE:%.*]] = arith.muli %{{.*}}, %c256_i32 : i32
// CHECK: [[BASE1:%.*]] = arith.addi [[BASE]], %c128_i32 : i32
// CHECK: [[S0:%.*]] = tt.splat [[BASE]] : i32
// CHECK: arith.addi [[S0]], [[R0]]
// CHECK-NOT: tt.splat [[BASE1]] :
// CHECK: [[S1:%.*]] = tt.splat [[BASE]] : i32
// CHECK: arith.addi [[S1]], [[R1]]
// CHECK-NOT: tt.splat [[BASE1]] :
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[BASE]], %{{.*}}]
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[BASE1]], %{{.*}}]

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ptr_a_dp2_tma_store(%A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<1024> : tensor<256x1xi64, #blocked>
    %cst_0 = arith.constant dense<1024> : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c32_i32 = arith.constant 32 : i32
    %grid_m = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x32xf32, #blocked>
    %true = arith.constant true
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %c_desc = tt.make_tensor_descriptor %C, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<256x128xf32, #shared>
    %b_desc = tt.make_tensor_descriptor %B, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<32x128xbf16, #shared1>
    %0 = tt.get_program_id x : i32
    %rm = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a = tt.splat %A : !tt.ptr<f32> -> tensor<256x1x!tt.ptr<f32>, #blocked>
    scf.for %tile_id = %0 to %c32_i32 step %c148_i32  : i32 {
      %pid_m = arith.remsi %tile_id, %grid_m : i32
      %pid_n = arith.divsi %tile_id, %grid_m : i32
      %rm_3 = arith.muli %pid_m, %c256_i32 : i32
      %rm_4 = tt.splat %rm_3 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %rm_5 = arith.addi %rm_4, %rm : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %a_6 = arith.cmpi slt, %rm_5, %cst_0 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %a_7 = tt.expand_dims %a_6 {axis = 1 : i32} : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi1, #blocked>
      %a_8 = tt.expand_dims %rm_5 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
      %a_9 = arith.extsi %a_8 : tensor<256x1xi32, #blocked> to tensor<256x1xi64, #blocked>
      %a_10 = arith.muli %a_9, %cst : tensor<256x1xi64, #blocked>
      %a_11 = tt.addptr %a, %a_10 : tensor<256x1x!tt.ptr<f32>, #blocked>, tensor<256x1xi64, #blocked>
      %a_12 = tt.broadcast %a_11 : tensor<256x1x!tt.ptr<f32>, #blocked> -> tensor<256x32x!tt.ptr<f32>, #blocked>
      %a_13 = tt.broadcast %a_7 : tensor<256x1xi1, #blocked> -> tensor<256x32xi1, #blocked>
      %b = arith.muli %pid_n, %c128_i32 : i32
      %acc, %acc_14 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_15 = ttng.tmem_store %cst_2, %acc[%acc_14], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_16:2 = scf.for %ki = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%acc_19 = %false, %acc_20 = %acc_15) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c32_i32 : i32
        %rk_21 = tt.splat %offs_k : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %rk_22 = arith.addi %rk_21, %rk : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %a_23 = tt.expand_dims %rk_22 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %a_24 = tt.broadcast %a_23 : tensor<1x32xi32, #blocked> -> tensor<256x32xi32, #blocked>
        %a_25 = tt.addptr %a_12, %a_24 : tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<256x32xi32, #blocked>
        %a_26 = tt.load %a_25, %a_13, %cst_1 : tensor<256x32x!tt.ptr<f32>, #blocked>
        %a_27 = arith.truncf %a_26 : tensor<256x32xf32, #blocked> to tensor<256x32xbf16, #blocked>
        %acc_28 = ttg.convert_layout %a_27 : tensor<256x32xbf16, #blocked> -> tensor<256x32xbf16, #linear1>
        %acc_29 = ttng.tmem_alloc %acc_28 : (tensor<256x32xbf16, #linear1>) -> !ttg.memdesc<256x32xbf16, #tmem1, #ttng.tensor_memory>
        %b_30 = tt.descriptor_load %b_desc[%offs_k, %b] : !tt.tensordesc<32x128xbf16, #shared1> -> tensor<32x128xbf16, #blocked1>
        %b_31 = ttg.local_alloc %b_30 : (tensor<32x128xbf16, #blocked1>) -> !ttg.memdesc<32x128xbf16, #shared1, #smem>
        %acc_32 = ttng.tc_gen5_mma %acc_29, %b_31, %acc[%acc_20], %acc_19, %true : !ttg.memdesc<256x32xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<32x128xbf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_32 : i1, !ttg.async.token
      }
      %acc_17, %acc_18 = ttng.tmem_load %acc[%acc_16#1] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %1 = ttg.convert_layout %acc_17 : tensor<256x128xf32, #linear> -> tensor<256x128xf32, #blocked2>
      tt.descriptor_store %c_desc[%rm_3, %b], %1 : !tt.tensordesc<256x128xf32, #shared>, tensor<256x128xf32, #blocked2>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
