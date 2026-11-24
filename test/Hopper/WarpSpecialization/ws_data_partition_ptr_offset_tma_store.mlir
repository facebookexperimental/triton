// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

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

// -----

// Same bug when partitioning along N. BLOCK_M=64 and BLOCK_N=256, with a
// pointer-loaded B whose column base pid_n * 256 is also the store's column
// coordinate. Partition 1's B columns are already shifted by the sliced
// tt.make_range {128, 256}, so their splat must use the unshifted base.

// CHECK-LABEL: @ptr_b_dp2_n_tma_store
// CHECK-DAG: [[R0:%.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0
// CHECK-DAG: [[R1:%.*]] = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0
// CHECK: scf.for
// CHECK: [[OM:%.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// CHECK: [[BASE:%.*]] = arith.muli %{{.*}}, %c256_i32 : i32
// CHECK: [[BASE1:%.*]] = arith.addi [[BASE]], %c128_i32 : i32
// CHECK: [[S0:%.*]] = tt.splat [[BASE]] : i32
// CHECK: arith.addi [[S0]], [[R0]]
// CHECK-NOT: tt.splat [[BASE1]] :
// CHECK: [[S1:%.*]] = tt.splat [[BASE]] : i32
// CHECK: arith.addi [[S1]], [[R1]]
// CHECK-NOT: tt.splat [[BASE1]] :
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[OM]], [[BASE]]]
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[OM]], [[BASE1]]]

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ptr_b_dp2_n_tma_store(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c32_i32 = arith.constant 32 : i32
    %grid_m = arith.constant 8 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_k = arith.constant dense<1024> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #linear>
    %c_desc = tt.make_tensor_descriptor %C, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x256xf32, #shared>
    %a_desc = tt.make_tensor_descriptor %A, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %0 = tt.get_program_id x : i32
    %rn = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bp = tt.splat %B : !tt.ptr<f32> -> tensor<32x256x!tt.ptr<f32>, #blocked>
    scf.for %tile_id = %0 to %c32_i32 step %c148_i32  : i32 {
      %pid_m = arith.remsi %tile_id, %grid_m : i32
      %pid_n = arith.divsi %tile_id, %grid_m : i32
      %om = arith.muli %pid_m, %c64_i32 : i32
      %on = arith.muli %pid_n, %c256_i32 : i32
      %rn_s = tt.splat %on : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %rn_a = arith.addi %rn_s, %rn : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %rn_e = tt.expand_dims %rn_a {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %rn_b = tt.broadcast %rn_e : tensor<1x256xi32, #blocked> -> tensor<32x256xi32, #blocked>
      %acc, %acc_t = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_s = ttng.tmem_store %cst_0, %acc[%acc_t], %true : tensor<64x256xf32, #linear> -> !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %r:2 = scf.for %ki = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use = %false, %tok = %acc_s) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c32_i32 : i32
        %a = tt.descriptor_load %a_desc[%om, %offs_k] : !tt.tensordesc<64x32xbf16, #shared1> -> tensor<64x32xbf16, #blocked1>
        %a_s = ttg.local_alloc %a : (tensor<64x32xbf16, #blocked1>) -> !ttg.memdesc<64x32xbf16, #shared1, #smem>
        %rk_s = tt.splat %offs_k : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %rk_a = arith.addi %rk_s, %rk : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %rk_e = tt.expand_dims %rk_a {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
        %rk_m = arith.muli %rk_e, %cst_k : tensor<32x1xi32, #blocked>
        %rk_b = tt.broadcast %rk_m : tensor<32x1xi32, #blocked> -> tensor<32x256xi32, #blocked>
        %off = arith.addi %rk_b, %rn_b : tensor<32x256xi32, #blocked>
        %ptrs = tt.addptr %bp, %off : tensor<32x256x!tt.ptr<f32>, #blocked>, tensor<32x256xi32, #blocked>
        %b = tt.load %ptrs : tensor<32x256x!tt.ptr<f32>, #blocked>
        %b16 = arith.truncf %b : tensor<32x256xf32, #blocked> to tensor<32x256xbf16, #blocked>
        %b_s = ttg.local_alloc %b16 : (tensor<32x256xbf16, #blocked>) -> !ttg.memdesc<32x256xbf16, #shared2, #smem>
        %mma = ttng.tc_gen5_mma %a_s, %b_s, %acc[%tok], %use, %true : !ttg.memdesc<64x32xbf16, #shared1, #smem>, !ttg.memdesc<32x256xbf16, #shared2, #smem>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %mma : i1, !ttg.async.token
      }
      %res, %res_t = ttng.tmem_load %acc[%r#1] : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %1 = ttg.convert_layout %res : tensor<64x256xf32, #linear> -> tensor<64x256xf32, #blocked2>
      tt.descriptor_store %c_desc[%om, %on], %1 : !tt.tensordesc<64x256xf32, #shared>, tensor<64x256xf32, #blocked2>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

// The same SSA value in both store indices. Only the partitioned dimension of
// partition 1's store may be shifted: [base + 128, base], not
// [base + 128, base + 128].

// CHECK-LABEL: @ptr_a_dp2_tma_store_same_coord
// CHECK: scf.for
// CHECK: [[BASE:%.*]] = arith.muli %{{.*}}, %c256_i32 : i32
// CHECK: [[BASE1:%.*]] = arith.addi [[BASE]], %c128_i32 : i32
// CHECK-NOT: tt.splat [[BASE1]] :
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[BASE]], [[BASE]]]
// CHECK: tt.descriptor_store %{{.*}}{{\[}}[[BASE1]], [[BASE]]]

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
  tt.func public @ptr_a_dp2_tma_store_same_coord(%A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
      tt.descriptor_store %c_desc[%rm_3, %rm_3], %1 : !tt.tensordesc<256x128xf32, #shared>, tensor<256x128xf32, #blocked2>
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
