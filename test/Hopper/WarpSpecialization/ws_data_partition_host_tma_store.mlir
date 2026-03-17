// RUN: not triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 2>&1 | FileCheck %s

// Reproducer for data partition with host-side TMA descriptor_store.
// When DATA_PARTITION_FACTOR=2, the pass slices the a_desc func arg
// (256x64 -> 128x64) but must also slice the c_desc func arg
// (256x128 -> 128x128) AND the stored tensor value consistently.
// Previously, the c_desc type was updated but the stored value was not
// partitioned, causing a type mismatch verification failure.

// CHECK: error: 'tt.descriptor_store' op tensor descriptor block and tensor types must match
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @host_tma_dp_store(
      %a_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64,
      %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64,
      %c_desc: !tt.tensordesc<tensor<256x128xf16, #shared>>,
      %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blocked>
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #blocked1>
    %start_pid = tt.get_program_id x : i32
    %0 = arith.addi %M, %c255_i32 : i32
    %num_pid_m = arith.divsi %0, %c256_i32 : i32
    %1 = arith.addi %N, %c127_i32 : i32
    %num_pid_n = arith.divsi %1, %c128_i32 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %k_tiles = arith.divsi %2, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m, %num_pid_n : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n, %c8_i32 : i32
    %3 = arith.cmpi eq, %k_tiles, %c0_i32 : i32
    scf.if %3 {
      %4 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%iter_tile_id_c = %tile_id_c) -> (i32)  : i32 {
        %5 = arith.addi %iter_tile_id_c, %c148_i32 : i32
        %6 = arith.divsi %5, %num_pid_in_group : i32
        %7 = arith.muli %6, %c8_i32 : i32
        %8 = arith.subi %num_pid_m, %7 : i32
        %9 = arith.minsi %8, %c8_i32 : i32
        %10 = arith.remsi %5, %9 : i32
        %11 = arith.addi %7, %10 : i32
        %12 = arith.remsi %5, %num_pid_in_group : i32
        %13 = arith.divsi %12, %9 : i32
        %offs_am_c = arith.muli %11, %c256_i32 : i32
        %offs_bn_c = arith.muli %13, %c128_i32 : i32
        tt.descriptor_store %c_desc[%offs_am_c, %offs_bn_c], %cst : !tt.tensordesc<tensor<256x128xf16, #shared>>, tensor<256x128xf16, #blocked>
        scf.yield %5 : i32
      } {tt.data_partition_factor = 2 : i32, tt.flatten, tt.smem_alloc_algo = 1 : i32}
    } else {
      %num_iters = arith.subi %num_tiles, %start_pid : i32
      %num_iters_ceildiv = arith.ceildivsi %num_iters, %c148_i32 : i32
      %k_tiles_clamped = arith.maxsi %k_tiles, %c1_i32 : i32
      %total_iters = arith.muli %num_iters_ceildiv, %k_tiles_clamped : i32
      %tile_id_c_init = arith.subi %start_pid, %c148_i32 : i32
      %k_tiles_m1 = arith.subi %k_tiles_clamped, %c1_i32 : i32
      %k_tiles_m1_2 = arith.subi %k_tiles_clamped, %c1_i32 : i32
      %tmem_acc:2 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %tmem_init = ttng.tmem_store %cst_12, %tmem_acc#0[%tmem_acc#1], %true : tensor<256x128xf32, #blocked1> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %results:8 = scf.for %iv = %c0_i32 to %total_iters step %c1_i32 iter_args(%ki = %c0_i32, %tile_iter = %tile_id_c_init, %store_tile = %tile_id_c, %k_idx = %c0_i32, %offs_am = %c0_i32, %offs_bn = %c0_i32, %use_acc = %false, %acc_tok = %tmem_init) -> (i32, i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
        %is_first_k = arith.cmpi eq, %ki, %c0_i32 : i32
        %k_idx_sel = arith.select %is_first_k, %c0_i32, %k_idx : i32
        %load_info:3 = scf.if %is_first_k -> (i32, i32, i32) {
          %new_tile = arith.addi %tile_iter, %c148_i32 : i32
          %gid = arith.divsi %new_tile, %num_pid_in_group : i32
          %first_m = arith.muli %gid, %c8_i32 : i32
          %gsm = arith.subi %num_pid_m, %first_m : i32
          %gsm_clamped = arith.minsi %gsm, %c8_i32 : i32
          %pm = arith.remsi %new_tile, %gsm_clamped : i32
          %pid_m = arith.addi %first_m, %pm : i32
          %pn_rem = arith.remsi %new_tile, %num_pid_in_group : i32
          %pid_n = arith.divsi %pn_rem, %gsm_clamped : i32
          %am = arith.muli %pid_m, %c256_i32 : i32
          %bn = arith.muli %pid_n, %c128_i32 : i32
          scf.yield %am, %bn, %new_tile : i32, i32, i32
        } else {
          scf.yield %offs_am, %offs_bn, %tile_iter : i32, i32, i32
        }
        %offs_k = arith.muli %k_idx_sel, %c64_i32 : i32
        %a = tt.descriptor_load %a_desc[%load_info#0, %offs_k] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked2>
        %a_smem = ttg.local_alloc %a : (tensor<256x64xf16, #blocked2>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%load_info#1, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked2>
        %b_smem = ttg.local_alloc %b : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b_trans = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %mma_tok = ttng.tc_gen5_mma %a_smem, %b_trans, %tmem_acc#0[%acc_tok], %use_acc, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %next_k = arith.addi %k_idx_sel, %c1_i32 : i32
        %is_last_k = arith.cmpi eq, %ki, %k_tiles_m1 : i32
        %next_use_acc = arith.select %is_last_k, %false, %true : i1
        %store_info:2 = scf.if %is_last_k -> (i32, !ttg.async.token) {
          %new_store_tile = arith.addi %store_tile, %c148_i32 : i32
          %gid = arith.divsi %new_store_tile, %num_pid_in_group : i32
          %first_m = arith.muli %gid, %c8_i32 : i32
          %gsm = arith.subi %num_pid_m, %first_m : i32
          %gsm_clamped = arith.minsi %gsm, %c8_i32 : i32
          %pm = arith.remsi %new_store_tile, %gsm_clamped : i32
          %pid_m = arith.addi %first_m, %pm : i32
          %pn_rem = arith.remsi %new_store_tile, %num_pid_in_group : i32
          %pid_n = arith.divsi %pn_rem, %gsm_clamped : i32
          %store_am = arith.muli %pid_m, %c256_i32 : i32
          %store_bn = arith.muli %pid_n, %c128_i32 : i32
          %loaded:2 = ttng.tmem_load %tmem_acc#0[%mma_tok] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked1>
          %truncated = arith.truncf %loaded#0 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
          %converted = ttg.convert_layout %truncated : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #blocked>
          tt.descriptor_store %c_desc[%store_am, %store_bn], %converted : !tt.tensordesc<tensor<256x128xf16, #shared>>, tensor<256x128xf16, #blocked>
          scf.yield %new_store_tile, %loaded#1 : i32, !ttg.async.token
        } else {
          scf.yield %store_tile, %mma_tok : i32, !ttg.async.token
        }
        %next_ki = arith.addi %ki, %c1_i32 : i32
        %reset_ki = arith.cmpi eq, %ki, %k_tiles_m1_2 : i32
        %ki_out = arith.select %reset_ki, %c0_i32, %next_ki : i32
        scf.yield %ki_out, %load_info#2, %store_info#0, %next_k, %load_info#0, %load_info#1, %next_use_acc, %store_info#1 : i32, i32, i32, i32, i32, i32, i1, !ttg.async.token
      } {tt.warp_specialize}
    }
    tt.return
  }
}
