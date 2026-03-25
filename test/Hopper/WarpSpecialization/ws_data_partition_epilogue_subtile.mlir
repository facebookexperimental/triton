// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

// Test that data partition handles unpartitioned descriptor_store ops whose
// source values are derived from a splat constant through a chain of
// element-preserving ops (split -> truncf -> convert_layout). This pattern
// arises with EPILOGUE_SUBTILE > 1 and FLATTEN=True when the persistent GEMM
// creates an scf.if with a k_tiles==0 zero-store path.

// CHECK-LABEL: @epilogue_subtile_dp
// Function signature should show sliced a_desc (256x64 -> 128x64) and c_desc (256x64 -> 128x64):
// CHECK-SAME: !tt.tensordesc<tensor<128x64xf16
// CHECK-SAME: !tt.tensordesc<tensor<128x64xf16

// The if-branch stores should be partitioned (4 stores: 2 subtiles x 2 partitions):
// CHECK: scf.if
// CHECK: scf.for
// CHECK-COUNT-4: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @epilogue_subtile_dp(
      %a_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64,
      %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>,
      %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64,
      %c_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    // The 3D zero constant that gets split for epilogue subtiling.
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64x2xf32, #blocked>
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
    %is_zero_k = arith.cmpi eq, %k_tiles, %c0_i32 : i32
    scf.if %is_zero_k {
      // Zero-K path: stores zeros via split -> truncf -> convert_layout chain.
      // These are NOT direct arith.constant ops — the pass must recognize them
      // as effectively splat through the element-preserving op chain.
      %outLHS, %outRHS = tt.split %cst : tensor<256x64x2xf32, #blocked> -> tensor<256x64xf32, #blocked2>
      %c0 = arith.truncf %outLHS : tensor<256x64xf32, #blocked2> to tensor<256x64xf16, #blocked2>
      %c0_cvt = ttg.convert_layout %c0 : tensor<256x64xf16, #blocked2> -> tensor<256x64xf16, #blocked3>
      %c1 = arith.truncf %outRHS : tensor<256x64xf32, #blocked2> to tensor<256x64xf16, #blocked2>
      %c1_cvt = ttg.convert_layout %c1 : tensor<256x64xf16, #blocked2> -> tensor<256x64xf16, #blocked3>
      %3 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%iter = %tile_id_c) -> (i32)  : i32 {
        %4 = arith.addi %iter, %c148_i32 : i32
        %gid = arith.divsi %4, %num_pid_in_group : i32
        %fm = arith.muli %gid, %c8_i32 : i32
        %gsm = arith.subi %num_pid_m, %fm : i32
        %gsm2 = arith.minsi %gsm, %c8_i32 : i32
        %pm = arith.remsi %4, %gsm2 : i32
        %pid_m = arith.addi %fm, %pm : i32
        %pn_r = arith.remsi %4, %num_pid_in_group : i32
        %pid_n = arith.divsi %pn_r, %gsm2 : i32
        %offs_am = arith.muli %pid_m, %c256_i32 : i32
        %offs_bn = arith.muli %pid_n, %c128_i32 : i32
        tt.descriptor_store %c_desc[%offs_am, %offs_bn], %c0_cvt : !tt.tensordesc<tensor<256x64xf16, #shared>>, tensor<256x64xf16, #blocked3>
        %5 = arith.addi %offs_bn, %c64_i32 : i32
        tt.descriptor_store %c_desc[%offs_am, %5], %c1_cvt : !tt.tensordesc<tensor<256x64xf16, #shared>>, tensor<256x64xf16, #blocked3>
        scf.yield %4 : i32
      } {tt.data_partition_factor = 2 : i32, tt.flatten, tt.smem_alloc_algo = 1 : i32}
    } else {
      %num_iters_raw = arith.subi %num_tiles, %start_pid : i32
      %num_iters = arith.ceildivsi %num_iters_raw, %c148_i32 : i32
      %k_clamped = arith.maxsi %k_tiles, %c1_i32 : i32
      %total_iters = arith.muli %num_iters, %k_clamped : i32
      %init_tile = arith.subi %start_pid, %c148_i32 : i32
      %km1 = arith.subi %k_clamped, %c1_i32 : i32
      %km1_2 = arith.subi %k_clamped, %c1_i32 : i32
      %tmem_acc:2 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %tmem_init = ttng.tmem_store %cst_12, %tmem_acc#0[%tmem_acc#1], %true : tensor<256x128xf32, #blocked1> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %r:8 = scf.for %iv = %c0_i32 to %total_iters step %c1_i32 iter_args(%ki = %c0_i32, %tile_iter = %init_tile, %store_tile = %tile_id_c, %k_idx = %c0_i32, %offs_am = %c0_i32, %offs_bn = %c0_i32, %use_acc = %false, %acc_tok = %tmem_init) -> (i32, i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
        %is_first_k = arith.cmpi eq, %ki, %c0_i32 : i32
        %k_sel = arith.select %is_first_k, %c0_i32, %k_idx : i32
        %li:3 = scf.if %is_first_k -> (i32, i32, i32) {
          %nt = arith.addi %tile_iter, %c148_i32 : i32
          %gid = arith.divsi %nt, %num_pid_in_group : i32
          %fm = arith.muli %gid, %c8_i32 : i32
          %gsm = arith.subi %num_pid_m, %fm : i32
          %gsm2 = arith.minsi %gsm, %c8_i32 : i32
          %pm = arith.remsi %nt, %gsm2 : i32
          %pid_m = arith.addi %fm, %pm : i32
          %pn_r = arith.remsi %nt, %num_pid_in_group : i32
          %pid_n = arith.divsi %pn_r, %gsm2 : i32
          %am = arith.muli %pid_m, %c256_i32 : i32
          %bn = arith.muli %pid_n, %c128_i32 : i32
          scf.yield %am, %bn, %nt : i32, i32, i32
        } else {
          scf.yield %offs_am, %offs_bn, %tile_iter : i32, i32, i32
        }
        %ok = arith.muli %k_sel, %c64_i32 : i32
        %a = tt.descriptor_load %a_desc[%li#0, %ok] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked3>
        %a_smem = ttg.local_alloc %a : (tensor<256x64xf16, #blocked3>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%ok, %li#1] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked4>
        %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #blocked4>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %mma_tok = ttng.tc_gen5_mma %a_smem, %b_smem, %tmem_acc#0[%acc_tok], %use_acc, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %nk = arith.addi %k_sel, %c1_i32 : i32
        %is_last_k = arith.cmpi eq, %ki, %km1 : i32
        %next_use = arith.select %is_last_k, %false, %true : i1
        %si:2 = scf.if %is_last_k -> (i32, !ttg.async.token) {
          %nst = arith.addi %store_tile, %c148_i32 : i32
          %gid = arith.divsi %nst, %num_pid_in_group : i32
          %fm = arith.muli %gid, %c8_i32 : i32
          %gsm = arith.subi %num_pid_m, %fm : i32
          %gsm2 = arith.minsi %gsm, %c8_i32 : i32
          %pm = arith.remsi %nst, %gsm2 : i32
          %pid_m = arith.addi %fm, %pm : i32
          %pn_r = arith.remsi %nst, %num_pid_in_group : i32
          %pid_n = arith.divsi %pn_r, %gsm2 : i32
          %sam = arith.muli %pid_m, %c256_i32 : i32
          %sbn = arith.muli %pid_n, %c128_i32 : i32
          %loaded:2 = ttng.tmem_load %tmem_acc#0[%mma_tok] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked1>
          %acc = tt.reshape %loaded#0 : tensor<256x128xf32, #blocked1> -> tensor<256x2x64xf32, #blocked5>
          %acc_t = tt.trans %acc {order = array<i32: 0, 2, 1>} : tensor<256x2x64xf32, #blocked5> -> tensor<256x64x2xf32, #blocked>
          %outLHS, %outRHS = tt.split %acc_t : tensor<256x64x2xf32, #blocked> -> tensor<256x64xf32, #blocked2>
          %c0 = arith.truncf %outLHS : tensor<256x64xf32, #blocked2> to tensor<256x64xf16, #blocked2>
          %c0_cvt = ttg.convert_layout %c0 : tensor<256x64xf16, #blocked2> -> tensor<256x64xf16, #blocked3>
          tt.descriptor_store %c_desc[%sam, %sbn], %c0_cvt : !tt.tensordesc<tensor<256x64xf16, #shared>>, tensor<256x64xf16, #blocked3>
          %c1 = arith.truncf %outRHS : tensor<256x64xf32, #blocked2> to tensor<256x64xf16, #blocked2>
          %c1_cvt = ttg.convert_layout %c1 : tensor<256x64xf16, #blocked2> -> tensor<256x64xf16, #blocked3>
          %off2 = arith.addi %sbn, %c64_i32 : i32
          tt.descriptor_store %c_desc[%sam, %off2], %c1_cvt : !tt.tensordesc<tensor<256x64xf16, #shared>>, tensor<256x64xf16, #blocked3>
          scf.yield %nst, %loaded#1 : i32, !ttg.async.token
        } else {
          scf.yield %store_tile, %mma_tok : i32, !ttg.async.token
        }
        %nki = arith.addi %ki, %c1_i32 : i32
        %reset = arith.cmpi eq, %ki, %km1_2 : i32
        %ki_out = arith.select %reset, %c0_i32, %nki : i32
        scf.yield %ki_out, %li#2, %si#0, %nk, %li#0, %li#1, %next_use, %si#1 : i32, i32, i32, i32, i32, i32, i1, !ttg.async.token
      } {tt.warp_specialize}
    }
    tt.return
  }
}
