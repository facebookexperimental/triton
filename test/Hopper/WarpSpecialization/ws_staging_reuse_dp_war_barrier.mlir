// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=1 smem-budget=232448" 2>/dev/null | FileCheck %s

// Persistent BM=256 GEMM with data_partition_factor=2, num_stages=1 and an
// A operand built in registers from two fp32 TMA tiles. SMEM is tight, so the
// planner aliases one output-staging group onto an fp32 operand landing
// buffer and also chains the two data partitions' stagings onto each other.
// The cross-tile WAR barrier must still be emitted for the operand alias: the
// staging->staging entries are ordered by the reuse group itself and must not
// make the reuse pairs look inconsistent. The release must ride on the wait of
// the last TMA store, which store-wait pipelining defers to the next tile
// (loop.stage = 1), not fire at the bottom of the body while that store still
// reads the aliased SMEM.

// Only the operand alias is realized: mergeStagingReuseIntoHost drops a
// reuseTarget that names another staging buffer, so that staging keeps its own
// local_alloc and needs no cross-tile edge. (Stagings that share SMEM do so as
// one buffer.id reuse group, which its own acquires order.)
// CHECK: ttg.local_alloc {{.*}}buffer.tmaStaging = 1
// CHECK: ttg.memdesc_reinterpret
// CHECK-NOT: ttg.memdesc_reinterpret
// CHECK: partition1(
// CHECK: scf.for
// CHECK: ttng.async_tma_store_token_wait {{.*}}, %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}] {{.*}}loop.stage = 1
// CHECK: partition2(
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
// CHECK: scf.for
// CHECK: ttng.async_tma_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [128, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [128, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kmin(%X: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %E2: !tt.ptr<f32>, %Y: !tt.ptr<bf16>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %num_pid_m = arith.constant 255 : i32
    %num_pid_n = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %x_desc = arith.extsi %M : i32 to i64
    %y_desc = arith.extsi %N : i32 to i64
    %y_desc_0 = tt.make_tensor_descriptor %Y, [%K, %N], [%y_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared>
    %num_pid_m_1 = arith.addi %M, %num_pid_m : i32
    %num_pid_m_2 = arith.divsi %num_pid_m_1, %c256_i32 : i32
    %num_pid_n_3 = arith.addi %N, %num_pid_n : i32
    %num_pid_n_4 = arith.divsi %num_pid_n_3, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_2, %num_pid_n_4 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %x_desc_5 = tt.make_tensor_descriptor %X, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %e1_desc = tt.make_tensor_descriptor %E1, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%y_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared1>
    %x_desc_6 = tt.make_tensor_descriptor %X, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %e1_desc_7 = tt.make_tensor_descriptor %E1, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %c_desc_8 = tt.make_tensor_descriptor %C, [%M, %N], [%y_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared1>
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_2 {ttg.partition = array<i32: 2, 3>} : i32
      %pid_n = arith.divsi %t, %num_pid_m_2 {ttg.partition = array<i32: 2, 3>} : i32
      %a_t = arith.muli %pid_m, %c256_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %a_t_9 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %a_t_10 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %4 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %5 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %b = arith.muli %pid_n, %c128_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %acc = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_11 = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_0, %acc_0_12 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_13 = ttng.tmem_store %acc, %acc_0[%acc_0_12], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_1, %acc_1_14 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_15 = ttng.tmem_store %acc_11, %acc_1[%acc_1_14], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_16:3 = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_37 = %false, %acc_38 = %acc_13, %acc_39 = %acc_15) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %a_t_40 = arith.muli %ki, %c64_i32 {ttg.partition = array<i32: 3>} : i32
        %b_41 = tt.descriptor_load %y_desc_0[%a_t_40, %b] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked>
        %a_t_42 = tt.descriptor_load %x_desc_5[%a_t_40, %a_t] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_43 = tt.descriptor_load %e1_desc[%a_t_40, %a_t] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_44 = tt.descriptor_load %x_desc_6[%a_t_40, %a_t_10] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_45 = tt.descriptor_load %e1_desc_7[%a_t_40, %a_t_9] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %b_46 = ttg.local_alloc %b_41 {ttg.partition = array<i32: 3>} : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %a_t_47 = arith.addf %a_t_42, %a_t_43 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1>
        %a_t_48 = arith.truncf %a_t_47 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %acc_0_49 = ttg.local_alloc %a_t_48 {ttg.partition = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_50 = ttg.memdesc_trans %acc_0_49 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %acc_51 = ttng.tc_gen5_mma %acc_50, %b_46, %acc_0[%acc_38], %acc_37, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %a_t_52 = arith.addf %a_t_44, %a_t_45 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1>
        %a_t_53 = arith.truncf %a_t_52 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %acc_1_54 = ttg.local_alloc %a_t_53 {ttg.partition = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_55 = ttg.memdesc_trans %acc_1_54 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %acc_56 = ttng.tc_gen5_mma %acc_55, %b_46, %acc_1[%acc_39], %acc_37, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_51, %acc_56 : i1, !ttg.async.token, !ttg.async.token
      }
      %6 = arith.addi %b, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %acc_17, %acc_18 = ttng.tmem_load %acc_0[%acc_16#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_19 = ttg.convert_layout %acc_17 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %sl = tt.reshape %acc_19 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %sl_20 = tt.trans %sl {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %sl_21, %sl_22 = tt.split %sl_20 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %sl_23 = ttg.convert_layout %sl_22 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %sl_24 = ttg.convert_layout %sl_21 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %c_desc_staging = ttg.local_alloc %sl_24 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %7 = ttng.async_tma_copy_local_to_global %c_desc[%a_t, %b] %c_desc_staging {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %7   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %c_desc_staging_25 = ttg.local_alloc %sl_23 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %8 = ttng.async_tma_copy_local_to_global %c_desc[%a_t, %6] %c_desc_staging_25 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %8   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %acc_26, %acc_27 = ttng.tmem_load %acc_1[%acc_16#2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_28 = ttg.convert_layout %acc_26 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %sl_29 = tt.reshape %acc_28 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %sl_30 = tt.trans %sl_29 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %sl_31, %sl_32 = tt.split %sl_30 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %sl_33 = ttg.convert_layout %sl_32 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %sl_34 = ttg.convert_layout %sl_31 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %c_desc_staging_35 = ttg.local_alloc %sl_34 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %9 = ttng.async_tma_copy_local_to_global %c_desc_8[%5, %b] %c_desc_staging_35 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %9   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %c_desc_staging_36 = ttg.local_alloc %sl_33 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %10 = ttng.async_tma_copy_local_to_global %c_desc_8[%4, %6] %c_desc_staging_36 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %10   {ttg.partition = array<i32: 2>} : !ttg.async.token
    } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Same kernel with an extra, conditional wait on the last store's token after
// the last unconditional wait. The release must stay on the top-level wait (a
// release inside the scf.if would be skipped on some tiles and hang the load
// partition's next acquire).

// CHECK: partition1(
// CHECK: scf.for
// CHECK: ttng.async_tma_store_token_wait {{.*}}, %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}] {{.*}}loop.stage = 1
// CHECK: scf.if
// CHECK-NEXT: ttng.async_tma_store_token_wait %{{[0-9a-z_]+}} {{ *}}{async_task_id
// CHECK: partition2(

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [128, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [128, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kmin(%X: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %E2: !tt.ptr<f32>, %Y: !tt.ptr<bf16>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %num_pid_m = arith.constant 255 : i32
    %num_pid_n = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %x_desc = arith.extsi %M : i32 to i64
    %y_desc = arith.extsi %N : i32 to i64
    %y_desc_0 = tt.make_tensor_descriptor %Y, [%K, %N], [%y_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared>
    %num_pid_m_1 = arith.addi %M, %num_pid_m : i32
    %num_pid_m_2 = arith.divsi %num_pid_m_1, %c256_i32 : i32
    %num_pid_n_3 = arith.addi %N, %num_pid_n : i32
    %num_pid_n_4 = arith.divsi %num_pid_n_3, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_2, %num_pid_n_4 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %x_desc_5 = tt.make_tensor_descriptor %X, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %e1_desc = tt.make_tensor_descriptor %E1, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%y_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared1>
    %x_desc_6 = tt.make_tensor_descriptor %X, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %e1_desc_7 = tt.make_tensor_descriptor %E1, [%K, %M], [%x_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %c_desc_8 = tt.make_tensor_descriptor %C, [%M, %N], [%y_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared1>
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_2 {ttg.partition = array<i32: 2, 3>} : i32
      %pid_n = arith.divsi %t, %num_pid_m_2 {ttg.partition = array<i32: 2, 3>} : i32
      %a_t = arith.muli %pid_m, %c256_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %a_t_9 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %a_t_10 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %4 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %5 = arith.addi %a_t, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %b = arith.muli %pid_n, %c128_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %acc = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_11 = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_0, %acc_0_12 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_13 = ttng.tmem_store %acc, %acc_0[%acc_0_12], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_1, %acc_1_14 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_15 = ttng.tmem_store %acc_11, %acc_1[%acc_1_14], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_16:3 = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_37 = %false, %acc_38 = %acc_13, %acc_39 = %acc_15) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %a_t_40 = arith.muli %ki, %c64_i32 {ttg.partition = array<i32: 3>} : i32
        %b_41 = tt.descriptor_load %y_desc_0[%a_t_40, %b] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked>
        %a_t_42 = tt.descriptor_load %x_desc_5[%a_t_40, %a_t] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_43 = tt.descriptor_load %e1_desc[%a_t_40, %a_t] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_44 = tt.descriptor_load %x_desc_6[%a_t_40, %a_t_10] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %a_t_45 = tt.descriptor_load %e1_desc_7[%a_t_40, %a_t_9] {ttg.partition = array<i32: 3>} : !tt.tensordesc<64x128xf32, #shared1> -> tensor<64x128xf32, #blocked1>
        %b_46 = ttg.local_alloc %b_41 {ttg.partition = array<i32: 3>} : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %a_t_47 = arith.addf %a_t_42, %a_t_43 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1>
        %a_t_48 = arith.truncf %a_t_47 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %acc_0_49 = ttg.local_alloc %a_t_48 {ttg.partition = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_50 = ttg.memdesc_trans %acc_0_49 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %acc_51 = ttng.tc_gen5_mma %acc_50, %b_46, %acc_0[%acc_38], %acc_37, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %a_t_52 = arith.addf %a_t_44, %a_t_45 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1>
        %a_t_53 = arith.truncf %a_t_52 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %acc_1_54 = ttg.local_alloc %a_t_53 {ttg.partition = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_55 = ttg.memdesc_trans %acc_1_54 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %acc_56 = ttng.tc_gen5_mma %acc_55, %b_46, %acc_1[%acc_39], %acc_37, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_51, %acc_56 : i1, !ttg.async.token, !ttg.async.token
      }
      %6 = arith.addi %b, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %acc_17, %acc_18 = ttng.tmem_load %acc_0[%acc_16#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_19 = ttg.convert_layout %acc_17 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %sl = tt.reshape %acc_19 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %sl_20 = tt.trans %sl {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %sl_21, %sl_22 = tt.split %sl_20 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %sl_23 = ttg.convert_layout %sl_22 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %sl_24 = ttg.convert_layout %sl_21 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %c_desc_staging = ttg.local_alloc %sl_24 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %7 = ttng.async_tma_copy_local_to_global %c_desc[%a_t, %b] %c_desc_staging {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %7   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %c_desc_staging_25 = ttg.local_alloc %sl_23 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %8 = ttng.async_tma_copy_local_to_global %c_desc[%a_t, %6] %c_desc_staging_25 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %8   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %acc_26, %acc_27 = ttng.tmem_load %acc_1[%acc_16#2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_28 = ttg.convert_layout %acc_26 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %sl_29 = tt.reshape %acc_28 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %sl_30 = tt.trans %sl_29 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %sl_31, %sl_32 = tt.split %sl_30 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %sl_33 = ttg.convert_layout %sl_32 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %sl_34 = ttg.convert_layout %sl_31 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %c_desc_staging_35 = ttg.local_alloc %sl_34 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %9 = ttng.async_tma_copy_local_to_global %c_desc_8[%5, %b] %c_desc_staging_35 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %9   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %c_desc_staging_36 = ttg.local_alloc %sl_33 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %10 = ttng.async_tma_copy_local_to_global %c_desc_8[%4, %6] %c_desc_staging_36 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %10   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %pz = arith.cmpi eq, %pid_n, %c0_i32 {ttg.partition = array<i32: 2>} : i32
      scf.if %pz {
        ttng.async_tma_store_token_wait %10   {ttg.partition = array<i32: 2>} : !ttg.async.token
      } {ttg.partition = array<i32: 2>}
    } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
