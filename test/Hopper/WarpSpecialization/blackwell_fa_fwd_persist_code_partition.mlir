// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s
// CHECK-LABEL: _attn_fwd_persist
// CHECK: ttg.warp_specialize
// CHECK-SAME: ttg.partition.types = ["correction", "gemm", "load", "epilogue_store", "computation", "computation"]
// CHECK: default
//
// partition0 = gemm
//
// Outer loop carries i64 counters initialized to 0.
// q0 phase uses divui by 1 (single-buffer, outer-loop-only counter).
// k/v phase uses divui by 3 (triple-buffer, inner-loop counter).
// Counter increments by 1 each outer iteration.
//
// CHECK: partition0
// CHECK: arith.constant {{.*}} 0 : i64
// Outer loop with i64 iter_args, first initialized to 0
// CHECK: scf.for %arg{{[0-9]+}} = {{.*}} iter_args(%[[ARG0:arg[0-9]+]] = %c0_i64, %[[ARG1:arg[0-9]+]] = %c0_i64{{.*}}, %[[ARG2:arg[0-9]+]] = %c0_i64{{.*}}) -> (i64, i64, i64)
//
// q0 phase: full data dependency chain from ARG0 to wait_barrier
//   ARG0 -> divui -> DIV -> andi -> PHASE_BIT -> trunci -> PHASE_I1 -> extui -> PHASE_I32 -> wait_barrier
// CHECK:   [[DIV0:%.*]] = arith.divui %[[ARG0]],
// CHECK-SAME: : i64
// CHECK:   [[PHASE_BIT0:%.*]] = arith.andi [[DIV0]],
// CHECK-SAME: : i64
// CHECK:   [[PHASE_I1_0:%.*]] = arith.trunci [[PHASE_BIT0]]
// CHECK-SAME: : i64 to i1
// Second q0 channel: also from ARG0
// CHECK:   [[DIV1:%.*]] = arith.divui %[[ARG0]],
// CHECK-SAME: : i64
// CHECK:   [[PHASE_BIT1:%.*]] = arith.andi [[DIV1]],
// CHECK-SAME: : i64
// CHECK:   [[PHASE_I1_1:%.*]] = arith.trunci [[PHASE_BIT1]]
// CHECK-SAME: : i64 to i1
//
// q0 consumer wait: extui(PHASE_I1) -> wait_barrier (no xori)
// CHECK:   [[PHASE_I32_1:%.*]] = arith.extui [[PHASE_I1_1]]
// CHECK-SAME: : i1 to i32
// CHECK-NOT: arith.xori
// CHECK:   ttng.wait_barrier {{.*}}, [[PHASE_I32_1]]
// CHECK:   [[PHASE_I32_0:%.*]] = arith.extui [[PHASE_I1_0]]
// CHECK-SAME: : i1 to i32
// CHECK-NOT: arith.xori
// CHECK:   ttng.wait_barrier {{.*}}, [[PHASE_I32_0]]
//
// Inner loop: k/v phase uses divui by 3 (buffer.copy=3)
// Inner loop iter_args: ARG3 for acc counter, ARG4 for k/v counter
// CHECK:   scf.for %arg{{[0-9]+}} = {{.*}} iter_args(%[[ARG3:arg[0-9]+]] = {{.*}}, %[[ARG4:arg[0-9]+]] = {{.*}}) -> (i64, i64)
// k/v phase: full data dependency chain from ARG4 to wait_barrier
//   ARG4 -> divui by 3 -> DIV_KV -> andi -> PHASE_KV -> trunci -> PHASE_KV_I1 -> extui -> wait_barrier
// CHECK:     [[C3:%.*]] = arith.constant {{.*}} 3 : i64
// CHECK:     [[DIV_KV:%.*]] = arith.divui %[[ARG4]], [[C3]]
// CHECK-SAME: : i64
// CHECK:     [[PHASE_KV_BIT:%.*]] = arith.andi [[DIV_KV]],
// CHECK-SAME: : i64
// CHECK:     [[PHASE_KV_I1:%.*]] = arith.trunci [[PHASE_KV_BIT]]
// CHECK-SAME: : i64 to i1
// k consumer wait with phase from ARG4
// CHECK:     [[PHASE_KV_I32:%.*]] = arith.extui [[PHASE_KV_I1]]
// CHECK-SAME: : i1 to i32
// CHECK:     ttng.wait_barrier {{.*}}, [[PHASE_KV_I32]]
// k/v counter update: ARG4 incremented by 2 (k+v each consume one buffer slot)
// CHECK:     [[KV_INC:%.*]] = arith.constant {{.*}} 2 : i64
// CHECK:     [[NEW_KV:%.*]] = arith.addi %[[ARG4]], [[KV_INC]]
// CHECK-SAME: : i64
// Inner acc counter update: ARG3 incremented by 1
// CHECK:     [[NEW_ACC:%.*]] = arith.addi %[[ARG3]],
// CHECK-SAME: : i64
// CHECK:     scf.yield {{.*}}[[NEW_ACC]], [[NEW_KV]]
//
// Outer counter update: ARG0 incremented by 1, yielded as first result
// CHECK:   [[NEW_CNT:%.*]] = arith.addi %[[ARG0]],
// CHECK-SAME: : i64
// CHECK:   scf.yield {{.*}}[[NEW_CNT]],
//
// partition1 = load: q0 producer uses inverted phase (xori)
// CHECK: partition1
// CHECK: scf.for
// CHECK:   arith.trunci {{.*}} : i64 to i1
// CHECK:   arith.xori
// CHECK:   arith.extui {{.*}} : i1 to i32
// CHECK:   ttng.wait_barrier
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK:   arith.trunci {{.*}} : i64 to i1
// CHECK:   arith.xori
// CHECK:   arith.extui {{.*}} : i1 to i32
// CHECK:   ttng.wait_barrier
// CHECK:   ttng.async_tma_copy_global_to_local
//
// CHECK: partition2
// CHECK: partition3
// CHECK: partition4

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_2 = arith.constant {async_task_id = array<i32: 4, 5>} 1.44269502 : f32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 0 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c128_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 128 : i32
    %c4096_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 4096 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 1 : i32
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 16 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %_0 = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %_1 = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %acc_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %acc_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha_1, %alpha_1_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alpha_0, %alpha_0_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_1, %qk_1_5 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_0, %qk_0_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m_ij_0, %m_ij_0_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_1, %l_i0_1_8 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_ij_1, %m_ij_1_9 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_0, %l_i0_0_10 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_1_11, %acc_1_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_0_13, %acc_0_14 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q0_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %q0_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles = arith.muli %Z, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles_15 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_15, %num_progs {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %0 = arith.remsi %total_tiles_15, %num_progs {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_27 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm_27 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>}
    %desc_q_16 = arith.muli %Z, %H {async_task_id = array<i32: 2, 3>} : i32
    %desc_q_17 = arith.muli %desc_q_16, %c4096_i32 {async_task_id = array<i32: 2, 3>} : i32
    %desc_q_18 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_19 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_20 = tt.make_tensor_descriptor %desc_k, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_21 = tt.make_tensor_descriptor %desc_v, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_22 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_23 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c4096_i32 {async_task_id = array<i32: 2, 3>} : i32
    %offs_m0 = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m0_24 = tt.make_range {async_task_id = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst_2 {async_task_id = array<i32: 4, 5>} : f32
    %m_ij = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_ij_25 = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128x128xf32, #blocked>
    %qk_26 = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128x128xf32, #blocked>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_27 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_27, %n_tile_num {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_hz = arith.divsi %tile_idx_27, %n_tile_num {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_z = arith.divsi %off_hz, %H {async_task_id = array<i32: 2, 3>} : i32
      %off_h = arith.remsi %off_hz, %H {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_28 = arith.muli %off_z, %offset_y {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_29 = arith.muli %off_h, %c4096_i32 {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_30 = arith.addi %offset_y_28, %offset_y_29 {async_task_id = array<i32: 2, 3>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_31 = arith.addi %offset_y_30, %qo_offset_y {async_task_id = array<i32: 2, 3>} : i32
      %3 = arith.addi %qo_offset_y_31, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %q0 = arith.addi %qo_offset_y_31, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_m0_32 = tt.splat %qo_offset_y {async_task_id = array<i32: 0>} : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_33 = tt.splat %qo_offset_y {async_task_id = array<i32: 0>} : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_34 = arith.addi %offs_m0_32, %offs_m0 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
      %offs_m0_35 = arith.addi %offs_m0_33, %offs_m0_24 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
      %q0_36 = tt.descriptor_load %desc_q_18[%qo_offset_y_31, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %q0_37 = tt.descriptor_load %desc_q_19[%q0, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %q0_36, %q0_0 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %q0_37, %q0_1 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %acc = ttng.tmem_store %cst_1, %acc_0_13[%acc_0_14], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_38 = ttng.tmem_store %cst_1, %acc_1_11[%acc_1_12], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y:9 = scf.for %offsetkv_y_81 = %c0_i32 to %c4096_i32 step %c128_i32 iter_args(%offset_y_82 = %offset_y_30, %arg12 = %cst, %arg13 = %cst_0, %qk_0_83 = %qk_0_6, %acc_84 = %acc, %arg16 = %cst, %arg17 = %cst_0, %qk_1_85 = %qk_1_5, %acc_86 = %acc_38) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_87 = tt.descriptor_load %desc_k_20[%offset_y_82, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        %v_88 = tt.descriptor_load %desc_v_21[%offset_y_82, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        ttg.local_store %k_87, %k {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_89 = ttg.memdesc_trans %k {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        ttg.local_store %v_88, %v {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_90 = ttng.tc_gen5_mma %q0_0, %k_89, %qk_0[%qk_0_83], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_91 = ttng.tc_gen5_mma %q0_1, %k_89, %qk_1[%qk_1_85], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_92, %qk_93 = ttng.tmem_load %qk_0[%qk_90] {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %qk_94, %qk_95 = ttng.tmem_load %qk_1[%qk_91] {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_96 = "tt.reduce"(%qk_92) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_162: f32, %m_ij_163: f32):
          %m_ij_164 = arith.maxnumf %m_ij_162, %m_ij_163 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %m_ij_164 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_97 = "tt.reduce"(%qk_94) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_162: f32, %m_ij_163: f32):
          %m_ij_164 = arith.maxnumf %m_ij_162, %m_ij_163 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %m_ij_164 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_98 = arith.mulf %m_ij_96, %m_ij {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_99 = arith.mulf %m_ij_97, %m_ij_25 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_100 = arith.maxnumf %arg13, %m_ij_98 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_101 = arith.maxnumf %arg17, %m_ij_99 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_102 = arith.mulf %qk_92, %qk {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_103 = arith.mulf %qk_94, %qk_26 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %qk_104 = tt.expand_dims %m_ij_100 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_105 = tt.expand_dims %m_ij_101 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_106 = tt.broadcast %qk_104 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_107 = tt.broadcast %qk_105 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_108 = arith.subf %qk_102, %qk_106 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_109 = arith.subf %qk_103, %qk_107 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_108 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %p_110 = math.exp2 %qk_109 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %alpha = arith.subf %arg13, %m_ij_100 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_111 = arith.subf %arg17, %m_ij_101 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_112 = math.exp2 %alpha {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_113 = tt.expand_dims %alpha_112 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_114 = ttg.convert_layout %alpha_113 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        ttng.tmem_store %alpha_114, %alpha_0, %true {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %alpha_115 = math.exp2 %alpha_111 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_116 = tt.expand_dims %alpha_115 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_117 = ttg.convert_layout %alpha_116 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        ttng.tmem_store %alpha_117, %alpha_1, %true {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_162: f32, %l_ij_163: f32):
          %l_ij_164 = arith.addf %l_ij_162, %l_ij_163 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %l_ij_164 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_118 = "tt.reduce"(%p_110) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_162: f32, %l_ij_163: f32):
          %l_ij_164 = arith.addf %l_ij_162, %l_ij_163 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %l_ij_164 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_119, %acc_120 = ttng.tmem_load %acc_0_13[%acc_84] {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_121, %acc_122 = ttng.tmem_load %acc_1_11[%acc_86] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %12 = tt.reshape %acc_119 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
        %13 = tt.reshape %acc_121 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
        %14 = tt.trans %12 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %15 = tt.trans %13 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %outLHS, %outRHS = tt.split %14 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %outLHS_123, %outRHS_124 = tt.split %15 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %16 = ttg.convert_layout %outRHS {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
        %17 = ttg.convert_layout %outRHS_124 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
        %18 = ttg.convert_layout %outLHS {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
        %19 = ttg.convert_layout %outLHS_123 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
        %acc0_125, %acc0_126 = ttng.tmem_load %alpha_0[] {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc0_127 = tt.reshape %acc0_125 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc0_128 = ttg.convert_layout %acc0_127 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc0_129 = tt.expand_dims %acc0_128 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_130, %acc0_131 = ttng.tmem_load %alpha_1[] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc0_132 = tt.reshape %acc0_130 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc0_133 = ttg.convert_layout %acc0_132 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc0_134 = tt.expand_dims %acc0_133 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_135 = tt.broadcast %acc0_129 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc0_136 = tt.broadcast %acc0_134 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc0_137 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", loop.cluster = 3 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true} %18, %acc0_135 : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc0_138 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", loop.cluster = 1 : i32, loop.stage = 2 : i32, packed_element = 2 : i32, pure = true} %19, %acc0_136 : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", loop.cluster = 3 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true} %16, %acc0_135 : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc1_139 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", loop.cluster = 1 : i32, loop.stage = 2 : i32, packed_element = 2 : i32, pure = true} %17, %acc0_136 : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked>
        %acc_140 = tt.join %acc0_137, %acc1 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked7>
        %acc_141 = tt.join %acc0_138, %acc1_139 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked7>
        %acc_142 = tt.trans %acc_140 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked7> -> tensor<128x2x64xf32, #blocked8>
        %acc_143 = tt.trans %acc_141 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked7> -> tensor<128x2x64xf32, #blocked8>
        %acc_144 = ttg.convert_layout %acc_142 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x2x64xf32, #blocked8> -> tensor<128x2x64xf32, #linear1>
        %acc_145 = ttg.convert_layout %acc_143 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x2x64xf32, #blocked8> -> tensor<128x2x64xf32, #linear1>
        %acc_146 = tt.reshape %acc_144 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x2x64xf32, #linear1> -> tensor<128x128xf32, #linear2>
        %acc_147 = tt.reshape %acc_145 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x2x64xf32, #linear1> -> tensor<128x128xf32, #linear2>
        %p_148 = arith.truncf %p {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %p_149 = arith.truncf %p_110 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %acc_150 = ttg.convert_layout %p_148 {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        ttng.tmem_store %acc_150, %acc_0, %true {async_task_id = array<i32: 5>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_151 = ttg.convert_layout %p_149 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        ttng.tmem_store %acc_151, %acc_1, %true {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_152 = ttg.convert_layout %acc_146 {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear2> -> tensor<128x128xf32, #blocked>
        %acc_153 = ttg.convert_layout %acc_147 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear2> -> tensor<128x128xf32, #blocked>
        %acc_154 = ttng.tmem_store %acc_152, %acc_0_13[%acc_120], %true {async_task_id = array<i32: 0>, loop.cluster = 3 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 16>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_155 = ttng.tmem_store %acc_153, %acc_1_11[%acc_122], %true {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.start = array<i32: 14>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_156 = ttng.tc_gen5_mma %acc_0, %v, %acc_0_13[%acc_154], %true, %true {async_task_id = array<i32: 1>, loop.cluster = 3 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 16>, tmem.start = array<i32: 17>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_157 = ttng.tc_gen5_mma %acc_1, %v, %acc_1_11[%acc_155], %true, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 14>, tmem.start = array<i32: 15>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg12, %alpha_112 {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_158 = arith.mulf %arg16, %alpha_115 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_159 = arith.addf %l_i0, %l_ij {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_160 = arith.addf %l_i0_158, %l_ij_118 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_161 = arith.addi %offset_y_82, %c128_i32 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 4, 5>} %offsetkv_y_161, %l_i0_159, %m_ij_100, %qk_93, %acc_156, %l_i0_160, %m_ij_101, %qk_95, %acc_157 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true}
      %offsetkv_y_39 = tt.expand_dims %offsetkv_y#6 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_40 = ttg.convert_layout %offsetkv_y_39 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      ttng.tmem_store %offsetkv_y_40, %m_ij_0, %true {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y#5 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_42 = ttg.convert_layout %offsetkv_y_41 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      ttng.tmem_store %offsetkv_y_42, %l_i0_1, %true {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y#2 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_44 = ttg.convert_layout %offsetkv_y_43 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      ttng.tmem_store %offsetkv_y_44, %m_ij_1, %true {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y#1 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_46 = ttg.convert_layout %offsetkv_y_45 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      ttng.tmem_store %offsetkv_y_46, %l_i0_0, %true {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %m_i0, %m_i0_47 = ttng.tmem_load %l_i0_0[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_48 = tt.reshape %m_i0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_49 = ttg.convert_layout %m_i0_48 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_50 = math.log2 %m_i0_49 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_51, %m_i0_52 = ttng.tmem_load %m_ij_1[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_53 = tt.reshape %m_i0_51 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_54 = ttg.convert_layout %m_i0_53 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_55 = arith.addf %m_i0_54, %m_i0_50 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %4 = ttg.convert_layout %m_i0_55 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0 = arith.muli %off_hz, %c4096_i32 {async_task_id = array<i32: 0>} : i32
      %m_ptrs0_56 = tt.addptr %M, %m_ptrs0 {async_task_id = array<i32: 0>} : !tt.ptr<f32>, i32
      %m_ptrs0_57 = tt.splat %m_ptrs0_56 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_58 = tt.addptr %m_ptrs0_57, %offs_m0_34 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_58, %4 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0 = tt.expand_dims %m_i0_49 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_59 = tt.broadcast %acc0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_60, %acc_61 = ttng.tmem_load %acc_0_13[%offsetkv_y#4] {async_task_id = array<i32: 0>, tmem.end = array<i32: 17>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_62 = arith.divf %acc_60, %acc0_59 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %5 = arith.truncf %acc0_62 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %6, %_1 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7 = ttg.local_load %_1 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_22[%qo_offset_y_31, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %m_i0_63, %m_i0_64 = ttng.tmem_load %l_i0_1[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_65 = tt.reshape %m_i0_63 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_66 = ttg.convert_layout %m_i0_65 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_67 = math.log2 %m_i0_66 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_68, %m_i0_69 = ttng.tmem_load %m_ij_0[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_70 = tt.reshape %m_i0_68 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_71 = ttg.convert_layout %m_i0_70 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_72 = arith.addf %m_i0_71, %m_i0_67 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %8 = ttg.convert_layout %m_i0_72 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0_73 = tt.splat %m_ptrs0_56 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_74 = tt.addptr %m_ptrs0_73, %offs_m0_35 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_74, %8 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0_75 = tt.expand_dims %m_i0_66 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_76 = tt.broadcast %acc0_75 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_77, %acc_78 = ttng.tmem_load %acc_1_11[%offsetkv_y#8] {async_task_id = array<i32: 0>, tmem.end = array<i32: 15>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_79 = arith.divf %acc_77, %acc0_76 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %9 = arith.truncf %acc0_79 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %10 = ttg.convert_layout %9 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %10, %_0 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %11 = ttg.local_load %_0 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_23[%3, %c0_i32], %11 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %tile_idx_80 = arith.addi %tile_idx_27, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_80 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "load", "epilogue_store", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
