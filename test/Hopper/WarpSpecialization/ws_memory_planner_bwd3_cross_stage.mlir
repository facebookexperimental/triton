// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Test case: Cross-stage consumer detection for SMEM buffers
//
// This test verifies that isSmemCrossStage correctly identifies buffers where
// actual consumers (following through memdesc_trans) are in different stages,
// AND the buffer is updated inside the innermost loop (srcOp has loop.stage).
//
// For buffer %dsT:
//   - Write (local_store): cluster=2, stage=0, task_id=3
//   - Read 1 (MMA via memdesc_trans): stage=1 (actual consumer after following trans)
//   - Read 2 (MMA direct): stage=1
//   - Both actual consumers are at stage 1 → NOT cross-stage
//
// For buffer %q:
//   - Write (local_store): cluster=1, stage=0, task_id=2 (inside innermost loop)
//   - Read 1 (MMA via memdesc_trans %qT): stage=0
//   - Read 2 (MMA direct %dsT, %q, %dk): stage=1
//   - Actual consumers at stages 0 and 1 → IS cross-stage → gets copy=2
//
// For buffer %k:
//   - Write (local_store): NO loop.stage (outside innermost loop)
//   - Even though consumers are at different stages, the buffer is not updated
//     inside the innermost loop, so it does NOT need double-buffering

// CHECK-LABEL: tt.func public @_attn_bwd_persist
//
// SMEM allocation: dsT - actual consumers both at stage 1, NOT cross-stage
// CHECK: %dsT = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32}
//
// SMEM allocation: do (TMA buffer)
// CHECK: %do = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32}
//
// SMEM allocation: q has actual consumers at stages 0 and 1, IS cross-stage
// CHECK: %q = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32}
//
// SMEM: v is not innermost, copy=1
// CHECK: %v = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
//
// SMEM: k store is outside innermost loop (no loop.stage), NOT cross-stage
// CHECK: %k = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}

// -----// WarpSpec internal IR Dump After: doBufferAllocation
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 32], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 32, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %sm_scale: f32, %desc_do: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dq: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %desc_dk: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dv: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dq, %dq_0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dpT, %dpT_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %ppT = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %qkT, %qkT_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dv, %dv_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 1 : i64
    %c128_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 128 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 32 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 64 : i32
    %c96_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 96 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_5 = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x32xf32, #blocked1>
    %n_tile_num_6 = arith.addi %N_CTX, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %n_tile_num_7 = arith.divsi %n_tile_num_6, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles = arith.muli %n_tile_num_7, %BATCH {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles_8 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_8, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %0 = arith.remsi %total_tiles_8, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_17 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm_17 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %y_dim = arith.muli %BATCH, %H {async_task_id = array<i32: 0, 2, 3>} : i32
    %y_dim_9 = arith.muli %y_dim, %N_CTX {async_task_id = array<i32: 0, 2, 3>} : i32
    %desc_q_10 = tt.make_tensor_descriptor %desc_q, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_do_11 = tt.make_tensor_descriptor %desc_do, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_dq_12 = tt.make_tensor_descriptor %desc_dq, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 0>} : !tt.ptr<f32>, !tt.tensordesc<tensor<128x32xf32, #shared1>>
    %desc_v_13 = tt.make_tensor_descriptor %desc_v, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_14 = tt.make_tensor_descriptor %desc_k, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_dv_15 = tt.make_tensor_descriptor %desc_dv, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x32xf16, #shared2>>
    %desc_dk_16 = tt.make_tensor_descriptor %desc_dk, [%y_dim_9, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x32xf16, #shared2>>
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x32xf32, #blocked1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_17 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_17, %n_tile_num_7 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %tile_idx_17, %n_tile_num_7 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 3>} : i32
      %off_chz_18 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh_19 = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_20 = arith.muli %stride_h, %off_bh_19 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_21 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_22 = arith.muli %stride_z, %off_bh_21 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_23 = arith.addi %off_bh_20, %off_bh_22 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_24 = arith.extsi %off_bh_23 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %off_bh_25 = arith.divsi %off_bh_24, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i64
      %M_26 = tt.addptr %M, %off_chz_18 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %D_27 = tt.addptr %D, %off_chz_18 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %k_28 = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_29 = arith.addi %off_bh_25, %k_28 {async_task_id = array<i32: 2, 3>} : i64
      %k_30 = arith.trunci %k_29 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_31 = tt.descriptor_load %desc_k_14[%k_30, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %k_31, %k {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v_32 = tt.descriptor_load %desc_v_13[%k_30, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %v_32, %v {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %m = tt.splat %M_26 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %Di = tt.splat %D_27 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %dk_33 = ttng.tmem_store %cst, %dk[%dk_4], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_34 = ttng.tmem_store %cst, %dv[%dv_3], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %curr_m_66 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg19 = %c0_i32, %arg20 = %false, %qkT_67 = %qkT_2, %dv_68 = %dv_34, %dpT_69 = %dpT_1, %dk_70 = %dk_33, %dq_71 = %dq_0) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_72 = arith.extsi %arg19 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_73 = arith.addi %off_bh_25, %q_72 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %q_74 = arith.trunci %q_73 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_75 = tt.descriptor_load %desc_q_10[%q_74, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %q_75, %q {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %offs_m_76 = tt.splat %arg19 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
        %offs_m_77 = arith.addi %offs_m_76, %offs_m {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
        %m_78 = tt.addptr %m, %offs_m_77 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %m_79 = tt.load %m_78 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %qkT_80 = ttng.tc_gen5_mma %k, %qT, %qkT[%qkT_67], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %pT = ttg.convert_layout %m_79 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %pT_81 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %pT_82 = tt.broadcast %pT_81 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qkT_83, %qkT_84 = ttng.tmem_load %qkT[%qkT_80] {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %pT_85 = arith.subf %qkT_83, %pT_82 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %pT_86 = math.exp2 %pT_85 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %do_87 = tt.descriptor_load %desc_do_11[%q_74, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %do_87, %do {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %ppT_88 = arith.truncf %pT_86 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %dv_89 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} true
        ttng.tmem_store %ppT_88, %ppT, %dv_89 {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dv_90 = ttng.tc_gen5_mma %ppT, %do, %dv[%dv_68], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 6 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_91 = tt.addptr %Di, %offs_m_77 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %Di_92 = tt.load %Di_91 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %dpT_93 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %dpT_94 = ttng.tc_gen5_mma %v, %dpT_93, %dpT[%dpT_69], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dsT_95 = ttg.convert_layout %Di_92 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %dsT_96 = tt.expand_dims %dsT_95 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %dsT_97 = tt.broadcast %dsT_96 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %dpT_98, %dpT_99 = ttng.tmem_load %dpT[%dpT_94] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dsT_100 = arith.subf %dpT_98, %dsT_97 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %dsT_101 = arith.mulf %pT_86, %dsT_100 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %dsT_102 = arith.truncf %dsT_101 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        ttg.local_store %dsT_102, %dsT {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %dk_103 = ttng.tc_gen5_mma %dsT, %q, %dk[%dk_70], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_104 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %dq_105 = ttng.tc_gen5_mma %dq_104, %k, %dq[%dq_71], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_106, %dq_107 = ttng.tmem_load %dq[%dq_105] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dqs = tt.reshape %dq_106 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
        %dqs_108 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %dqs_109, %dqs_110 = tt.split %dqs_108 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %dqs_111 = tt.reshape %dqs_109 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_112 = tt.trans %dqs_111 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_113, %dqs_114 = tt.split %dqs_112 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqs_115 = tt.reshape %dqs_110 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_116 = tt.trans %dqs_115 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_117, %dqs_118 = tt.split %dqs_116 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqN = arith.mulf %dqs_113, %cst_5 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_119 = ttg.convert_layout %dqN {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_12[%q_74, %c0_i32], %dqN_119 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_120 = arith.mulf %dqs_114, %cst_5 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_121 = ttg.convert_layout %dqN_120 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_12[%q_74, %c32_i32], %dqN_121 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_122 = arith.mulf %dqs_117, %cst_5 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_123 = ttg.convert_layout %dqN_122 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_12[%q_74, %c64_i32], %dqN_123 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_124 = arith.mulf %dqs_118, %cst_5 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_125 = ttg.convert_layout %dqN_124 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_12[%q_74, %c96_i32], %dqN_125 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %curr_m_126 = arith.addi %arg19, %c128_i32 {async_task_id = array<i32: 0, 2, 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_126, %true, %qkT_84, %dv_90, %dpT_99, %dk_103, %dq_107 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %dv_35, %dv_36 = ttng.tmem_load %dv[%curr_m#3] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dvs = tt.reshape %dv_35 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dvs_37 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dvs_38, %dvs_39 = tt.split %dvs_37 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dvs_40 = tt.reshape %dvs_39 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_41 = tt.reshape %dvs_38 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_42 = tt.trans %dvs_41 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_43, %dvs_44 = tt.split %dvs_42 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %3 = arith.truncf %dvs_44 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %4 = arith.truncf %dvs_43 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %dvs_45 = tt.trans %dvs_40 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_46, %dvs_47 = tt.split %dvs_45 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %5 = arith.truncf %dvs_47 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %6 = arith.truncf %dvs_46 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %7 = ttg.convert_layout %4 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv_15[%k_30, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %8 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv_15[%k_30, %c32_i32], %8 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %9 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv_15[%k_30, %c64_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %10 = ttg.convert_layout %5 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv_15[%k_30, %c96_i32], %10 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dk_48, %dk_49 = ttng.tmem_load %dk[%curr_m#5] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dks = tt.reshape %dk_48 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dks_50 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dks_51, %dks_52 = tt.split %dks_50 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dks_53 = tt.reshape %dks_52 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_54 = tt.reshape %dks_51 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_55 = tt.trans %dks_54 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_56, %dks_57 = tt.split %dks_55 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dkN_58 = arith.mulf %dks_57, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dkN_59 = arith.mulf %dks_56, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dks_60 = tt.trans %dks_53 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_61, %dks_62 = tt.split %dks_60 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dkN_63 = arith.mulf %dks_62, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dkN_64 = arith.mulf %dks_61, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %11 = arith.truncf %dkN_59 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %12 = ttg.convert_layout %11 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk_16[%k_30, %c0_i32], %12 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %13 = arith.truncf %dkN_58 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %14 = ttg.convert_layout %13 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk_16[%k_30, %c32_i32], %14 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %15 = arith.truncf %dkN_64 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %16 = ttg.convert_layout %15 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk_16[%k_30, %c64_i32], %16 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %17 = arith.truncf %dkN_63 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %18 = ttg.convert_layout %17 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk_16[%k_30, %c96_i32], %18 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %tile_idx_65 = arith.addi %tile_idx_17, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_65 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.split_mma, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
