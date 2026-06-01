// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Test case: Memory planner with user-provided tt.autows channel annotations.
//
// Each tc_gen5_mma op carries a tt.autows JSON attribute with a "channels"
// array specifying per-operand buffer assignments. The memory planner reads
// these annotations and pre-assigns buffer.id and buffer.copy accordingly.
//
// Annotations per MMA:
//   qkT: opndA,smem,1,0 / opndB,smem,2,1 / opndD,tmem,1,2
//   dpT: opndA,smem,1,3 / opndB,smem,1,4 / opndD,tmem,1,5
//   dv:  opndA,tmem,1,2 / opndD,tmem,1,7
//   dq:  opndA,smem,1,8 / opndD,tmem,1,5
//   dk:  opndD,tmem,1,10
//
// SMEM buffers:
//   k  (qkT opndA): smem,1,0 → buffer.id=0, copy=1 (pinned)
//   q  (qkT opndB): smem,2,1 → buffer.id=1, copy=2 (pinned)
//   v  (dpT opndA): smem,1,3 → buffer.id=3, copy=1 (pinned)
//   do (dpT opndB): smem,1,4 → buffer.id=4, copy=1 (pinned)
//   dsT (dq opndA): smem,1,8 → buffer.id=8, copy=1 (pinned)
//   dsT: also used by dk (no annotation) → heuristic would assign, but
//        pinned by dq's annotation
//
// TMEM buffers (pre-assigned):
//   qkT opndD: tmem,1,2 (owner)
//   ppT (dv opndA): tmem,1,2 (reuses qkT, offset=0)
//   dpT opndD: tmem,1,5 (owner)
//   dq  opndD: tmem,1,5 (reuses dpT, offset=0)
//   dv  opndD: tmem,1,7
//   dk  opndD: tmem,1,10

// CHECK-LABEL: tt.func public @_attn_bwd_persist
//
// TMEM: dq pre-assigned by annotation (opndD) → buffer.id=5, reuses dpT
// CHECK: {{%[A-Za-z0-9_]+}}, {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32}
//
// SMEM: dsT pinned by annotation (dq opndA) → buffer.id=8, buffer.copy=1
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32}
//
// TMEM: dpT pre-assigned by annotation (opndD) → buffer.id=5 (owner)
// CHECK: {{%[A-Za-z0-9_]+}}, {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32}
//
// TMEM: ppT pre-assigned by annotation (dv opndA) → buffer.id=2, reuses qkT
// CHECK: {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32}
//
// SMEM: do pinned by annotation (dpT opndB) → buffer.id=4, buffer.copy=1
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}
//
// TMEM: qkT pre-assigned by annotation (opndD) → buffer.id=2 (owner)
// CHECK: {{%[A-Za-z0-9_]+}}, {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
//
// SMEM: q pinned by annotation (qkT opndB) → buffer.id=1, buffer.copy=2
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
//
// TMEM: dv pre-assigned by annotation (opndD) → buffer.id=7
// CHECK: {{%[A-Za-z0-9_]+}}, {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
//
// TMEM: dk pre-assigned by annotation (opndD) → buffer.id=10
// CHECK: {{%[A-Za-z0-9_]+}}, {{%[A-Za-z0-9_]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32}
//
// SMEM: v pinned by annotation (dpT opndA) → buffer.id=3, buffer.copy=1
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
//
// SMEM: k pinned by annotation (qkT opndA) → buffer.id=0, buffer.copy=1
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32}

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
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_do_12: i32, %desc_do_13: i32, %desc_do_14: i64, %desc_do_15: i64, %desc_dq: !tt.tensordesc<tensor<128x32xf32, #shared1>>, %desc_dq_16: i32, %desc_dq_17: i32, %desc_dq_18: i64, %desc_dq_19: i64, %desc_dk: !tt.tensordesc<tensor<128x32xf16, #shared2>>, %desc_dk_20: i32, %desc_dk_21: i32, %desc_dk_22: i64, %desc_dk_23: i64, %desc_dv: !tt.tensordesc<tensor<128x32xf16, #shared2>>, %desc_dv_24: i32, %desc_dv_25: i32, %desc_dv_26: i64, %desc_dv_27: i64, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dq, %dq_28 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dpT, %dpT_29 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %ppT = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %qkT, %qkT_30 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dv, %dv_31 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_32 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 32 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 64 : i32
    %c96_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 96 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_33 = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x32xf32, #blocked1>
    %n_tile_num_34 = arith.addi %N_CTX, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %n_tile_num_35 = arith.divsi %n_tile_num_34, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles = arith.muli %n_tile_num_35, %BATCH {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles_36 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_36, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %0 = arith.remsi %total_tiles_36, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_37 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm_37 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x32xf32, #blocked1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_37 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_37, %n_tile_num_35 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %tile_idx_37, %n_tile_num_35 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 3>} : i32
      %off_chz_38 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh_39 = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_40 = arith.muli %stride_h, %off_bh_39 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_41 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_42 = arith.muli %stride_z, %off_bh_41 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_43 = arith.addi %off_bh_40, %off_bh_42 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_44 = arith.extsi %off_bh_43 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %off_bh_45 = arith.divsi %off_bh_44, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i64
      %M_46 = tt.addptr %M, %off_chz_38 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %D_47 = tt.addptr %D, %off_chz_38 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %k_48 = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_49 = arith.addi %off_bh_45, %k_48 {async_task_id = array<i32: 2, 3>} : i64
      %k_50 = arith.trunci %k_49 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_51 = tt.descriptor_load %desc_k[%k_50, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %k_51, %k {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v_52 = tt.descriptor_load %desc_v[%k_50, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %v_52, %v {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %m = tt.splat %M_46 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %Di = tt.splat %D_47 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %dk_53 = ttng.tmem_store %cst, %dk[%dk_32], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_54 = ttng.tmem_store %cst, %dv[%dv_31], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %curr_m_86 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg47 = %c0_i32, %arg48 = %false, %qkT_87 = %qkT_30, %dpT_88 = %dpT_29, %dv_89 = %dv_54, %dq_90 = %dq_28, %dk_91 = %dk_53) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_92 = arith.extsi %arg47 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_93 = arith.addi %off_bh_45, %q_92 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %q_94 = arith.trunci %q_93 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_95 = tt.descriptor_load %desc_q[%q_94, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %q_95, %q {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %offs_m_96 = tt.splat %arg47 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
        %offs_m_97 = arith.addi %offs_m_96, %offs_m {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
        %m_98 = tt.addptr %m, %offs_m_97 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %m_99 = tt.load %m_98 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %qkT_100 = ttng.tc_gen5_mma %k, %qT, %qkT[%qkT_87], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %pT = ttg.convert_layout %m_99 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %pT_101 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %pT_102 = tt.broadcast %pT_101 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qkT_103, %qkT_104 = ttng.tmem_load %qkT[%qkT_100] {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %pT_105 = arith.subf %qkT_103, %pT_102 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %pT_106 = math.exp2 %pT_105 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %do_107 = tt.descriptor_load %desc_do[%q_94, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %do_107, %do {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %ppT_108 = arith.truncf %pT_106 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %dv_109 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} true
        ttng.tmem_store %ppT_108, %ppT, %dv_109 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dpT_110 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %dpT_111 = ttng.tc_gen5_mma %v, %dpT_110, %dpT[%dpT_88], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_112 = tt.addptr %Di, %offs_m_97 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %Di_113 = tt.load %Di_112 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %dv_114 = ttng.tc_gen5_mma %ppT, %do, %dv[%dv_89], %arg48, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dsT_115 = ttg.convert_layout %Di_113 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %dsT_116 = tt.expand_dims %dsT_115 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %dsT_117 = tt.broadcast %dsT_116 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %dpT_118, %dpT_119 = ttng.tmem_load %dpT[%dpT_111] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dsT_120 = arith.subf %dpT_118, %dsT_117 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %dsT_121 = arith.mulf %pT_106, %dsT_120 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %dsT_122 = arith.truncf %dsT_121 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        ttg.local_store %dsT_122, %dsT {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %dq_123 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %dq_124 = ttng.tc_gen5_mma %dq_123, %k, %dq[%dq_90], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}"} : !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_125 = ttng.tc_gen5_mma %dsT, %q, %dk[%dk_91], %arg48, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_126, %dq_127 = ttng.tmem_load %dq[%dq_124] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dqs = tt.reshape %dq_126 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
        %dqs_128 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %dqs_129, %dqs_130 = tt.split %dqs_128 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %dqs_131 = tt.reshape %dqs_129 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_132 = tt.trans %dqs_131 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_133, %dqs_134 = tt.split %dqs_132 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqs_135 = tt.reshape %dqs_130 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_136 = tt.trans %dqs_135 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_137, %dqs_138 = tt.split %dqs_136 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqN = arith.mulf %dqs_133, %cst_33 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_139 = ttg.convert_layout %dqN {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_94, %c0_i32], %dqN_139 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_140 = arith.mulf %dqs_134, %cst_33 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_141 = ttg.convert_layout %dqN_140 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_94, %c32_i32], %dqN_141 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_142 = arith.mulf %dqs_137, %cst_33 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_143 = ttg.convert_layout %dqN_142 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_94, %c64_i32], %dqN_143 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_144 = arith.mulf %dqs_138, %cst_33 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_145 = ttg.convert_layout %dqN_144 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_94, %c96_i32], %dqN_145 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %curr_m_146 = arith.addi %arg47, %c128_i32 {async_task_id = array<i32: 0, 2, 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_146, %true, %qkT_104, %dpT_119, %dv_114, %dq_127, %dk_125 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %dv_55, %dv_56 = ttng.tmem_load %dv[%curr_m#4] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dvs = tt.reshape %dv_55 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dvs_57 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dvs_58, %dvs_59 = tt.split %dvs_57 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dvs_60 = tt.reshape %dvs_59 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_61 = tt.reshape %dvs_58 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_62 = tt.trans %dvs_61 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_63, %dvs_64 = tt.split %dvs_62 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %3 = arith.truncf %dvs_64 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %4 = arith.truncf %dvs_63 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %dvs_65 = tt.trans %dvs_60 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_66, %dvs_67 = tt.split %dvs_65 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %5 = arith.truncf %dvs_67 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %6 = arith.truncf %dvs_66 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %7 = ttg.convert_layout %4 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_50, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %8 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_50, %c32_i32], %8 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %9 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_50, %c64_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %10 = ttg.convert_layout %5 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_50, %c96_i32], %10 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dk_68, %dk_69 = ttng.tmem_load %dk[%curr_m#6] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dks = tt.reshape %dk_68 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dks_70 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dks_71, %dks_72 = tt.split %dks_70 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dks_73 = tt.reshape %dks_72 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_74 = tt.reshape %dks_71 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_75 = tt.trans %dks_74 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_76, %dks_77 = tt.split %dks_75 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dkN_78 = arith.mulf %dks_77, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dkN_79 = arith.mulf %dks_76, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dks_80 = tt.trans %dks_73 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_81, %dks_82 = tt.split %dks_80 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dkN_83 = arith.mulf %dks_82, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %dkN_84 = arith.mulf %dks_81, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1>
      %11 = arith.truncf %dkN_79 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %12 = ttg.convert_layout %11 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_50, %c0_i32], %12 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %13 = arith.truncf %dkN_78 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %14 = ttg.convert_layout %13 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_50, %c32_i32], %14 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %15 = arith.truncf %dkN_84 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %16 = ttg.convert_layout %15 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_50, %c64_i32], %16 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %17 = arith.truncf %dkN_83 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %18 = ttg.convert_layout %17 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_50, %c96_i32], %18 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %tile_idx_85 = arith.addi %tile_idx_37, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_85 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
