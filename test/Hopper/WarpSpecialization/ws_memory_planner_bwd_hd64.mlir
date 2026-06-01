// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Test case: FA BWD with HEAD_DIM=64 — dq reuses a larger tmem buffer at a col offset.
//
// When HEAD_DIM=64, dk/dv/dq are 128x64 while qkT/dpT remain 128x128.
// The memory planner assigns dq as a sub-allocation within one of the
// 128x128 tmem buffers (buffer ID and offset may vary).
//
// CHECK-LABEL: tt.func public @_attn_bwd
// CHECK: %dq, %dq_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32, buffer.offset = {{[0-9]+}} : i32}
// CHECK: %dpT, %dpT_1 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 8 : i32}
// CHECK: %dv = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32}
// CHECK: %qkT, %qkT_2 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 7 : i32}
// CHECK: %dv_3, %dv_4 = ttng.tmem_alloc {{{.*}}buffer.copy = 2 : i32, buffer.id = 6 : i32}
// CHECK: %dk, %dk_5 = ttng.tmem_alloc {{{.*}}buffer.copy = 2 : i32, buffer.id = 5 : i32}

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 32], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 32, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 2, 16], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %sm_scale: f32, %desc_do: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dq: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %desc_dk: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dv: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dq, %dq_0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dpT, %dpT_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dv = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %qkT, %qkT_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %dv_3, %dv_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %c48_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 48 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 32 : i32
    %c16_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 16 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c64_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 1 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x16xf32, #blocked>
    %cst_6 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x64xf32, #blocked1>
    %n_tile_num_7 = arith.addi %N_CTX, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %n_tile_num_8 = arith.divsi %n_tile_num_7, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles = arith.muli %n_tile_num_8, %BATCH {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles_9 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_9, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %0 = arith.remsi %total_tiles_9, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_18 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm_18 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %y_dim = arith.muli %BATCH, %H {async_task_id = array<i32: 0, 2, 3>} : i32
    %y_dim_10 = arith.muli %y_dim, %N_CTX {async_task_id = array<i32: 0, 2, 3>} : i32
    %desc_q_11 = tt.make_tensor_descriptor %desc_q, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_do_12 = tt.make_tensor_descriptor %desc_do, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_dq_13 = tt.make_tensor_descriptor %desc_dq, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 0>} : !tt.ptr<f32>, !tt.tensordesc<tensor<128x16xf32, #shared1>>
    %desc_v_14 = tt.make_tensor_descriptor %desc_v, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_k_15 = tt.make_tensor_descriptor %desc_k, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_dv_16 = tt.make_tensor_descriptor %desc_dv, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x16xf16, #shared2>>
    %desc_dk_17 = tt.make_tensor_descriptor %desc_dk, [%y_dim_10, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x16xf16, #shared2>>
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x16xf32, #blocked>
    %tile_idx = scf.for %arg16 = %c0_i32 to %2 step %c1_i32 iter_args(%arg17 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %arg17, %n_tile_num_8 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %arg17, %n_tile_num_8 {async_task_id = array<i32: 0, 2, 3>} : i32
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
      %k_31 = tt.descriptor_load %desc_k_15[%k_30, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %k_31, %k {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %v_32 = tt.descriptor_load %desc_v_14[%k_30, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %v_32, %v {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %m = tt.splat %M_26 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %Di = tt.splat %D_27 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %dk_33 = ttng.tmem_store %cst_6, %dk[%dk_5], %true {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_34 = ttng.tmem_store %cst_6, %dv_3[%dv_4], %true {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %arg18 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg19 = %c0_i32, %arg20 = %false, %arg21 = %qkT_2, %arg22 = %dv_34, %arg23 = %dpT_1, %arg24 = %dk_33, %arg25 = %dq_0) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_66 = arith.extsi %arg19 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_67 = arith.addi %off_bh_25, %q_66 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %q_68 = arith.trunci %q_67 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_69 = tt.descriptor_load %desc_q_11[%q_68, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
        ttg.local_store %q_69, %q {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>
        %offs_m_70 = tt.splat %arg19 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
        %offs_m_71 = arith.addi %offs_m_70, %offs_m {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
        %m_72 = tt.addptr %m, %offs_m_71 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %m_73 = tt.load %m_72 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %qkT_74 = ttng.tc_gen5_mma %k, %qT, %qkT[%arg21], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %pT = ttg.convert_layout %m_73 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %pT_75 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xf32, #blocked4>
        %pT_76 = tt.broadcast %pT_75 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked4> -> tensor<128x128xf32, #blocked4>
        %qkT_77, %qkT_78 = ttng.tmem_load %qkT[%qkT_74] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked4>
        %pT_79 = arith.subf %qkT_77, %pT_76 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4>
        %pT_80 = math.exp2 %pT_79 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4>
        %do_81 = tt.descriptor_load %desc_do_12[%q_68, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
        ttg.local_store %do_81, %do {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %ppT = arith.truncf %pT_80 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4> to tensor<128x128xf16, #blocked4>
        %dv_82 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %ppT, %dv, %dv_82 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked4> -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
        %dv_83 = ttng.tc_gen5_mma %dv, %do, %dv_3[%arg22], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_84 = tt.addptr %Di, %offs_m_71 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %Di_85 = tt.load %Di_84 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %dpT_86 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>
        %dpT_87 = ttng.tc_gen5_mma %v, %dpT_86, %dpT[%arg23], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dsT_88 = ttg.convert_layout %Di_85 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %dsT_89 = tt.expand_dims %dsT_88 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xf32, #blocked4>
        %dsT_90 = tt.broadcast %dsT_89 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked4> -> tensor<128x128xf32, #blocked4>
        %dpT_91, %dpT_92 = ttng.tmem_load %dpT[%dpT_87] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked4>
        %dsT_93 = arith.subf %dpT_91, %dsT_90 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4>
        %dsT_94 = arith.mulf %pT_80, %dsT_93 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4>
        %dsT_95 = arith.truncf %dsT_94 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked4> to tensor<128x128xf16, #blocked4>
        ttg.local_store %dsT_95, %dsT {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked4> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %dk_96 = ttng.tc_gen5_mma %dsT, %q, %dk[%arg24], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_97 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %dq_98 = ttng.tc_gen5_mma %dq_97, %k, %dq[%arg25], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_99, %dq_100 = ttng.tmem_load %dq[%dq_98] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
        %dqs = tt.reshape %dq_99 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked1> -> tensor<128x2x32xf32, #blocked5>
        %dqs_101 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x32x2xf32, #blocked6>
        %dqs_102, %dqs_103 = tt.split %dqs_101 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked6> -> tensor<128x32xf32, #blocked7>
        %dqs_104 = tt.reshape %dqs_102 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
        %dqs_105 = tt.trans %dqs_104 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
        %dqs_106, %dqs_107 = tt.split %dqs_105 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
        %dqs_108 = tt.reshape %dqs_103 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
        %dqs_109 = tt.trans %dqs_108 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
        %dqs_110, %dqs_111 = tt.split %dqs_109 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
        %dqN = arith.mulf %dqs_106, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %dqN_112 = ttg.convert_layout %dqN {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #blocked10>
        tt.descriptor_reduce add, %desc_dq_13[%q_68, %c0_i32], %dqN_112 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x16xf32, #shared1>>, tensor<128x16xf32, #blocked10>
        %dqN_113 = arith.mulf %dqs_107, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %dqN_114 = ttg.convert_layout %dqN_113 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #blocked10>
        tt.descriptor_reduce add, %desc_dq_13[%q_68, %c16_i32], %dqN_114 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x16xf32, #shared1>>, tensor<128x16xf32, #blocked10>
        %dqN_115 = arith.mulf %dqs_110, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %dqN_116 = ttg.convert_layout %dqN_115 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #blocked10>
        tt.descriptor_reduce add, %desc_dq_13[%q_68, %c32_i32], %dqN_116 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x16xf32, #shared1>>, tensor<128x16xf32, #blocked10>
        %dqN_117 = arith.mulf %dqs_111, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %dqN_118 = ttg.convert_layout %dqN_117 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #blocked10>
        tt.descriptor_reduce add, %desc_dq_13[%q_68, %c48_i32], %dqN_118 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x16xf32, #shared1>>, tensor<128x16xf32, #blocked10>
        %curr_m_119 = arith.addi %arg19, %c128_i32 {async_task_id = array<i32: 0, 2, 3>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_119, %true, %qkT_78, %dv_83, %dpT_92, %dk_96, %dq_100 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %dv_35, %dv_36 = ttng.tmem_load %dv_3[%curr_m#3] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
      %dvs = tt.reshape %dv_35 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked1> -> tensor<128x2x32xf32, #blocked5>
      %dvs_37 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x32x2xf32, #blocked6>
      %dvs_38, %dvs_39 = tt.split %dvs_37 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked6> -> tensor<128x32xf32, #blocked7>
      %dvs_40 = tt.reshape %dvs_39 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
      %dvs_41 = tt.reshape %dvs_38 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
      %dvs_42 = tt.trans %dvs_41 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
      %dvs_43, %dvs_44 = tt.split %dvs_42 {async_task_id = array<i32: 3>} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
      %3 = arith.truncf %dvs_44 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %4 = arith.truncf %dvs_43 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %dvs_45 = tt.trans %dvs_40 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
      %dvs_46, %dvs_47 = tt.split %dvs_45 {async_task_id = array<i32: 3>} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
      %5 = arith.truncf %dvs_47 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %6 = arith.truncf %dvs_46 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %7 = ttg.convert_layout %4 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dv_16[%k_30, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %8 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dv_16[%k_30, %c16_i32], %8 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %9 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dv_16[%k_30, %c32_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %10 = ttg.convert_layout %5 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dv_16[%k_30, %c48_i32], %10 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %dk_48, %dk_49 = ttng.tmem_load %dk[%curr_m#5] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
      %dks = tt.reshape %dk_48 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked1> -> tensor<128x2x32xf32, #blocked5>
      %dks_50 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x32x2xf32, #blocked6>
      %dks_51, %dks_52 = tt.split %dks_50 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked6> -> tensor<128x32xf32, #blocked7>
      %dks_53 = tt.reshape %dks_52 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
      %dks_54 = tt.reshape %dks_51 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked7> -> tensor<128x2x16xf32, #blocked8>
      %dks_55 = tt.trans %dks_54 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
      %dks_56, %dks_57 = tt.split %dks_55 {async_task_id = array<i32: 3>} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
      %dkN_58 = arith.mulf %dks_57, %dkN {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked>
      %dkN_59 = arith.mulf %dks_56, %dkN {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked>
      %dks_60 = tt.trans %dks_53 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked8> -> tensor<128x16x2xf32, #blocked9>
      %dks_61, %dks_62 = tt.split %dks_60 {async_task_id = array<i32: 3>} : tensor<128x16x2xf32, #blocked9> -> tensor<128x16xf32, #blocked>
      %dkN_63 = arith.mulf %dks_62, %dkN {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked>
      %dkN_64 = arith.mulf %dks_61, %dkN {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked>
      %11 = arith.truncf %dkN_59 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %12 = ttg.convert_layout %11 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dk_17[%k_30, %c0_i32], %12 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %13 = arith.truncf %dkN_58 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %14 = ttg.convert_layout %13 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dk_17[%k_30, %c16_i32], %14 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %15 = arith.truncf %dkN_64 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %16 = ttg.convert_layout %15 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dk_17[%k_30, %c32_i32], %16 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %17 = arith.truncf %dkN_63 {async_task_id = array<i32: 3>} : tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
      %18 = ttg.convert_layout %17 {async_task_id = array<i32: 3>} : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #blocked10>
      tt.descriptor_store %desc_dk_17[%k_30, %c48_i32], %18 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x16xf16, #shared2>>, tensor<128x16xf16, #blocked10>
      %tile_idx_65 = arith.addi %arg17, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_65 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
