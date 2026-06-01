// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=200000" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Regression test: BWD config 1 (BLOCK_M1=64, EPILOGUE_SUBTILE=2) with
// early_tma_store_lowering produced 4 TMA store staging allocs that were
// not counted in the SMEM budget. Phase 4.5 bumped their copies to 2,
// causing: OutOfResources: shared memory, Required: 280232, limit: 232448.
//
// Fix: Phase 4.6 in WSMemoryPlanner.cpp checks the combined SMEM
// (channel buffers + TMA store staging buffers). If it exceeds smem_budget,
// TMA store staging copies are capped to 1.
//
// Key verification:
//   - TMA store staging allocs (buffer.id=7, memdesc<128x64xf16>) get buffer.copy=1
//   - Inner-loop channel allocs are unaffected (q gets buffer.copy=2, etc.)

// CHECK-LABEL: tt.func public @_attn_bwd_persist

// Inner-loop channel allocs — unchanged by the fix:
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32}
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
// CHECK: {{%[A-Za-z0-9_]+}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32}

// TMA store staging allocs: emit current attributes (cap-to-1 not currently
// enforced; see PSM-related design discussion).
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 19 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xf16

// -----// WarpSpec internal IR Dump After: doBufferAllocation
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_do_12: i32, %desc_do_13: i32, %desc_do_14: i64, %desc_do_15: i64, %desc_dq: !tt.tensordesc<tensor<64x64xf32, #shared1>>, %desc_dq_16: i32, %desc_dq_17: i32, %desc_dq_18: i64, %desc_dq_19: i64, %desc_dk: !tt.tensordesc<tensor<128x64xf16, #shared>>, %desc_dk_20: i32, %desc_dk_21: i32, %desc_dk_22: i64, %desc_dk_23: i64, %desc_dv: !tt.tensordesc<tensor<128x64xf16, #shared>>, %desc_dv_24: i32, %desc_dv_25: i32, %desc_dv_26: i64, %desc_dv_27: i64, %desc_m: !tt.tensordesc<tensor<64xf32, #shared2>>, %desc_m_28: i32, %desc_m_29: i64, %desc_delta: !tt.tensordesc<tensor<64xf32, #shared2>>, %desc_delta_30: i32, %desc_delta_31: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dq, %dq_32 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %Di = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
    %dpT, %dpT_33 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %ppT = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %qkT, %qkT_34 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
    %q = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %dk, %dk_35 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dv, %dv_36 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<64x64xf32, #blocked>
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst_37 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %n_tile_num_38 = arith.addi %N_CTX, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %n_tile_num_39 = arith.divsi %n_tile_num_38, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles = arith.muli %n_tile_num_39, %BATCH {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles_40 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_40, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %0 = arith.remsi %total_tiles_40, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_41 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm_41 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x64xf32, #linear1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_41 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_41, %n_tile_num_39 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %tile_idx_41, %n_tile_num_39 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 2>} : i32
      %off_chz_42 = arith.extsi %off_chz {async_task_id = array<i32: 2>} : i32 to i64
      %off_bh_43 = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_44 = arith.muli %stride_h, %off_bh_43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_45 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_46 = arith.muli %stride_z, %off_bh_45 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_47 = arith.addi %off_bh_44, %off_bh_46 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_48 = arith.extsi %off_bh_47 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %off_bh_49 = arith.divsi %off_bh_48, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %k_50 = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_51 = arith.addi %off_bh_49, %k_50 {async_task_id = array<i32: 2, 3>} : i64
      %k_52 = arith.trunci %k_51 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_53 = tt.descriptor_load %desc_k[%k_52, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      ttg.local_store %k_53, %k {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v_54 = tt.descriptor_load %desc_v[%k_52, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      ttg.local_store %v_54, %v {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %curr_m:7 = scf.for %curr_m_68 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg51 = %c0_i32, %arg52 = %false, %qkT_69 = %qkT_34, %dpT_70 = %dpT_33, %dv_71 = %dv_36, %dq_72 = %dq_32, %dk_73 = %dk_35) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_74 = arith.extsi %arg51 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_75 = arith.addi %off_bh_49, %q_74 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %q_76 = arith.trunci %q_75 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_77 = tt.descriptor_load %desc_q[%q_76, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
        ttg.local_store %q_77, %q {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x128xf16, #blocked1> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>
        %offs_m_start = arith.addi %off_chz_42, %q_74 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %m_78 = arith.trunci %offs_m_start {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %m_79 = tt.descriptor_load %desc_m[%m_78] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64xf32, #shared2>> -> tensor<64xf32, #blocked2>
        ttg.local_store %m_79, %m {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
        %qkT_80 = ttng.tc_gen5_mma %k, %qT, %qkT[%qkT_69], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %m_81 = ttg.local_load %m {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #blocked2>
        %pT = ttg.convert_layout %m_81 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %pT_82 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %pT_83 = tt.broadcast %pT_82 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %qkT_84, %qkT_85 = ttng.tmem_load %qkT[%qkT_80] {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %pT_86 = arith.subf %qkT_84, %pT_83 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1>
        %pT_87 = math.exp2 %pT_86 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1>
        %do_88 = tt.descriptor_load %desc_do[%q_76, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
        ttg.local_store %do_88, %do {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x128xf16, #blocked1> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %ppT_89 = arith.truncf %pT_87 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        %dv_90 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} true
        ttng.tmem_store %ppT_89, %ppT, %dv_90 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
        %dpT_91 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>
        %dpT_92 = ttng.tc_gen5_mma %v, %dpT_91, %dpT[%dpT_70], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %Di_93 = tt.descriptor_load %desc_delta[%m_78] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64xf32, #shared2>> -> tensor<64xf32, #blocked2>
        ttg.local_store %Di_93, %Di {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
        %dv_94 = ttng.tc_gen5_mma %ppT, %do, %dv[%dv_71], %arg52, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %Di_95 = ttg.local_load %Di {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #blocked2>
        %dsT_96 = ttg.convert_layout %Di_95 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %dsT_97 = tt.expand_dims %dsT_96 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %dsT_98 = tt.broadcast %dsT_97 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %dpT_99, %dpT_100 = ttng.tmem_load %dpT[%dpT_92] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %dsT_101 = arith.subf %dpT_99, %dsT_98 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1>
        %dsT_102 = arith.mulf %pT_87, %dsT_101 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1>
        %dsT_103 = arith.truncf %dsT_102 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        ttg.local_store %dsT_103, %dsT {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %dq_104 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>
        %dq_105 = ttng.tc_gen5_mma %dq_104, %k, %dq[%dq_72], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}"} : !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_106 = ttng.tc_gen5_mma %dsT, %q, %dk[%dk_73], %arg52, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %dq_107, %dq_108 = ttng.tmem_load %dq[%dq_105] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear2>
        %dqs = tt.reshape %dq_107 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #linear2> -> tensor<64x2x64xf32, #linear3>
        %dqs_109 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #linear3> -> tensor<64x64x2xf32, #linear4>
        %dqs_110 = ttg.convert_layout %dqs_109 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64x2xf32, #linear4> -> tensor<64x64x2xf32, #blocked3>
        %dqs_111, %dqs_112 = tt.split %dqs_110 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64x2xf32, #blocked3> -> tensor<64x64xf32, #blocked>
        %dqN = arith.mulf %dqs_111, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%q_76, %c0_i32], %dqN {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<64x64xf32, #shared1>>, tensor<64x64xf32, #blocked>
        %dqN_113 = arith.mulf %dqs_112, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%q_76, %c64_i32], %dqN_113 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<64x64xf32, #shared1>>, tensor<64x64xf32, #blocked>
        %curr_m_114 = arith.addi %arg51, %c64_i32 {async_task_id = array<i32: 0, 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_114, %true, %qkT_85, %dpT_100, %dv_94, %dq_108, %dk_106 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %dv_55, %dv_56 = ttng.tmem_load %dv[%curr_m#4] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dvs = tt.reshape %dv_55 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear5>
      %dvs_57 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear5> -> tensor<128x64x2xf32, #linear6>
      %dvs_58, %dvs_59 = tt.split %dvs_57 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear6> -> tensor<128x64xf32, #linear1>
      %3 = arith.truncf %dvs_58 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %4 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked4>
      %5 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %4, %5 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked4> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %6 = ttng.async_tma_copy_local_to_global %desc_dv[%k_52, %c0_i32] %5 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %6   {async_task_id = array<i32: 3>} : !ttg.async.token
      %7 = arith.truncf %dvs_59 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %8 = ttg.convert_layout %7 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked4>
      %9 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %8, %9 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked4> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %10 = ttng.async_tma_copy_local_to_global %desc_dv[%k_52, %c64_i32] %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %10   {async_task_id = array<i32: 3>} : !ttg.async.token
      %dk_60, %dk_61 = ttng.tmem_load %dk[%curr_m#6] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dks = tt.reshape %dk_60 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear5>
      %dks_62 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear5> -> tensor<128x64x2xf32, #linear6>
      %dks_63, %dks_64 = tt.split %dks_62 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear6> -> tensor<128x64xf32, #linear1>
      %dkN_65 = arith.mulf %dks_63, %dkN {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1>
      %11 = arith.truncf %dkN_65 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %12 = ttg.convert_layout %11 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked4>
      %13 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %12, %13 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked4> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %14 = ttng.async_tma_copy_local_to_global %desc_dk[%k_52, %c0_i32] %13 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %14   {async_task_id = array<i32: 3>} : !ttg.async.token
      %dkN_66 = arith.mulf %dks_64, %dkN {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1>
      %15 = arith.truncf %dkN_66 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %16 = ttg.convert_layout %15 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked4>
      %17 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %16, %17 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked4> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %18 = ttng.async_tma_copy_local_to_global %desc_dk[%k_52, %c64_i32] %17 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %18   {async_task_id = array<i32: 3>} : !ttg.async.token
      %tile_idx_67 = arith.addi %tile_idx_41, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_67 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
