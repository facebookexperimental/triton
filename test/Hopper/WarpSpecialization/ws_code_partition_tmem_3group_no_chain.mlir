// RUN: not --crash triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1" 2>&1 | FileCheck %s
//
// A TMEM reuse group with >= 3 buffers must have a unique dependency-chain
// order over the shared slot, or code partitioning must reject it at compile
// time. Here {dpT, dsT, dq} share one TMEM slot (buffer.id = 8):
//   dpT (gemm, task1)  -> dsT (computation tmem_store, task3) -> dq (gemm, task1)
// dsT is read by the dk MMA (task1). The dq MMA (which overwrites the shared
// slot) is emitted BEFORE dk reads dsT, so dsT and dq are incomparable and
// orderReuseGroupChain finds no unique order. Without the guard this would fall
// to the same-block path and emit a WAR making dq wait on dk's release - but dk
// runs after dq in the same partition -> deadlock. The guard turns it into a
// compile-time error instead.
//
// CHECK: TMEM reuse group with >= 3 buffers has no unique dependency-chain order

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 32], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 32, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %sm_scale: f32, %desc_do: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dq: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %desc_dk: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_dv: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dq, %dq_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dq")
    %dsT = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("dsT")
    %dpT, %dpT_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dpT")
    %dsT_t, %dsT_t_tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dsT")
    %dv = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %qkT, %qkT_2 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %dv_3, %dv_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_5 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c128_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 128 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 1 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x32xf32, #blocked>
    %cst_6 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
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
    %desc_q_11 = tt.make_tensor_descriptor %desc_q, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_do_12 = tt.make_tensor_descriptor %desc_do, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_dq_13 = tt.make_tensor_descriptor %desc_dq, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 0>} : !tt.ptr<f32>, !tt.tensordesc<128x32xf32, #shared1>
    %desc_v_14 = tt.make_tensor_descriptor %desc_v, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_k_15 = tt.make_tensor_descriptor %desc_k, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_dv_16 = tt.make_tensor_descriptor %desc_dv, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<128x32xf16, #shared2>
    %desc_dk_17 = tt.make_tensor_descriptor %desc_dk, [%y_dim_10, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<128x32xf16, #shared2>
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x32xf32, #blocked>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_18 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_18, %n_tile_num_8 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %tile_idx_18, %n_tile_num_8 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 3>} : i32
      %off_chz_19 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh_20 = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_21 = arith.muli %stride_h, %off_bh_20 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_22 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_23 = arith.muli %stride_z, %off_bh_22 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_24 = arith.addi %off_bh_21, %off_bh_23 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_25 = arith.extsi %off_bh_24 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %off_bh_26 = arith.divsi %off_bh_25, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i64
      %M_27 = tt.addptr %M, %off_chz_19 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %D_28 = tt.addptr %D, %off_chz_19 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %k_29 = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_30 = arith.addi %off_bh_26, %k_29 {async_task_id = array<i32: 2, 3>} : i64
      %k_31 = arith.trunci %k_30 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_32 = tt.descriptor_load %desc_k_15[%k_31, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %k_32, %k {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v_33 = tt.descriptor_load %desc_v_14[%k_31, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked3>
      ttg.local_store %v_33, %v {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %m = tt.splat %M_27 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %Di = tt.splat %D_28 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %dk_34 = ttng.tmem_store %cst_6, %dk[%dk_5], %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 9>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_35 = ttng.tmem_store %cst_6, %dv_3[%dv_4], %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 7>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %curr_m_67 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg19 = %c0_i32, %arg20 = %false, %qkT_68 = %qkT_2, %dv_69 = %dv_35, %dpT_70 = %dpT_1, %dk_71 = %dk_34, %dq_72 = %dq_0) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_73 = arith.extsi %arg19 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_74 = arith.addi %off_bh_26, %q_73 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
        %q_75 = arith.trunci %q_74 {async_task_id = array<i32: 0, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_76 = tt.descriptor_load %desc_q_11[%q_75, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %q_76, %q {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %offs_m_77 = tt.splat %arg19 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
        %offs_m_78 = arith.addi %offs_m_77, %offs_m {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
        %m_79 = tt.addptr %m, %offs_m_78 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %m_80 = tt.load %m_79 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %qkT_81 = ttng.tc_gen5_mma %k, %qT, %qkT[%qkT_68], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %pT = ttg.convert_layout %m_80 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %pT_82 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xf32, #blocked1>
        %pT_83 = tt.broadcast %pT_82 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
        %qkT_84, %qkT_85 = ttng.tmem_load %qkT[%qkT_81] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        %pT_86 = arith.subf %qkT_84, %pT_83 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
        %pT_87 = math.exp2 %pT_86 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
        %do_88 = tt.descriptor_load %desc_do_12[%q_75, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked3>
        ttg.local_store %do_88, %do {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %ppT = arith.truncf %pT_87 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
        %dv_89 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %ppT, %dv, %dv_89 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dv_90 = ttng.tc_gen5_mma %dv, %do, %dv_3[%dv_69], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 7>, tmem.start = array<i32: 8>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_91 = tt.addptr %Di, %offs_m_78 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %Di_92 = tt.load %Di_91 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %dpT_93 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable> loc("dpT")
        %dpT_94 = ttng.tc_gen5_mma %v, %dpT_93, %dpT[%dpT_70], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc("dpT")
        %dsT_95 = ttg.convert_layout %Di_92 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc("dsT")
        %dsT_96 = tt.expand_dims %dsT_95 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xf32, #blocked1> loc("dsT")
        %dsT_97 = tt.broadcast %dsT_96 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked1> -> tensor<128x128xf32, #blocked1> loc("dsT")
        %dpT_98, %dpT_99 = ttng.tmem_load %dpT[%dpT_94] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1> loc("dpT")
        %dsT_100 = arith.subf %dpT_98, %dsT_97 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> loc("dsT")
        %dsT_101 = arith.mulf %pT_87, %dsT_100 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> loc("dsT")
        %dsT_102 = arith.truncf %dsT_101 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1> loc("dsT")
        ttg.local_store %dsT_102, %dsT {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("dsT")
        // computation (task3) materializes dsT into the shared buf8 TMEM slot.
        ttng.tmem_store %dsT_102, %dsT_t, %true {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> loc("dsT")
        // HAZARD: dq (writes buf8) emitted BEFORE dk reads dsT from buf8 -> the
        // {dpT,dsT,dq} chain has no unique order (dsT and dq are incomparable).
        %dq_104 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable> loc("dq")
        %dq_105 = ttng.tc_gen5_mma %dq_104, %k, %dq[%dq_72], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc("dq")
        %dk_103 = ttng.tc_gen5_mma %dsT_t, %q, %dk[%dk_71], %arg20, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 10>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_106, %dq_107 = ttng.tmem_load %dq[%dq_105] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1> loc("dq")
        %dqs = tt.reshape %dq_106 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
        %dqs_108 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %dqs_109, %dqs_110 = tt.split %dqs_108 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %dqs_111 = tt.reshape %dqs_109 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_112 = tt.trans %dqs_111 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_113, %dqs_114 = tt.split %dqs_112 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
        %dqs_115 = tt.reshape %dqs_110 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_116 = tt.trans %dqs_115 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_117, %dqs_118 = tt.split %dqs_116 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
        %dqN = arith.mulf %dqs_113, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
        %dqN_119 = ttg.convert_layout %dqN {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_13[%q_75, %c0_i32], %dqN_119 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, tensor<128x32xf32, #blocked9>
        %dqN_120 = arith.mulf %dqs_114, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
        %dqN_121 = ttg.convert_layout %dqN_120 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_13[%q_75, %c0_i32], %dqN_121 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, tensor<128x32xf32, #blocked9>
        %dqN_122 = arith.mulf %dqs_117, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
        %dqN_123 = ttg.convert_layout %dqN_122 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_13[%q_75, %c0_i32], %dqN_123 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, tensor<128x32xf32, #blocked9>
        %dqN_124 = arith.mulf %dqs_118, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
        %dqN_125 = ttg.convert_layout %dqN_124 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq_13[%q_75, %c0_i32], %dqN_125 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, tensor<128x32xf32, #blocked9>
        %curr_m_126 = arith.addi %arg19, %c128_i32 {async_task_id = array<i32: 0, 2, 3>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_126, %true, %qkT_85, %dv_90, %dpT_99, %dk_103, %dq_107 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %dv_36, %dv_37 = ttng.tmem_load %dv_3[%curr_m#3] {async_task_id = array<i32: 3>, tmem.end = array<i32: 8>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dvs = tt.reshape %dv_36 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %dvs_38 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dvs_39, %dvs_40 = tt.split %dvs_38 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dvs_41 = tt.reshape %dvs_40 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_42 = tt.reshape %dvs_39 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_43 = tt.trans %dvs_42 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_44, %dvs_45 = tt.split %dvs_43 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %3 = arith.truncf %dvs_45 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %4 = arith.truncf %dvs_44 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %dvs_46 = tt.trans %dvs_41 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_47, %dvs_48 = tt.split %dvs_46 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %5 = arith.truncf %dvs_48 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %6 = arith.truncf %dvs_47 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %7 = ttg.convert_layout %4 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dv_16[%k_31, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %8 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dv_16[%k_31, %c0_i32], %8 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %9 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dv_16[%k_31, %c0_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %10 = ttg.convert_layout %5 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dv_16[%k_31, %c0_i32], %10 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %dk_49, %dk_50 = ttng.tmem_load %dk[%curr_m#5] {async_task_id = array<i32: 3>, tmem.end = array<i32: 10>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dks = tt.reshape %dk_49 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %dks_51 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dks_52, %dks_53 = tt.split %dks_51 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dks_54 = tt.reshape %dks_53 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_55 = tt.reshape %dks_52 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_56 = tt.trans %dks_55 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_57, %dks_58 = tt.split %dks_56 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %dkN_59 = arith.mulf %dks_58, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
      %dkN_60 = arith.mulf %dks_57, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
      %dks_61 = tt.trans %dks_54 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_62, %dks_63 = tt.split %dks_61 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %dkN_64 = arith.mulf %dks_63, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
      %dkN_65 = arith.mulf %dks_62, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
      %11 = arith.truncf %dkN_60 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %12 = ttg.convert_layout %11 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dk_17[%k_31, %c0_i32], %12 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %13 = arith.truncf %dkN_59 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %14 = ttg.convert_layout %13 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dk_17[%k_31, %c0_i32], %14 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %15 = arith.truncf %dkN_65 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %16 = ttg.convert_layout %15 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dk_17[%k_31, %c0_i32], %16 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %17 = arith.truncf %dkN_64 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %18 = ttg.convert_layout %17 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #blocked9>
      tt.descriptor_store %desc_dk_17[%k_31, %c0_i32], %18 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x32xf16, #shared2>, tensor<128x32xf16, #blocked9>
      %tile_idx_66 = arith.addi %tile_idx_18, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_66 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
