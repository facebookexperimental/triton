// RUN: triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=2 smem-budget=300000" --tritongpu-pipeline="num-stages=2" --canonicalize | FileCheck %s
//
// Reduced from the post-doTaskIdPropagate dump of the HSTU self-attention
// CLC backward benchmark with runtime masked/unmasked activation branches.
// Both result-producing scf.if operations and their complete then/else bodies
// are assigned to computation task 3. After specialization, the first masked
// tile must be peeled from the unmasked remainder loop in partition3.
//
// CHECK-LABEL: @_hstu_attn_bwd_clc
// CHECK: ttg.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "gemm", "load", "computation"]
// Task 3 is the third physical WS region (partition2; the default region is
// not numbered).
// CHECK: partition2
// The zero-trip guard encloses a straight-line masked first tile. Its element
// predicate and selects survive, but its scalar masked/unmasked IfOps fold.
// CHECK: %[[HAS_FIRST:.*]] = arith.cmpi slt
// CHECK: scf.if %[[HAS_FIRST]]
// CHECK: tt.expand_dims
// CHECK: arith.select
// The remainder starts at lb + step and contains only the unmasked activation
// path: no scalar branch and no per-element activation select.
// CHECK: arith.addi
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK-NOT: arith.select
// CHECK: scf.yield

// -----// WarpSpec internal IR Dump After: doTaskIdPropagate
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 8], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], warp = [[0, 0, 32], [0, 1, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[0, 32, 0], [0, 0, 1]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [0, 4], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_bwd_clc(%Q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %K: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %seq_offsets: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %seq_offsets_q: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %DOut: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DQ: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DK: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DV: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %TILE_IDS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %LOCK: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %stride_qm: i32 {tt.divisibility = 16 : i32}, %stride_qh: i32 {tt.divisibility = 16 : i32}, %stride_kn: i32 {tt.divisibility = 16 : i32}, %stride_kh: i32 {tt.divisibility = 16 : i32}, %stride_vn: i32 {tt.divisibility = 16 : i32}, %stride_vh: i32 {tt.divisibility = 16 : i32}, %stride_dom: i32 {tt.divisibility = 16 : i32}, %stride_doh: i32 {tt.divisibility = 16 : i32}, %stride_dqm: i32 {tt.divisibility = 16 : i32}, %stride_dqh: i32 {tt.divisibility = 16 : i32}, %stride_dkn: i32 {tt.divisibility = 16 : i32}, %stride_dkh: i32 {tt.divisibility = 16 : i32}, %stride_dvn: i32 {tt.divisibility = 16 : i32}, %stride_dvh: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %attn_scale: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %AUTOTUNE_Z: i32, %H: i32, %max_q_len: i32 {tt.divisibility = 16 : i32}, %AUTOTUNE_MAX_SEQ_LEN: i32 {tt.divisibility = 16 : i32}, %DimQ: i32 {tt.divisibility = 16 : i32}, %DimV: i32 {tt.divisibility = 16 : i32}, %max_attn_len: i32 {tt.divisibility = 16 : i32}, %contextual_seq_len: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %cst = arith.constant {async_task_id = array<i32: 0, 3>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_0 = arith.constant {async_task_id = array<i32: 3>} dense<1.000000e+00> : tensor<128x128xf32, #linear>
    %c96_i64 = arith.constant {async_task_id = array<i32: 0>} 96 : i64
    %c64_i64 = arith.constant {async_task_id = array<i32: 0>} 64 : i64
    %c32_i64 = arith.constant {async_task_id = array<i32: 0>} 32 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 1 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %num_n_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %cst_1 = arith.constant {async_task_id = array<i32: 3>} dense<0> : tensor<128x128xi32, #linear>
    %cst_2 = arith.constant {async_task_id = array<i32: 3>} dense<true> : tensor<128x128xi1, #linear>
    %total_kv = tt.addptr %seq_offsets, %Z {async_task_id = array<i32: 2, 3>} : !tt.ptr<i64>, i32
    %total_kv_3 = tt.load %total_kv {async_task_id = array<i32: 2, 3>} : !tt.ptr<i64>
    %total_q = tt.addptr %seq_offsets_q, %Z {async_task_id = array<i32: 0, 2>} : !tt.ptr<i64>, i32
    %total_q_4 = tt.load %total_q {async_task_id = array<i32: 0, 2>} : !tt.ptr<i64>
    %desc_q = arith.muli %H, %DimQ {async_task_id = array<i32: 0, 2, 3>} : i32
    %desc_q_5 = arith.trunci %total_q_4 {async_task_id = array<i32: 0, 2>} : i64 to i32
    %desc_q_6 = arith.extsi %desc_q {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %desc_q_7 = tt.make_tensor_descriptor %Q, [%desc_q_5, %desc_q], [%desc_q_6, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %desc_k = arith.trunci %total_kv_3 {async_task_id = array<i32: 2, 3>} : i64 to i32
    %desc_k_8 = tt.make_tensor_descriptor %K, [%desc_k, %desc_q], [%desc_q_6, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %desc_v = arith.muli %H, %DimV {async_task_id = array<i32: 2, 3>} : i32
    %desc_v_9 = arith.extsi %desc_v {async_task_id = array<i32: 2, 3>} : i32 to i64
    %desc_v_10 = tt.make_tensor_descriptor %V, [%desc_k, %desc_v], [%desc_v_9, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %desc_do = tt.make_tensor_descriptor %DOut, [%desc_q_5, %desc_v], [%desc_v_9, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %desc_dq = tt.make_tensor_descriptor %DQ, [%desc_q_5, %desc_q], [%desc_q_6, %c1_i64] {async_task_id = array<i32: 0>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared1>
    %desc_dk = tt.make_tensor_descriptor %DK, [%desc_k, %desc_q], [%desc_q_6, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %desc_dv = tt.make_tensor_descriptor %DV, [%desc_k, %desc_v], [%desc_v_9, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %num_n_tiles_11 = arith.addi %max_q_len, %num_n_tiles {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_n_tiles_12 = arith.divsi %num_n_tiles_11, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %sched = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %offs_m_13 = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %k = arith.extsi %stride_kh {async_task_id = array<i32: 2>} : i32 to i64
    %v = arith.extsi %stride_vh {async_task_id = array<i32: 2>} : i32 to i64
    %q = arith.extsi %stride_qh {async_task_id = array<i32: 2>} : i32 to i64
    %do = arith.extsi %stride_doh {async_task_id = array<i32: 2>} : i32 to i64
    %dq_trans = tt.splat %alpha {async_task_id = array<i32: 0, 3>} : f32 -> tensor<128x128xf32, #linear>
    %0 = arith.extsi %stride_dqh {async_task_id = array<i32: 0>} : i32 to i64
    %1 = arith.extsi %stride_dvh {async_task_id = array<i32: 3>} : i32 to i64
    %2 = arith.extsi %stride_dkh {async_task_id = array<i32: 3>} : i32 to i64
    %sched._z = scf.while (%sched._valid = %true, %sched._x = %sched) : (i1, i32) -> i32 {
      scf.condition(%sched._valid) {async_task_id = array<i32: 0, 1, 2, 3>} %sched._x : i32
    } do {
    ^bb0(%sched._x: i32):
      %sched_14 = ttng.clc_try_cancel_async {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.async.token
      %tile_id = tt.addptr %TILE_IDS, %sched._x {async_task_id = array<i32: 0, 2, 3>} : !tt.ptr<i32>, i32
      %tile_id_15 = tt.load %tile_id {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i32>
      %off_hz = arith.divsi %tile_id_15, %num_n_tiles_12 {async_task_id = array<i32: 0, 2, 3>} : i32
      %start_n = arith.remsi %tile_id_15, %num_n_tiles_12 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %start_n_16 = arith.muli %start_n, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %off_z = arith.divsi %off_hz, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_h = arith.remsi %off_hz, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_h_17 = arith.extsi %off_h {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %seq_start_kv = tt.addptr %seq_offsets, %off_z {async_task_id = array<i32: 2, 3>} : !tt.ptr<i64>, i32
      %seq_start_kv_18 = tt.load %seq_start_kv {async_task_id = array<i32: 2, 3>} : !tt.ptr<i64>
      %seq_start_q = tt.addptr %seq_offsets_q, %off_z {async_task_id = array<i32: 0, 2>} : !tt.ptr<i64>, i32
      %seq_start_q_19 = tt.load %seq_start_q {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
      %seq_end_q = tt.addptr %seq_start_q, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>, i32
      %seq_end_q_20 = tt.load %seq_end_q {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
      %seq_len_q = arith.subi %seq_end_q_20, %seq_start_q_19 {async_task_id = array<i32: 0, 1, 2, 3>} : i64
      %seq_len_q_21 = arith.trunci %seq_len_q {async_task_id = array<i32: 0, 1, 2, 3>} : i64 to i32
      %k_22 = arith.extsi %start_n_16 {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_23 = arith.addi %seq_start_kv_18, %k_22 {async_task_id = array<i32: 2, 3>} : i64
      %k_24 = arith.trunci %k_23 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_25 = arith.muli %off_h_17, %k {async_task_id = array<i32: 2>} : i64
      %k_26 = arith.trunci %k_25 {async_task_id = array<i32: 2>} : i64 to i32
      %k_27 = tt.descriptor_load %desc_k_8[%k_24, %k_26] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %k_28 = ttg.local_alloc %k_27 {async_task_id = array<i32: 2>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %v_29 = arith.muli %off_h_17, %v {async_task_id = array<i32: 2>} : i64
      %v_30 = arith.trunci %v_29 {async_task_id = array<i32: 2>} : i64 to i32
      %v_31 = tt.descriptor_load %desc_v_10[%k_24, %v_30] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %v_32 = ttg.local_alloc %v_31 {async_task_id = array<i32: 2>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %offs_n = tt.splat %start_n_16 {async_task_id = array<i32: 3>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offs_n_33 = arith.addi %offs_n, %offs_m_13 {async_task_id = array<i32: 3>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %q_34 = arith.muli %off_h_17, %q {async_task_id = array<i32: 2>} : i64
      %q_35 = arith.trunci %q_34 {async_task_id = array<i32: 2>} : i64 to i32
      %apply_mask = arith.addi %start_n_16, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %do_36 = arith.muli %off_h_17, %do {async_task_id = array<i32: 2>} : i64
      %do_37 = arith.trunci %do_36 {async_task_id = array<i32: 2>} : i64 to i32
      %dq_trans_38 = ttg.memdesc_trans %k_28 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %3 = arith.muli %off_h_17, %0 {async_task_id = array<i32: 0>} : i64
      %4 = arith.trunci %3 {async_task_id = array<i32: 0>} : i64 to i32
      %5 = arith.addi %3, %c32_i64 {async_task_id = array<i32: 0>} : i64
      %6 = arith.trunci %5 {async_task_id = array<i32: 0>} : i64 to i32
      %7 = arith.addi %3, %c64_i64 {async_task_id = array<i32: 0>} : i64
      %8 = arith.trunci %7 {async_task_id = array<i32: 0>} : i64 to i32
      %9 = arith.addi %3, %c96_i64 {async_task_id = array<i32: 0>} : i64
      %10 = arith.trunci %9 {async_task_id = array<i32: 0>} : i64 to i32
      %dv, %dv_39 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk, %dk_40 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_41 = ttng.tmem_store %cst, %dk[%dk_40], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_42 = ttng.tmem_store %cst, %dv[%dv_39], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dk_43:3 = scf.for %start_m = %start_n_16 to %seq_len_q_21 step %c128_i32 iter_args(%dv_53 = %false, %dv_54 = %dv_42, %dk_55 = %dk_41) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_m_56 = tt.splat %start_m {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %offs_m_57 = arith.addi %offs_m, %offs_m_56 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %scale = tt.load %attn_scale {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.ptr<f32>
        %q_58 = arith.extsi %start_m {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_59 = arith.extsi %start_m {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_60 = arith.addi %seq_start_q_19, %q_58 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %q_61 = arith.addi %seq_start_q_19, %q_59 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64
        %q_62 = arith.trunci %q_60 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_63 = arith.trunci %q_61 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_64 = tt.descriptor_load %desc_q_7[%q_62, %q_35] {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %q_65 = ttg.local_alloc %q_64 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %q_trans = ttg.memdesc_trans %q_65 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
        %qk_trans, %qk_trans_66 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %qk_trans_67 = ttng.tc_gen5_mma %k_28, %q_trans, %qk_trans[%qk_trans_66], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,2\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %apply_mask_68 = arith.cmpi slt, %start_m, %apply_mask {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
        %qk_trans_69, %qk_trans_70 = ttng.tmem_load %qk_trans[%qk_trans_67] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %21:4 = scf.if %apply_mask_68 -> (tensor<128x128xf32, #linear>, tensor<128x128xbf16, #linear>, tensor<128x128xf32, #linear>, tensor<128x128xi1, #linear>) {
          %valid_mask_trans = tt.expand_dims %offs_m_57 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
          %valid_mask_trans_103 = tt.expand_dims %offs_n_33 {async_task_id = array<i32: 3>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
          %valid_mask_trans_104 = tt.broadcast %valid_mask_trans {async_task_id = array<i32: 3>} : tensor<1x128xi32, #linear> -> tensor<128x128xi32, #linear>
          %valid_mask_trans_105 = tt.broadcast %valid_mask_trans_103 {async_task_id = array<i32: 3>} : tensor<128x1xi32, #linear> -> tensor<128x128xi32, #linear>
          %valid_mask_trans_106 = arith.cmpi eq, %valid_mask_trans_104, %valid_mask_trans_105 {async_task_id = array<i32: 3>} : tensor<128x128xi32, #linear>
          %pos_offs_m_minus_n = arith.subi %valid_mask_trans_104, %valid_mask_trans_105 {async_task_id = array<i32: 3>} : tensor<128x128xi32, #linear>
          %valid_mask_trans_107 = arith.cmpi sgt, %pos_offs_m_minus_n, %cst_1 {async_task_id = array<i32: 3>} : tensor<128x128xi32, #linear>
          %valid_mask_trans_108 = arith.ori %valid_mask_trans_106, %valid_mask_trans_107 {async_task_id = array<i32: 3>} : tensor<128x128xi1, #linear>
          %qk_trans_109 = arith.mulf %qk_trans_69, %dq_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans = arith.negf %qk_trans_109 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_110 = math.exp %sig_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_111 = arith.addf %sig_trans_110, %cst_0 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_112 = tt.extern_elementwise %cst_0, %sig_trans_111 {async_task_id = array<i32: 3>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear>) -> tensor<128x128xf32, #linear>
          %silu_trans = arith.mulf %qk_trans_109, %sig_trans_112 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %silu_trans_113 = tt.splat %scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x128xf32, #linear>
          %silu_trans_114 = arith.mulf %silu_trans, %silu_trans_113 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %act_qk_trans = arith.select %valid_mask_trans_108, %silu_trans_114, %cst {async_task_id = array<i32: 3>} : tensor<128x128xi1, #linear>, tensor<128x128xf32, #linear>
          %act_qk_trans_115 = arith.truncf %act_qk_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
          scf.yield {async_task_id = array<i32: 3>} %qk_trans_109, %act_qk_trans_115, %sig_trans_112, %valid_mask_trans_108 : tensor<128x128xf32, #linear>, tensor<128x128xbf16, #linear>, tensor<128x128xf32, #linear>, tensor<128x128xi1, #linear>
        } else {
          %qk_trans_103 = arith.mulf %qk_trans_69, %dq_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans = arith.negf %qk_trans_103 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_104 = math.exp %sig_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_105 = arith.addf %sig_trans_104, %cst_0 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %sig_trans_106 = tt.extern_elementwise %cst_0, %sig_trans_105 {async_task_id = array<i32: 3>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear>) -> tensor<128x128xf32, #linear>
          %act_qk_trans = arith.mulf %qk_trans_103, %sig_trans_106 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %act_qk_trans_107 = tt.splat %scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x128xf32, #linear>
          %act_qk_trans_108 = arith.mulf %act_qk_trans, %act_qk_trans_107 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %act_qk_trans_109 = arith.truncf %act_qk_trans_108 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
          scf.yield {async_task_id = array<i32: 3>} %qk_trans_103, %act_qk_trans_109, %sig_trans_106, %cst_2 : tensor<128x128xf32, #linear>, tensor<128x128xbf16, #linear>, tensor<128x128xf32, #linear>, tensor<128x128xi1, #linear>
        } {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 0 : i32}
        %22 = ttg.local_alloc %21#1 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %do_71 = tt.descriptor_load %desc_do[%q_62, %do_37] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %do_72 = ttg.local_alloc %do_71 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %dact_qk_trans = ttg.memdesc_trans %do_72 {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
        %dact_qk_trans_73, %dact_qk_trans_74 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dact_qk_trans_75 = ttng.tc_gen5_mma %v_32, %dact_qk_trans, %dact_qk_trans_73[%dact_qk_trans_74], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dv_76 = ttng.tc_gen5_mma %22, %do_72, %dv[%dv_54], %dv_53, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,11\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dact_qk_trans_77, %dact_qk_trans_78 = ttng.tmem_load %dact_qk_trans_73[%dact_qk_trans_75] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %23 = scf.if %apply_mask_68 -> (tensor<128x128xf32, #linear>) {
          %dqk_trans_103 = arith.mulf %dact_qk_trans_77, %21#2 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_104 = arith.subf %cst_0, %21#2 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_105 = arith.mulf %21#0, %dqk_trans_104 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_106 = arith.addf %dqk_trans_105, %cst_0 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_107 = arith.mulf %dqk_trans_103, %dqk_trans_106 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_108 = tt.splat %scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x128xf32, #linear>
          %dqk_trans_109 = arith.mulf %dqk_trans_107, %dqk_trans_108 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_110 = arith.select %21#3, %dqk_trans_109, %cst {async_task_id = array<i32: 3>} : tensor<128x128xi1, #linear>, tensor<128x128xf32, #linear>
          scf.yield {async_task_id = array<i32: 3>} %dqk_trans_110 : tensor<128x128xf32, #linear>
        } else {
          %dqk_trans_103 = arith.mulf %dact_qk_trans_77, %21#2 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_104 = arith.subf %cst_0, %21#2 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_105 = arith.mulf %21#0, %dqk_trans_104 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_106 = arith.addf %dqk_trans_105, %cst_0 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_107 = arith.mulf %dqk_trans_103, %dqk_trans_106 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          %dqk_trans_108 = tt.splat %scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x128xf32, #linear>
          %dqk_trans_109 = arith.mulf %dqk_trans_107, %dqk_trans_108 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
          scf.yield {async_task_id = array<i32: 3>} %dqk_trans_109 : tensor<128x128xf32, #linear>
        } {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32}
        %dqk_trans = arith.truncf %23 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
        %dqk_trans_79 = ttg.local_alloc %dqk_trans {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %dk_80 = ttng.tc_gen5_mma %dqk_trans_79, %q_65, %dk[%dk_55], %dv_53, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,10\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_trans_81, %dq_trans_82 = ttng.tmem_alloc {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dq_trans_83 = ttng.tc_gen5_mma %dq_trans_38, %dqk_trans_79, %dq_trans_81[%dq_trans_82], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_trans_84, %dq_trans_85 = ttng.tmem_load %dq_trans_81[%dq_trans_83] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %dq_trans_86 = arith.mulf %dq_trans_84, %dq_trans {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %dq = tt.trans %dq_trans_86 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
        %dq_87 = arith.truncf %dq {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> to tensor<128x128xbf16, #linear1>
        %dqs = tt.reshape %dq_87 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #linear1> -> tensor<128x2x64xbf16, #linear2>
        %dqs_88 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xbf16, #linear2> -> tensor<128x64x2xbf16, #linear3>
        %dqs_89 = ttg.convert_layout %dqs_88 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64x2xbf16, #linear3> -> tensor<128x64x2xbf16, #linear4>
        %dqs_90, %dqs_91 = tt.split %dqs_89 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64x2xbf16, #linear4> -> tensor<128x64xbf16, #linear5>
        %dqs_92 = tt.reshape %dqs_90 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xbf16, #linear5> -> tensor<128x2x32xbf16, #blocked1>
        %dqs_93 = tt.trans %dqs_92 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xbf16, #blocked1> -> tensor<128x32x2xbf16, #blocked2>
        %dqs_94, %dqs_95 = tt.split %dqs_93 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xbf16, #blocked2> -> tensor<128x32xbf16, #blocked3>
        %dqs_96 = tt.reshape %dqs_91 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xbf16, #linear5> -> tensor<128x2x32xbf16, #blocked1>
        %dqs_97 = tt.trans %dqs_96 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xbf16, #blocked1> -> tensor<128x32x2xbf16, #blocked2>
        %dqs_98, %dqs_99 = tt.split %dqs_97 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xbf16, #blocked2> -> tensor<128x32xbf16, #blocked3>
        %desc_dq_reduce_staging = ttg.local_alloc %dqs_94 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x32xbf16, #blocked3>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %24 = ttng.async_tma_reduce add, %desc_dq[%q_63, %4] %desc_dq_reduce_staging {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dq_reduce_staging   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_100 = ttg.local_alloc %dqs_95 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x32xbf16, #blocked3>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %25 = ttng.async_tma_reduce add, %desc_dq[%q_63, %6] %desc_dq_reduce_staging_100 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dq_reduce_staging_100   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_101 = ttg.local_alloc %dqs_98 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x32xbf16, #blocked3>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %26 = ttng.async_tma_reduce add, %desc_dq[%q_63, %8] %desc_dq_reduce_staging_101 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dq_reduce_staging_101   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_102 = ttg.local_alloc %dqs_99 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x32xbf16, #blocked3>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        %27 = ttng.async_tma_reduce add, %desc_dq[%q_63, %10] %desc_dq_reduce_staging_102 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dq_reduce_staging_102   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %dv_76, %dk_80 : i1, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.loop_unroll_factor = 1 : i32, tt.merge_epilogue_to_computation = true, tt.scheduled_max_stage = 1 : i32}
      %dv_44, %dv_45 = ttng.tmem_load %dv[%dk_43#1] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dk_46, %dk_47 = ttng.tmem_load %dk[%dk_43#2] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dk_48 = arith.mulf %dk_46, %dq_trans {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear>
      %11 = arith.muli %off_h_17, %1 {async_task_id = array<i32: 3>} : i64
      %12 = arith.trunci %11 {async_task_id = array<i32: 3>} : i64 to i32
      %13 = arith.truncf %dv_44 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %14 = ttg.convert_layout %13 {async_task_id = array<i32: 3>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      %desc_dv_staging = ttg.local_alloc %14 {async_task_id = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %15 = ttng.async_tma_copy_local_to_global %desc_dv[%k_24, %12] %desc_dv_staging {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      nvws.tma_store_wait %desc_dv_staging   {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %16 = arith.muli %off_h_17, %2 {async_task_id = array<i32: 3>} : i64
      %17 = arith.trunci %16 {async_task_id = array<i32: 3>} : i64 to i32
      %18 = arith.truncf %dk_48 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %19 = ttg.convert_layout %18 {async_task_id = array<i32: 3>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      %desc_dk_staging = ttg.local_alloc %19 {async_task_id = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %20 = ttng.async_tma_copy_local_to_global %desc_dk[%k_24, %17] %desc_dk_staging {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      nvws.tma_store_wait %desc_dk_staging   {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %sched_49, %sched_50, %sched_51, %sched_52 = ttng.clc_read %sched_14 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %sched_49, %sched_50 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 2 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
