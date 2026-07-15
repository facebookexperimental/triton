// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3 post-channel-creation=1" | FileCheck %s

// Regression test for the Operand-D back-edge channel bug.
// Pattern: FA fwd accumulator with all three triggering conditions:
//   1. Out-of-loop tmem_store initializing the operand-D alloc (task 0).
//   2. In-body order on the same alloc:
//        tmem_load (task 0) -> tmem_store (task 0) -> tc_gen5_mma (task 1).
//   3. tmem_load and tmem_store both in correction partition (task 0),
//      gen5 in MMA partition (task 1).
//
// Bug: handleOperandD's pre-loop scan (added in d35e994174) seeds
// currentProds with the out-of-loop init store, which short-circuits
// the deferred channel-creation path that emits the back-edge channel
// (gen5 -> in-body tmem_load). Without that channel, the in-body
// tmem_load has no synchronization with the previous iteration's MMA
// commit and reads stale data on Blackwell.
//
// The fix-up signal: each in-body tmem_load on the 128x128xf32
// accumulator must carry a tmem.end attribute pointing to the
// back-edge channel id.

// CHECK-LABEL: @_attn_fwd_persist
// In-body tmem_load on the accumulator with loop.cluster = 4 / stage = 1
// must carry a tmem.end attribute (back-edge channel exists).
// CHECK: ttng.tmem_load %{{.+}}[] {loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.end = array<i32: {{.+}}>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32
// In-body tmem_load on the accumulator with loop.cluster = 1 / stage = 2
// must also carry a tmem.end attribute.
// CHECK: ttng.tmem_load %{{.+}}[] {loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.end = array<i32: {{.+}}>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {ttg.partition = array<i32: 0, 4, 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_0 = arith.constant {ttg.partition = array<i32: 0, 4, 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_1 = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_2 = arith.constant {ttg.partition = array<i32: 4, 5>} 1.44269502 : f32
    %c256_i32 = arith.constant {ttg.partition = array<i32: 0, 2, 3>} 256 : i32
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 0 : i32
    %c1_i64 = arith.constant {ttg.partition = array<i32: 2, 3>} 1 : i64
    %c128_i64 = arith.constant {ttg.partition = array<i32: 2, 3>} 128 : i64
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 128 : i32
    %c1024_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 1024 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 1 : i32
    %n_tile_num = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 4 : i32
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 1>} false
    %_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %acc_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %acc_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha_0, %alpha_0_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alpha_1, %alpha_1_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_1, %qk_1_5 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_0, %qk_0_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m_ij_1, %m_ij_1_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_0, %l_i0_0_8 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_ij_0, %m_ij_0_9 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_1, %l_i0_1_10 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_1_11, %acc_1_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_0_13, %acc_0_14 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q0_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %q0_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %prog_id = tt.get_program_id x {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %num_progs = tt.get_num_programs x {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles = arith.muli %Z, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles_15 = arith.muli %total_tiles, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_15, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %0 = arith.remsi %total_tiles_15, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_27 = arith.addi %tiles_per_sm, %c1_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm_27 : i32
    } else {
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm : i32
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>}
    %desc_q_16 = arith.muli %Z, %H {ttg.partition = array<i32: 2, 3>} : i32
    %desc_q_17 = arith.muli %desc_q_16, %c1024_i32 {ttg.partition = array<i32: 2, 3>} : i32
    %desc_q_18 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_19 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_20 = tt.make_tensor_descriptor %desc_k, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_21 = tt.make_tensor_descriptor %desc_v, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_22 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_23 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c1024_i32 {ttg.partition = array<i32: 2, 3>} : i32
    %offs_m0 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m0_24 = tt.make_range {ttg.partition = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst_2 {ttg.partition = array<i32: 4, 5>} : f32
    %m_ij = tt.splat %qk_scale {ttg.partition = array<i32: 5>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_ij_25 = tt.splat %qk_scale {ttg.partition = array<i32: 4>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale {ttg.partition = array<i32: 5>} : f32 -> tensor<128x128xf32, #linear>
    %qk_26 = tt.splat %qk_scale {ttg.partition = array<i32: 4>} : f32 -> tensor<128x128xf32, #linear>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_27 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_27, %n_tile_num {ttg.partition = array<i32: 0, 2, 3>} : i32
      %off_hz = arith.divsi %tile_idx_27, %n_tile_num {ttg.partition = array<i32: 0, 2, 3>} : i32
      %off_z = arith.divsi %off_hz, %H {ttg.partition = array<i32: 2, 3>} : i32
      %off_h = arith.remsi %off_hz, %H {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_28 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_29 = arith.muli %off_h, %c1024_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_30 = arith.addi %offset_y_28, %offset_y_29 {ttg.partition = array<i32: 2, 3>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_31 = arith.addi %offset_y_30, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %3 = arith.addi %qo_offset_y_31, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %q0 = arith.addi %qo_offset_y_31, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_m0_32 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_33 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_34 = arith.addi %offs_m0_32, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m0_35 = arith.addi %offs_m0_33, %offs_m0_24 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %q0_36 = tt.descriptor_load %desc_q_18[%qo_offset_y_31, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %q0_37 = tt.descriptor_load %desc_q_19[%q0, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      ttg.local_store %q0_36, %q0_0 {ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %q0_37, %q0_1 {ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %acc = ttg.convert_layout %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_38 = ttng.tmem_store %acc, %acc_0_13[%acc_0_14], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_39 = ttg.convert_layout %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %acc_40 = ttng.tmem_store %acc_39, %acc_1_11[%acc_1_12], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_81 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%offset_y_82 = %offset_y_30, %arg12 = %false, %arg13 = %cst, %arg14 = %cst_0, %qk_0_83 = %qk_0_6, %acc_84 = %acc_38, %arg17 = %cst, %arg18 = %cst_0, %qk_1_85 = %qk_1_5, %acc_86 = %acc_40) -> (i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_87 = tt.descriptor_load %desc_k_20[%offset_y_82, %c0_i32] {ttg.partition = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        %v_88 = tt.descriptor_load %desc_v_21[%offset_y_82, %c0_i32] {ttg.partition = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        ttg.local_store %k_87, %k {ttg.partition = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_89 = ttg.memdesc_trans %k {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        ttg.local_store %v_88, %v {ttg.partition = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_90 = ttng.tc_gen5_mma %q0_0, %k_89, %qk_0[%qk_0_83], %false, %true {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_91 = ttng.tc_gen5_mma %q0_1, %k_89, %qk_1[%qk_1_85], %false, %true {ttg.partition = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_92, %qk_93 = ttng.tmem_load %qk_0[%qk_90] {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_94 = ttg.convert_layout %qk_92 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %qk_95, %qk_96 = ttng.tmem_load %qk_1[%qk_91] {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_97 = ttg.convert_layout %qk_95 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %m_ij_98 = "tt.reduce"(%qk_94) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_164: f32, %m_ij_165: f32):
          %m_ij_166 = arith.maxnumf %m_ij_164, %m_ij_165 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_166 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_99 = "tt.reduce"(%qk_97) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_164: f32, %m_ij_165: f32):
          %m_ij_166 = arith.maxnumf %m_ij_164, %m_ij_165 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_166 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_100 = arith.mulf %m_ij_98, %m_ij {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_101 = arith.mulf %m_ij_99, %m_ij_25 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_102 = arith.maxnumf %arg14, %m_ij_100 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_103 = arith.maxnumf %arg18, %m_ij_101 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_104 = arith.mulf %qk_94, %qk {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %qk_105 = arith.mulf %qk_97, %qk_26 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear>
        %qk_106 = tt.expand_dims %m_ij_102 {ttg.partition = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_107 = tt.expand_dims %m_ij_103 {ttg.partition = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_108 = tt.broadcast %qk_106 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_109 = tt.broadcast %qk_107 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_110 = arith.subf %qk_104, %qk_108 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %qk_111 = arith.subf %qk_105, %qk_109 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_110 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %p_112 = math.exp2 %qk_111 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear>
        %alpha = arith.subf %arg14, %m_ij_102 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_113 = arith.subf %arg18, %m_ij_103 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_114 = math.exp2 %alpha {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_115 = tt.expand_dims %alpha_114 {ttg.partition = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        ttng.tmem_store %alpha_115, %alpha_1, %true {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %alpha_116 = math.exp2 %alpha_113 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_117 = tt.expand_dims %alpha_116 {ttg.partition = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        ttng.tmem_store %alpha_117, %alpha_0, %true {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_164: f32, %l_ij_165: f32):
          %l_ij_166 = arith.addf %l_ij_164, %l_ij_165 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_166 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij_118 = "tt.reduce"(%p_112) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_164: f32, %l_ij_165: f32):
          %l_ij_166 = arith.addf %l_ij_164, %l_ij_165 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_166 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_119, %acc_120 = ttng.tmem_load %acc_0_13[%acc_84] {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %acc_121 = ttg.convert_layout %acc_119 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %acc_122, %acc_123 = ttng.tmem_load %acc_1_11[%acc_86] {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %acc_124 = ttg.convert_layout %acc_122 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %12 = tt.reshape %acc_121 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
        %13 = tt.reshape %acc_124 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
        %14 = tt.trans %12 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %15 = tt.trans %13 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %14 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_125, %outRHS_126 = tt.split %15 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %acc0_127, %acc0_128 = ttng.tmem_load %alpha_1[] {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %acc0_129 = tt.reshape %acc0_127 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %acc0_130 = ttg.convert_layout %acc0_129 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_131 = tt.expand_dims %acc0_130 {ttg.partition = array<i32: 0>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_132, %acc0_133 = ttng.tmem_load %alpha_0[] {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %acc0_134 = tt.reshape %acc0_132 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %acc0_135 = ttg.convert_layout %acc0_134 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_136 = tt.expand_dims %acc0_135 {ttg.partition = array<i32: 0>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_137 = ttg.convert_layout %acc0_131 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_138 = ttg.convert_layout %acc0_136 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_139 = tt.broadcast %acc0_137 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_140 = tt.broadcast %acc0_138 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_141 = arith.mulf %outLHS, %acc0_139 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear4>
        %acc0_142 = arith.mulf %outLHS_125, %acc0_140 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear4>
        %acc1 = arith.mulf %outRHS, %acc0_139 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear4>
        %acc1_143 = arith.mulf %outRHS_126, %acc0_140 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear4>
        %acc_144 = tt.join %acc0_141, %acc1 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_145 = tt.join %acc0_142, %acc1_143 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_146 = tt.trans %acc_144 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_147 = tt.trans %acc_145 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_148 = tt.reshape %acc_146 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %acc_149 = tt.reshape %acc_147 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %p_150 = arith.truncf %p {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %p_151 = arith.truncf %p_112 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %acc_152 = ttg.convert_layout %p_150 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #linear>
        ttng.tmem_store %acc_152, %acc_0, %true {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_153 = ttg.convert_layout %p_151 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #linear>
        ttng.tmem_store %acc_153, %acc_1, %true {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_154 = ttg.convert_layout %acc_148 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
        %acc_155 = ttng.tmem_store %acc_154, %acc_0_13[%acc_120], %true {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_156 = ttg.convert_layout %acc_149 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
        %acc_157 = ttng.tmem_store %acc_156, %acc_1_11[%acc_123], %true {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_158 = ttng.tc_gen5_mma %acc_0, %v, %acc_0_13[%acc_155], %arg12, %true {ttg.partition = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_159 = ttng.tc_gen5_mma %acc_1, %v, %acc_1_11[%acc_157], %arg12, %true {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg13, %alpha_114 {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_160 = arith.mulf %arg17, %alpha_116 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_161 = arith.addf %l_i0, %l_ij {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_162 = arith.addf %l_i0_160, %l_ij_118 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_163 = arith.addi %offset_y_82, %c128_i32 {ttg.partition = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 1 : i32} : i32
        scf.yield {ttg.partition = array<i32: 0>} %offsetkv_y_163, %true, %l_i0_161, %m_ij_102, %qk_93, %acc_158, %l_i0_162, %m_ij_103, %qk_96, %acc_159 : i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true}
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y#7 {ttg.partition = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_41, %m_ij_1, %true {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_42 = tt.expand_dims %offsetkv_y#6 {ttg.partition = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_42, %l_i0_0, %true {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y#3 {ttg.partition = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_43, %m_ij_0, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_44 = tt.expand_dims %offsetkv_y#2 {ttg.partition = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_44, %l_i0_1, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %m_i0, %m_i0_45 = ttng.tmem_load %l_i0_1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %m_i0_46 = tt.reshape %m_i0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %m_i0_47 = ttg.convert_layout %m_i0_46 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_48 = math.log2 %m_i0_47 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_49, %m_i0_50 = ttng.tmem_load %m_ij_0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %m_i0_51 = tt.reshape %m_i0_49 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %m_i0_52 = ttg.convert_layout %m_i0_51 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_53 = arith.addf %m_i0_52, %m_i0_48 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %4 = ttg.convert_layout %m_i0_53 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %m_ptrs0 = arith.muli %off_hz, %c1024_i32 {ttg.partition = array<i32: 0>} : i32
      %m_ptrs0_54 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %m_ptrs0_55 = tt.splat %m_ptrs0_54 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_56 = tt.addptr %m_ptrs0_55, %offs_m0_34 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %m_ptrs0_56, %4 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      %acc0 = tt.expand_dims %m_i0_47 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_57 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc_58, %acc_59 = ttng.tmem_load %acc_0_13[%offsetkv_y#5] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_60 = ttg.convert_layout %acc_58 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_61 = arith.divf %acc_60, %acc0_57 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %5 = arith.truncf %acc0_61 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %5, %_1 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = ttg.local_load %_1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %7 = ttg.convert_layout %6 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_22[%qo_offset_y_31, %c0_i32], %7 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      %m_i0_62, %m_i0_63 = ttng.tmem_load %l_i0_0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %m_i0_64 = tt.reshape %m_i0_62 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %m_i0_65 = ttg.convert_layout %m_i0_64 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_66 = math.log2 %m_i0_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_67, %m_i0_68 = ttng.tmem_load %m_ij_1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %m_i0_69 = tt.reshape %m_i0_67 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %m_i0_70 = ttg.convert_layout %m_i0_69 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_71 = arith.addf %m_i0_70, %m_i0_66 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %8 = ttg.convert_layout %m_i0_71 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %m_ptrs0_72 = tt.splat %m_ptrs0_54 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_73 = tt.addptr %m_ptrs0_72, %offs_m0_35 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %m_ptrs0_73, %8 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      %acc0_74 = tt.expand_dims %m_i0_65 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_75 = tt.broadcast %acc0_74 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc_76, %acc_77 = ttng.tmem_load %acc_1_11[%offsetkv_y#9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_78 = ttg.convert_layout %acc_76 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_79 = arith.divf %acc_78, %acc0_75 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %9 = arith.truncf %acc0_79 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %9, %_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %10 = ttg.local_load %_0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %11 = ttg.convert_layout %10 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_23[%3, %c0_i32], %11 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      %tile_idx_80 = arith.addi %tile_idx_27, %num_progs {ttg.partition = array<i32: 0, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 2, 3>} %tile_idx_80 : i32
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
