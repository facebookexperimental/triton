// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=3 --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Test case: FA FWD persistent pattern with num-buffers=3.
// With num-buffers=3, cross-stage TMA buffers (k, v) get copy=3.
// Non-cross-stage buffers retain copy=1.
//
// The key buffers in allocation order:
//   [0] _1: output staging (SMEM), copy=1
//   [1] _0: output staging (SMEM), copy=1
//   [2] v/k: cross-stage KV buffers (SMEM), copy=3 (share buffer.id)
//   [3] q0_1: query buffer (SMEM), copy=1
//   [4] q0_0: query buffer (SMEM), copy=1
//
// TMEM allocations with packing:
//   [5] acc_0_10: f32 accumulator, owns buffer 5
//   [6] acc_1_8: f32 accumulator, owns buffer 6
//   [7] qk_0/alpha_0/m_ij_0/l_i0_1: packed in buffer 7
//       - qk_0 owns buffer 7
//       - acc_0 (f16) reuses at offset 0
//       - alpha_0 at offset 64
//       - m_ij_0 at offset 65
//       - l_i0_1 at offset 66
//   [8] qk_1/alpha_1/m_ij_1/l_i0_0: packed in buffer 8
//       - qk_1 owns buffer 8
//       - acc_1 (f16) reuses at offset 0
//       - alpha_1 at offset 64
//       - m_ij_1 at offset 65
//       - l_i0_0 at offset 66

// CHECK-LABEL: tt.func public @_attn_fwd_persist
//
// SMEM allocations
// CHECK: %_1 = ttg.local_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 0 : i32}
// CHECK: %_0 = ttg.local_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 1 : i32}
//
// TMEM allocations: acc_1 (f16) reuses qk_1's buffer at offset 0
// CHECK: %acc_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32}
//
// TMEM allocations: acc_0 (f16) reuses qk_0's buffer at offset 0
// CHECK: %acc_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32}
//
// TMEM allocations: alpha_1 packed in buffer 8 at offset 64
// CHECK: %alpha_1, %alpha_1_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 64 : i32}
//
// TMEM allocations: alpha_0 packed in buffer 7 at offset 64
// CHECK: %alpha_0, %alpha_0_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 64 : i32}
//
// TMEM allocations: qk_1 owns buffer 8
// CHECK: %qk_1, %qk_1_2 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32}
//
// TMEM allocations: qk_0 owns buffer 7
// CHECK: %qk_0, %qk_0_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
//
// SMEM allocations: v and k get copy=3 with num-buffers=3, sharing buffer.id=2
// CHECK: %v = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32}
// CHECK: %k = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32}
//
// TMEM allocations: m_ij_1 packed in buffer 8 at offset 65
// CHECK: %m_ij_1, %m_ij_1_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 65 : i32}
//
// TMEM allocations: l_i0_0 packed in buffer 8 at offset 66
// CHECK: %l_i0_0, %l_i0_0_5 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 66 : i32}
//
// TMEM allocations: m_ij_0 packed in buffer 7 at offset 65
// CHECK: %m_ij_0, %m_ij_0_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 65 : i32}
//
// TMEM allocations: l_i0_1 packed in buffer 7 at offset 66
// CHECK: %l_i0_1, %l_i0_1_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 66 : i32}
//
// TMEM allocations: acc_1_8 (f32 accumulator) owns buffer 6
// CHECK: %acc_1_8, %acc_1_9 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 6 : i32}
//
// TMEM allocations: acc_0_10 (f32 accumulator) owns buffer 5
// CHECK: %acc_0_10, %acc_0_11 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 5 : i32}
//
// SMEM allocations: query buffers
// CHECK: %q0_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
// CHECK: %q0_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}

// -----// WarpSpec internal IR Dump After: doBufferAllocation
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %_1 = ttg.local_alloc {async_task_id = array<i32: 0>} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %_0 = ttg.local_alloc {async_task_id = array<i32: 0>} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %acc_1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %acc_0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha_1, %alpha_1_0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alpha_0, %alpha_0_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_1, %qk_1_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_0, %qk_0_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m_ij_1, %m_ij_1_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_0, %l_i0_0_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_ij_0, %m_ij_0_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0_1, %l_i0_1_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_1_8, %acc_1_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_0_10, %acc_0_11 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q0_1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %q0_0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 4 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 1 : i32
    %c1024_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 1024 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 128 : i32
    %c128_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} 0 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %cst = arith.constant {async_task_id = array<i32: 4, 5>} 1.44269502 : f32
    %cst_12 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_13 = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_14 = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
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
    %desc_q_17 = arith.muli %desc_q_16, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
    %desc_q_18 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_19 = tt.make_tensor_descriptor %desc_q, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_20 = tt.make_tensor_descriptor %desc_k, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_21 = tt.make_tensor_descriptor %desc_v, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_22 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_23 = tt.make_tensor_descriptor %desc_o, [%desc_q_17, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
    %offs_m0 = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m0_24 = tt.make_range {async_task_id = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst {async_task_id = array<i32: 4, 5>} : f32
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
      %offset_y_29 = arith.muli %off_h, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
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
      %acc = ttng.tmem_store %cst_12, %acc_0_10[%acc_0_11], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_38 = ttng.tmem_store %cst_12, %acc_1_8[%acc_1_9], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_85 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%offset_y_86 = %offset_y_30, %arg12 = %false, %arg13 = %cst_14, %arg14 = %cst_13, %qk_0_87 = %qk_0_3, %acc_88 = %acc, %arg17 = %cst_14, %arg18 = %cst_13, %qk_1_89 = %qk_1_2, %acc_90 = %acc_38) -> (i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_91 = tt.descriptor_load %desc_k_20[%offset_y_86, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        ttg.local_store %k_91, %k {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_92 = ttg.memdesc_trans %k {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        %v_93 = tt.descriptor_load %desc_v_21[%offset_y_86, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        ttg.local_store %v_93, %v {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_94 = ttng.tc_gen5_mma %q0_0, %k_92, %qk_0[%qk_0_87], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_95 = ttng.tc_gen5_mma %q0_1, %k_92, %qk_1[%qk_1_89], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_96, %qk_97 = ttng.tmem_load %qk_0[%qk_94] {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %qk_98, %qk_99 = ttng.tmem_load %qk_1[%qk_95] {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_100 = "tt.reduce"(%qk_96) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_157: f32, %m_ij_158: f32):
          %m_ij_159 = arith.maxnumf %m_ij_157, %m_ij_158 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %m_ij_159 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_101 = "tt.reduce"(%qk_98) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_157: f32, %m_ij_158: f32):
          %m_ij_159 = arith.maxnumf %m_ij_157, %m_ij_158 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %m_ij_159 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_102 = arith.mulf %m_ij_100, %m_ij {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_103 = arith.mulf %m_ij_101, %m_ij_25 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_104 = arith.maxnumf %arg14, %m_ij_102 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_105 = arith.maxnumf %arg18, %m_ij_103 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_106 = arith.mulf %qk_96, %qk {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_107 = arith.mulf %qk_98, %qk_26 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %qk_108 = tt.expand_dims %m_ij_104 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_109 = tt.expand_dims %m_ij_105 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_110 = tt.broadcast %qk_108 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_111 = tt.broadcast %qk_109 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_112 = arith.subf %qk_106, %qk_110 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_113 = arith.subf %qk_107, %qk_111 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_112 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %p_114 = math.exp2 %qk_113 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %alpha = arith.subf %arg14, %m_ij_104 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_108 = arith.subf %arg18, %m_ij_105 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_109 = math.exp2 %alpha {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_110 = tt.expand_dims %alpha_109 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_111 = ttg.convert_layout %alpha_110 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        %alpha_112 = arith.constant {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %alpha_111, %alpha_0, %alpha_112 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %alpha_113 = math.exp2 %alpha_108 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_114 = tt.expand_dims %alpha_113 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_115 = ttg.convert_layout %alpha_114 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        %alpha_116 = arith.constant {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} true
        ttng.tmem_store %alpha_115, %alpha_1, %alpha_116 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_157: f32, %l_ij_158: f32):
          %l_ij_159 = arith.addf %l_ij_157, %l_ij_158 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %l_ij_159 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_124 = "tt.reduce"(%p_114) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_157: f32, %l_ij_158: f32):
          %l_ij_159 = arith.addf %l_ij_157, %l_ij_158 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %l_ij_159 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_125, %acc_126 = ttng.tmem_load %alpha_0[] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc_127 = tt.reshape %acc_125 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc_128 = ttg.convert_layout %acc_127 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_129 = tt.expand_dims %acc_128 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_130, %acc_131 = ttng.tmem_load %alpha_1[] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc_132 = tt.reshape %acc_130 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc_133 = ttg.convert_layout %acc_132 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_134 = tt.expand_dims %acc_133 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_135 = tt.broadcast %acc_129 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_136 = tt.broadcast %acc_134 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_137, %acc_138 = ttng.tmem_load %acc_0_10[%acc_88] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_139, %acc_140 = ttng.tmem_load %acc_1_8[%acc_90] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_141 = arith.mulf %acc_137, %acc_135 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %acc_142 = arith.mulf %acc_139, %acc_136 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %p_143 = arith.truncf %p {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %p_144 = arith.truncf %p_114 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %acc_145 = ttg.convert_layout %p_143 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        %acc_146 = arith.constant {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %acc_145, %acc_0, %acc_146 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_147 = ttg.convert_layout %p_144 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        %acc_148 = arith.constant {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} true
        ttng.tmem_store %acc_147, %acc_1, %acc_148 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_149 = ttng.tmem_store %acc_141, %acc_0_10[%acc_138], %true {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_150 = ttng.tmem_store %acc_142, %acc_1_8[%acc_140], %true {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_151 = ttng.tc_gen5_mma %acc_0, %v, %acc_0_10[%acc_149], %arg12, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_152 = ttng.tc_gen5_mma %acc_1, %v, %acc_1_8[%acc_150], %arg12, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg13, %alpha_109 {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_153 = arith.mulf %arg17, %alpha_113 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_154 = arith.addf %l_i0, %l_ij {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_155 = arith.addf %l_i0_153, %l_ij_124 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_156 = arith.addi %offset_y_86, %c128_i32 {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 4, 5>} %offsetkv_y_156, %true, %l_i0_154, %m_ij_104, %qk_97, %acc_151, %l_i0_155, %m_ij_105, %qk_99, %acc_152 : i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.scheduled_max_stage = 2 : i32}
      %offsetkv_y_39 = tt.expand_dims %offsetkv_y#7 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_40 = ttg.convert_layout %offsetkv_y_39 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_41 = arith.constant {async_task_id = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_40, %m_ij_1, %offsetkv_y_41 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_42 = tt.expand_dims %offsetkv_y#6 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_43 = ttg.convert_layout %offsetkv_y_42 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_44 = arith.constant {async_task_id = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_43, %l_i0_0, %offsetkv_y_44 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y#3 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_46 = ttg.convert_layout %offsetkv_y_45 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_47 = arith.constant {async_task_id = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_46, %m_ij_0, %offsetkv_y_47 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_48 = tt.expand_dims %offsetkv_y#2 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_49 = ttg.convert_layout %offsetkv_y_48 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_50 = arith.constant {async_task_id = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_49, %l_i0_1, %offsetkv_y_50 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %m_i0, %m_i0_51 = ttng.tmem_load %l_i0_1[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_52 = tt.reshape %m_i0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_53 = ttg.convert_layout %m_i0_52 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_54 = math.log2 %m_i0_53 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_55, %m_i0_56 = ttng.tmem_load %m_ij_0[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_57 = tt.reshape %m_i0_55 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_58 = ttg.convert_layout %m_i0_57 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_59 = arith.addf %m_i0_58, %m_i0_54 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %4 = ttg.convert_layout %m_i0_59 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0 = arith.muli %off_hz, %c1024_i32 {async_task_id = array<i32: 0>} : i32
      %m_ptrs0_60 = tt.addptr %M, %m_ptrs0 {async_task_id = array<i32: 0>} : !tt.ptr<f32>, i32
      %m_ptrs0_61 = tt.splat %m_ptrs0_60 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_62 = tt.addptr %m_ptrs0_61, %offs_m0_34 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_62, %4 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0 = tt.expand_dims %m_i0_53 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_63 = tt.broadcast %acc0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_64, %acc_65 = ttng.tmem_load %acc_0_10[%offsetkv_y#5] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_66 = arith.divf %acc_64, %acc0_63 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %5 = arith.truncf %acc0_66 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttg.local_store %5, %_0 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = ttg.local_load %_0 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %7 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_22[%qo_offset_y_31, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %m_i0_67, %m_i0_68 = ttng.tmem_load %l_i0_0[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_69 = tt.reshape %m_i0_67 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_70 = ttg.convert_layout %m_i0_69 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_71 = math.log2 %m_i0_70 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_72, %m_i0_73 = ttng.tmem_load %m_ij_1[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_74 = tt.reshape %m_i0_72 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_75 = ttg.convert_layout %m_i0_74 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_76 = arith.addf %m_i0_75, %m_i0_71 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %8 = ttg.convert_layout %m_i0_76 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0_77 = tt.splat %m_ptrs0_60 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_78 = tt.addptr %m_ptrs0_77, %offs_m0_35 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_78, %8 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0_79 = tt.expand_dims %m_i0_70 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_80 = tt.broadcast %acc0_79 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_81, %acc_82 = ttng.tmem_load %acc_1_8[%offsetkv_y#9] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_83 = arith.divf %acc_81, %acc0_80 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %9 = arith.truncf %acc0_83 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttg.local_store %9, %_1 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %10 = ttg.local_load %_1 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %11 = ttg.convert_layout %10 {async_task_id = array<i32: 3>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_23[%3, %c0_i32], %11 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %tile_idx_84 = arith.addi %tile_idx_27, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_84 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
