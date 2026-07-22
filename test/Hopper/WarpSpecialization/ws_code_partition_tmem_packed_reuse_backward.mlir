// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=2" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix | FileCheck %s

// Regression test for whole-allocation overwrite with packed TMEM reuse,
// exercised by the FA-fwd-persistent TMEM aliasing race.
//
// The memory planner packs the scalar buffers alpha/m_ij/l_i0 into the spare
// columns (offsets 64/65/66) of the QK accumulator (buffer.id 8 for dp0,
// buffer.id 9 for dp1). They form a single-copy reuse group whose representative
// (the QK accumulator) is produced by a `tc_gen5_mma` with useC=false, which
// ZEROS all 128 columns -- clobbering the packed scalar columns.
//
// CADENCE MATTERS:
//   * The QK MMA runs at INNER-loop (KV-block) cadence.
//   * alpha (col 64) is produced/consumed INSIDE the inner loop (same cadence),
//     so it is ordered by its own per-iteration barriers and needs NO extra edge.
//   * m_ij (col 65) / l_i0 (col 66) are produced/consumed in the OUTER tile
//     epilogue (outer cadence). The NEXT tile's first inner QK MMA zeros their
//     columns before the default partition (task 0) finishes reading them in the
//     previous tile's epilogue -> the race.
//
// The fix (isWholeAllocationOverwriteReuseOwner + the overwrite-owner "hub" case)
// emits, in the gemm partition (ttg.partition = 1) and BEFORE the inner loop
// (once per outer tile), a producer_acquire on each OUTER-cadence packed
// sibling's token (dstTask = 0), using the sibling's OWN outer-loop phase. This
// makes the next tile's first QK MMA wait for the previous tile's epilogue read.
// Placing it before the inner loop (not at the MMA) and using the sibling's outer
// phase (not the owner's inner phase) is essential: the naive per-iteration / owner
// -phase version deadlocks.
//
// REQUIRED BARRIERS (the four edges the fix adds, all ttg.partition = 1,
// dstTask = 0, all appearing BEFORE the inner `scf.for`). Program order is
// m_ij / l_i0 for dp1 (buffer.id 9) then dp0 (buffer.id 8). The test asserts ALL
// four are present and precede the useC=false QK `tc_gen5_mma`. They are NOT
// adjacent (index/phase arith ops sit between them), so we use ordered CHECK
// (not CHECK-NEXT).
//
// alpha is deliberately NOT acquired by task 1 (inner cadence; safe via its own
// barriers). The pre-existing QK backward (dstTask = 5 for dp0, 4 for dp1) lives
// INSIDE the inner loop and is unchanged by this fix.
//
// NOTE ON BARRIER FUSION: if a future pass fuses these backward barriers (e.g.
// onto a combined token, or merges them with the QK acquire; see
// BarrierFusion.md), these CHECKs WILL fail. That failure is expected -- to fix
// the test, assert that the (now fused) barrier still makes the next tile's
// useC=false QK MMA wait on BOTH outer-cadence consumers (m_ij AND l_i0) of BOTH
// dp groups before the overwrite. Dropping the wait on ANY of the four
// reintroduces the TMEM aliasing race, so every one must remain represented.

// CHECK-LABEL: tt.func public @_attn_fwd_persist

// Four outer-cadence back-edges, before the inner loop (dp1 then dp0):
// CHECK: nvws.producer_acquire %m_ij_{{[0-9_]+}}, {{.*}}dstTask = 0{{.*}}ttg.partition = array<i32: 1>
// CHECK: nvws.producer_acquire %l_i0_{{[0-9_]+}}, {{.*}}dstTask = 0{{.*}}ttg.partition = array<i32: 1>
// CHECK: nvws.producer_acquire %m_ij_{{[0-9_]+}}, {{.*}}dstTask = 0{{.*}}ttg.partition = array<i32: 1>
// CHECK: nvws.producer_acquire %l_i0_{{[0-9_]+}}, {{.*}}dstTask = 0{{.*}}ttg.partition = array<i32: 1>
// Then the inner KV loop, which contains the useC=false QK MMA the back-edges gate:
// CHECK: scf.for
// CHECK: tc_gen5_mma {{.*}}, %false, %true

// -----// WarpSpec internal IR Dump After: doMemoryPlanner
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<128x128xbf16, #shared>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<128x128xbf16, #shared>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<128x128xbf16, #shared>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<128x128xbf16, #shared>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64) attributes {noinline = false} {
    %cst = arith.constant {ttg.partition = array<i32: 4, 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_16 = arith.constant {ttg.partition = array<i32: 4, 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_17 = arith.constant {ttg.partition = array<i32: 4, 5>} dense<0.000000e+00> : tensor<128x1xf32, #linear>
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 128 : i32
    %cst_18 = arith.constant {ttg.partition = array<i32: 4, 5>} 1.44269502 : f32
    %c256_i32 = arith.constant {ttg.partition = array<i32: 0, 2, 3>} 256 : i32
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 0 : i32
    %c8192_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 8192 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 1 : i32
    %num_pid_m = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} 32 : i32
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 1>} false
    %0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %acc_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha, %alpha_19 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk, %qk_20 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha_21, %alpha_22 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_23, %qk_24 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %k = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %acc_1_25, %acc_1_26 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_0_27, %acc_0_28 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_ij_0, %m_ij_0_29 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("m_ij_0")
    %m_ij_1, %m_ij_1_30 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("m_ij_1")
    %l_i0_1, %l_i0_1_31 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("l_i0_1")
    %l_i0_0, %l_i0_0_32 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("l_i0_0")
    %q1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %q0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %prog_id = tt.get_program_id x {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %num_progs = tt.get_num_programs x {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %num_pid_n = arith.muli %Z, %H {ttg.partition = array<i32: 0, 2, 3>} : i32
    %total_tiles = arith.muli %Z, %num_pid_m {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles_33 = arith.muli %total_tiles, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_33, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %2 = arith.remsi %total_tiles_33, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %3 = arith.cmpi slt, %prog_id, %2 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %4 = scf.if %3 -> (i32) {
      %tiles_per_sm_35 = arith.addi %tiles_per_sm, %c1_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm_35 : i32
    } else {
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm : i32
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>}
    %offset_y = arith.muli %H, %c8192_i32 {ttg.partition = array<i32: 2, 3>} : i32
    %offs_m0 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m1 = tt.make_range {ttg.partition = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst_18 {ttg.partition = array<i32: 4, 5>} : f32
    %m_ij = tt.splat %qk_scale {ttg.partition = array<i32: 4, 5>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk_34 = tt.splat %qk_scale {ttg.partition = array<i32: 4, 5>} : f32 -> tensor<128x128xf32, #linear>
    %tile_idx = scf.for %_ = %c0_i32 to %4 step %c1_i32 iter_args(%tile_idx_35 = %prog_id) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_idx_35, %num_pid_m {ttg.partition = array<i32: 0, 2, 3>} : i32
      %group_size_n = arith.subi %num_pid_n, %group_id {ttg.partition = array<i32: 0, 2, 3>} : i32
      %group_size_n_36 = arith.minsi %group_size_n, %c1_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %off_hz = arith.remsi %tile_idx_35, %group_size_n_36 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %off_hz_37 = arith.addi %group_id, %off_hz {ttg.partition = array<i32: 0, 2, 3>} : i32
      %pid = arith.remsi %tile_idx_35, %num_pid_m {ttg.partition = array<i32: 0, 2, 3>} : i32
      %pid_38 = arith.divsi %pid, %group_size_n_36 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %off_z = arith.divsi %off_hz_37, %H {ttg.partition = array<i32: 2, 3>} : i32
      %off_h = arith.remsi %off_hz_37, %H {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_39 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_40 = arith.muli %off_h, %c8192_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %offset_y_41 = arith.addi %offset_y_39, %offset_y_40 {ttg.partition = array<i32: 2, 3>} : i32
      %qo_offset_y = arith.muli %pid_38, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_42 = arith.addi %offset_y_41, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %offs_m0_43 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_44 = arith.addi %offs_m0_43, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m1_45 = arith.addi %offs_m0_43, %offs_m1 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %q0_46 = tt.descriptor_load %desc_q[%qo_offset_y_42, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked1>
      ttg.local_store %q0_46, %q0 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %q1_47 = arith.addi %qo_offset_y_42, %c128_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %q1_48 = tt.descriptor_load %desc_q[%q1_47, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked1>
      ttg.local_store %q1_48, %q1 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_86 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg27 = %cst, %arg28 = %cst, %arg29 = %cst_16, %arg30 = %cst_16, %offset_y_87 = %offset_y_41, %acc_88 = %false, %qk_89 = %qk_24, %acc_90 = %acc_0_28, %qk_91 = %qk_20, %acc_92 = %acc_1_26) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_93 = tt.descriptor_load %desc_k[%offset_y_87, %c0_i32] {ttg.partition = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked1>
        ttg.local_store %k_93, %k {ttg.partition = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %k_94 = ttg.memdesc_trans %k {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %v_95 = tt.descriptor_load %desc_v[%offset_y_87, %c0_i32] {ttg.partition = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked1>
        ttg.local_store %v_95, %v {ttg.partition = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %qk_96 = ttng.tc_gen5_mma %q0, %k_94, %qk_23[%qk_89], %false, %true {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_97, %qk_98 = ttng.tmem_load %qk_23[%qk_96] {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %m_ij_99 = "tt.reduce"(%qk_97) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_167: f32, %m_ij_168: f32):
          %m_ij_169 = arith.maxnumf %m_ij_167, %m_ij_168 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_169 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_100 = arith.mulf %m_ij_99, %m_ij {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_101 = arith.maxnumf %arg29, %m_ij_100 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_102 = tt.expand_dims %m_ij_101 {ttg.partition = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_103 = arith.subf %cst_17, %qk_102 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear>
        %qk_104 = tt.broadcast %qk_103 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_105 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc, rd;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mov.b64 rc, { $6, $7 };\0A            fma.rn.f32x2 rd, ra, rb, rc;\0A            mov.b64 { $0, $1 }, rd;\0A        }\0A        " {ttg.partition = array<i32: 5>, constraints = "=r,=r,r,r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true} %qk_97, %qk_34, %qk_104 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %13 = tt.reshape %qk_105 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
        %14 = tt.trans %13 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
        %outLHS, %outRHS = tt.split %14 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
        %p0 = math.exp2 %outLHS {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear3>
        %p0_bf16 = arith.truncf %p0 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
        %p1 = math.exp2 %outRHS {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear3>
        %p1_bf16 = arith.truncf %p1 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
        %p = tt.join %p0, %p1 {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3> -> tensor<128x64x2xf32, #linear2>
        %p_106 = tt.trans %p {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #linear2> -> tensor<128x2x64xf32, #linear1>
        %p_107 = tt.reshape %p_106 {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x2x64xf32, #linear1> -> tensor<128x128xf32, #linear>
        %alpha_108 = arith.subf %arg29, %m_ij_101 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_109 = math.exp2 %alpha_108 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_110 = tt.expand_dims %alpha_109 {ttg.partition = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        ttng.tmem_store %alpha_110, %alpha_21, %true {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p_107) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_167: f32, %l_ij_168: f32):
          %l_ij_169 = arith.addf %l_ij_167, %l_ij_168 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_169 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_111, %acc_112 = ttng.tmem_load %alpha_21[] {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
        %acc_113 = tt.reshape %acc_111 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
        %acc_114 = ttg.convert_layout %acc_113 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_115 = tt.expand_dims %acc_114 {ttg.partition = array<i32: 0>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc_116 = tt.broadcast %acc_115 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %acc_117, %acc_118 = ttng.tmem_load %acc_0_27[%acc_90] {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 13>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %acc_119 = arith.mulf %acc_117, %acc_116 {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %p_bf16 = tt.join %p0_bf16, %p1_bf16 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x64xbf16, #linear3> -> tensor<128x64x2xbf16, #linear6>
        %p_bf16_120 = tt.trans %p_bf16 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xbf16, #linear6> -> tensor<128x2x64xbf16, #linear7>
        %p_bf16_121 = tt.reshape %p_bf16_120 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x2x64xbf16, #linear7> -> tensor<128x128xbf16, #linear8>
        %acc_122 = ttg.convert_layout %p_bf16_121 {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #linear8> -> tensor<128x128xbf16, #linear>
        ttng.tmem_store %acc_122, %acc_1, %true {ttg.partition = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_123 = ttng.tmem_store %acc_119, %acc_0_27[%acc_118], %true {ttg.partition = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 14>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_124 = ttng.tc_gen5_mma %acc_1, %v, %acc_0_27[%acc_123], %acc_88, %true {ttg.partition = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 14>, tmem.start = array<i32: 13, 15>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg27, %alpha_109 {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_125 = arith.addf %l_i0, %l_ij {ttg.partition = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_126 = ttng.tc_gen5_mma %q1, %k_94, %qk[%qk_91], %false, %true {ttg.partition = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_127, %qk_128 = ttng.tmem_load %qk[%qk_126] {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %m_ij_129 = "tt.reduce"(%qk_127) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_167: f32, %m_ij_168: f32):
          %m_ij_169 = arith.maxnumf %m_ij_167, %m_ij_168 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_169 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_130 = arith.mulf %m_ij_129, %m_ij {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_131 = arith.maxnumf %arg30, %m_ij_130 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_132 = tt.expand_dims %m_ij_131 {ttg.partition = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_133 = arith.subf %cst_17, %qk_132 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear>
        %qk_134 = tt.broadcast %qk_133 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_135 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc, rd;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mov.b64 rc, { $6, $7 };\0A            fma.rn.f32x2 rd, ra, rb, rc;\0A            mov.b64 { $0, $1 }, rd;\0A        }\0A        " {ttg.partition = array<i32: 4>, constraints = "=r,=r,r,r,r,r,r,r", loop.cluster = 1 : i32, loop.stage = 2 : i32, packed_element = 2 : i32, pure = true} %qk_127, %qk_34, %qk_134 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %15 = tt.reshape %qk_135 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
        %16 = tt.trans %15 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
        %outLHS_136, %outRHS_137 = tt.split %16 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
        %p0_138 = math.exp2 %outLHS_136 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3>
        %p0_bf16_139 = arith.truncf %p0_138 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
        %p1_140 = math.exp2 %outRHS_137 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3>
        %p1_bf16_141 = arith.truncf %p1_140 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
        %p_142 = tt.join %p0_138, %p1_140 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xf32, #linear3> -> tensor<128x64x2xf32, #linear2>
        %p_143 = tt.trans %p_142 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #linear2> -> tensor<128x2x64xf32, #linear1>
        %p_144 = tt.reshape %p_143 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x2x64xf32, #linear1> -> tensor<128x128xf32, #linear>
        %alpha_145 = arith.subf %arg30, %m_ij_131 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_146 = math.exp2 %alpha_145 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_147 = tt.expand_dims %alpha_146 {ttg.partition = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        ttng.tmem_store %alpha_147, %alpha, %true {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij_148 = "tt.reduce"(%p_144) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_167: f32, %l_ij_168: f32):
          %l_ij_169 = arith.addf %l_ij_167, %l_ij_168 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_169 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_149, %acc_150 = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
        %acc_151 = tt.reshape %acc_149 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
        %acc_152 = ttg.convert_layout %acc_151 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_153 = tt.expand_dims %acc_152 {ttg.partition = array<i32: 0>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc_154 = tt.broadcast %acc_153 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %acc_155, %acc_156 = ttng.tmem_load %acc_1_25[%acc_92] {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 10>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %acc_157 = arith.mulf %acc_155, %acc_154 {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #linear>
        %p_bf16_158 = tt.join %p0_bf16_139, %p1_bf16_141 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x64xbf16, #linear3> -> tensor<128x64x2xbf16, #linear6>
        %p_bf16_159 = tt.trans %p_bf16_158 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>} : tensor<128x64x2xbf16, #linear6> -> tensor<128x2x64xbf16, #linear7>
        %p_bf16_160 = tt.reshape %p_bf16_159 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x2x64xbf16, #linear7> -> tensor<128x128xbf16, #linear8>
        %acc_161 = ttg.convert_layout %p_bf16_160 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xbf16, #linear8> -> tensor<128x128xbf16, #linear>
        ttng.tmem_store %acc_161, %acc_0, %true {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_162 = ttng.tmem_store %acc_157, %acc_1_25[%acc_156], %true {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.start = array<i32: 11>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_163 = ttng.tc_gen5_mma %acc_0, %v, %acc_1_25[%acc_162], %acc_88, %true {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 11>, tmem.start = array<i32: 10, 12>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0_164 = arith.mulf %arg28, %alpha_146 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_165 = arith.addf %l_i0_164, %l_ij_148 {ttg.partition = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_166 = arith.addi %offset_y_87, %c128_i32 {ttg.partition = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 1 : i32} : i32
        scf.yield {ttg.partition = array<i32: 0>} %l_i0_125, %l_i0_165, %m_ij_101, %m_ij_131, %offsetkv_y_166, %true, %qk_98, %acc_124, %qk_128, %acc_163 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, tt.disallow_acc_multi_buffer, tt.merge_epilogue = true, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true}
      %offsetkv_y_49 = tt.expand_dims %offsetkv_y#3 {ttg.partition = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_49, %m_ij_0, %true {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_50 = tt.expand_dims %offsetkv_y#2 {ttg.partition = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_50, %m_ij_1, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_51 = tt.expand_dims %offsetkv_y#1 {ttg.partition = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_51, %l_i0_1, %true {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_52 = tt.expand_dims %offsetkv_y#0 {ttg.partition = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      ttng.tmem_store %offsetkv_y_52, %l_i0_0, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %acc, %acc_53 = ttng.tmem_load %acc_1_25[%offsetkv_y#9] {ttg.partition = array<i32: 0>, tmem.end = array<i32: 12>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %offsetkv_y_54 = ttg.convert_layout %acc {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear8>
      %offsetkv_y_55, %offsetkv_y_56 = ttng.tmem_load %l_i0_1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
      %offsetkv_y_57 = tt.reshape %offsetkv_y_55 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
      %offsetkv_y_58 = ttg.convert_layout %offsetkv_y_57 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offsetkv_y_59 = ttg.convert_layout %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear8}>>
      %acc1 = tt.expand_dims %offsetkv_y_59 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear8}>> -> tensor<128x1xf32, #linear8>
      %acc_60, %acc_61 = ttng.tmem_load %acc_0_27[%offsetkv_y#7] {ttg.partition = array<i32: 0>, tmem.end = array<i32: 15>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %offsetkv_y_62 = ttg.convert_layout %acc_60 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear8>
      %offsetkv_y_63, %offsetkv_y_64 = ttng.tmem_load %l_i0_0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
      %offsetkv_y_65 = tt.reshape %offsetkv_y_63 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
      %offsetkv_y_66 = ttg.convert_layout %offsetkv_y_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offsetkv_y_67 = ttg.convert_layout %offsetkv_y_66 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear8}>>
      %acc0 = tt.expand_dims %offsetkv_y_67 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear8}>> -> tensor<128x1xf32, #linear8>
      %m_i0 = math.log2 %offsetkv_y_66 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_68, %m_i0_69 = ttng.tmem_load %m_ij_1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
      %m_i0_70 = tt.reshape %m_i0_68 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
      %m_i0_71 = ttg.convert_layout %m_i0_70 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_72 = arith.addf %m_i0_71, %m_i0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %5 = ttg.convert_layout %m_i0_72 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %acc0_73 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear8> -> tensor<128x128xf32, #linear8>
      %acc0_74 = arith.divf %offsetkv_y_62, %acc0_73 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8>
      %6 = arith.truncf %acc0_74 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8> to tensor<128x128xbf16, #linear8>
      %7 = ttg.convert_layout %6 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #linear8> -> tensor<128x128xbf16, #blocked1>
      %m_ptrs0 = arith.muli %off_hz_37, %c8192_i32 {ttg.partition = array<i32: 0>} : i32
      %m_ptrs0_75 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %m_ptrs0_76 = tt.splat %m_ptrs0_75 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_77 = tt.addptr %m_ptrs0_76, %offs_m0_44 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %m_ptrs0_77, %5 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %7, %1 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %8 = ttng.async_tma_copy_local_to_global %desc_o[%qo_offset_y_42, %c0_i32] %1 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %8   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %m_i1 = math.log2 %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i1_78, %m_i1_79 = ttng.tmem_load %m_ij_0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear4>
      %m_i1_80 = tt.reshape %m_i1_78 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128xf32, #linear5>
      %m_i1_81 = ttg.convert_layout %m_i1_80 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear5> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i1_82 = arith.addf %m_i1_81, %m_i1 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %9 = ttg.convert_layout %m_i1_82 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %acc1_83 = tt.broadcast %acc1 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear8> -> tensor<128x128xf32, #linear8>
      %acc1_84 = arith.divf %offsetkv_y_54, %acc1_83 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8>
      %10 = arith.truncf %acc1_84 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8> to tensor<128x128xbf16, #linear8>
      %11 = ttg.convert_layout %10 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #linear8> -> tensor<128x128xbf16, #blocked1>
      %m_ptrs1 = tt.addptr %m_ptrs0_76, %offs_m1_45 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %m_ptrs1, %9 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %11, %0 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %12 = ttng.async_tma_copy_local_to_global %desc_o[%q1_47, %c0_i32] %0 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %12   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %tile_idx_85 = arith.addi %tile_idx_35, %num_progs {ttg.partition = array<i32: 0, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 2, 3>} %tile_idx_85 : i32
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
