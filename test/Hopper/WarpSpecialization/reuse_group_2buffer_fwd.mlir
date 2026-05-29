// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix | FileCheck %s
//
// Regression test: verify that 2-buffer reuse group logic does NOT
// incorrectly move the accumulator MMA's producer_acquire in the
// forward persistent attention kernel.
//
// In the FWD persistent FA kernel, the accumulator TMEM buffers form
// reuse groups (buffer.id 7 and 8, with buffer.copy=1).  The tmem_store
// (computation partition, task 0) writes the softmax-corrected
// accumulator, and tc_gen5_mma (gemm partition, task 1) consumes it as
// operand D.
//
// The correct ordering within task 1's inner loop is:
//
//   qk MMA (cluster 0) → qk MMA (cluster 2) →
//     consumer_wait (cluster 4) → acc MMA (cluster 4) →
//     consumer_wait (cluster 1) → acc MMA (cluster 1)
//
// The 2-buffer reuse group logic should NOT fire for this pattern.
// If it incorrectly fires, producer_acquire for the acc MMA channels
// gets inserted between the qk MMAs and the consumer_waits,
// causing the MMA to read stale/corrupted TMEM data.
//
// Operand-D race fix same-task guard:
// The operand-D race fix must NOT fire for FA fwd because the tmem_store
// (task 0, computation) and tmem_load (task 0, computation) for the
// accumulator are in the same partition.  If it fires, a token-based
// ProducerAcquire is inserted before the tmem_store which creates a
// deadlock.  Instead, a WaitBarrierOp (from desyncTCGen5MMAOp) must
// appear before the accumulator tmem_store.
//
// Verify: inside the inner scf.for, wait_barrier (NOT producer_acquire
// with create_token) appears before the accumulator tmem_store with
// tmem.start in the default partition.
//
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}}loop.cluster = 4{{.*}}loop.stage = 1
// CHECK: ttng.tmem_store {{.*}}loop.cluster = 4{{.*}}loop.stage = 1{{.*}}tmem.start
//
// Verify: no producer_acquire appears between qk MMA
// (cluster 2) and the acc consumer_wait (cluster 4).
//
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 2{{.*}}loop.stage = 1
// CHECK-NOT: nvws.producer_acquire
// CHECK: nvws.consumer_wait {{.*}}loop.cluster = 4{{.*}}loop.stage = 1
// The accumulator MMA now carries 2 distinct channel ids in tmem.start
// (the gen5 -> tmem_load back-edge channel + the gen5 -> post-loop
// tmem_load forward channel), in addition to tmem.end for the in-body
// tmem_store -> gen5 forward channel.
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 4{{.*}}loop.stage = 1{{.*}}tmem.end = array<i32: {{.+}}>, tmem.start = array<i32: {{.+}}, {{.+}}>
//
// Same check for cluster 1, stage 2:
// CHECK-NOT: nvws.producer_acquire
// CHECK: nvws.consumer_wait {{.*}}loop.cluster = 1{{.*}}loop.stage = 2
// CHECK: ttng.tc_gen5_mma {{.*}}loop.cluster = 1{{.*}}loop.stage = 2{{.*}}tmem.end = array<i32: {{.+}}>, tmem.start = array<i32: {{.+}}, {{.+}}>
//
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %acc = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %acc_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %alpha, %alpha_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alpha_2, %alpha_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk, %qk_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %qk_5, %qk_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %offsetkv_y, %offsetkv_y_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %offsetkv_y_8, %offsetkv_y_9 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %offsetkv_y_10, %offsetkv_y_11 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %offsetkv_y_12, %offsetkv_y_13 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_14, %acc_15 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_16, %acc_17 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %q0_18 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
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
    %cst_19 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_20 = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_21 = arith.constant {async_task_id = array<i32: 0, 4, 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles = arith.muli %Z, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %total_tiles_22 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_22, %num_progs {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %2 = arith.remsi %total_tiles_22, %num_progs {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %3 = arith.cmpi slt, %prog_id, %2 {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
    %4 = scf.if %3 -> (i32) {
      %tiles_per_sm_35 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm_35 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>}
    %desc_q_23 = arith.muli %Z, %H {async_task_id = array<i32: 2, 3>} : i32
    %desc_q_24 = arith.muli %desc_q_23, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
    %desc_q_25 = tt.make_tensor_descriptor %desc_q, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_26 = tt.make_tensor_descriptor %desc_q, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_27 = tt.make_tensor_descriptor %desc_k, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_28 = tt.make_tensor_descriptor %desc_v, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_29 = tt.make_tensor_descriptor %desc_o, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_30 = tt.make_tensor_descriptor %desc_o, [%desc_q_24, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
    %offs_m0 = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m0_31 = tt.make_range {async_task_id = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst {async_task_id = array<i32: 4, 5>} : f32
    %m_ij = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_ij_32 = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk_33 = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128x128xf32, #blocked>
    %qk_34 = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128x128xf32, #blocked>
    %tile_idx = scf.for %_ = %c0_i32 to %4 step %c1_i32 iter_args(%tile_idx_35 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_35, %n_tile_num {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_hz = arith.divsi %tile_idx_35, %n_tile_num {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_z = arith.divsi %off_hz, %H {async_task_id = array<i32: 2, 3>} : i32
      %off_h = arith.remsi %off_hz, %H {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_36 = arith.muli %off_z, %offset_y {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_37 = arith.muli %off_h, %c1024_i32 {async_task_id = array<i32: 2, 3>} : i32
      %offset_y_38 = arith.addi %offset_y_36, %offset_y_37 {async_task_id = array<i32: 2, 3>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_39 = arith.addi %offset_y_38, %qo_offset_y {async_task_id = array<i32: 2, 3>} : i32
      %5 = arith.addi %qo_offset_y_39, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %q0_40 = arith.addi %qo_offset_y_39, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_m0_41 = tt.splat %qo_offset_y {async_task_id = array<i32: 0>} : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_42 = tt.splat %qo_offset_y {async_task_id = array<i32: 0>} : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_43 = arith.addi %offs_m0_41, %offs_m0 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
      %offs_m0_44 = arith.addi %offs_m0_42, %offs_m0_31 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
      %q0_45 = tt.descriptor_load %desc_q_25[%qo_offset_y_39, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %q0_46 = tt.descriptor_load %desc_q_26[%q0_40, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %q0_45, %q0_18 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %q0_46, %q0 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %acc_47 = ttng.tmem_store %cst_19, %acc_16[%acc_17], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_48 = ttng.tmem_store %cst_19, %acc_14[%acc_15], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y_49:10 = scf.for %offsetkv_y_96 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%offset_y_97 = %offset_y_38, %arg12 = %false, %arg13 = %cst_21, %arg14 = %cst_20, %qk_98 = %qk_6, %acc_99 = %acc_47, %arg17 = %cst_21, %arg18 = %cst_20, %qk_100 = %qk_4, %acc_101 = %acc_48) -> (i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_102 = tt.descriptor_load %desc_k_27[%offset_y_97, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        ttg.local_store %k_102, %k {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_103 = ttg.memdesc_trans %k {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        %v_104 = tt.descriptor_load %desc_v_28[%offset_y_97, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 6 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        ttg.local_store %v_104, %v {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_105 = ttng.tc_gen5_mma %q0_18, %k_103, %qk_5[%qk_98], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_106 = ttng.tc_gen5_mma %q0, %k_103, %qk[%qk_100], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_107, %qk_108 = ttng.tmem_load %qk_5[%qk_105] {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %qk_109, %qk_110 = ttng.tmem_load %qk[%qk_106] {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_111 = "tt.reduce"(%qk_107) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_169: f32, %m_ij_170: f32):
          %m_ij_171 = arith.maxnumf %m_ij_169, %m_ij_170 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %m_ij_171 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_112 = "tt.reduce"(%qk_109) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_169: f32, %m_ij_170: f32):
          %m_ij_171 = arith.maxnumf %m_ij_169, %m_ij_170 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %m_ij_171 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_113 = arith.mulf %m_ij_111, %m_ij {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_114 = arith.mulf %m_ij_112, %m_ij_32 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_115 = arith.maxnumf %arg14, %m_ij_113 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_116 = arith.maxnumf %arg18, %m_ij_114 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_117 = arith.mulf %qk_107, %qk_33 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_118 = arith.mulf %qk_109, %qk_34 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %qk_119 = tt.expand_dims %m_ij_115 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_120 = tt.expand_dims %m_ij_116 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_121 = tt.broadcast %qk_119 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_122 = tt.broadcast %qk_120 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_123 = arith.subf %qk_117, %qk_121 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %qk_124 = arith.subf %qk_118, %qk_122 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_123 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %p_125 = math.exp2 %qk_124 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %alpha_126 = arith.subf %arg14, %m_ij_115 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_127 = arith.subf %arg18, %m_ij_116 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_128 = math.exp2 %alpha_126 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_129 = tt.expand_dims %alpha_128 {async_task_id = array<i32: 5>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_130 = ttg.convert_layout %alpha_129 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        %alpha_131 = arith.constant {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %alpha_130, %alpha_2, %alpha_131 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %alpha_132 = math.exp2 %alpha_127 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_133 = tt.expand_dims %alpha_132 {async_task_id = array<i32: 4>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %alpha_134 = ttg.convert_layout %alpha_133 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
        %alpha_135 = arith.constant {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} true
        ttng.tmem_store %alpha_134, %alpha, %alpha_135 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_169: f32, %l_ij_170: f32):
          %l_ij_171 = arith.addf %l_ij_169, %l_ij_170 {async_task_id = array<i32: 5>} : f32
          tt.reduce.return %l_ij_171 {async_task_id = array<i32: 5>} : f32
        }) {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_136 = "tt.reduce"(%p_125) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_169: f32, %l_ij_170: f32):
          %l_ij_171 = arith.addf %l_ij_169, %l_ij_170 {async_task_id = array<i32: 4>} : f32
          tt.reduce.return %l_ij_171 {async_task_id = array<i32: 4>} : f32
        }) {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_137, %acc_138 = ttng.tmem_load %alpha_2[] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc_139 = tt.reshape %acc_137 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc_140 = ttg.convert_layout %acc_139 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_141 = tt.expand_dims %acc_140 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_142, %acc_143 = ttng.tmem_load %alpha[] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        %acc_144 = tt.reshape %acc_142 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc_145 = ttg.convert_layout %acc_144 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_146 = tt.expand_dims %acc_145 {async_task_id = array<i32: 0>, axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc_147 = tt.broadcast %acc_141 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_148 = tt.broadcast %acc_146 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_149, %acc_150 = ttng.tmem_load %acc_16[%acc_99] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_151, %acc_152 = ttng.tmem_load %acc_14[%acc_101] {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_153 = arith.mulf %acc_149, %acc_147 {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %acc_154 = arith.mulf %acc_151, %acc_148 {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
        %p_155 = arith.truncf %p {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %p_156 = arith.truncf %p_125 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %acc_157 = ttg.convert_layout %p_155 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        %acc_158 = arith.constant {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} true
        ttng.tmem_store %acc_157, %acc_0, %acc_158 {async_task_id = array<i32: 5>, loop.cluster = 4 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_159 = ttg.convert_layout %p_156 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
        %acc_160 = arith.constant {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} true
        ttng.tmem_store %acc_159, %acc, %acc_160 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %acc_161 = ttng.tmem_store %acc_153, %acc_16[%acc_150], %true {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 16>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_162 = ttng.tmem_store %acc_154, %acc_14[%acc_152], %true {async_task_id = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.start = array<i32: 14>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_163 = ttng.tc_gen5_mma %acc_0, %v, %acc_16[%acc_161], %arg12, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 16>, tmem.start = array<i32: 17>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_164 = ttng.tc_gen5_mma %acc, %v, %acc_14[%acc_162], %arg12, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 14>, tmem.start = array<i32: 15>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg13, %alpha_128 {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_165 = arith.mulf %arg17, %alpha_132 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_166 = arith.addf %l_i0, %l_ij {async_task_id = array<i32: 5>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i0_167 = arith.addf %l_i0_165, %l_ij_136 {async_task_id = array<i32: 4>, loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_168 = arith.addi %offset_y_97, %c128_i32 {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 4, 5>} %offsetkv_y_168, %true, %l_i0_166, %m_ij_115, %qk_108, %acc_163, %l_i0_167, %m_ij_116, %qk_110, %acc_164 : i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.scheduled_max_stage = 2 : i32}
      %offsetkv_y_50 = tt.expand_dims %offsetkv_y_49#7 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_51 = ttg.convert_layout %offsetkv_y_50 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_52 = arith.constant {async_task_id = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_51, %offsetkv_y, %offsetkv_y_52 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_53 = tt.expand_dims %offsetkv_y_49#6 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_54 = ttg.convert_layout %offsetkv_y_53 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_55 = arith.constant {async_task_id = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_54, %offsetkv_y_8, %offsetkv_y_55 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_56 = tt.expand_dims %offsetkv_y_49#3 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_57 = ttg.convert_layout %offsetkv_y_56 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_58 = arith.constant {async_task_id = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_57, %offsetkv_y_10, %offsetkv_y_58 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y_59 = tt.expand_dims %offsetkv_y_49#2 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %offsetkv_y_60 = ttg.convert_layout %offsetkv_y_59 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked3>
      %offsetkv_y_61 = arith.constant {async_task_id = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_60, %offsetkv_y_12, %offsetkv_y_61 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      %m_i0, %m_i0_62 = ttng.tmem_load %offsetkv_y_12[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_63 = tt.reshape %m_i0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_64 = ttg.convert_layout %m_i0_63 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_65 = math.log2 %m_i0_64 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_66, %m_i0_67 = ttng.tmem_load %offsetkv_y_10[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_68 = tt.reshape %m_i0_66 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_69 = ttg.convert_layout %m_i0_68 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_70 = arith.addf %m_i0_69, %m_i0_65 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %6 = ttg.convert_layout %m_i0_70 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0 = arith.muli %off_hz, %c1024_i32 {async_task_id = array<i32: 0>} : i32
      %m_ptrs0_71 = tt.addptr %M, %m_ptrs0 {async_task_id = array<i32: 0>} : !tt.ptr<f32>, i32
      %m_ptrs0_72 = tt.splat %m_ptrs0_71 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_73 = tt.addptr %m_ptrs0_72, %offs_m0_43 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_73, %6 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0 = tt.expand_dims %m_i0_64 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_74 = tt.broadcast %acc0 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_75, %acc_76 = ttng.tmem_load %acc_16[%offsetkv_y_49#5] {async_task_id = array<i32: 0>, tmem.end = array<i32: 17>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_77 = arith.divf %acc_75, %acc0_74 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %7 = arith.truncf %acc0_77 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttg.local_store %7, %1 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %8 = ttg.local_load %1 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %9 = ttg.convert_layout %8 {async_task_id = array<i32: 3>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_29[%qo_offset_y_39, %c0_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %m_i0_78, %m_i0_79 = ttng.tmem_load %offsetkv_y_8[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_80 = tt.reshape %m_i0_78 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_81 = ttg.convert_layout %m_i0_80 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_82 = math.log2 %m_i0_81 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_83, %m_i0_84 = ttng.tmem_load %offsetkv_y[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
      %m_i0_85 = tt.reshape %m_i0_83 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
      %m_i0_86 = ttg.convert_layout %m_i0_85 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_87 = arith.addf %m_i0_86, %m_i0_82 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = ttg.convert_layout %m_i0_87 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      %m_ptrs0_88 = tt.splat %m_ptrs0_71 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_89 = tt.addptr %m_ptrs0_88, %offs_m0_44 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      tt.store %m_ptrs0_89, %10 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
      %acc0_90 = tt.expand_dims %m_i0_81 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_91 = tt.broadcast %acc0_90 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc_92, %acc_93 = ttng.tmem_load %acc_14[%offsetkv_y_49#9] {async_task_id = array<i32: 0>, tmem.end = array<i32: 15>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc0_94 = arith.divf %acc_92, %acc0_91 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
      %11 = arith.truncf %acc0_94 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttg.local_store %11, %0 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %12 = ttg.local_load %0 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      %13 = ttg.convert_layout %12 {async_task_id = array<i32: 3>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.descriptor_store %desc_o_30[%5, %c0_i32], %13 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
      %tile_idx_95 = arith.addi %tile_idx_35, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_95 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
