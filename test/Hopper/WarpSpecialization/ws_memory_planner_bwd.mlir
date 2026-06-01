// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s --check-prefix=OPERANDD

// Test case: FA BWD pattern with budget-aware SMEM allocation (algo=1).
// With smem_budget=200000, only one of the two cross-stage TMA buffers
// (do, q) can get copy=2 before exceeding budget. The other stays at copy=1.
//
// The key buffers in allocation order:
//   [0] dk: liveness=[44-112) size=128x128 - accumulator, long-lived
//   [1] dv: liveness=[45-110) size=128x128 - accumulator, long-lived
//   [2] qkT: liveness=[56-61) size=128x128 - temp buffer, short-lived
//   [3] dpT: liveness=[72-77) size=128x128 - temp buffer, short-lived
//   [4] dq: liveness=[83-85) size=128x128 - output buffer, short-lived
//   [5] dv_interm: liveness=[67-69) size=128x64 - intermediate, short-lived
//
// The hasPotentialReuse matrix (non-zero entries):
//   hasPotentialReuse(qkT, dq) = 2  (exact size match, has dependency)
//   hasPotentialReuse(qkT, dv_interm) = 1  (partial size, has dependency)
//   hasPotentialReuse(dpT, dq) = 2  (exact size match, has dependency)
//   hasPotentialReuse(dq, qkT) = 2  (bidirectional)
//   hasPotentialReuse(dq, dpT) = 2  (bidirectional)
//   NOTE: hasPotentialReuse(dpT, dv_interm) = 0 (NO dependency!)
//
// With backtracking search, the algorithm finds:
//   - dq first tries qkT, but that blocks dv_interm → backtrack
//   - dq then reuses dpT (buffer.id=6)
//   - dv_interm reuses qkT (buffer.id=5)

// CHECK-LABEL: tt.func public @_attn_bwd
//
// SMEM allocations
// CHECK: %dsT = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32}
//
// TMEM allocation: dv (bf16) reuses qkT's buffer at offset 0
// CHECK: %dv = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32}
//
// SMEM allocations
// CHECK: %do = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// CHECK: %q = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// CHECK: %k_42 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
// CHECK: %v_43 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}
//
// TMEM allocations: qkT owns buffer 7
// CHECK: %qkT, %qkT_44 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 7 : i32}
//
// TMEM allocation: dv_45 (f32 accumulator) owns buffer 6
// CHECK: %dv_45, %dv_46 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 6 : i32}
//
// TMEM allocation: dpT owns buffer 8
// CHECK: %dpT, %dpT_47 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 8 : i32}
//
// TMEM allocation: dk owns buffer 5
// CHECK: %dk, %dk_48 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 5 : i32}
//
// TMEM allocation: dq reuses dpT (buffer.id=8, buffer.offset=0) — key verification
// CHECK: %dq, %dq_49 = ttng.tmem_alloc {{{.*}}buffer.copy = 1 : i32, buffer.id = 8 : i32, buffer.offset = 0 : i32}

// -----// WarpSpec internal IR Dump After: doBufferAllocation
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
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
  tt.func public @_attn_bwd(%desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_do_12: i32, %desc_do_13: i32, %desc_do_14: i64, %desc_do_15: i64, %desc_dq: !tt.tensordesc<tensor<128x32xf32, #shared1>>, %desc_dq_16: i32, %desc_dq_17: i32, %desc_dq_18: i64, %desc_dq_19: i64, %desc_dk: !tt.tensordesc<tensor<128x32xbf16, #shared2>>, %desc_dk_20: i32, %desc_dk_21: i32, %desc_dk_22: i64, %desc_dk_23: i64, %desc_dv: !tt.tensordesc<tensor<128x32xbf16, #shared2>>, %desc_dv_24: i32, %desc_dv_25: i32, %desc_dv_26: i64, %desc_dv_27: i64, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %dv = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
    %do = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 0>} false
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 2>} dense<0.693147182> : tensor<128x32xf32, #blocked>
    %cst_28 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %bhid = tt.get_program_id z {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 3>} : i32
    %off_chz_29 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
    %off_bh = arith.remsi %bhid, %H {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_bh_30 = arith.muli %stride_h, %off_bh {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_bh_31 = arith.divsi %bhid, %H {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_bh_32 = arith.muli %stride_z, %off_bh_31 {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_bh_33 = arith.addi %off_bh_30, %off_bh_32 {async_task_id = array<i32: 1, 2, 3>} : i32
    %off_bh_34 = arith.extsi %off_bh_33 {async_task_id = array<i32: 1, 2, 3>} : i32 to i64
    %off_bh_35 = arith.extsi %stride_tok {async_task_id = array<i32: 1, 2, 3>} : i32 to i64
    %off_bh_36 = arith.divsi %off_bh_34, %off_bh_35 {async_task_id = array<i32: 1, 2, 3>} : i64
    %pid = tt.get_program_id x {async_task_id = array<i32: 1, 3>} : i32
    %M_37 = tt.addptr %M, %off_chz_29 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
    %D_38 = tt.addptr %D, %off_chz_29 {async_task_id = array<i32: 3>} : !tt.ptr<f32>, i64
    %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 1, 3>} : i32
    %k = arith.extsi %start_n {async_task_id = array<i32: 1, 3>} : i32 to i64
    %k_39 = arith.addi %off_bh_36, %k {async_task_id = array<i32: 1, 3>} : i64
    %k_40 = arith.trunci %k_39 {async_task_id = array<i32: 1, 3>} : i64 to i32
    %k_41 = tt.descriptor_load %desc_k[%k_40, %c0_i32] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
    %k_42 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    ttg.local_store %k_41, %k_42 {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %v = tt.descriptor_load %desc_v[%k_40, %c0_i32] {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
    %v_43 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    ttg.local_store %v, %v_43 {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %offs_m = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %m = tt.splat %M_37 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked3>
    %Di = tt.splat %D_38 {async_task_id = array<i32: 3>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked3>
    %qkT, %qkT_44 = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dv_45, %dv_46 = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dpT, %dpT_47 = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_48 = ttng.tmem_alloc {async_task_id = array<i32: 0, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dq, %dq_49 = ttng.tmem_alloc {async_task_id = array<i32: 0, 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk_50 = ttng.tmem_store %cst_28, %dk[%dk_48], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dv_51 = ttng.tmem_store %cst_28, %dv_45[%dv_46], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %curr_m:7 = scf.for %curr_m_82 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg45 = %c0_i32, %arg46 = %false, %qkT_83 = %qkT_44, %dv_84 = %dv_51, %dpT_85 = %dpT_47, %dk_86 = %dk_50, %dq_87 = %dq_49) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %q_88 = arith.extsi %arg45 {async_task_id = array<i32: 1, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
      %q_89 = arith.addi %off_bh_36, %q_88 {async_task_id = array<i32: 1, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
      %q_90 = arith.trunci %q_89 {async_task_id = array<i32: 1, 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
      %q_91 = tt.descriptor_load %desc_q[%q_90, %c0_i32] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      ttg.local_store %q_91, %q {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>
      %offs_m_92 = tt.splat %arg45 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked3>
      %offs_m_93 = arith.addi %offs_m_92, %offs_m {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked3>
      %m_94 = tt.addptr %m, %offs_m_93 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked3>, tensor<128xi32, #blocked3>
      %m_95 = tt.load %m_94 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked3>
      %qkT_96 = ttng.tc_gen5_mma %k_42, %qT, %qkT[%qkT_83], %false, %true {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %pT = ttg.convert_layout %m_95 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %pT_97 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xf32, #blocked1>
      %pT_98 = tt.broadcast %pT_97 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %qkT_99, %qkT_100 = ttng.tmem_load %qkT[%qkT_96] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %pT_101 = arith.subf %qkT_99, %pT_98 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
      %pT_102 = math.exp2 %pT_101 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
      %do_103 = tt.descriptor_load %desc_do[%q_90, %c0_i32] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      ttg.local_store %do_103, %do {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xbf16, #blocked2> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %ppT = arith.truncf %pT_102 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xbf16, #blocked1>
      %dv_104 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} true
      ttng.tmem_store %ppT, %dv, %dv_104 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
      %dv_105 = ttng.tc_gen5_mma %dv, %do, %dv_45[%dv_84], %arg46, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %Di_106 = tt.addptr %Di, %offs_m_93 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked3>, tensor<128xi32, #blocked3>
      %Di_107 = tt.load %Di_106 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked3>
      %dpT_108 = ttg.memdesc_trans %do {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>
      %dpT_109 = ttng.tc_gen5_mma %v_43, %dpT_108, %dpT[%dpT_85], %false, %true {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dsT_110 = ttg.convert_layout %Di_107 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %dsT_111 = tt.expand_dims %dsT_110 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xf32, #blocked1>
      %dsT_112 = tt.broadcast %dsT_111 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %dpT_113, %dpT_114 = ttng.tmem_load %dpT[%dpT_109] {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dsT_115 = arith.subf %dpT_113, %dsT_112 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
      %dsT_116 = arith.mulf %pT_102, %dsT_115 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1>
      %dsT_117 = arith.truncf %dsT_116 {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xbf16, #blocked1>
      ttg.local_store %dsT_117, %dsT {async_task_id = array<i32: 3>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %dk_118 = ttng.tc_gen5_mma %dsT, %q, %dk[%dk_86], %arg46, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dq_119 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>
      %dq_120 = ttng.tc_gen5_mma %dq_119, %k_42, %dq[%dq_87], %false, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dq_121, %dq_122 = ttng.tmem_load %dq[%dq_120] {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dqs = tt.reshape %dq_121 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %dqs_123 = tt.trans %dqs {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dqs_124, %dqs_125 = tt.split %dqs_123 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dqs_126 = tt.reshape %dqs_124 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dqs_127 = tt.trans %dqs_126 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dqs_128, %dqs_129 = tt.split %dqs_127 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %dqs_130 = tt.reshape %dqs_125 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dqs_131 = tt.trans %dqs_130 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dqs_132, %dqs_133 = tt.split %dqs_131 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
      %dqN = arith.mulf %dqs_128, %cst {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
      %dqN_134 = ttg.convert_layout %dqN {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
      tt.descriptor_reduce add, %desc_dq[%q_90, %c0_i32], %dqN_134 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
      %dqN_135 = arith.mulf %dqs_129, %cst {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
      %dqN_136 = ttg.convert_layout %dqN_135 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
      tt.descriptor_reduce add, %desc_dq[%q_90, %c0_i32], %dqN_136 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
      %dqN_137 = arith.mulf %dqs_132, %cst {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
      %dqN_138 = ttg.convert_layout %dqN_137 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
      tt.descriptor_reduce add, %desc_dq[%q_90, %c0_i32], %dqN_138 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
      %dqN_139 = arith.mulf %dqs_133, %cst {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked>
      %dqN_140 = ttg.convert_layout %dqN_139 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #blocked9>
      tt.descriptor_reduce add, %desc_dq[%q_90, %c0_i32], %dqN_140 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
      %curr_m_141 = arith.addi %arg45, %c128_i32 {async_task_id = array<i32: 1, 2, 3>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_141, %true, %qkT_100, %dv_105, %dpT_114, %dk_118, %dq_122 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, "tt.smem_alloc_algo" = 1 : i32, "tt.smem_budget" = 200000 : i32, "tt.tmem_alloc_algo" = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %dv_52, %dv_53 = ttng.tmem_load %dv_45[%curr_m#3] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %dvs = tt.reshape %dv_52 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
    %dk_54, %dk_55 = ttng.tmem_load %dk[%curr_m#5] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %dks = tt.reshape %dk_54 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
    %dvs_56 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
    %dvs_57, %dvs_58 = tt.split %dvs_56 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
    %dvs_59 = tt.reshape %dvs_58 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
    %dvs_60 = tt.reshape %dvs_57 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
    %dvs_61 = tt.trans %dvs_60 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
    %dvs_62, %dvs_63 = tt.split %dvs_61 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
    %0 = arith.truncf %dvs_63 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %1 = arith.truncf %dvs_62 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %dvs_64 = tt.trans %dvs_59 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
    %dvs_65, %dvs_66 = tt.split %dvs_64 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
    %2 = arith.truncf %dvs_66 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %3 = arith.truncf %dvs_65 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %4 = ttg.convert_layout %1 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dv[%k_40, %c0_i32], %4 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %5 = ttg.convert_layout %0 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dv[%k_40, %c0_i32], %5 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %6 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dv[%k_40, %c0_i32], %6 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %7 = ttg.convert_layout %2 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dv[%k_40, %c0_i32], %7 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %dks_67 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
    %dks_68, %dks_69 = tt.split %dks_67 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
    %dks_70 = tt.reshape %dks_69 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
    %dks_71 = tt.reshape %dks_68 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
    %dks_72 = tt.trans %dks_71 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
    %dks_73, %dks_74 = tt.split %dks_72 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
    %dks_75 = tt.trans %dks_70 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
    %dks_76, %dks_77 = tt.split %dks_75 {async_task_id = array<i32: 3>} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked>
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x32xf32, #blocked>
    %dkN_78 = arith.mulf %dks_77, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
    %dkN_79 = arith.mulf %dks_76, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
    %dkN_80 = arith.mulf %dks_74, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
    %dkN_81 = arith.mulf %dks_73, %dkN {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked>
    %8 = arith.truncf %dkN_81 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %9 = ttg.convert_layout %8 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dk[%k_40, %c0_i32], %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %10 = arith.truncf %dkN_80 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %11 = ttg.convert_layout %10 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dk[%k_40, %c0_i32], %11 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %12 = arith.truncf %dkN_79 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %13 = ttg.convert_layout %12 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dk[%k_40, %c0_i32], %13 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    %14 = arith.truncf %dkN_78 {async_task_id = array<i32: 3>} : tensor<128x32xf32, #blocked> to tensor<128x32xbf16, #blocked>
    %15 = ttg.convert_layout %14 {async_task_id = array<i32: 3>} : tensor<128x32xbf16, #blocked> -> tensor<128x32xbf16, #blocked9>
    tt.descriptor_store %desc_dk[%k_40, %c0_i32], %15 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xbf16, #shared2>>, tensor<128x32xbf16, #blocked9>
    tt.return
  }
}

// ----
// Operand-D race fix: verify token-based producer_acquire fires for the
// dk/dv zeroing tmem_stores (tmem.start) in the BWD kernel.
//
// The dk zeroing tmem_store (task 0, gemm) and dk tmem_load (task 3,
// computation) are in DIFFERENT partitions, creating a cross-partition
// race. The operand-D race fix detects this and inserts:
//   tmem_load → consumer_release(tok) → producer_acquire(tok) → tmem_store
//
// Verify: producer_acquire (token) before dk and dv zeroing tmem_stores
// appear BEFORE the inner scf.for loop (they are initial zeroing ops).
//
// OPERANDD-LABEL: tt.func public @_attn_bwd
// OPERANDD: ttg.warp_specialize
// OPERANDD: default
// OPERANDD: nvws.producer_acquire
// OPERANDD: ttng.tmem_store {{.*}}tmem.start
// OPERANDD: nvws.producer_acquire
// OPERANDD: ttng.tmem_store {{.*}}tmem.start
// OPERANDD: scf.for
