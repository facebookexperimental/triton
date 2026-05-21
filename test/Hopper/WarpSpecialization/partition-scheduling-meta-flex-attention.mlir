// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta="merge-epilogue" | FileCheck %s

// Tests that flex attention (dpFactor=2, no epilogue stores, scf.if masking)
// gets two separate computation partitions with symmetric split.
// Without the fix, the pass collapses all computation ops into a single
// partition because:
// 1. No epilogue stores → hasEpilogue=false → no defaultPartition created
// 2. Without defaultPartition, Phase 4 load user propagation is skipped
// 3. Phase 5's greedy scheduleUsers absorbs all ops through the scf.if merge
// 4. Shared ops (scf.if) form cross-partition clusters in propagatePartitions
//
// The fix:
// 1. Creates defaultPartition when numDataPartitions > 1
// 2. Pre-assigns DataPartition ops to separate computation partitions
// 3. Pre-assigns shared MMA backward-slice ops to the default partition

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @flex_attention_data_partition_split
//
// --- Anchor ops: loads → load partition, MMAs → gemm partition ---
// CHECK: tt.descriptor_load {{.*}} ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}} ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: [[GEMM]]>
//
// --- QK tmem_loads go to two DIFFERENT computation partitions ---
// CHECK: ttng.tmem_load {{.*}} ttg.partition = array<i32: [[COMP_A:[0-9]+]]>
// CHECK: ttng.tmem_load {{.*}} ttg.partition = array<i32: [[COMP_B:[0-9]+]]>
//
// --- Split scf.if: the condition and else-yield operands defined outside the
//     scf.if must NOT have a ttg.partition attribute. If they keep partition 0
//     (from the original shared scf.if), doTaskIdPropagate expands the split
//     scf.if's task set to include partition 0, creating a cross-partition
//     SMEM channel that overflows shared memory. ---
// CHECK: arith.mulf
// CHECK-NOT: ttg.partition
// CHECK: arith.mulf
// CHECK-NOT: ttg.partition
// CHECK: arith.cmpi
// CHECK-NOT: ttg.partition
//
// --- Split scf.if: ops inside then-block get the parent's computation partition.
//     The scf.if itself inherits loop scheduling attributes from the original so
//     the downstream pipeliner can schedule channel ops derived from it. ---
// CHECK: scf.if
// CHECK: tt.splat {{.*}} ttg.partition = array<i32: [[COMP_A]]>
// CHECK: } {
// CHECK-SAME: loop.cluster
// CHECK-SAME: loop.stage
// CHECK-SAME: ttg.partition = array<i32: [[COMP_A]]>
// CHECK: scf.if
// CHECK: tt.splat {{.*}} ttg.partition = array<i32: [[COMP_B]]>
// CHECK: } {
// CHECK-SAME: loop.cluster
// CHECK-SAME: loop.stage
// CHECK-SAME: ttg.partition = array<i32: [[COMP_B]]>
//
// --- Correction/rescale ops (acc tmem_load, tmem_store) go to correction (partition 0) ---
// CHECK: ttng.tmem_load {{.*}} ttg.partition = array<i32: 0>
// CHECK: ttng.tmem_load {{.*}} ttg.partition = array<i32: 0>
// CHECK: ttng.tmem_store {{.*}} ttg.partition = array<i32: 0>
// CHECK: ttng.tmem_store {{.*}} ttg.partition = array<i32: 0>
//
// --- PV MMAs go to gemm partition ---
// CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: [[GEMM]]>
// CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: [[GEMM]]>
//
// --- Partition types: correction + gemm + load + two computation partitions ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types =
// CHECK-SAME: "correction"
// CHECK-SAME: "gemm"
// CHECK-SAME: "load"
// CHECK-SAME: "computation"
// CHECK-SAME: "computation"
//
// --- Post-loop ops go to correction partition (partition 0) ---
// CHECK: tmem_load {{.*}}ttg.partition = array<i32: 0>
// CHECK: tmem_load {{.*}}ttg.partition = array<i32: 0>
// CHECK: tt.store {{.*}}ttg.partition = array<i32: 0>

tt.func public @flex_attention_data_partition_split(
  %Q: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  %K: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  %V: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  %Out: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
  %LSE: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %KV_IDX: !tt.ptr<i32> {tt.divisibility = 16 : i32},
  %stride_qm: i32 {tt.divisibility = 16 : i32},
  %stride_kn: i32 {tt.divisibility = 16 : i32},
  %stride_vn: i32 {tt.divisibility = 16 : i32},
  %stride_om: i32 {tt.divisibility = 16 : i32},
  %Q_LEN: i32 {tt.divisibility = 16 : i32},
  %KV_LEN: i32 {tt.divisibility = 16 : i32},
  %SM_SCALE: f32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst_neg_inf = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %cst_zero_f = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %cst_zero_2d = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %cst_neg_inf_2d = arith.constant dense<0xFF800000> : tensor<128x128xf32, #blocked>
  %cst_scale = arith.constant dense<1.44269502> : tensor<128x128xf32, #blocked>
  %n_iters = arith.constant 8 : i32

  // Q descriptor and loads for two data partitions
  %desc_q_stride = arith.extsi %stride_qm : i32 to i64
  %desc_q = tt.make_tensor_descriptor %Q, [%Q_LEN, %c128_i32], [%desc_q_stride, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
  %q_0_data = tt.descriptor_load %desc_q[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
  %q_1_data = tt.descriptor_load %desc_q[%c128_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
  %q_0 = ttg.local_alloc %q_0_data : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
  %q_1 = ttg.local_alloc %q_1_data : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>

  // K/V descriptors
  %desc_k_stride = arith.extsi %stride_kn : i32 to i64
  %desc_k = tt.make_tensor_descriptor %K, [%KV_LEN, %c128_i32], [%desc_k_stride, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
  %desc_v_stride = arith.extsi %stride_vn : i32 to i64
  %desc_v = tt.make_tensor_descriptor %V, [%KV_LEN, %c128_i32], [%desc_v_stride, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>

  // QK and ACC TMEM allocations
  %qk_0, %qk_0_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %qk_1, %qk_1_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %acc_0, %acc_0_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %acc_1, %acc_1_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

  // Init accumulators
  %acc_0_init = ttng.tmem_store %cst_zero_2d, %acc_0[%acc_0_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %acc_1_init = ttng.tmem_store %cst_zero_2d, %acc_1[%acc_1_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

  // Sparse block index load (outside loop, used for masking)
  %kv_idx_val = tt.load %KV_IDX : !tt.ptr<i32>

  // Main attention loop — no epilogue stores inside, pointer-based stores
  // after the loop (like flex attention).
  %loop:8 = scf.for %i = %c0_i32 to %n_iters step %c1_i32
      iter_args(
        %l_i_0 = %cst_zero_f, %m_i_0 = %cst_neg_inf,
        %qk_tok_0 = %qk_0_tok, %acc_tok_0 = %acc_0_init,
        %l_i_1 = %cst_zero_f, %m_i_1 = %cst_neg_inf,
        %qk_tok_1 = %qk_1_tok, %acc_tok_1 = %acc_1_init
      ) -> (
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        !ttg.async.token, !ttg.async.token,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        !ttg.async.token, !ttg.async.token
      ) : i32 {

    // Load K and V
    %kv_offset = arith.muli %i, %c128_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %k_data = tt.descriptor_load %desc_k[%kv_offset, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
    %v_data = tt.descriptor_load %desc_v[%kv_offset, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
    %k_smem = ttg.local_alloc %k_data {loop.cluster = 3 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %k_trans = ttg.memdesc_trans %k_smem {loop.cluster = 3 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
    %v_smem = ttg.local_alloc %v_data {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>

    // QK MMA for both data partitions
    %qk_mma_0 = ttng.tc_gen5_mma %q_0, %k_trans, %qk_0[%qk_tok_0], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qk_mma_1 = ttng.tc_gen5_mma %q_1, %k_trans, %qk_1[%qk_tok_1], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Load QK results
    %qk_val_0, %qk_val_0_tok = ttng.tmem_load %qk_0[%qk_mma_0] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %qk_val_1, %qk_val_1_tok = ttng.tmem_load %qk_1[%qk_mma_1] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    // Scale QK
    %scores_0 = arith.mulf %qk_val_0, %cst_scale {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %scores_1 = arith.mulf %qk_val_1, %cst_scale {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>

    // scf.if for masking — this is the merge point that causes both data
    // partitions to collapse into one computation partition without the fix.
    // The then-block has transitive dependencies (splat → mulf) to test that
    // splitDataPartitionedIfOps clones the entire backward slice, not just
    // the immediate defining ops of the yield operands.
    %is_full = arith.cmpi sge, %i, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
    %masked:2 = scf.if %is_full -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>) {
      %full_scale = tt.splat %SM_SCALE {loop.cluster = 1 : i32, loop.stage = 1 : i32} : f32 -> tensor<128x128xf32, #blocked>
      %full_0 = arith.mulf %scores_0, %full_scale {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
      %full_1 = arith.mulf %scores_1, %full_scale {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
      scf.yield %full_0, %full_1 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
    } else {
      // Yield values defined OUTSIDE the scf.if — this exercises the fix that
      // removes stale partition assignments from such ops.
      scf.yield %scores_0, %scores_1 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
    } {loop.cluster = 1 : i32, loop.stage = 1 : i32}

    // Online softmax: m_ij, alpha, p, l_i — per data partition
    %m_ij_0 = "tt.reduce"(%masked#0) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %b0: f32):
      %max0 = arith.maxnumf %a0, %b0 : f32
      tt.reduce.return %max0 : f32
    }) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_ij_1 = "tt.reduce"(%masked#1) <{axis = 1 : i32}> ({
    ^bb0(%a1: f32, %b1: f32):
      %max1 = arith.maxnumf %a1, %b1 : f32
      tt.reduce.return %max1 : f32
    }) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %new_m_0 = arith.maxnumf %m_i_0, %m_ij_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %new_m_1 = arith.maxnumf %m_i_1, %m_ij_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha_0 = arith.subf %m_i_0, %new_m_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha_1 = arith.subf %m_i_1, %new_m_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha_exp_0 = math.exp2 %alpha_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha_exp_1 = math.exp2 %alpha_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // p = exp2(scores - m)
    %m_bcast_0 = tt.expand_dims %new_m_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %m_bcast2d_0 = tt.broadcast %m_bcast_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %m_bcast_1 = tt.expand_dims %new_m_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %m_bcast2d_1 = tt.broadcast %m_bcast_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %p_sub_0 = arith.subf %masked#0, %m_bcast2d_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %p_sub_1 = arith.subf %masked#1, %m_bcast2d_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %p_0 = math.exp2 %p_sub_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %p_1 = math.exp2 %p_sub_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>

    // l_i update
    %l_scaled_0 = arith.mulf %l_i_0, %alpha_exp_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_scaled_1 = arith.mulf %l_i_1, %alpha_exp_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_sum_0 = "tt.reduce"(%p_0) <{axis = 1 : i32}> ({
    ^bb0(%a2: f32, %b2: f32):
      %s0 = arith.addf %a2, %b2 : f32
      tt.reduce.return %s0 : f32
    }) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_sum_1 = "tt.reduce"(%p_1) <{axis = 1 : i32}> ({
    ^bb0(%a3: f32, %b3: f32):
      %s1 = arith.addf %a3, %b3 : f32
      tt.reduce.return %s1 : f32
    }) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %new_l_0 = arith.addf %l_scaled_0, %l_sum_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %new_l_1 = arith.addf %l_scaled_1, %l_sum_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // Rescale acc and accumulate P*V
    %alpha_1d_0 = tt.expand_dims %alpha_exp_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %alpha_2d_0 = tt.broadcast %alpha_1d_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %alpha_1d_1 = tt.expand_dims %alpha_exp_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %alpha_2d_1 = tt.broadcast %alpha_1d_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %acc_old_0, %acc_old_0_tok = ttng.tmem_load %acc_0[%acc_tok_0] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %acc_old_1, %acc_old_1_tok = ttng.tmem_load %acc_1[%acc_tok_1] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %acc_scaled_0 = arith.mulf %acc_old_0, %alpha_2d_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %acc_scaled_1 = arith.mulf %acc_old_1, %alpha_2d_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
    %acc_store_0 = ttng.tmem_store %acc_scaled_0, %acc_0[%acc_old_0_tok], %true {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_store_1 = ttng.tmem_store %acc_scaled_1, %acc_1[%acc_old_1_tok], %true {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // p → bf16 → tmem for PV MMA
    %p_bf16_0 = arith.truncf %p_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %p_bf16_1 = arith.truncf %p_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %p_tmem_0 = ttng.tmem_alloc %p_bf16_0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>
    %p_tmem_1 = ttng.tmem_alloc %p_bf16_1 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>

    // PV MMA
    %pv_0 = ttng.tc_gen5_mma %p_tmem_0, %v_smem, %acc_0[%acc_store_0], %true, %true {loop.cluster = 1 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %pv_1 = ttng.tc_gen5_mma %p_tmem_1, %v_smem, %acc_1[%acc_store_1], %true, %true {loop.cluster = 1 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    scf.yield %new_l_0, %new_m_0, %qk_val_0_tok, %pv_0,
              %new_l_1, %new_m_1, %qk_val_1_tok, %pv_1
      : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        !ttg.async.token, !ttg.async.token,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
        !ttg.async.token, !ttg.async.token
  } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}

  // Post-loop: pointer-based stores (NOT descriptor stores)
  // This is the key difference from FA — no epilogue stores.
  %final_acc_0, %_ = ttng.tmem_load %acc_0[%loop#3] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
  %final_acc_1, %__ = ttng.tmem_load %acc_1[%loop#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
  %out_bf16_0 = arith.truncf %final_acc_0 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
  %out_bf16_1 = arith.truncf %final_acc_1 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
  // Use pointer-based store (tt.store), not descriptor store
  %out_ptr = tt.splat %Out : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
  tt.store %out_ptr, %out_bf16_0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
  tt.store %out_ptr, %out_bf16_1 : tensor<128x128x!tt.ptr<bf16>, #blocked>

  tt.return
}

}
