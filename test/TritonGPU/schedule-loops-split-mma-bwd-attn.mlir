// RUN: triton-opt %s "-tritongpu-schedule-loops=num-stages=2" | FileCheck %s

// Backward attention kernel with 5 MMA ops. Verify that tt.split_mma on the
// ForOp produces the desired interleaved ordering:
//   qkT(iter i) -> dQ(iter i-1) -> dK(iter i-1) -> dpT(iter i) -> dV(iter i)
//
// dQ and dK are "deferred MMAs" whose operand A depends on tmem_load results
// of both qkT and dpT (via the dsT computation chain). They are placed at
// stage 1 (iter i-1) while qkT/dpT/dV are at stage 0 (iter i).

// CHECK-LABEL: @_attn_bwd_split_mma

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_split_mma(%desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_do: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_dq: !tt.tensordesc<tensor<128x128xf32, #shared1>>, %K_smem: !ttg.memdesc<128x128xbf16, #shared, #smem>, %V_smem: !ttg.memdesc<128x128xbf16, #shared, #smem>, %M_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.693147182> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %num_steps = arith.divsi %N_CTX, %c128_i32 : i32
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %M_splat = tt.splat %M_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %D_splat = tt.splat %D_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_1, %token_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %26 = ttng.tmem_store %cst_0, %result_5[%token_6], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %27 = ttng.tmem_store %cst_0, %result_1[%token_2], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %28:7 = scf.for %arg43 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg44 = %c0_i32, %arg45 = %false, %arg46 = %token, %arg47 = %27, %arg48 = %token_4, %arg49 = %26, %arg50 = %token_8) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      // --- Loads (latency ops) ---
      // q descriptor_load
      %q = tt.descriptor_load %desc_q[%arg44, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      %q_smem = ttg.local_alloc %q : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %qT = ttg.memdesc_trans %q_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %offs = tt.splat %arg44 : i32 -> tensor<128xi32, #blocked2>
      %offs_m = arith.addi %offs, %range : tensor<128xi32, #blocked2>
      %m_ptr = tt.addptr %M_splat, %offs_m : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %m = tt.load %m_ptr {tt.latency = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>

      // --- qkT MMA: K * qT (latency=1, self_latency=1) ---
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %result[%arg{{.*}}]
      // CHECK-SAME: loop.cluster = 1 : i32
      // CHECK-SAME: loop.stage = 0 : i32
      %qkT = ttng.tc_gen5_mma %K_smem, %qT, %result[%arg46], %false, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // --- softmax: tmem_load(qkT) -> subf -> exp2 ---
      %m_cvt = ttg.convert_layout %m : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %m_exp = tt.expand_dims %m_cvt {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
      %m_bcast = tt.broadcast %m_exp : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %qkT_val, %qkT_tok = ttng.tmem_load %result[%qkT] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %pT_sub = arith.subf %qkT_val, %m_bcast : tensor<128x128xf32, #blocked>
      %pT = math.exp2 %pT_sub : tensor<128x128xf32, #blocked>

      // --- do descriptor_load ---
      %do = tt.descriptor_load %desc_do[%arg44, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      %do_smem = ttg.local_alloc %do : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %pT_trunc = arith.truncf %pT : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %pT_tmem = ttng.tmem_alloc %pT_trunc : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>

      // --- dV MMA: pT * dO (latency=0, self_latency=1) ---
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %result_1[%arg{{.*}}]
      // CHECK-SAME: loop.cluster = 6 : i32
      // CHECK-SAME: loop.stage = 0 : i32
      %dV = ttng.tc_gen5_mma %pT_tmem, %do_smem, %result_1[%arg47], %arg45, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // --- Di load ---
      %d_ptr = tt.addptr %D_splat, %offs_m : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %di = tt.load %d_ptr {tt.latency = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
      %doT = ttg.memdesc_trans %do_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>

      // --- dpT MMA: V * doT (latency=1, self_latency=1) ---
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %result_3[%arg{{.*}}]
      // CHECK-SAME: loop.cluster = 4 : i32
      // CHECK-SAME: loop.stage = 0 : i32
      %dpT = ttng.tc_gen5_mma %V_smem, %doT, %result_3[%arg48], %false, %true {tt.latency = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // --- dsT computation: depends on BOTH tmem_load(qkT) and tmem_load(dpT) ---
      %di_cvt = ttg.convert_layout %di : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %di_exp = tt.expand_dims %di_cvt {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
      %di_bcast = tt.broadcast %di_exp : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %dpT_val, %dpT_tok = ttng.tmem_load %result_3[%dpT] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %ds_sub = arith.subf %dpT_val, %di_bcast : tensor<128x128xf32, #blocked>
      %ds_mul = arith.mulf %pT, %ds_sub : tensor<128x128xf32, #blocked>
      %dsT = arith.truncf %ds_mul : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %dsT_smem = ttg.local_alloc %dsT : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>

      // --- dK MMA: dsT * q (latency=0, self_latency=1) ---
      // Operand A (dsT_smem) depends on tmem_load of BOTH qkT and dpT -> deferred MMA
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %result_5[%arg{{.*}}]
      // CHECK-SAME: loop.cluster = 3 : i32
      // CHECK-SAME: loop.stage = 1 : i32
      %dK = ttng.tc_gen5_mma %dsT_smem, %q_smem, %result_5[%arg49], %arg45, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // --- dQ MMA: dsTᵀ * K (no tt.latency, has tmem_load after) ---
      // Operand A (dsTᵀ) depends on tmem_load of BOTH qkT and dpT -> deferred MMA
      // hasLoadsAfterMMA=true -> ordered before dK in cluster
      %dsTT = ttg.memdesc_trans %dsT_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      // CHECK: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %result_7[%arg{{.*}}]
      // CHECK-SAME: loop.cluster = 2 : i32
      // CHECK-SAME: loop.stage = 1 : i32
      %dQ = ttng.tc_gen5_mma %dsTT, %K_smem, %result_7[%arg50], %false, %true : !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // --- dQ consumers: tmem_load -> mulf -> descriptor_reduce (inside loop) ---
      %dQ_val, %dQ_tok = ttng.tmem_load %result_7[%dQ] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dQ_scaled = arith.mulf %dQ_val, %cst : tensor<128x128xf32, #blocked>
      %dQ_cvt = ttg.convert_layout %dQ_scaled : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
      tt.descriptor_reduce add, %desc_dq[%arg44, %c0_i32], %dQ_cvt : !tt.tensordesc<tensor<128x128xf32, #shared1>>, tensor<128x128xf32, #blocked1>

      %next = arith.addi %arg44, %c128_i32 : i32
      scf.yield %next, %true, %qkT_tok, %dV, %dpT_tok, %dK, %dQ_tok : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.split_mma}
    %result_9, %token_10 = ttng.tmem_load %result_1[%28#3] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %result_11, %token_12 = ttng.tmem_load %result_5[%28#5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %30 = ttg.convert_layout %result_9 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    tt.descriptor_store %desc_dq[%c0_i32, %c0_i32], %30 : !tt.tensordesc<tensor<128x128xf32, #shared1>>, tensor<128x128xf32, #blocked1>
    tt.return
  }
}
