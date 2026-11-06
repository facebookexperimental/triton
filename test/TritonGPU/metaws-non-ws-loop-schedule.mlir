// RUN: triton-opt %s -allow-unregistered-dialect "-tritongpu-assign-latencies=num-stages=3 use-meta-ws=true" "-tritongpu-schedule-loops=num-stages=3 use-meta-ws=true" -o %t.metaws
// RUN: triton-opt %s -allow-unregistered-dialect "-tritongpu-assign-latencies=num-stages=3 use-meta-ws=false" "-tritongpu-schedule-loops=num-stages=3 use-meta-ws=false" -o %t.ref
// RUN: diff %t.metaws %t.ref

// use-meta-ws only changes latencies and schedules of loops AutoWS
// warp-specializes. This loop has no tt.warp_specialize, so it is
// software-pipelined exactly as with use-meta-ws=false.

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 168 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_conditional_mask(%desc_q: !tt.tensordesc<128x128xf16, #shared>, %desc_k: !tt.tensordesc<64x128xf16, #shared>, %desc_v: !tt.tensordesc<64x128xf16, #shared>, %desc_o: !tt.tensordesc<128x128xf16, #shared>, %N_CTX: i32 {tt.divisibility = 16 : i32}, %cond: i1) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst_neg = arith.constant dense<-1.000000e+06> : tensor<128x64xf32, #blocked>
    %l_i = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_i = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %q = tt.descriptor_load %desc_q[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked3>
    %q_buf = ttg.local_alloc %q : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %qk_tmem, %qk_tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_tmem, %acc_tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_stored = ttng.tmem_store %acc_init, %acc_tmem[%acc_tok0], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
    %res:5 = scf.for %iv = %c0_i32 to %N_CTX step %c64_i32 iter_args(%li = %l_i, %mi = %m_i, %off = %c0_i32, %qk_tok = %qk_tok0, %acc_tok = %acc_stored) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.async.token, !ttg.async.token) : i32 {
      %k = tt.descriptor_load %desc_k[%off, %c0_i32] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked3>
      %k_buf = ttg.local_alloc %k : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %k_t = ttg.memdesc_trans %k_buf {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
      %qk_done = ttng.tc_gen5_mma %q_buf, %k_t, %qk_tmem[%qk_tok], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %qk_val, %qk_tok_out = ttng.tmem_load %qk_tmem[%qk_done] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      %masked_qk = scf.if %cond -> (tensor<128x64xf32, #blocked>) {
        %masked = arith.addf %qk_val, %cst_neg : tensor<128x64xf32, #blocked>
        scf.yield %masked : tensor<128x64xf32, #blocked>
      } else {
        scf.yield %qk_val : tensor<128x64xf32, #blocked>
      }
      %m_ij = "tt.reduce"(%masked_qk) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %mx = arith.maxnumf %a, %b : f32
        tt.reduce.return %mx : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_new = arith.maxnumf %mi, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_exp = tt.expand_dims %m_new {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %m_bc = tt.broadcast %m_exp : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %qk_sub = arith.subf %masked_qk, %m_bc : tensor<128x64xf32, #blocked>
      %p = math.exp2 %qk_sub : tensor<128x64xf32, #blocked>
      %alpha = arith.subf %mi, %m_new : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_exp = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%a2: f32, %b2: f32):
        %s = arith.addf %a2, %b2 : f32
        tt.reduce.return %s : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_val, %acc_tok_ld = ttng.tmem_load %acc_tmem[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %rs = tt.reshape %acc_val : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %tr = tt.trans %rs {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %lhs, %rhs = tt.split %tr : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked>
      %a_exp = tt.expand_dims %alpha_exp {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %a_bc = tt.broadcast %a_exp : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %lhs_s = arith.mulf %lhs, %a_bc : tensor<128x64xf32, #blocked>
      %rhs_s = arith.mulf %rhs, %a_bc : tensor<128x64xf32, #blocked>
      %joined = tt.join %lhs_s, %rhs_s : tensor<128x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked5>
      %tr2 = tt.trans %joined {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x2x64xf32, #blocked4>
      %acc_new = tt.reshape %tr2 : tensor<128x2x64xf32, #blocked4> -> tensor<128x128xf32, #blocked1>
      %v = tt.descriptor_load %desc_v[%off, %c0_i32] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked3>
      %v_buf = ttg.local_alloc %v : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %p_f16 = arith.truncf %p : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %p_tmem = ttng.tmem_alloc %p_f16 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>
      %acc_st = ttng.tmem_store %acc_new, %acc_tmem[%acc_tok_ld], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %pv_done = ttng.tc_gen5_mma %p_tmem, %v_buf, %acc_tmem[%acc_st], %true, %true : !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %l_new = arith.mulf %li, %alpha_exp : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_upd = arith.addf %l_new, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %off_next = arith.addi %off, %c64_i32 : i32
      scf.yield %l_upd, %m_new, %off_next, %qk_tok_out, %pv_done : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}
