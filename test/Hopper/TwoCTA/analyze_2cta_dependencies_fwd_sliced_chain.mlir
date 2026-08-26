// RUN: triton-opt %s --nvgpu-analyze-2cta-dependencies | FileCheck %s --check-prefix=DEP

// Full TTGIR captured immediately before NVGPUAnalyze2CTADependencies from
// test_dot_2cta.py::test_tl_dot_2cta_sliced_dependent_chain: a 2-CTA qK MMA
// whose result is statically subtiled by reshape -> rank-3 permute -> split and
// fed to two serial 2-CTA PV MMAs.
//
// A rank-3 transpose here only exposes a size-two axis to tt.split; it does not
// transpose the logical MMA operand across CTAs, so both dependent MMAs are
// collective contractions. Classifying them as peer gathers instead makes
// Plan2CTAExchange fail with "requires_peer_gather expects a transposed
// local_alloc operand with a tensor source". Only the rank-2 matrix transpose
// of the dQ path needs a peer gather.
// DEP-COUNT-2: ttng.two_cta_dependency = "collective_contraction"
// DEP-NOT: requires_peer_gather

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#loc = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":56:1)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#loc17 = loc("q_ptr"(#loc))
#loc18 = loc("k_ptr"(#loc))
#loc19 = loc("v_ptr"(#loc))
#loc20 = loc("o_ptr"(#loc))
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_tl_dot_2cta_sliced_dependent_chain_kernel(%q_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("q_ptr"(#loc)), %k_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("k_ptr"(#loc)), %v_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("v_ptr"(#loc)), %o_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("o_ptr"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear> loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %v1 = arith.constant 64 : i32 loc(#loc21)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c128_i64 = arith.constant 128 : i64 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %q_desc = tt.make_tensor_descriptor %q_ptr, [%c256_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16> loc(#loc22)
    %k_desc = tt.make_tensor_descriptor %k_ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16> loc(#loc23)
    %v_desc = tt.make_tensor_descriptor %v_ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16> loc(#loc24)
    %o_desc = tt.make_tensor_descriptor %o_ptr, [%c256_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16> loc(#loc25)
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<256x128xbf16> -> tensor<256x128xbf16, #blocked> loc(#loc26)
    %q_0 = ttg.local_alloc %q : (tensor<256x128xbf16, #blocked>) -> !ttg.memdesc<256x128xbf16, #shared, #smem> loc(#loc26)
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked> loc(#loc27)
    %v0 = tt.descriptor_load %v_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16, #blocked> loc(#loc28)
    %v0_1 = ttg.local_alloc %v0 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem> loc(#loc28)
    %v1_2 = tt.descriptor_load %v_desc[%v1, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16, #blocked> loc(#loc21)
    %v1_3 = ttg.local_alloc %v1_2 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem> loc(#loc21)
    %qk = ttg.local_alloc %k : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc29)
    %qk_4 = ttg.memdesc_trans %qk {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem> loc(#loc29)
    %qk_5, %qk_6 = ttng.tmem_alloc %cst : (tensor<256x128xf32, #linear>) -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc30)
    %qk_7 = ttng.tc_gen5_mma %q_0, %qk_4, %qk_5[%qk_6], %true, %true {two_ctas} : !ttg.memdesc<256x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc30)
    %qk_8, %qk_9 = ttng.tmem_load %qk_5[%qk_7] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear> loc(#loc30)
    %qk_10 = arith.truncf %qk_8 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear> loc(#loc30)
    %0 = tt.reshape %qk_10 : tensor<256x128xbf16, #linear> -> tensor<256x2x64xbf16, #linear1> loc(#loc12)
    %1 = tt.trans %0 {order = array<i32: 0, 2, 1>} : tensor<256x2x64xbf16, #linear1> -> tensor<256x64x2xbf16, #linear2> loc(#loc12)
    %outLHS, %outRHS = tt.split %1 : tensor<256x64x2xbf16, #linear2> -> tensor<256x64xbf16, #linear3> loc(#loc12)
    %2 = ttg.local_alloc %outLHS : (tensor<256x64xbf16, #linear3>) -> !ttg.memdesc<256x64xbf16, #shared, #smem> loc(#loc12)
    %3 = ttg.local_alloc %outRHS : (tensor<256x64xbf16, #linear3>) -> !ttg.memdesc<256x64xbf16, #shared, #smem> loc(#loc12)
    %acc, %acc_11 = ttng.tmem_alloc %cst : (tensor<256x128xf32, #linear>) -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc31)
    %acc_12 = ttng.tc_gen5_mma %2, %v0_1, %acc[%acc_11], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc31)
    %acc_13, %acc_14 = ttng.tmem_load %acc[%acc_12] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear> loc(#loc31)
    %acc_15, %acc_16 = ttng.tmem_alloc %acc_13 : (tensor<256x128xf32, #linear>) -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc32)
    %acc_17 = ttng.tc_gen5_mma %3, %v1_3, %acc_15[%acc_16], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc32)
    %acc_18, %acc_19 = ttng.tmem_load %acc_15[%acc_17] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear> loc(#loc32)
    %4 = arith.truncf %acc_18 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear> loc(#loc15)
    %5 = ttg.convert_layout %4 : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked> loc(#loc16)
    tt.descriptor_store %o_desc[%c0_i32, %c0_i32], %5 : !tt.tensordesc<256x128xbf16>, tensor<256x128xbf16, #blocked> loc(#loc16)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":65:10)
#loc3 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":57:14)
#loc4 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":58:14)
#loc5 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":59:14)
#loc6 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":60:14)
#loc7 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":62:9)
#loc8 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":63:9)
#loc9 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":64:10)
#loc10 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":66:20)
#loc11 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":66:10)
#loc12 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":68:14)
#loc13 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":70:11)
#loc14 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":71:11)
#loc15 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":72:26)
#loc16 = loc("/home/mren/MetaMain2/triton-splits/python/test/unit/language/test_dot_2cta.py":72:5)
#loc21 = loc("v1"(#loc2))
#loc22 = loc("q_desc"(#loc3))
#loc23 = loc("k_desc"(#loc4))
#loc24 = loc("v_desc"(#loc5))
#loc25 = loc("o_desc"(#loc6))
#loc26 = loc("q"(#loc7))
#loc27 = loc("k"(#loc8))
#loc28 = loc("v0"(#loc9))
#loc29 = loc("qk"(#loc10))
#loc30 = loc("qk"(#loc11))
#loc31 = loc("acc"(#loc13))
#loc32 = loc("acc"(#loc14))



