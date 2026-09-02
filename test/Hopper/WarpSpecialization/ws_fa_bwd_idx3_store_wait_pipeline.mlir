#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#loc = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1113:1)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#loc60 = loc("desc_q"(#loc))
#loc61 = loc("desc_q.shape.0"(#loc))
#loc62 = loc("desc_q.shape.1"(#loc))
#loc63 = loc("desc_q.stride.0"(#loc))
#loc64 = loc("desc_q.stride.1"(#loc))
#loc65 = loc("desc_k"(#loc))
#loc66 = loc("desc_k.shape.0"(#loc))
#loc67 = loc("desc_k.shape.1"(#loc))
#loc68 = loc("desc_k.stride.0"(#loc))
#loc69 = loc("desc_k.stride.1"(#loc))
#loc70 = loc("desc_v"(#loc))
#loc71 = loc("desc_v.shape.0"(#loc))
#loc72 = loc("desc_v.shape.1"(#loc))
#loc73 = loc("desc_v.stride.0"(#loc))
#loc74 = loc("desc_v.stride.1"(#loc))
#loc75 = loc("sm_scale"(#loc))
#loc76 = loc("desc_do"(#loc))
#loc77 = loc("desc_do.shape.0"(#loc))
#loc78 = loc("desc_do.shape.1"(#loc))
#loc79 = loc("desc_do.stride.0"(#loc))
#loc80 = loc("desc_do.stride.1"(#loc))
#loc81 = loc("desc_dq"(#loc))
#loc82 = loc("desc_dq.shape.0"(#loc))
#loc83 = loc("desc_dq.shape.1"(#loc))
#loc84 = loc("desc_dq.stride.0"(#loc))
#loc85 = loc("desc_dq.stride.1"(#loc))
#loc86 = loc("desc_dk"(#loc))
#loc87 = loc("desc_dk.shape.0"(#loc))
#loc88 = loc("desc_dk.shape.1"(#loc))
#loc89 = loc("desc_dk.stride.0"(#loc))
#loc90 = loc("desc_dk.stride.1"(#loc))
#loc91 = loc("desc_dv"(#loc))
#loc92 = loc("desc_dv.shape.0"(#loc))
#loc93 = loc("desc_dv.shape.1"(#loc))
#loc94 = loc("desc_dv.stride.0"(#loc))
#loc95 = loc("desc_dv.stride.1"(#loc))
#loc96 = loc("desc_m"(#loc))
#loc97 = loc("desc_m.shape.0"(#loc))
#loc98 = loc("desc_m.stride.0"(#loc))
#loc99 = loc("desc_delta"(#loc))
#loc100 = loc("desc_delta.shape.0"(#loc))
#loc101 = loc("desc_delta.stride.0"(#loc))
#loc102 = loc("stride_z"(#loc))
#loc103 = loc("stride_h"(#loc))
#loc104 = loc("stride_tok"(#loc))
#loc105 = loc("BATCH"(#loc))
#loc106 = loc("H"(#loc))
#loc107 = loc("N_CTX"(#loc))
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd(%desc_q: !tt.tensordesc<64x128xf16, #shared> loc("desc_q"(#loc)), %desc_q.shape.0: i32 loc("desc_q.shape.0"(#loc)), %desc_q.shape.1: i32 loc("desc_q.shape.1"(#loc)), %desc_q.stride.0: i64 loc("desc_q.stride.0"(#loc)), %desc_q.stride.1: i64 loc("desc_q.stride.1"(#loc)), %desc_k: !tt.tensordesc<128x128xf16, #shared> loc("desc_k"(#loc)), %desc_k.shape.0: i32 loc("desc_k.shape.0"(#loc)), %desc_k.shape.1: i32 loc("desc_k.shape.1"(#loc)), %desc_k.stride.0: i64 loc("desc_k.stride.0"(#loc)), %desc_k.stride.1: i64 loc("desc_k.stride.1"(#loc)), %desc_v: !tt.tensordesc<128x128xf16, #shared> loc("desc_v"(#loc)), %desc_v.shape.0: i32 loc("desc_v.shape.0"(#loc)), %desc_v.shape.1: i32 loc("desc_v.shape.1"(#loc)), %desc_v.stride.0: i64 loc("desc_v.stride.0"(#loc)), %desc_v.stride.1: i64 loc("desc_v.stride.1"(#loc)), %sm_scale: f32 loc("sm_scale"(#loc)), %desc_do: !tt.tensordesc<64x128xf16, #shared> loc("desc_do"(#loc)), %desc_do.shape.0: i32 loc("desc_do.shape.0"(#loc)), %desc_do.shape.1: i32 loc("desc_do.shape.1"(#loc)), %desc_do.stride.0: i64 loc("desc_do.stride.0"(#loc)), %desc_do.stride.1: i64 loc("desc_do.stride.1"(#loc)), %desc_dq: !tt.tensordesc<64x32xf32, #shared1> loc("desc_dq"(#loc)), %desc_dq.shape.0: i32 loc("desc_dq.shape.0"(#loc)), %desc_dq.shape.1: i32 loc("desc_dq.shape.1"(#loc)), %desc_dq.stride.0: i64 loc("desc_dq.stride.0"(#loc)), %desc_dq.stride.1: i64 loc("desc_dq.stride.1"(#loc)), %desc_dk: !tt.tensordesc<128x64xf16, #shared> loc("desc_dk"(#loc)), %desc_dk.shape.0: i32 loc("desc_dk.shape.0"(#loc)), %desc_dk.shape.1: i32 loc("desc_dk.shape.1"(#loc)), %desc_dk.stride.0: i64 loc("desc_dk.stride.0"(#loc)), %desc_dk.stride.1: i64 loc("desc_dk.stride.1"(#loc)), %desc_dv: !tt.tensordesc<128x64xf16, #shared> loc("desc_dv"(#loc)), %desc_dv.shape.0: i32 loc("desc_dv.shape.0"(#loc)), %desc_dv.shape.1: i32 loc("desc_dv.shape.1"(#loc)), %desc_dv.stride.0: i64 loc("desc_dv.stride.0"(#loc)), %desc_dv.stride.1: i64 loc("desc_dv.stride.1"(#loc)), %desc_m: !tt.tensordesc<64xf32, #shared2> loc("desc_m"(#loc)), %desc_m.shape.0: i32 loc("desc_m.shape.0"(#loc)), %desc_m.stride.0: i64 loc("desc_m.stride.0"(#loc)), %desc_delta: !tt.tensordesc<64xf32, #shared2> loc("desc_delta"(#loc)), %desc_delta.shape.0: i32 loc("desc_delta.shape.0"(#loc)), %desc_delta.stride.0: i64 loc("desc_delta.stride.0"(#loc)), %stride_z: i32 {tt.divisibility = 16 : i32} loc("stride_z"(#loc)), %stride_h: i32 {tt.divisibility = 16 : i32} loc("stride_h"(#loc)), %stride_tok: i32 {tt.divisibility = 16 : i32} loc("stride_tok"(#loc)), %BATCH: i32 loc("BATCH"(#loc)), %H: i32 {tt.divisibility = 16 : i32} loc("H"(#loc)), %N_CTX: i32 {tt.divisibility = 16 : i32} loc("N_CTX"(#loc))) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false loc(#loc1)
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<64x32xf32, #blocked> loc(#loc163)
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32 loc(#loc163)
    %c96_i32 = arith.constant {async_task_id = array<i32: 0>} 96 : i32 loc(#loc163)
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32 loc(#loc164)
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32 loc(#loc109)
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32 loc(#loc109)
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32 loc(#loc109)
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true loc(#loc1)
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear> loc(#loc1)
    %bhid = tt.get_program_id z {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc110)
    %pid = tt.get_program_id x {async_task_id = array<i32: 2, 3>} : i32 loc(#loc111)
    %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 2>} : i32 loc(#loc165)
    %off_chz_1 = arith.extsi %off_chz {async_task_id = array<i32: 2>} : i32 to i64 loc(#loc166)
    %off_bh = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc167)
    %off_bh_2 = arith.muli %stride_h, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc168)
    %off_bh_3 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc169)
    %off_bh_4 = arith.muli %stride_z, %off_bh_3 {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc170)
    %off_bh_5 = arith.addi %off_bh_2, %off_bh_4 {async_task_id = array<i32: 0, 2, 3>} : i32 loc(#loc168)
    %off_bh_6 = arith.extsi %off_bh_5 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64 loc(#loc171)
    %off_bh_7 = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64 loc(#loc172)
    %off_bh_8 = arith.divsi %off_bh_6, %off_bh_7 {async_task_id = array<i32: 0, 2, 3>} : i64 loc(#loc172)
    %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32 loc(#loc173)
    %k = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64 loc(#loc174)
    %k_9 = arith.addi %off_bh_8, %k {async_task_id = array<i32: 2, 3>} : i64 loc(#loc174)
    %k_10 = arith.trunci %k_9 {async_task_id = array<i32: 2, 3>} : i64 to i32 loc(#loc175)
    %k_11 = tt.descriptor_load %desc_k[%k_10, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1> loc(#loc176)
    %k_12 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc176)
    ttg.local_store %k_11, %k_12 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc176)
    %v = tt.descriptor_load %desc_v[%k_10, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1> loc(#loc177)
    %v_13 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc177)
    ttg.local_store %v, %v_13 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked1> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc177)
    %num_steps = arith.divsi %N_CTX, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 loc(#loc178)
    %qkT, %qkT_14 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>} : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc186)
    %dpT, %dpT_15 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>} : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc187)
    %dv, %dv_16 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc188)
    %dk, %dk_17 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1, 3>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc189)
    %q = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc190)
    %m = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared2, #smem, mutable> loc(#loc191)
    %do = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc192)
    %ppT = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc193)
    %Di = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared2, #smem, mutable> loc(#loc194)
    %dsT_0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc195)
    %dsT_1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc196)
    %dq, %dq_18 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc197)
    %dk_19 = ttng.tmem_store %cst_0, %dk[%dk_17], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc189)
    %dv_20 = ttng.tmem_store %cst_0, %dv[%dv_16], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc188)
    %curr_m:7 = scf.for %blk_idx = %c0_i32 to %num_steps step %c1_i32 iter_args(%curr_m_35 = %c0_i32, %dv_36 = %false, %qkT_37 = %qkT_14, %dpT_38 = %dpT_15, %dv_39 = %dv_20, %dk_40 = %dk_19, %dq_41 = %dq_18) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %q_42 = arith.extsi %curr_m_35 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64 loc(#loc199)
      %q_43 = arith.addi %off_bh_8, %q_42 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 loc(#loc199)
      %q_44 = arith.trunci %q_43 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32 loc(#loc200)
      %q_45 = tt.descriptor_load %desc_q[%q_44, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1> loc(#loc190)
      ttg.local_store %q_45, %q {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x128xf16, #blocked1> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc190)
      %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable> loc(#loc201)
      %offs_m_start = arith.addi %off_chz_1, %q_42 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64 loc(#loc202)
      %m_46 = arith.trunci %offs_m_start {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : i64 to i32 loc(#loc203)
      %m_47 = tt.descriptor_load %desc_m[%m_46] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64xf32, #shared2> -> tensor<64xf32, #blocked2> loc(#loc191)
      ttg.local_store %m_47, %m {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable> loc(#loc191)
      %qkT_48 = ttng.tc_gen5_mma %k_12, %qT, %qkT[%qkT_37], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc186)
      %m_49 = ttg.local_load %m {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #blocked2> loc(#loc191)
      %pT = ttg.convert_layout %m_49 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc204)
      %pT_50 = ttg.convert_layout %m_47 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc204)
      %pT_51 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc205)
      %pT_52 = tt.expand_dims %pT_50 {async_task_id = array<i32: 2>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc205)
      %pT_53 = tt.broadcast %pT_51 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1> loc(#loc204)
      %qkT_54, %qkT_55 = ttng.tmem_load %qkT[%qkT_48] {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1> loc(#loc186)
      %pT_56 = arith.subf %qkT_54, %pT_53 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1> loc(#loc204)
      %pT_57 = math.exp2 %pT_56 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1> loc(#loc206)
      %do_58 = tt.descriptor_load %desc_do[%q_44, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1> loc(#loc192)
      ttg.local_store %do_58, %do {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x128xf16, #blocked1> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc192)
      %ppT_59 = arith.truncf %pT_57 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc193)
      %dv_60 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} true loc(#loc188)
      ttng.tmem_store %ppT_59, %ppT, %dv_60 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc188)
      %dpT_61 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable> loc(#loc207)
      %dpT_62 = ttng.tc_gen5_mma %v_13, %dpT_61, %dpT[%dpT_38], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc187)
      %Di_63 = tt.descriptor_load %desc_delta[%m_46] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<64xf32, #shared2> -> tensor<64xf32, #blocked2> loc(#loc194)
      ttg.local_store %Di_63, %Di {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #blocked2> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable> loc(#loc194)
      %dv_64 = ttng.tc_gen5_mma %ppT, %do, %dv[%dv_39], %dv_36, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc188)
      %Di_65 = ttg.local_load %Di {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #blocked2> loc(#loc194)
      %dsT = ttg.convert_layout %Di_65 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc208)
      %dsT_66 = ttg.convert_layout %Di_63 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc208)
      %dsT_67 = tt.expand_dims %dsT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc209)
      %dsT_68 = tt.expand_dims %dsT_66 {async_task_id = array<i32: 2>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc209)
      %dsT_69 = tt.broadcast %dsT_67 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1> loc(#loc208)
      %dpT_70, %dpT_71 = ttng.tmem_load %dpT[%dpT_62] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1> loc(#loc187)
      %dsT_72 = arith.subf %dpT_70, %dsT_69 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> loc(#loc208)
      %dsT_73 = arith.mulf %pT_57, %dsT_72 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> loc(#loc210)
      %dsT_74 = arith.truncf %dsT_73 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc211)
      ttg.local_store %dsT_74, %dsT_0 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc211)
      %dk_75 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} true loc(#loc189)
      ttng.tmem_store %dsT_74, %dsT_1, %dk_75 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> loc(#loc189)
      %dk_76 = ttng.tc_gen5_mma %dsT_1, %q, %dk[%dk_40], %dv_36, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc189)
      %dq_77 = ttg.memdesc_trans %dsT_0 {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared3, #smem, mutable> loc(#loc212)
      %dq_78 = ttng.tc_gen5_mma %dq_77, %k_12, %dq[%dq_41], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}"} : !ttg.memdesc<64x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> loc(#loc197)
      %dq_79, %dq_80 = ttng.tmem_load %dq[%dq_78] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear2> loc(#loc197)
      %dqs = tt.reshape %dq_79 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #linear2> -> tensor<64x2x64xf32, #linear3> loc(#loc220)
      %dqs_81 = tt.trans %dqs {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #linear3> -> tensor<64x64x2xf32, #linear4> loc(#loc220)
      %dqs_82 = ttg.convert_layout %dqs_81 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64x2xf32, #linear4> -> tensor<64x64x2xf32, #linear5> loc(#loc220)
      %dqs_83, %dqs_84 = tt.split %dqs_82 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64x2xf32, #linear5> -> tensor<64x64xf32, #linear6> loc(#loc220)
      %dqs_85 = tt.reshape %dqs_83 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #linear6> -> tensor<64x2x32xf32, #blocked3> loc(#loc224)
      %dqs_86 = tt.trans %dqs_85 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #blocked3> -> tensor<64x32x2xf32, #blocked4> loc(#loc224)
      %dqs_87, %dqs_88 = tt.split %dqs_86 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x2xf32, #blocked4> -> tensor<64x32xf32, #blocked> loc(#loc224)
      %dqs_89 = tt.reshape %dqs_84 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #linear6> -> tensor<64x2x32xf32, #blocked3> loc(#loc225)
      %dqs_90 = tt.trans %dqs_89 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #blocked3> -> tensor<64x32x2xf32, #blocked4> loc(#loc225)
      %dqs_91, %dqs_92 = tt.split %dqs_90 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x2xf32, #blocked4> -> tensor<64x32xf32, #blocked> loc(#loc225)
      %dqN = arith.mulf %dqs_87, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> loc(#loc214)
      %desc_dq_reduce_staging = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      ttg.local_store %dqN, %desc_dq_reduce_staging {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      %12 = ttng.async_tma_reduce add, %desc_dq[%q_44, %c0_i32] %desc_dq_reduce_staging {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<64x32xf32, #shared1>, !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> !ttg.async.token loc(#loc215)
      ttng.async_tma_store_token_wait %12   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token loc(#loc215)
      %dqN_93 = arith.mulf %dqs_88, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> loc(#loc214)
      %desc_dq_reduce_staging_94 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      ttg.local_store %dqN_93, %desc_dq_reduce_staging_94 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      %13 = ttng.async_tma_reduce add, %desc_dq[%q_44, %c32_i32] %desc_dq_reduce_staging_94 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<64x32xf32, #shared1>, !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> !ttg.async.token loc(#loc215)
      ttng.async_tma_store_token_wait %13   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token loc(#loc215)
      %dqN_95 = arith.mulf %dqs_91, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> loc(#loc214)
      %desc_dq_reduce_staging_96 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      ttg.local_store %dqN_95, %desc_dq_reduce_staging_96 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      %14 = ttng.async_tma_reduce add, %desc_dq[%q_44, %c64_i32] %desc_dq_reduce_staging_96 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<64x32xf32, #shared1>, !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> !ttg.async.token loc(#loc215)
      ttng.async_tma_store_token_wait %14   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token loc(#loc215)
      %dqN_97 = arith.mulf %dqs_92, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> loc(#loc214)
      %desc_dq_reduce_staging_98 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      ttg.local_store %dqN_97, %desc_dq_reduce_staging_98 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32xf32, #blocked> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> loc(#loc223)
      %15 = ttng.async_tma_reduce add, %desc_dq[%q_44, %c96_i32] %desc_dq_reduce_staging_98 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<64x32xf32, #shared1>, !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> !ttg.async.token loc(#loc215)
      ttng.async_tma_store_token_wait %15   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token loc(#loc215)
      %curr_m_99 = arith.addi %curr_m_35, %c64_i32 {async_task_id = array<i32: 0, 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32 loc(#loc216)
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_99, %true, %qkT_55, %dpT_71, %dv_64, %dk_76, %dq_80 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token loc(#loc164)
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue_to_computation = true, tt.scheduled_max_stage = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 202000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32} loc(#loc219)
    %dv_21, %dv_22 = ttng.tmem_load %dv[%curr_m#4] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear> loc(#loc188)
    %dk_23, %dk_24 = ttng.tmem_load %dk[%curr_m#5] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear> loc(#loc189)
    %dvs = tt.reshape %dv_21 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear7> loc(#loc217)
    %dvs_25 = tt.trans %dvs {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear7> -> tensor<128x64x2xf32, #linear8> loc(#loc217)
    %dvs_26, %dvs_27 = tt.split %dvs_25 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear8> -> tensor<128x64xf32, #linear1> loc(#loc217)
    %0 = arith.truncf %dvs_26 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc157)
    %1 = ttg.convert_layout %0 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc157)
    %desc_dv_staging = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc182)
    ttg.local_store %1, %desc_dv_staging {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked5> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc182)
    %2 = ttng.async_tma_copy_local_to_global %desc_dv[%k_10, %c0_i32] %desc_dv_staging {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc158)
    ttng.async_tma_store_token_wait %2   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc158)
    %3 = arith.truncf %dvs_27 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc157)
    %4 = ttg.convert_layout %3 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc157)
    %desc_dv_staging_28 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc182)
    ttg.local_store %4, %desc_dv_staging_28 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked5> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc182)
    %5 = ttng.async_tma_copy_local_to_global %desc_dv[%k_10, %c64_i32] %desc_dv_staging_28 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc158)
    ttng.async_tma_store_token_wait %5   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc158)
    %dks = tt.reshape %dk_23 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear7> loc(#loc218)
    %dks_29 = tt.trans %dks {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear7> -> tensor<128x64x2xf32, #linear8> loc(#loc218)
    %dks_30, %dks_31 = tt.split %dks_29 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear8> -> tensor<128x64xf32, #linear1> loc(#loc218)
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x64xf32, #linear1> loc(#loc184)
    %dkN_32 = arith.mulf %dks_30, %dkN {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> loc(#loc184)
    %6 = arith.truncf %dkN_32 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc161)
    %7 = ttg.convert_layout %6 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc161)
    %desc_dk_staging = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc185)
    ttg.local_store %7, %desc_dk_staging {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked5> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc185)
    %8 = ttng.async_tma_copy_local_to_global %desc_dk[%k_10, %c0_i32] %desc_dk_staging {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc162)
    ttng.async_tma_store_token_wait %8   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc162)
    %dkN_33 = arith.mulf %dks_31, %dkN {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> loc(#loc184)
    %9 = arith.truncf %dkN_33 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc161)
    %10 = ttg.convert_layout %9 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc161)
    %desc_dk_staging_34 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc185)
    ttg.local_store %10, %desc_dk_staging_34 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked5> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc185)
    %11 = ttng.async_tma_copy_local_to_global %desc_dk[%k_10, %c64_i32] %desc_dk_staging_34 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc162)
    ttng.async_tma_store_token_wait %11   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc162)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1061:14)
#loc3 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1148:5)
#loc4 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":790:9)
#loc5 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1145:12)
#loc6 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1146:11)
#loc7 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1049:16)
#loc8 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1049:15)
#loc9 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:28)
#loc10 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:16)
#loc11 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:52)
#loc12 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:40)
#loc13 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:15)
#loc14 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1050:14)
#loc15 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1055:15)
#loc16 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1058:23)
#loc17 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1058:22)
#loc18 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1058:9)
#loc19 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1059:9)
#loc20 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1060:17)
#loc21 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":707:15)
#loc22 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":799:30)
#loc23 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":719:15)
#loc24 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":721:15)
#loc25 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":733:15)
#loc26 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":702:9)
#loc27 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":705:9)
#loc28 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":715:10)
#loc29 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":717:11)
#loc30 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":720:14)
#loc31 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":727:11)
#loc32 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":734:14)
#loc33 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":702:23)
#loc34 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":702:22)
#loc35 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":703:10)
#loc36 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":704:20)
#loc37 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":705:22)
#loc38 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":710:23)
#loc39 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":710:29)
#loc40 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":710:10)
#loc41 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":719:25)
#loc42 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":726:17)
#loc43 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":726:23)
#loc44 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":726:11)
#loc45 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":734:21)
#loc46 = loc("/data/users/mren/MetaMain2/triton/python/triton/language/extra/subtile_ops.py":10:18)
#loc47 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":738:11)
#loc48 = loc("/data/users/mren/MetaMain2/triton/python/triton/language/extra/subtile_ops.py":11:16)
#loc49 = loc("/data/users/mren/MetaMain2/triton/python/triton/language/extra/subtile_ops.py":11:53)
#loc50 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":741:15)
#loc51 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":742:9)
#loc52 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":743:5)
#loc53 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1093:11)
#loc54 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1099:13)
#loc55 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1097:9)
#loc56 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1102:11)
#loc57 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1104:15)
#loc58 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1107:13)
#loc59 = loc("/data/users/mren/MetaMain2/triton/third_party/tlx/tutorials/fused_attention_ws_device_tma.py":1105:9)
#loc108 = loc(callsite(#loc2 at #loc3))
#loc109 = loc(callsite(#loc1 at #loc3))
#loc110 = loc("bhid"(#loc5))
#loc111 = loc("pid"(#loc6))
#loc112 = loc("off_chz"(#loc7))
#loc113 = loc("off_chz"(#loc8))
#loc114 = loc("off_bh"(#loc9))
#loc115 = loc("off_bh"(#loc10))
#loc116 = loc("off_bh"(#loc11))
#loc117 = loc("off_bh"(#loc12))
#loc118 = loc("off_bh"(#loc13))
#loc119 = loc("off_bh"(#loc14))
#loc120 = loc("start_n"(#loc15))
#loc121 = loc("k"(#loc16))
#loc122 = loc("k"(#loc17))
#loc123 = loc("k"(#loc18))
#loc124 = loc("v"(#loc19))
#loc125 = loc("num_steps"(#loc20))
#loc126 = loc("qkT"(#loc21))
#loc127 = loc("dpT"(#loc23))
#loc128 = loc("dv"(#loc24))
#loc129 = loc("dk"(#loc25))
#loc130 = loc("q"(#loc26))
#loc131 = loc("m"(#loc27))
#loc132 = loc("do"(#loc28))
#loc133 = loc("ppT"(#loc29))
#loc134 = loc("Di"(#loc30))
#loc135 = loc("dsT_0"(#loc31))
#loc136 = loc("dsT_1"(#loc31))
#loc137 = loc("dq"(#loc32))
#loc138 = loc("dk"(#loc4))
#loc139 = loc("q"(#loc33))
#loc140 = loc("q"(#loc34))
#loc141 = loc("qT"(#loc35))
#loc142 = loc("offs_m_start"(#loc36))
#loc143 = loc("m"(#loc37))
#loc144 = loc("pT"(#loc38))
#loc145 = loc("pT"(#loc39))
#loc146 = loc("pT"(#loc40))
#loc147 = loc("dpT"(#loc41))
#loc148 = loc("dsT"(#loc42))
#loc149 = loc("dsT"(#loc43))
#loc150 = loc("dsT"(#loc44))
#loc151 = loc("dsT"(#loc31))
#loc152 = loc("dq"(#loc45))
#loc153 = loc("dqs"(#loc47))
#loc154 = loc("dqN"(#loc50))
#loc155 = loc("curr_m"(#loc52))
#loc156 = loc("dvs"(#loc53))
#loc157 = loc(callsite(#loc54 at #loc3))
#loc158 = loc(callsite(#loc55 at #loc3))
#loc159 = loc("dks"(#loc56))
#loc160 = loc("dkN"(#loc57))
#loc161 = loc(callsite(#loc58 at #loc3))
#loc162 = loc(callsite(#loc59 at #loc3))
#loc163 = loc(callsite(#loc1 at #loc108))
#loc164 = loc(callsite(#loc4 at #loc108))
#loc165 = loc(callsite(#loc112 at #loc3))
#loc166 = loc(callsite(#loc113 at #loc3))
#loc167 = loc(callsite(#loc114 at #loc3))
#loc168 = loc(callsite(#loc115 at #loc3))
#loc169 = loc(callsite(#loc116 at #loc3))
#loc170 = loc(callsite(#loc117 at #loc3))
#loc171 = loc(callsite(#loc118 at #loc3))
#loc172 = loc(callsite(#loc119 at #loc3))
#loc173 = loc(callsite(#loc120 at #loc3))
#loc174 = loc(callsite(#loc121 at #loc3))
#loc175 = loc(callsite(#loc122 at #loc3))
#loc176 = loc(callsite(#loc123 at #loc3))
#loc177 = loc(callsite(#loc124 at #loc3))
#loc178 = loc(callsite(#loc125 at #loc3))
#loc179 = loc(callsite(#loc22 at #loc108))
#loc180 = loc("dv"(#loc138))
#loc181 = loc(callsite(#loc156 at #loc3))
#loc182 = loc("desc_dv_staging"(#loc158))
#loc183 = loc(callsite(#loc159 at #loc3))
#loc184 = loc(callsite(#loc160 at #loc3))
#loc185 = loc("desc_dk_staging"(#loc162))
#loc186 = loc(callsite(#loc126 at #loc179))
#loc187 = loc(callsite(#loc127 at #loc179))
#loc188 = loc(callsite(#loc128 at #loc179))
#loc189 = loc(callsite(#loc129 at #loc179))
#loc190 = loc(callsite(#loc130 at #loc179))
#loc191 = loc(callsite(#loc131 at #loc179))
#loc192 = loc(callsite(#loc132 at #loc179))
#loc193 = loc(callsite(#loc133 at #loc179))
#loc194 = loc(callsite(#loc134 at #loc179))
#loc195 = loc(callsite(#loc135 at #loc179))
#loc196 = loc(callsite(#loc136 at #loc179))
#loc197 = loc(callsite(#loc137 at #loc179))
#loc198 = loc("curr_m"(#loc180))
#loc199 = loc(callsite(#loc139 at #loc179))
#loc200 = loc(callsite(#loc140 at #loc179))
#loc201 = loc(callsite(#loc141 at #loc179))
#loc202 = loc(callsite(#loc142 at #loc179))
#loc203 = loc(callsite(#loc143 at #loc179))
#loc204 = loc(callsite(#loc144 at #loc179))
#loc205 = loc(callsite(#loc145 at #loc179))
#loc206 = loc(callsite(#loc146 at #loc179))
#loc207 = loc(callsite(#loc147 at #loc179))
#loc208 = loc(callsite(#loc148 at #loc179))
#loc209 = loc(callsite(#loc149 at #loc179))
#loc210 = loc(callsite(#loc150 at #loc179))
#loc211 = loc(callsite(#loc151 at #loc179))
#loc212 = loc(callsite(#loc152 at #loc179))
#loc213 = loc(callsite(#loc153 at #loc179))
#loc214 = loc(callsite(#loc154 at #loc179))
#loc215 = loc(callsite(#loc51 at #loc179))
#loc216 = loc(callsite(#loc155 at #loc179))
#loc217 = loc(callsite(#loc46 at #loc181))
#loc218 = loc(callsite(#loc46 at #loc183))
#loc219 = loc(callsite(#loc198 at #loc108))
#loc220 = loc(callsite(#loc46 at #loc213))
#loc221 = loc(callsite(#loc48 at #loc213))
#loc222 = loc(callsite(#loc49 at #loc213))
#loc223 = loc("desc_dq_reduce_staging"(#loc215))
#loc224 = loc(callsite(#loc46 at #loc221))
#loc225 = loc(callsite(#loc46 at #loc222))


// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=202000" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s --check-prefix=PLANNER
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=202000" --nvgpu-test-annotate-tma-store-waits --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s --check-prefix=ANNOTATE
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=202000" --nvgpu-test-annotate-tma-store-waits --nvgpu-test-tma-store-token-wait-reorder --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s --check-prefix=REORDER
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=202000" --nvgpu-test-annotate-tma-store-waits --nvgpu-test-tma-store-token-wait-reorder --nvgpu-tma-store-token-wait-lowering --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s --check-prefix=LOWER

// Generated from FA backward config idx 3 immediately after
// doBufferAllocation: BLOCK_M1=64, EPILOGUE_SUBTILE=2, DQ_SUBTILE=4,
// BWD_DOT_ATTRS=_BWD_DOT_ATTRS_BM64_TMEM.
//
// The budget comes from tt.smem_budget on the loop, not from the RUN-line
// option; the two are kept in sync only for readability. It is 202000 rather
// than the shipped 200000 because the point of this fixture is the dV
// two-copy store-wait ring: at 200000 the effective budget is 195676 (2 KiB of
// auxiliary SMEM) and Phase 3.7's dV bump lands at 197632, so the planner
// reverts dV to one copy and there is no rotation left to check. dK must still
// miss the bump for the asymmetry below to hold, so do not raise this further.
// Whether the shipped 200000 config should still fit a two-copy dV is tracked
// separately (T284939236).

// PLANNER-LABEL: tt.func public @_attn_bwd
// PLANNER-COUNT-4: %desc_dq_reduce_staging{{.*}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = [[DQ:[0-9]+]] : i32, buffer.tmaStaging = 2 : i32}
// PLANNER-COUNT-2: %desc_dv_staging{{.*}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = [[DV:[0-9]+]] : i32, buffer.tmaStaging = 1 : i32}
// PLANNER-COUNT-2: %desc_dk_staging{{.*}} = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[DK:[0-9]+]] : i32, buffer.tmaStaging = 1 : i32}

// Only dQ is inside the merged computation loop. Its four token waits inherit
// the two-copy ring and the stable wait_group(1) lowering contract.
// ANNOTATE-LABEL: tt.func public @_attn_bwd
// ANNOTATE-COUNT-4: ttng.async_tma_store_token_wait {{.*}}can_rotate_by_buffer_count = 2 : i32
// ANNOTATE-NOT: planned_pending_count

// Reordering materializes the two wraparound waits, keeps the two
// same-iteration token waits at their future slot writers, and adds one drain.
// REORDER-LABEL: tt.func public @_attn_bwd
// REORDER-COUNT-2: ttng.async_tma_store_wait {{.*}}pendings = 1 : i32
// REORDER-COUNT-2: ttng.async_tma_store_token_wait {{.*}}planned_pending_count = 1 : i32
// REORDER: ttng.async_tma_store_wait {{.*}}pendings = 0 : i32

// Final lowering matches the one-CTA TLX topology: dQ rotates two staging
// slots with wait_group(1); dV/dK use wait_group(0) at their overwrite and
// drain points because idx 3 has one effective straight-line staging view.
// LOWER-LABEL: tt.func public @_attn_bwd
// Input descriptor loads must still land in their shared-memory buffers before
// the corresponding consumers run. nvws.descriptor_load names the destination
// buffer directly, so the load and its store are one op here rather than the
// tt.descriptor_load + ttg.local_store pair this fixture was captured with.
// LOWER: nvws.descriptor_load %desc_q{{.*}} %q
// LOWER: nvws.descriptor_load %desc_do{{.*}} %do

// dQ loads its accumulator, then rotates four reductions over two staging
// slots. Every wait_group(1) is immediately before the next staging writer;
// the loop is followed by a wait_group(0) drain.
// LOWER: ttng.tmem_load %dq
// LOWER: ttng.async_tma_store_wait {{.*}}pendings = 1 : i32
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DQ0:desc_dq_reduce_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_reduce {{.*}} %[[DQ0]]
// LOWER: ttng.async_tma_store_wait {{.*}}pendings = 1 : i32
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DQ1:desc_dq_reduce_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_reduce {{.*}} %[[DQ1]]
// LOWER: ttng.async_tma_store_wait {pendings = 1 : i32}
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DQ2:desc_dq_reduce_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_reduce {{.*}} %[[DQ2]]
// LOWER: ttng.async_tma_store_wait {pendings = 1 : i32}
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DQ3:desc_dq_reduce_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_reduce {{.*}} %[[DQ3]]
// LOWER: scf.yield
// LOWER: ttng.async_tma_store_wait {{.*}}pendings = 0 : i32

// The straight-line epilogue loads dV and dK before staging either output.
// dV's final drain is delayed across dK preparation to the first dK writer.
// LOWER: {{.*}} = ttng.tmem_load %dv
// LOWER-NEXT: {{.*}} = ttng.tmem_load %dk
// LOWER: ttg.local_store {{.*}}, %[[DV0:desc_dv_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_copy_local_to_global %desc_dv{{.*}} %[[DV0]]
// LOWER: ttng.async_tma_store_wait {pendings = 0 : i32}
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DV1:desc_dv_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_copy_local_to_global %desc_dv{{.*}} %[[DV1]]
// LOWER: ttng.async_tma_store_wait {pendings = 0 : i32}
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DK0:desc_dk_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_copy_local_to_global %desc_dk{{.*}} %[[DK0]]
// LOWER: ttng.async_tma_store_wait {pendings = 0 : i32}
// LOWER-NEXT: ttg.local_store {{.*}}, %[[DK1:desc_dk_staging[^ ]*]]
// LOWER-NEXT: {{.*}} = ttng.async_tma_copy_local_to_global %desc_dk{{.*}} %[[DK1]]
// LOWER-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
// LOWER-NOT: ttng.async_tma_store_token_wait
