// FA backward (_attn_bwd) BM64_TMEM modulo-exhaustive top-K test (name-loc CHECKs).
// Input: pre-modulo TTGIR with the config's tt.autows stripped so the modulo
// scheduler owns the loop and annotates the tc_gen5_mma ops. STANDALONE_MODULO
// exhaustive emits ONLY the SWP schedule and defers partitioning to PSM.
// Printed with -mlir-print-debuginfo -mlir-print-local-scope so each key op is
// matched by its NAME location (q/k/do/qk_trans/dq/dk/...) alongside its
// loop.stage/loop.cluster, across the top-4 picks.

// RUN: env STANDALONE_MODULO=1 TRITON_USE_MODULO_SCHEDULE=contracted \
// RUN:   TRITON_MODULO_TOPK=5 TRITON_MODULO_PICK=1 \
// RUN:   triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix=CONTRACTED \
// RUN:       --implicit-check-not=tt.autows --implicit-check-not=tt.modulo_ii --implicit-check-not=ttg.partition

// RUN: env STANDALONE_MODULO=1 STANDALONE_MODULO_MAX_STAGE_DIFF=2 \
// RUN:   TRITON_USE_MODULO_SCHEDULE=exhaustive TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=0 \
// RUN:   triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix=P0 \
// RUN:       --implicit-check-not=tt.autows --implicit-check-not=tt.modulo_ii --implicit-check-not=ttg.partition

// RUN: env STANDALONE_MODULO=1 STANDALONE_MODULO_MAX_STAGE_DIFF=2 \
// RUN:   TRITON_USE_MODULO_SCHEDULE=exhaustive TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=1 \
// RUN:   triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix=P1 \
// RUN:       --implicit-check-not=tt.autows --implicit-check-not=tt.modulo_ii --implicit-check-not=ttg.partition

// RUN: env STANDALONE_MODULO=1 STANDALONE_MODULO_MAX_STAGE_DIFF=2 \
// RUN:   TRITON_USE_MODULO_SCHEDULE=exhaustive TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=2 \
// RUN:   triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix=P2 \
// RUN:       --implicit-check-not=tt.autows --implicit-check-not=tt.modulo_ii --implicit-check-not=ttg.partition

// RUN: env STANDALONE_MODULO=1 STANDALONE_MODULO_MAX_STAGE_DIFF=2 \
// RUN:   TRITON_USE_MODULO_SCHEDULE=exhaustive TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=3 \
// RUN:   triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix=P3 \
// RUN:       --implicit-check-not=tt.autows --implicit-check-not=tt.modulo_ii --implicit-check-not=ttg.partition

// Contracted pick 1: GEMM stages 0/1 with global modulo cluster order
// qkT < dk <= dq < dpT <= dv. Loads and stores are deliberately unchecked.
// CONTRACTED-LABEL: @_attn_bwd
// CONTRACTED: ttng.tc_gen5_mma {{.*}}loop.cluster = 4 : i32, loop.stage = 0 : i32{{.*}}"qkT"
// CONTRACTED: ttng.tc_gen5_mma {{.*}}loop.cluster = 36 : i32, loop.stage = 0 : i32{{.*}}"dpT"
// CONTRACTED: ttng.tc_gen5_mma {{.*}}loop.cluster = 37 : i32, loop.stage = 0 : i32{{.*}}"dv"
// CONTRACTED: ttng.tc_gen5_mma {{.*}}loop.cluster = 20 : i32, loop.stage = 1 : i32{{.*}}"dk"
// CONTRACTED: ttng.tc_gen5_mma {{.*}}loop.cluster = 22 : i32, loop.stage = 1 : i32{{.*}}"dq"
// CONTRACTED: tt.scheduled_max_stage = 1 : i32

// pick 0: each key op by NAME loc + loop.stage/loop.cluster, in program order.
// P0-LABEL: @_attn_bwd
// P0:      tt.descriptor_load {{.*}}loop.cluster = 3 : i32, loop.stage = 0 : i32{{.*}}"q"
// P0:      tt.descriptor_load {{.*}}loop.cluster = 6 : i32, loop.stage = 0 : i32{{.*}}"m"
// P0:      ttng.tc_gen5_mma {{.*}}loop.cluster = 11 : i32, loop.stage = 0 : i32{{.*}}"qkT"
// P0:      tt.descriptor_load {{.*}}loop.cluster = 7 : i32, loop.stage = 0 : i32{{.*}}"do"
// P0:      ttng.tc_gen5_mma {{.*}}loop.cluster = 12 : i32, loop.stage = 0 : i32{{.*}}"dpT"
// P0:      tt.descriptor_load {{.*}}loop.cluster = 8 : i32, loop.stage = 0 : i32{{.*}}"Di"
// P0:      ttng.tc_gen5_mma {{.*}}loop.cluster = 7 : i32, loop.stage = 1 : i32{{.*}}"dv"
// P0:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 1 : i32{{.*}}"dk"
// P0:      ttng.tc_gen5_mma {{.*}}loop.cluster = 10 : i32, loop.stage = 1 : i32{{.*}}"dq"
// P0:      tt.scheduled_max_stage = 2 : i32

// pick 1: each key op by NAME loc + loop.stage/loop.cluster, in program order.
// P1-LABEL: @_attn_bwd
// P1:      tt.descriptor_load {{.*}}loop.cluster = 3 : i32, loop.stage = 0 : i32{{.*}}"q"
// P1:      tt.descriptor_load {{.*}}loop.cluster = 6 : i32, loop.stage = 0 : i32{{.*}}"m"
// P1:      ttng.tc_gen5_mma {{.*}}loop.cluster = 11 : i32, loop.stage = 0 : i32{{.*}}"qkT"
// P1:      tt.descriptor_load {{.*}}loop.cluster = 7 : i32, loop.stage = 0 : i32{{.*}}"do"
// P1:      ttng.tc_gen5_mma {{.*}}loop.cluster = 12 : i32, loop.stage = 0 : i32{{.*}}"dpT"
// P1:      tt.descriptor_load {{.*}}loop.cluster = 8 : i32, loop.stage = 0 : i32{{.*}}"Di"
// P1:      ttng.tc_gen5_mma {{.*}}loop.cluster = 7 : i32, loop.stage = 1 : i32{{.*}}"dv"
// P1:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 1 : i32{{.*}}"dk"
// P1:      ttng.tc_gen5_mma {{.*}}loop.cluster = 0 : i32, loop.stage = 2 : i32{{.*}}"dq"
// P1:      tt.scheduled_max_stage = 2 : i32

// pick 2: each key op by NAME loc + loop.stage/loop.cluster, in program order.
// P2-LABEL: @_attn_bwd
// P2:      tt.descriptor_load {{.*}}loop.cluster = 3 : i32, loop.stage = 0 : i32{{.*}}"q"
// P2:      tt.descriptor_load {{.*}}loop.cluster = 6 : i32, loop.stage = 0 : i32{{.*}}"m"
// P2:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 0 : i32{{.*}}"qkT"
// P2:      tt.descriptor_load {{.*}}loop.cluster = 7 : i32, loop.stage = 0 : i32{{.*}}"do"
// P2:      ttng.tc_gen5_mma {{.*}}loop.cluster = 10 : i32, loop.stage = 0 : i32{{.*}}"dpT"
// P2:      tt.descriptor_load {{.*}}loop.cluster = 0 : i32, loop.stage = 1 : i32{{.*}}"Di"
// P2:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 1 : i32{{.*}}"dv"
// P2:      ttng.tc_gen5_mma {{.*}}loop.cluster = 11 : i32, loop.stage = 1 : i32{{.*}}"dk"
// P2:      ttng.tc_gen5_mma {{.*}}loop.cluster = 12 : i32, loop.stage = 1 : i32{{.*}}"dq"
// P2:      tt.scheduled_max_stage = 2 : i32

// pick 3: each key op by NAME loc + loop.stage/loop.cluster, in program order.
// P3-LABEL: @_attn_bwd
// P3:      tt.descriptor_load {{.*}}loop.cluster = 3 : i32, loop.stage = 0 : i32{{.*}}"q"
// P3:      tt.descriptor_load {{.*}}loop.cluster = 6 : i32, loop.stage = 0 : i32{{.*}}"m"
// P3:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 0 : i32{{.*}}"qkT"
// P3:      tt.descriptor_load {{.*}}loop.cluster = 7 : i32, loop.stage = 0 : i32{{.*}}"do"
// P3:      ttng.tc_gen5_mma {{.*}}loop.cluster = 10 : i32, loop.stage = 0 : i32{{.*}}"dpT"
// P3:      tt.descriptor_load {{.*}}loop.cluster = 0 : i32, loop.stage = 1 : i32{{.*}}"Di"
// P3:      ttng.tc_gen5_mma {{.*}}loop.cluster = 9 : i32, loop.stage = 1 : i32{{.*}}"dv"
// P3:      ttng.tc_gen5_mma {{.*}}loop.cluster = 11 : i32, loop.stage = 1 : i32{{.*}}"dk"
// P3:      ttng.tc_gen5_mma {{.*}}loop.cluster = 0 : i32, loop.stage = 2 : i32{{.*}}"dq"
// P3:      tt.scheduled_max_stage = 2 : i32

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
#loc = loc(unknown)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd(%desc_q: !tt.tensordesc<64x128xf16, #shared>, %desc_q.shape.0: i32, %desc_q.shape.1: i32, %desc_q.stride.0: i64, %desc_q.stride.1: i64, %desc_k: !tt.tensordesc<128x128xf16, #shared>, %desc_k.shape.0: i32, %desc_k.shape.1: i32, %desc_k.stride.0: i64, %desc_k.stride.1: i64, %desc_v: !tt.tensordesc<128x128xf16, #shared>, %desc_v.shape.0: i32, %desc_v.shape.1: i32, %desc_v.stride.0: i64, %desc_v.stride.1: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<64x128xf16, #shared>, %desc_do.shape.0: i32, %desc_do.shape.1: i32, %desc_do.stride.0: i64, %desc_do.stride.1: i64, %desc_dq: !tt.tensordesc<64x32xf32, #shared1>, %desc_dq.shape.0: i32, %desc_dq.shape.1: i32, %desc_dq.stride.0: i64, %desc_dq.stride.1: i64, %desc_dk: !tt.tensordesc<128x64xf16, #shared>, %desc_dk.shape.0: i32, %desc_dk.shape.1: i32, %desc_dk.stride.0: i64, %desc_dk.stride.1: i64, %desc_dv: !tt.tensordesc<128x64xf16, #shared>, %desc_dv.shape.0: i32, %desc_dv.shape.1: i32, %desc_dv.stride.0: i64, %desc_dv.stride.1: i64, %desc_m: !tt.tensordesc<64xf32, #shared2>, %desc_m.shape.0: i32, %desc_m.stride.0: i64, %desc_delta: !tt.tensordesc<64xf32, #shared2>, %desc_delta.shape.0: i32, %desc_delta.stride.0: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false loc(#loc)
    %cst = arith.constant dense<0.693147182> : tensor<64x32xf32, #blocked> loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c96_i32 = arith.constant 96 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c128_i32 = arith.constant 128 : i32 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %true = arith.constant true loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear> loc(#loc)
    %bhid = tt.get_program_id z : i32 loc(#loc)
    %pid = tt.get_program_id x : i32 loc(#loc)
    %off_chz = arith.muli %bhid, %N_CTX : i32 loc(#loc)
    %off_chz_1 = arith.extsi %off_chz : i32 to i64 loc(#loc)
    %off_bh = arith.remsi %bhid, %H : i32 loc(#loc)
    %off_bh_2 = arith.muli %stride_h, %off_bh : i32 loc(#loc)
    %off_bh_3 = arith.divsi %bhid, %H : i32 loc(#loc)
    %off_bh_4 = arith.muli %stride_z, %off_bh_3 : i32 loc(#loc)
    %off_bh_5 = arith.addi %off_bh_2, %off_bh_4 : i32 loc(#loc)
    %off_bh_6 = arith.extsi %off_bh_5 : i32 to i64 loc(#loc)
    %off_bh_7 = arith.extsi %stride_tok : i32 to i64 loc(#loc)
    %off_bh_8 = arith.divsi %off_bh_6, %off_bh_7 : i64 loc(#loc)
    %start_n = arith.muli %pid, %c128_i32 : i32 loc(#loc)
    %k = arith.extsi %start_n : i32 to i64 loc(#loc)
    %k_9 = arith.addi %off_bh_8, %k : i64 loc(#loc)
    %k_10 = arith.trunci %k_9 : i64 to i32 loc(#loc)
    %k_11 = tt.descriptor_load %desc_k[%k_10, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1> loc(#loc)
    %k_12 = ttg.local_alloc %k_11 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem> loc(#loc)
    %v = tt.descriptor_load %desc_v[%k_10, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1> loc(#loc)
    %v_13 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem> loc(#loc)
    %num_steps = arith.divsi %N_CTX, %c64_i32 : i32 loc(#loc)
    %qkT, %qkT_14 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("qkT")
    %dpT, %dpT_15 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dpT")
    %dv, %dv_16 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dv")
    %dk, %dk_17 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dk")
    %dq, %dq_18 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token) loc("dq")
    %dk_19 = ttng.tmem_store %cst_0, %dk[%dk_17], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc("dk")
    %dv_20 = ttng.tmem_store %cst_0, %dv[%dv_16], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc("dv")
    %curr_m:7 = scf.for %blk_idx = %c0_i32 to %num_steps step %c1_i32 iter_args(%curr_m_33 = %c0_i32, %dv_34 = %false, %qkT_35 = %qkT_14, %dpT_36 = %dpT_15, %dv_37 = %dv_20, %dk_38 = %dk_19, %dq_39 = %dq_18) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %q = arith.extsi %curr_m_33 : i32 to i64 loc("q")
      %q_40 = arith.addi %off_bh_8, %q : i64 loc("q")
      %q_41 = arith.trunci %q_40 : i64 to i32 loc("q")
      %q_42 = tt.descriptor_load %desc_q[%q_41, %c0_i32] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1> loc("q")
      %q_43 = ttg.local_alloc %q_42 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem> loc("q")
      %qT = ttg.memdesc_trans %q_43 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared3, #smem> loc(#loc)
      %offs_m_start = arith.addi %off_chz_1, %q : i64 loc(#loc)
      %m = arith.trunci %offs_m_start : i64 to i32 loc("m")
      %m_44 = tt.descriptor_load %desc_m[%m] : !tt.tensordesc<64xf32, #shared2> -> tensor<64xf32, #blocked2> loc("m")
      %qkT_45 = ttng.tc_gen5_mma %k_12, %qT, %qkT[%qkT_35], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared3, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> loc("qkT")
      %pT = ttg.convert_layout %m_44 : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc)
      %pT_46 = tt.expand_dims %pT {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc)
      %pT_47 = tt.broadcast %pT_46 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1> loc(#loc)
      %qkT_48, %qkT_49 = ttng.tmem_load %qkT[%qkT_45] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1> loc("qkT")
      %pT_50 = arith.subf %qkT_48, %pT_47 : tensor<128x64xf32, #linear1> loc(#loc)
      %pT_51 = math.exp2 %pT_50 : tensor<128x64xf32, #linear1> loc(#loc)
      %do = tt.descriptor_load %desc_do[%q_41, %c0_i32] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1> loc("do")
      %do_52 = ttg.local_alloc %do : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem> loc("do")
      %ppT = arith.truncf %pT_51 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
      %dv_53 = ttng.tmem_alloc %ppT : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory> loc("dv")
      %dpT_54 = ttg.memdesc_trans %do_52 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared3, #smem> loc("dpT")
      %dpT_55 = ttng.tc_gen5_mma %v_13, %dpT_54, %dpT[%dpT_36], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared3, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> loc("dpT")
      %Di = tt.descriptor_load %desc_delta[%m] : !tt.tensordesc<64xf32, #shared2> -> tensor<64xf32, #blocked2> loc("Di")
      %dv_56 = ttng.tc_gen5_mma %dv_53, %do_52, %dv[%dv_37], %dv_34, %true : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc("dv")
      %dsT = ttg.convert_layout %Di : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> loc(#loc)
      %dsT_57 = tt.expand_dims %dsT {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1> loc(#loc)
      %dsT_58 = tt.broadcast %dsT_57 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1> loc(#loc)
      %dpT_59, %dpT_60 = ttng.tmem_load %dpT[%dpT_55] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1> loc("dpT")
      %dsT_61 = arith.subf %dpT_59, %dsT_58 : tensor<128x64xf32, #linear1> loc(#loc)
      %dsT_62 = arith.mulf %pT_51, %dsT_61 : tensor<128x64xf32, #linear1> loc(#loc)
      %dsT_63 = arith.truncf %dsT_62 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
      %dsT_64 = ttg.local_alloc %dsT_63 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem> loc(#loc)
      %dk_65 = ttng.tmem_alloc %dsT_63 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory> loc("dk")
      %dk_66 = ttng.tc_gen5_mma %dk_65, %q_43, %dk[%dk_38], %dv_34, %true : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> loc("dk")
      %dq_67 = ttg.memdesc_trans %dsT_64 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared3, #smem> loc("dq")
      %dq_68 = ttng.tc_gen5_mma %dq_67, %k_12, %dq[%dq_39], %false, %true : !ttg.memdesc<64x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> loc("dq")
      %dq_69, %dq_70 = ttng.tmem_load %dq[%dq_68] : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear2> loc("dq")
      %dqs = tt.reshape %dq_69 : tensor<64x128xf32, #linear2> -> tensor<64x2x64xf32, #linear3> loc(#loc)
      %dqs_71 = tt.trans %dqs {order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #linear3> -> tensor<64x64x2xf32, #linear4> loc(#loc)
      %dqs_72 = ttg.convert_layout %dqs_71 : tensor<64x64x2xf32, #linear4> -> tensor<64x64x2xf32, #linear5> loc(#loc)
      %dqs_73, %dqs_74 = tt.split %dqs_72 : tensor<64x64x2xf32, #linear5> -> tensor<64x64xf32, #linear6> loc(#loc)
      %dqs_75 = tt.reshape %dqs_73 : tensor<64x64xf32, #linear6> -> tensor<64x2x32xf32, #blocked3> loc(#loc)
      %dqs_76 = tt.trans %dqs_75 {order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #blocked3> -> tensor<64x32x2xf32, #blocked4> loc(#loc)
      %dqs_77, %dqs_78 = tt.split %dqs_76 : tensor<64x32x2xf32, #blocked4> -> tensor<64x32xf32, #blocked> loc(#loc)
      %dqs_79 = tt.reshape %dqs_74 : tensor<64x64xf32, #linear6> -> tensor<64x2x32xf32, #blocked3> loc(#loc)
      %dqs_80 = tt.trans %dqs_79 {order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #blocked3> -> tensor<64x32x2xf32, #blocked4> loc(#loc)
      %dqs_81, %dqs_82 = tt.split %dqs_80 : tensor<64x32x2xf32, #blocked4> -> tensor<64x32xf32, #blocked> loc(#loc)
      %dqN = arith.mulf %dqs_77, %cst : tensor<64x32xf32, #blocked> loc(#loc)
      tt.descriptor_reduce add, %desc_dq[%q_41, %c0_i32], %dqN : !tt.tensordesc<64x32xf32, #shared1>, tensor<64x32xf32, #blocked> loc(#loc)
      %dqN_83 = arith.mulf %dqs_78, %cst : tensor<64x32xf32, #blocked> loc(#loc)
      tt.descriptor_reduce add, %desc_dq[%q_41, %c32_i32], %dqN_83 : !tt.tensordesc<64x32xf32, #shared1>, tensor<64x32xf32, #blocked> loc(#loc)
      %dqN_84 = arith.mulf %dqs_81, %cst : tensor<64x32xf32, #blocked> loc(#loc)
      tt.descriptor_reduce add, %desc_dq[%q_41, %c64_i32], %dqN_84 : !tt.tensordesc<64x32xf32, #shared1>, tensor<64x32xf32, #blocked> loc(#loc)
      %dqN_85 = arith.mulf %dqs_82, %cst : tensor<64x32xf32, #blocked> loc(#loc)
      tt.descriptor_reduce add, %desc_dq[%q_41, %c96_i32], %dqN_85 : !tt.tensordesc<64x32xf32, #shared1>, tensor<64x32xf32, #blocked> loc(#loc)
      %curr_m_86 = arith.addi %curr_m_33, %c64_i32 : i32 loc(#loc)
      scf.yield %curr_m_86, %true, %qkT_49, %dpT_60, %dv_56, %dk_66, %dq_70 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token loc(#loc)
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize} loc("dv")
    %dv_21, %dv_22 = ttng.tmem_load %dv[%curr_m#4] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear> loc("dv")
    %dk_23, %dk_24 = ttng.tmem_load %dk[%curr_m#5] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear> loc("dk")
    %dvs = tt.reshape %dv_21 : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear7> loc(#loc)
    %dvs_25 = tt.trans %dvs {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear7> -> tensor<128x64x2xf32, #linear8> loc(#loc)
    %dvs_26, %dvs_27 = tt.split %dvs_25 : tensor<128x64x2xf32, #linear8> -> tensor<128x64xf32, #linear1> loc(#loc)
    %0 = arith.truncf %dvs_26 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
    %1 = ttg.convert_layout %0 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc)
    tt.descriptor_store %desc_dv[%k_10, %c0_i32], %1 : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked5> loc(#loc)
    %2 = arith.truncf %dvs_27 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
    %3 = ttg.convert_layout %2 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc)
    tt.descriptor_store %desc_dv[%k_10, %c64_i32], %3 : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked5> loc(#loc)
    %dks = tt.reshape %dk_23 : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear7> loc(#loc)
    %dks_28 = tt.trans %dks {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear7> -> tensor<128x64x2xf32, #linear8> loc(#loc)
    %dks_29, %dks_30 = tt.split %dks_28 : tensor<128x64x2xf32, #linear8> -> tensor<128x64xf32, #linear1> loc(#loc)
    %dkN = tt.splat %sm_scale : f32 -> tensor<128x64xf32, #linear1> loc(#loc)
    %dkN_31 = arith.mulf %dks_29, %dkN : tensor<128x64xf32, #linear1> loc(#loc)
    %4 = arith.truncf %dkN_31 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
    %5 = ttg.convert_layout %4 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc)
    tt.descriptor_store %desc_dk[%k_10, %c0_i32], %5 : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked5> loc(#loc)
    %dkN_32 = arith.mulf %dks_30, %dkN : tensor<128x64xf32, #linear1> loc(#loc)
    %6 = arith.truncf %dkN_32 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1> loc(#loc)
    %7 = ttg.convert_layout %6 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5> loc(#loc)
    tt.descriptor_store %desc_dk[%k_10, %c64_i32], %7 : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked5> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
