// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=200000" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Regression test: BWD config 1 (BLOCK_M=128) with early_tma_store_lowering.
//
// `q` (128x128xf16) is cross-stage — consumed by the qkT MMA at loop.stage 0
// and the dk MMA at loop.stage 1 — so it REQUIRES buffer.copy=2. Before the
// fix, Phase 2 tentatively set q's 2nd copy, found the total over the (soft)
// smem_budget once the early-TMA staging allocs were counted at full size, and
// silently reverted q to copy=1 — which deadlocks at runtime (the TMA-load
// producer and the dk consumer collide on a single slot one iteration apart).
//
// The fix makes the cross-stage depth a strict floor that is applied FIRST and
// is never reverted for budget. Phase 3.6 may reclaim discretionary TMA
// store-staging SMEM to fit, but for this 128-wide config the store-staging is
// a 128x32 subtile with swizzle=64 (#shared2) while the operands it could alias
// are 128x128 with swizzle=128 (#shared) — incompatible encodings, so the reuse
// is unrealizable (mergeStagingReuseIntoHost would drop it). Phase 3.6 therefore
// does NOT mark it (areReuseEncodingsCompatible), and the floor is honored by
// the ship-anyway backstop (the real HW SMEM limit, not the soft budget).
//
// This is a trimmed doBufferAllocation dump: source-location metadata is
// dropped and the dq/dv/dk subtile epilogue arithmetic (which only computes the
// values written into the TMA staging buffers) is replaced by constants. Every
// buffer alloc and the producer/consumer ops that drive the allocation decision
// are kept verbatim.
//
// Key verification:
//   - q keeps its cross-stage floor: buffer.copy = 2 (the fix; was 1), even
//     though the budget can only be met by the ship-anyway backstop.
//   - TMA store-staging allocs (buffer.tmaStaging = 1) are NOT reuse-marked:
//     their swizzle=64 encoding is incompatible with the swizzle=128 operands,
//     so Phase 3.6 leaves them standalone (no allocation.reuseTarget /
//     allocation.shareGroup) rather than marking an unrealizable reuse.
//   - In-loop TMA reduce-staging allocs (buffer.tmaStaging = 2) are innermost
//     and left as dedicated copy=1.

// CHECK-LABEL: tt.func public @_attn_bwd_persist
// CHECK: %q = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32}
// The store-staging alloc starts its attr dict directly with `buffer.copy`
// (attrs sort alphabetically), confirming no `allocation.*` reuse attribute is
// stamped — the encoding-incompatible reuse is correctly not marked.
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32, buffer.tmaStaging = 1 : i32}

// -----// WarpSpec internal IR Dump After: doBufferAllocation
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<tensor<128x128xf16, #shared>> , %desc_q.shape.0: i32 , %desc_q.shape.1: i32 , %desc_q.stride.0: i64 , %desc_q.stride.1: i64 , %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>> , %desc_k.shape.0: i32 , %desc_k.shape.1: i32 , %desc_k.stride.0: i64 , %desc_k.stride.1: i64 , %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>> , %desc_v.shape.0: i32 , %desc_v.shape.1: i32 , %desc_v.stride.0: i64 , %desc_v.stride.1: i64 , %sm_scale: f32 , %desc_do: !tt.tensordesc<tensor<128x128xf16, #shared>> , %desc_do.shape.0: i32 , %desc_do.shape.1: i32 , %desc_do.stride.0: i64 , %desc_do.stride.1: i64 , %desc_dq: !tt.tensordesc<tensor<128x32xf32, #shared1>> , %desc_dq.shape.0: i32 , %desc_dq.shape.1: i32 , %desc_dq.stride.0: i64 , %desc_dq.stride.1: i64 , %desc_dk: !tt.tensordesc<tensor<128x32xf16, #shared2>> , %desc_dk.shape.0: i32 , %desc_dk.shape.1: i32 , %desc_dk.stride.0: i64 , %desc_dk.stride.1: i64 , %desc_dv: !tt.tensordesc<tensor<128x32xf16, #shared2>> , %desc_dv.shape.0: i32 , %desc_dv.shape.1: i32 , %desc_dv.stride.0: i64 , %desc_dv.stride.1: i64 , %desc_m: !tt.tensordesc<tensor<128xf32, #shared3>> , %desc_m.shape.0: i32 , %desc_m.stride.0: i64 , %desc_delta: !tt.tensordesc<tensor<128xf32, #shared3>> , %desc_delta.shape.0: i32 , %desc_delta.stride.0: i64 , %stride_z: i32 {tt.divisibility = 16 : i32} , %stride_h: i32 {tt.divisibility = 16 : i32} , %stride_tok: i32 {tt.divisibility = 16 : i32} , %BATCH: i32 , %H: i32 {tt.divisibility = 16 : i32} , %N_CTX: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("k")
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("v")
    %dv, %dv_0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk, %dk_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("q")
    %m = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared3, #smem, mutable> loc("m")
    %qkT, %qkT_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %do = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("do")
    %ppT = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %dpT, %dpT_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %Di = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared3, #smem, mutable> loc("Di")
    %dsT = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc("dsT")
    %dq, %dq_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %n_tile_num = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 32 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 64 : i32
    %c96_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 96 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_5 = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x32xf32, #linear1>
    %cst_dvk = arith.constant {async_task_id = array<i32: 3>} dense<0.000000e+00> : tensor<128x32xf16, #blocked3>
    %cst_dq = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x32xf32, #blocked2>
    %n_tile_num_6 = arith.addi %N_CTX, %n_tile_num {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %n_tile_num_7 = arith.divsi %n_tile_num_6, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %prog_id = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles = arith.muli %n_tile_num_7, %BATCH {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %total_tiles_8 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %tiles_per_sm = arith.divsi %total_tiles_8, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %0 = arith.remsi %total_tiles_8, %num_progs {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_9 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm_9 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %tiles_per_sm : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %dkN = tt.splat %sm_scale {async_task_id = array<i32: 3>} : f32 -> tensor<128x32xf32, #linear1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_9 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_9, %n_tile_num_7 {async_task_id = array<i32: 2, 3>} : i32
      %bhid = arith.divsi %tile_idx_9, %n_tile_num_7 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {async_task_id = array<i32: 2>} : i32
      %off_chz_10 = arith.extsi %off_chz {async_task_id = array<i32: 2>} : i32 to i64
      %off_bh_11 = arith.remsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_12 = arith.muli %stride_h, %off_bh_11 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_13 = arith.divsi %bhid, %H {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_14 = arith.muli %stride_z, %off_bh_13 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_15 = arith.addi %off_bh_12, %off_bh_14 {async_task_id = array<i32: 0, 2, 3>} : i32
      %off_bh_16 = arith.extsi %off_bh_15 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %off_bh_17 = arith.divsi %off_bh_16, %off_bh {async_task_id = array<i32: 0, 2, 3>} : i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %k_18 = arith.extsi %start_n {async_task_id = array<i32: 2, 3>} : i32 to i64
      %k_19 = arith.addi %off_bh_17, %k_18 {async_task_id = array<i32: 2, 3>} : i64
      %k_20 = arith.trunci %k_19 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %k_21 = tt.descriptor_load %desc_k[%k_20, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
      ttg.local_store %k_21, %k {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v_22 = tt.descriptor_load %desc_v[%k_20, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
      ttg.local_store %v_22, %v {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %curr_m:7 = scf.for %curr_m_54 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg51 = %c0_i32, %dv_55 = %false, %qkT_56 = %qkT_2, %dpT_57 = %dpT_3, %dv_58 = %dv_0, %dq_59 = %dq_4, %dk_60 = %dk_1) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q_61 = arith.extsi %arg51 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_62 = arith.addi %off_bh_17, %q_61 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %q_63 = arith.trunci %q_62 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_64 = tt.descriptor_load %desc_q[%q_63, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
        ttg.local_store %q_64, %q {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qT = ttg.memdesc_trans %q {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>
        %offs_m_start = arith.addi %off_chz_10, %q_61 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %m_65 = arith.trunci %offs_m_start {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %m_66 = tt.descriptor_load %desc_m[%m_65] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128xf32, #shared3>> -> tensor<128xf32, #blocked1>
        ttg.local_store %m_66, %m {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
        %qkT_67 = ttng.tc_gen5_mma %k, %qT, %qkT[%qkT_56], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %m_68 = ttg.local_load %m {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked1>
        %pT = ttg.convert_layout %m_68 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %pT_69 = tt.expand_dims %pT {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %pT_70 = tt.broadcast %pT_69 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %qkT_71, %qkT_72 = ttng.tmem_load %qkT[%qkT_67] {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %pT_73 = arith.subf %qkT_71, %pT_70 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %pT_74 = math.exp2 %pT_73 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %do_75 = tt.descriptor_load %desc_do[%q_63, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
        ttg.local_store %do_75, %do {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %ppT_76 = arith.truncf %pT_74 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %dv_77 = arith.constant {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} true
        ttng.tmem_store %ppT_76, %ppT, %dv_77 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dpT_78 = ttg.memdesc_trans %do {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>
        %dpT_79 = ttng.tc_gen5_mma %v, %dpT_78, %dpT[%dpT_57], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_80 = tt.descriptor_load %desc_delta[%m_65] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128xf32, #shared3>> -> tensor<128xf32, #blocked1>
        ttg.local_store %Di_80, %Di {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
        %dv_81 = ttng.tc_gen5_mma %ppT, %do, %dv[%dv_58], %dv_55, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_82 = ttg.local_load %Di {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked1>
        %dsT_83 = ttg.convert_layout %Di_82 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %dsT_84 = tt.expand_dims %dsT_83 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %dsT_85 = tt.broadcast %dsT_84 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %dpT_86, %dpT_87 = ttng.tmem_load %dpT[%dpT_79] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %dsT_88 = arith.subf %dpT_86, %dsT_85 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %dsT_89 = arith.mulf %pT_74, %dsT_88 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear>
        %dsT_90 = arith.truncf %dsT_89 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        ttg.local_store %dsT_90, %dsT {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %dq_91 = ttg.memdesc_trans %dsT {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>
        %dq_92 = ttng.tc_gen5_mma %dq_91, %k, %dq[%dq_59], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22}"} : !ttg.memdesc<128x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_93 = ttng.tc_gen5_mma %dsT, %q, %dk[%dk_60], %dv_55, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_94, %dq_95 = ttng.tmem_load %dq[%dq_92] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        // dq is read out of TMEM and TMA-reduced as four 128x32 subtiles. The
        // subtile reshape/scale math is irrelevant to memory planning, so the
        // values written to the reduce-staging buffers are replaced by a constant.
        %35 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        ttg.local_store %cst_dq, %35 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %36 = ttng.async_tma_reduce add, %desc_dq[%q_63, %c0_i32] %35 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %36   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %37 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        ttg.local_store %cst_dq, %37 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %38 = ttng.async_tma_reduce add, %desc_dq[%q_63, %c32_i32] %37 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %38   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %39 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        ttg.local_store %cst_dq, %39 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %40 = ttng.async_tma_reduce add, %desc_dq[%q_63, %c64_i32] %39 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %40   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %41 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        ttg.local_store %cst_dq, %41 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %42 = ttng.async_tma_reduce add, %desc_dq[%q_63, %c96_i32] %41 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %42   {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %curr_m_114 = arith.addi %arg51, %c128_i32 {async_task_id = array<i32: 0, 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %curr_m_114, %true, %qkT_72, %dpT_87, %dv_81, %dq_95, %dk_93 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      // dv/dk are read out of TMEM and TMA-stored as four 128x32 subtiles each.
      // The subtile reshape/scale math is irrelevant to memory planning, so the
      // values written to the store-staging buffers are replaced by a constant.
      %dv_23, %dv_24 = ttng.tmem_load %dv[%curr_m#4] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %5 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %5 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %6 = ttng.async_tma_copy_local_to_global %desc_dv[%k_20, %c0_i32] %5 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %6   {async_task_id = array<i32: 3>} : !ttg.async.token
      %9 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %9 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %10 = ttng.async_tma_copy_local_to_global %desc_dv[%k_20, %c32_i32] %9 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %10   {async_task_id = array<i32: 3>} : !ttg.async.token
      %13 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %13 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %14 = ttng.async_tma_copy_local_to_global %desc_dv[%k_20, %c64_i32] %13 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %14   {async_task_id = array<i32: 3>} : !ttg.async.token
      %17 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %17 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %18 = ttng.async_tma_copy_local_to_global %desc_dv[%k_20, %c96_i32] %17 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %18   {async_task_id = array<i32: 3>} : !ttg.async.token
      %dk_36, %dk_37 = ttng.tmem_load %dk[%curr_m#6] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %21 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %21 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %22 = ttng.async_tma_copy_local_to_global %desc_dk[%k_20, %c0_i32] %21 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %22   {async_task_id = array<i32: 3>} : !ttg.async.token
      %25 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %25 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %26 = ttng.async_tma_copy_local_to_global %desc_dk[%k_20, %c32_i32] %25 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %26   {async_task_id = array<i32: 3>} : !ttg.async.token
      %29 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %29 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %30 = ttng.async_tma_copy_local_to_global %desc_dk[%k_20, %c64_i32] %29 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %30   {async_task_id = array<i32: 3>} : !ttg.async.token
      %33 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      ttg.local_store %cst_dvk, %33 {async_task_id = array<i32: 3>} : tensor<128x32xf16, #blocked3> -> !ttg.memdesc<128x32xf16, #shared2, #smem, mutable>
      %34 = ttng.async_tma_copy_local_to_global %desc_dk[%k_20, %c96_i32] %33 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared2>>, !ttg.memdesc<128x32xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %34   {async_task_id = array<i32: 3>} : !ttg.async.token
      %tile_idx_53 = arith.addi %tile_idx_9, %num_progs {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %tile_idx_53 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue_to_computation = true, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
