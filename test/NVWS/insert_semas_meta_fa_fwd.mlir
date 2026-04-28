// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// Meta flash attention forward (persistent) IR, captured from
// meta-aws-logs/run-22may26-nvws-meta-tmem-crash/passes/
// 062-anonymous_VerifyWarpSpecializationPartitions.mlir (loc() stripped).
// High-coverage exercise of native point-of-use insert-semas with a persistent
// outer loop wrapping a pipelined inner loop. Native POU opens buffer.id=5 at
// its first inner-loop use and closes the recurrence with releases from the
// stage-1 uses; no buffer.id=5 token is carried through either loop.
// buffer.id=4 stays behind its own in-loop gate. The
// buffer.id=2/3 per-iteration accumulators thread tokens through the outer
// loop only (bottom acquire at the final readout) and re-acquire in-body
// inside the inner loop. The Q/K/V descriptor SMEM, stats stores, and epilogue
// O buffers exercise the remaining protocols. Hand-curated, semaphore-focused
// CHECKs.
//
// NOTE: the pass canonicalizes the TMEM encoding aliases, so in the captured
// output #tmem is blockN=128 (the 128x128 acc/result/f16-view buffers) and
// #tmem1 is blockN=1 (the alpha/l_i stats subslices) — the opposite of the
// input aliases below. CHECK lines therefore use the *output* alias names.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// CHECK: module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32,
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL:   tt.func public @_attn_fwd_persist(
// ASP-LABEL:     tt.func public @_attn_fwd_persist(
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %Z, %n_tile_num : i32
    %total_tiles_3 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs : i32
    %0 = arith.remsi %total_tiles_3, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_19 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_19 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %desc_q_4 = arith.muli %Z, %H : i32
    %desc_q_5 = arith.muli %desc_q_4, %c16384_i32 : i32
    %desc_q_6 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_7 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_8 = tt.make_tensor_descriptor %desc_k, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_9 = tt.make_tensor_descriptor %desc_v, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_10 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_11 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c16384_i32 : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m0_12 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_ij_13 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %qk_14 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>

    // The single-buffer Q/K/V SMEM allocs each get a true(EMPTY)/false(FULL)
    // semaphore pair (pending_count = 1).
    // CHECK:           [[Q0:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q0_E:%.*]] = nvws.semaphore.create [[Q0]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[Q0_F:%.*]] = nvws.semaphore.create [[Q0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %q0_0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[Q1_E:%.*]] = nvws.semaphore.create [[Q1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[Q1_F:%.*]] = nvws.semaphore.create [[Q1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %q0_1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[K:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[K_E:%.*]] = nvws.semaphore.create [[K]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[K_F:%.*]] = nvws.semaphore.create [[K]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[V:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[V_E:%.*]] = nvws.semaphore.create [[V]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[V_F:%.*]] = nvws.semaphore.create [[V]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The buffer.id=4 accumulator-class TMEM allocation is hoisted to a single
    // 1x128x128 alloc whose subslices (alpha/offsetkv stats + acc + f16 view)
    // form one multi-member semaphore group: two released gates (pending_count =
    // 2) plus five false(FULL) phases (pending_count = 1).
    // CHECK:           [[R4:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[R4_IN:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} released = -1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_E:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} released = -1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F1:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F2:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F3:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F4:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R4_F5:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R4]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    %alpha = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %alpha_15 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_16 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // The buffer.id=5 accumulator-class TMEM allocation forms the second
    // multi-member group: two released gates (pending_count = 2) plus five
    // false(FULL) semaphores (pending_count = 1).
    // CHECK:           [[R5:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[R5_IN:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} released = -1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_E:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} released = -1 {pending_count = 2 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F1:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F2:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F3:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F4:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    // CHECK:           [[R5_F5:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}}, %{{.*}}, [[R5]], %{{.*}} {pending_count = 1 : i32} : <[{{.*}}]>
    %offsetkv_y_17 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_18 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // Epilogue-store SMEM scratch (%3,%4) each get a true/false pair.
    // CHECK:           [[O0:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O0_E:%.*]] = nvws.semaphore.create [[O0]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[O0_F:%.*]] = nvws.semaphore.create [[O0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK:           [[O1_E:%.*]] = nvws.semaphore.create [[O1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK:           [[O1_F:%.*]] = nvws.semaphore.create [[O1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The two per-iteration acc TMEM allocs (buffer.id 2/3) are hoisted to
    // single-component 1x buffers with a true/false pair each.
    // CHECK:           [[ACC0:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[ACC0_E:%.*]] = nvws.semaphore.create [[ACC0]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC0_F:%.*]] = nvws.semaphore.create [[ACC0]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC1:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[ACC1_E:%.*]] = nvws.semaphore.create [[ACC1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[ACC1_F:%.*]] = nvws.semaphore.create [[ACC1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>

    // Initial outer-gate acquires for R4/R5 (partition 1), followed by the two
    // per-iteration accumulator tokens. Only ACC0/ACC1 thread through the
    // outer loop; the R5 token does not.
    // CHECK:           [[IA_R4:%.*]] = nvws.semaphore.acquire [[R4_E]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[{{.*}}]> -> !ttg.async.token
    // CHECK:           [[IA_R5:%.*]] = nvws.semaphore.acquire [[R5_E]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[{{.*}}]> -> !ttg.async.token
    // CHECK:           [[IA_ACC0:%.*]] = nvws.semaphore.acquire [[ACC0_E]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK:           [[IA_ACC1:%.*]] = nvws.semaphore.acquire [[ACC1_E]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token

    // R5 is not carried through the outer persistent loop. R4 also needs no
    // carried token.
    // CHECK-NOT:       scf.for {{.*}}[[IA_R5]]
    // CHECK:           scf.for {{.*}} iter_args({{.*}}, [[C_ACC0:%.*]] = [[IA_ACC0]], [[C_ACC1:%.*]] = [[IA_ACC1]]) -> (i32, !ttg.async.token, !ttg.async.token)  : i32 {
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_19 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_hz = arith.divsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_z = arith.divsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_h = arith.remsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_20 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_21 = arith.muli %off_h, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %5 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %q0 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_m0_24 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_25 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_26 = arith.addi %offs_m0_24, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m0_27 = arith.addi %offs_m0_25, %offs_m0_12 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>

      // Q0 load: acquire EMPTY, point-of-use buffer feeds the descriptor_load,
      // release FULL with a tma_load arrive.
      // CHECK:           [[Q0_AE:%.*]] = nvws.semaphore.acquire [[Q0_E]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[Q0_BUF:%.*]] = nvws.semaphore.buffer [[Q0_E]], [[Q0_AE]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[Q0_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK:           nvws.semaphore.release [[Q0_F]], [[Q0_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_q_6[%qo_offset_y_23, %c0_i32] 32768 %q0_0 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Q1 load mirrors Q0.
      // CHECK:           [[Q1_AE:%.*]] = nvws.semaphore.acquire [[Q1_E]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[Q1_BUF:%.*]] = nvws.semaphore.buffer [[Q1_E]], [[Q1_AE]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[Q1_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK:           nvws.semaphore.release [[Q1_F]], [[Q1_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_q_7[%q0, %c0_i32] 32768 %q0_1 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qk_0, %qk_0_32 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_1, %qk_1_33 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 4>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // The acc_0 / acc_1 init stores reuse the carried ACC0/ACC1 EMPTY tokens
      // (no re-acquire here), write zero into the point-of-use buffer, then
      // release the EMPTY gate for the first in-body acquire in the inner loop.
      // CHECK:           [[ACC0_BUF0:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[C_ACC0]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_store %{{.*}}, [[ACC0_BUF0]], %{{.*}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           nvws.semaphore.release [[ACC0_E]], [[C_ACC0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_0, %acc_0_34 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK:           [[ACC1_BUF0:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[C_ACC1]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_store %{{.*}}, [[ACC1_BUF0]], %{{.*}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           nvws.semaphore.release [[ACC1_E]], [[C_ACC1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_1, %acc_1_35 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Outside the inner loop, the K/V FULL tokens consumed by the trailing
      // tc5mma releases are acquired up front (partition 1) on the Q FULL sems.
      // CHECK:           [[K_PRE:%.*]] = nvws.semaphore.acquire [[Q0_F]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[V_PRE:%.*]] = nvws.semaphore.acquire [[Q1_F]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token

      // R5 is not carried through the inner pipelined loop either. The R4
      // inner gate and Q FULL tokens are loop-invariant captures; ACC0/ACC1
      // are re-acquired in-body.
      // CHECK:           [[INNER:%.*]]:5 = scf.for {{.*}} iter_args({{.*}}) -> (i32, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>, tensor<128xf32, {{.*}}>)  : i32 {
      %offsetkv_y_40:9 = scf.for %offsetkv_y_88 = %c0_i32 to %c16384_i32 step %c128_i32 iter_args(%offset_y_89 = %offset_y_22, %arg12 = %cst_2, %arg13 = %cst_1, %qk_0_90 = %qk_0_32, %acc_91 = %acc_0_34, %arg16 = %cst_2, %arg17 = %cst_1, %qk_1_92 = %qk_1_33, %acc_93 = %acc_1_35) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        // K descriptor load: acquire EMPTY (K_E), point-of-use buffer, release FULL.
        // CHECK:             [[KIN_AE:%.*]] = nvws.semaphore.acquire [[K_E]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[KIN_BUF:%.*]] = nvws.semaphore.buffer [[K_E]], [[KIN_AE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[KIN_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
        // CHECK:             nvws.semaphore.release [[K_F]], [[KIN_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.descriptor_load %desc_k_8[%offset_y_89, %c0_i32] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_97 = ttg.memdesc_reinterpret %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %k_98 = ttg.memdesc_trans %k_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
        // V descriptor load: acquire EMPTY (V_E), point-of-use buffer, release FULL.
        // CHECK:             [[VIN_AE:%.*]] = nvws.semaphore.acquire [[V_E]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[VIN_BUF:%.*]] = nvws.semaphore.buffer [[V_E]], [[VIN_AE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[VIN_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
        // CHECK:             nvws.semaphore.release [[V_F]], [[VIN_AE]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.descriptor_load %desc_v_9[%offset_y_89, %c0_i32] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // First QK MMA (partition 1): acquire and buffer the R4 inner gate,
        // buffer the Q0 FULL token (lhs), then acquire+buffer the inner K FULL
        // token (rhs, transposed). MMA lhs is the Q0 buffer, acc is R4_QK#3.
        // CHECK:             [[R4_QK_A:%.*]] = nvws.semaphore.acquire [[R4_IN]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[R4_QK:%.*]]:5 = nvws.semaphore.buffer [[R4_IN]], [[R4_QK_A]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[Q0_QK:%.*]] = nvws.semaphore.buffer [[Q0_F]], [[K_PRE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             [[KMMA_AF:%.*]] = nvws.semaphore.acquire [[K_F]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[KMMA_BUF:%.*]] = nvws.semaphore.buffer [[K_F]], [[KMMA_AF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_101 = ttng.tc_gen5_mma %q0_0, %k_98, %qk_0[%qk_0_90], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tc_gen5_mma [[Q0_QK]], %{{.*}}, [[R4_QK]]#3[], %{{.*}}, %{{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[R4_F1]], [[R4_QK_A]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // Second QK MMA mirrors the first on the R5 (id=5) acc-class set. Its
        // R5 token is acquired at this first use, not carried into the loop.
        // After it, the inner K_E EMPTY is released using the K consumer token.
        // CHECK:             [[R5_QK_A:%.*]] = nvws.semaphore.acquire [[R5_IN]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[R5_QK:%.*]]:5 = nvws.semaphore.buffer [[R5_IN]], [[R5_QK_A]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[Q1_QK:%.*]] = nvws.semaphore.buffer [[Q1_F]], [[V_PRE]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_102 = ttng.tc_gen5_mma %q0_1, %k_98, %qk_1[%qk_1_92], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tc_gen5_mma [[Q1_QK]], %{{.*}}, [[R5_QK]]#3[], %{{.*}}, %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[K_E]], [[KMMA_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK:             nvws.semaphore.release [[R5_F1]], [[R5_QK_A]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // QK tmem_load (partition 5): acquire R4_F1, buffer, load #3.
        // CHECK:             [[QK0_AF:%.*]] = nvws.semaphore.acquire [[R4_F1]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[QK0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F1]], [[QK0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[QK0_BUF]]#3[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        %qk_103, %qk_104 = ttng.tmem_load %qk_0[%qk_101] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_105 = ttg.convert_layout %qk_103 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        // QK tmem_load (partition 4): acquire R5_F1, buffer, load #3.
        // CHECK:             [[QK1_AF:%.*]] = nvws.semaphore.acquire [[R5_F1]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[QK1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F1]], [[QK1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[QK1_BUF]]#3[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        %qk_106, %qk_107 = ttng.tmem_load %qk_1[%qk_102] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_108 = ttg.convert_layout %qk_106 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %m_ij_109 = "tt.reduce"(%qk_105) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_110 = "tt.reduce"(%qk_108) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_111 = arith.mulf %m_ij_109, %m_ij {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_112 = arith.mulf %m_ij_110, %m_ij_13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_113 = arith.maxnumf %arg13, %m_ij_111 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_114 = arith.maxnumf %arg17, %m_ij_112 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_115 = arith.mulf %qk_105, %qk {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_116 = arith.mulf %qk_108, %qk_14 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %qk_117 = tt.expand_dims %m_ij_113 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_118 = tt.expand_dims %m_ij_114 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_119 = tt.broadcast %qk_117 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_120 = tt.broadcast %qk_118 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_121 = arith.subf %qk_115, %qk_119 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_122 = arith.subf %qk_116, %qk_120 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_121 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %p_123 = math.exp2 %qk_122 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %alpha_124 = arith.subf %arg13, %m_ij_113 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_125 = arith.subf %arg17, %m_ij_114 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_126 = math.exp2 %alpha_124 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_127 = tt.expand_dims %alpha_126 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_128 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} true
        // alpha stats store (partition 5) writes into the R4 subslice #0 buffer
        // already opened by [[QK0_BUF]] (point-of-use), then releases R4 FULL.
        // CHECK:             ttng.tmem_store %{{.*}}, [[QK0_BUF]]#0, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        // CHECK:             nvws.semaphore.release [[R4_F2]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        // CHECK:             nvws.semaphore.release [[R4_IN]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        ttng.tmem_store %alpha_127, %alpha, %alpha_128 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %alpha_129 = math.exp2 %alpha_125 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_130 = tt.expand_dims %alpha_129 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_131 = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} true
        // alpha_15 stats store (partition 4) mirrors on R5 subslice #0.
        // CHECK:             ttng.tmem_store %{{.*}}, [[QK1_BUF]]#0, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        // CHECK:             nvws.semaphore.release [[R5_F2]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        // CHECK:             nvws.semaphore.release [[R5_IN]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        ttng.tmem_store %alpha_130, %alpha_15, %alpha_131 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij_132 = "tt.reduce"(%p_123) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        // acc_0 tmem_load (partition 0) re-acquires ACC0 EMPTY in-body and
        // buffers under the fresh token.
        // CHECK:             [[ACC0_AE:%.*]] = nvws.semaphore.acquire [[ACC0_E]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[ACC0_LD:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[ACC0_AE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tmem_load [[ACC0_LD]][] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>}
        %acc_133, %acc_134 = ttng.tmem_load %acc_0[%acc_91] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        // CHECK:             [[ACC1_AE:%.*]] = nvws.semaphore.acquire [[ACC1_E]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[ACC1_LD:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[ACC1_AE]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tmem_load [[ACC1_LD]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>}
        %acc_135, %acc_136 = ttng.tmem_load %acc_1[%acc_93] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %18 = tt.reshape %acc_133 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %19 = tt.reshape %acc_135 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %20 = tt.trans %18 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %21 = tt.trans %19 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %20 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_137, %outRHS_138 = tt.split %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        // alpha stats reload (partition 0): acquire R4_F2, buffer, load subslice
        // #0, release the second arrival to R4_IN.
        // CHECK:             [[A0_AF:%.*]] = nvws.semaphore.acquire [[R4_F2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[A0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F2]], [[A0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[A0_BUF]]#0[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[R4_IN]], [[A0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
        %alpha_139, %alpha_140 = ttng.tmem_load %alpha[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_141 = tt.reshape %alpha_139 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_142 = ttg.convert_layout %alpha_141 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_143 = tt.expand_dims %alpha_142 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        // alpha_15 stats reload (partition 0): acquire R5_F2, buffer, load, and
        // release the second arrival to R5_IN for the next iteration's POU
        // acquire.
        // CHECK:             [[A1_AF:%.*]] = nvws.semaphore.acquire [[R5_F2]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[A1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F2]], [[A1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_load [[A1_BUF]]#0[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[R5_IN]], [[A1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
        %alpha_144, %alpha_145 = ttng.tmem_load %alpha_15[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_146 = tt.reshape %alpha_144 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_147 = ttg.convert_layout %alpha_146 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_148 = tt.expand_dims %alpha_147 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_149 = ttg.convert_layout %acc0_143 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_150 = ttg.convert_layout %acc0_148 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_151 = tt.broadcast %acc0_149 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_152 = tt.broadcast %acc0_150 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_153 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_154 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_137, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1_155 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_138, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc_156 = tt.join %acc0_153, %acc1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_157 = tt.join %acc0_154, %acc1_155 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_158 = tt.trans %acc_156 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_159 = tt.trans %acc_157 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_160 = tt.reshape %acc_158 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %acc_161 = tt.reshape %acc_159 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %p_162 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %p_163 = arith.truncf %p_123 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        // p (f16) store into the R4 f16 view subslice #4 (partition 5) reuses
        // the retained R4_F1 token and releases R4_F3.
        // CHECK:             [[P0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F1]], [[QK0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_store %{{.*}}, [[P0_BUF]]#4, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             nvws.semaphore.release [[R4_F3]], [[QK0_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>}
        %acc_164 = ttng.tmem_alloc %p_162 {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // p_123 (f16) store into R5 f16 view subslice #4 (partition 4) reuses
        // the retained R5_F1 token and releases R5_F3.
        // CHECK:             [[P1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F1]], [[QK1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             ttng.tmem_store %{{.*}}, [[P1_BUF]]#4, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             nvws.semaphore.release [[R5_F3]], [[QK1_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>}
        %acc_165 = ttng.tmem_alloc %p_163 {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // acc_0 update store (partition 0) into the in-body acquired ACC0
        // buffer, release ACC0_F on the same token.
        // CHECK:             ttng.tmem_store %{{.*}}, [[ACC0_LD]][], %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[ACC0_F]], [[ACC0_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_166 = ttng.tmem_store %acc_160, %acc_0[%acc_134], %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // CHECK:             ttng.tmem_store %{{.*}}, [[ACC1_LD]][], %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>}
        // CHECK:             nvws.semaphore.release [[ACC1_F]], [[ACC1_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_167 = ttng.tmem_store %acc_161, %acc_1[%acc_136], %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #1 (partition 1): buffer R4_F3 (p f16 view #4), acquire ACC0_F
        // FULL, acquire V_F FULL, tc5mma, release ACC0_E.
        // CHECK:             [[PV0_AF:%.*]] = nvws.semaphore.acquire [[R4_F3]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[PV0_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F3]], [[PV0_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[PV0_ACC_AF:%.*]] = nvws.semaphore.acquire [[ACC0_F]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV0_ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC0_F]], [[PV0_ACC_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             [[PV0_V_AF:%.*]] = nvws.semaphore.acquire [[V_F]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV0_V_BUF:%.*]] = nvws.semaphore.buffer [[V_F]], [[PV0_V_AF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // CHECK:             ttng.tc_gen5_mma [[PV0_BUF]]#4, [[PV0_V_BUF]], [[PV0_ACC_BUF]][], %{{.*}}, %{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[ACC0_E]], [[PV0_ACC_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_170 = ttng.tc_gen5_mma %acc_164, %v, %acc_0[%acc_166], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #2 (partition 1): R5_F3 / ACC1_F / re-use V buffer, release the
        // inner V_E EMPTY and ACC1_E.
        // CHECK:             [[PV1_AF:%.*]] = nvws.semaphore.acquire [[R5_F3]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
        // CHECK:             [[PV1_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F3]], [[PV1_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
        // CHECK:             [[PV1_ACC_AF:%.*]] = nvws.semaphore.acquire [[ACC1_F]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:             [[PV1_ACC_BUF:%.*]] = nvws.semaphore.buffer [[ACC1_F]], [[PV1_ACC_AF]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:             ttng.tc_gen5_mma [[PV1_BUF]]#4, [[PV0_V_BUF]], [[PV1_ACC_BUF]][], %{{.*}}, %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK:             nvws.semaphore.release [[V_E]], [[PV0_V_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK:             nvws.semaphore.release [[ACC1_E]], [[PV1_ACC_AF]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %acc_171 = ttng.tc_gen5_mma %acc_165, %v, %acc_1[%acc_167], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg12, %alpha_126 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_172 = arith.mulf %arg16, %alpha_129 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_173 = arith.addf %l_i0, %l_ij {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_174 = arith.addf %l_i0_172, %l_ij_132 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_175 = arith.addi %offset_y_89, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
        // No R5 token is re-acquired at the loop boundary or yielded.
        // CHECK-NOT:         nvws.semaphore.acquire [[R5_IN]]
        // CHECK:             scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} :
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %offsetkv_y_175, %l_i0_173, %m_ij_113, %qk_104, %acc_170, %l_i0_174, %m_ij_114, %qk_107, %acc_171 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token
      // Inner-loop close: pinned pipelining attrs.
      // CHECK:           } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = {{\[}}array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 4>, array<i32: 4>]}
      } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>]}

      // Post-inner-loop epilogue (still inside the persistent outer loop). The
      // inner Q FULL tokens release back as EMPTY for the next outer iteration.
      // R5 is acquired at the post-inner use, while R4 uses its point-of-use
      // drain; each opens one final FULL semaphore.
      // CHECK:           nvws.semaphore.release [[Q1_E]], [[V_PRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           nvws.semaphore.release [[Q0_E]], [[K_PRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           [[R5_POST:%.*]] = nvws.semaphore.acquire [[R5_IN]] {ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R5_F4]], [[R5_POST]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}

      // Both post-loop stats stores (partition 4) land in the single R5_F4
      // phase: one acquire, one buffer, store #2 then #1.
      // CHECK:           [[OK18_AF:%.*]] = nvws.semaphore.acquire [[R5_F4]] {ttg.partition = array<i32: 4>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK18_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F4]], [[OK18_AF]] {ttg.partition = array<i32: 4>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK18_BUF]]#2, %{{.*}} {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y_40#6 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_42 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_41, %offsetkv_y_18, %offsetkv_y_42 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y_17 stats store (partition 4) reuses the same token for
      // buffer #1, then releases R5_F5 and the first R5_E arrival.
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK18_BUF]]#1, %{{.*}} {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      // CHECK:           nvws.semaphore.release [[R5_F5]], [[OK18_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      // CHECK:           nvws.semaphore.release [[R5_E]], [[OK18_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y_40#5 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_44 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_43, %offsetkv_y_17, %offsetkv_y_44 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK:           [[R4_DRAIN:%.*]] = nvws.semaphore.acquire [[R4_IN]] {ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R4_F4]], [[R4_DRAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}

      // Both post-loop stats stores (partition 5) land in the single R4_F4
      // phase: acquire, buffer, store #2 then #1.
      // CHECK:           [[OK16_AF:%.*]] = nvws.semaphore.acquire [[R4_F4]] {ttg.partition = array<i32: 5>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK16_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F4]], [[OK16_AF]] {ttg.partition = array<i32: 5>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK16_BUF]]#2, %{{.*}} {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y_40#2 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_46 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_45, %offsetkv_y_16, %offsetkv_y_46 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y stats store (partition 5) reuses the same token for buffer
      // #1, then releases R4_F5 and the first R4_E arrive.
      // CHECK:           ttng.tmem_store %{{.*}}, [[OK16_BUF]]#1, %{{.*}} {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      // CHECK:           nvws.semaphore.release [[R4_F5]], [[OK16_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>}
      // CHECK:           nvws.semaphore.release [[R4_E]], [[OK16_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 5>}
      %offsetkv_y_47 = tt.expand_dims %offsetkv_y_40#1 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_48 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_47, %offsetkv_y, %offsetkv_y_48 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y reload (partition 0): acquire R4_F5, buffer, load #1.
      // CHECK:           [[OKR_AF:%.*]] = nvws.semaphore.acquire [[R4_F5]] {ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OKR_BUF:%.*]]:5 = nvws.semaphore.buffer [[R4_F5]], [[OKR_AF]] {ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_load [[OKR_BUF]]#1[] {ttg.partition = array<i32: 0>}
      %offsetkv_y_49, %offsetkv_y_50 = ttng.tmem_load %offsetkv_y[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_51 = tt.reshape %offsetkv_y_49 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_52 = ttg.convert_layout %offsetkv_y_51 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0 = math.log2 %offsetkv_y_52 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_17 reload (partition 0): acquire R5_F5, buffer, load #1.
      // CHECK:           [[OK17R_AF:%.*]] = nvws.semaphore.acquire [[R5_F5]] {ttg.partition = array<i32: 0>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           [[OK17R_BUF:%.*]]:5 = nvws.semaphore.buffer [[R5_F5]], [[OK17R_AF]] {ttg.partition = array<i32: 0>} : <[{{.*}}]>, !ttg.async.token -> {{.*}}
      // CHECK:           ttng.tmem_load [[OK17R_BUF]]#1[] {ttg.partition = array<i32: 0>}
      %offsetkv_y_53, %offsetkv_y_54 = ttng.tmem_load %offsetkv_y_17[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_55 = tt.reshape %offsetkv_y_53 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_56 = ttg.convert_layout %offsetkv_y_55 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_57 = math.log2 %offsetkv_y_56 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_16 reload (partition 0): load #2 from the R4_F5 buffer,
      // then the second R4_E arrive.
      // CHECK:           ttng.tmem_load [[OKR_BUF]]#2[] {ttg.partition = array<i32: 0>}
      // CHECK:           nvws.semaphore.release [[R4_E]], [[OKR_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %offsetkv_y_58, %offsetkv_y_59 = ttng.tmem_load %offsetkv_y_16[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_60 = tt.reshape %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_61 = ttg.convert_layout %offsetkv_y_60 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_62 = arith.addf %offsetkv_y_61, %m_i0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_18 reload (partition 0): load #2 from the R5_F5 buffer,
      // then the second R5_E arrive.
      // CHECK:           ttng.tmem_load [[OK17R_BUF]]#2[] {ttg.partition = array<i32: 0>}
      // CHECK:           nvws.semaphore.release [[R5_E]], [[OK17R_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %offsetkv_y_63, %offsetkv_y_64 = ttng.tmem_load %offsetkv_y_18[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_65 = tt.reshape %offsetkv_y_63 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_66 = ttg.convert_layout %offsetkv_y_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_67 = arith.addf %offsetkv_y_66, %m_i0_57 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = tt.expand_dims %offsetkv_y_52 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_68 = tt.expand_dims %offsetkv_y_56 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_69 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc0_70 = tt.broadcast %acc0_68 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      // Final acc_0 readout (partition 0): bottom re-acquire of ACC0 EMPTY
      // (carried to the next outer iteration via yield), buffer, load.
      // CHECK:           [[ACCF0_AE:%.*]] = nvws.semaphore.acquire [[ACC0_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:           [[ACCF0_BUF:%.*]] = nvws.semaphore.buffer [[ACC0_E]], [[ACCF0_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_load [[ACCF0_BUF]][] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>}
      %acc, %acc_71 = ttng.tmem_load %acc_0[%offsetkv_y_40#4] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_72 = ttg.convert_layout %acc {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      // Final acc_1 readout (partition 0) mirrors acc_0.
      // CHECK:           [[ACCF1_AE:%.*]] = nvws.semaphore.acquire [[ACC1_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK:           [[ACCF1_BUF:%.*]] = nvws.semaphore.buffer [[ACC1_E]], [[ACCF1_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK:           ttng.tmem_load [[ACCF1_BUF]][] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>}
      %acc_73, %acc_74 = ttng.tmem_load %acc_1[%offsetkv_y_40#8] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_75 = ttg.convert_layout %acc_73 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_76 = arith.divf %acc_72, %acc0_69 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %acc0_77 = arith.divf %acc_75, %acc0_70 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %m_ptrs0 = arith.muli %off_hz, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %m_ptrs0_78 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : !tt.ptr<f32>, i32
      %m_ptrs0_79 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_80 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_81 = tt.addptr %m_ptrs0_79, %offs_m0_26 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %m_ptrs0_82 = tt.addptr %m_ptrs0_80, %offs_m0_27 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %6 = ttg.convert_layout %m_i0_62 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %7 = ttg.convert_layout %m_i0_67 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      tt.store %m_ptrs0_81, %6 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %m_ptrs0_82, %7 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      // Epilogue O0 local_store (partition 0): acquire O0 EMPTY, point-of-use
      // buffer, store, release O0 FULL.
      // CHECK:           [[O0_AE:%.*]] = nvws.semaphore.acquire [[O0_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O0_BUF:%.*]] = nvws.semaphore.buffer [[O0_E]], [[O0_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_store %{{.*}}, [[O0_BUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.semaphore.release [[O0_F]], [[O0_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %8 = arith.truncf %acc0_76 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %8, %3 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O1 local_store (partition 0) mirrors O0.
      // CHECK:           [[O1_AE:%.*]] = nvws.semaphore.acquire [[O1_E]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O1_BUF:%.*]] = nvws.semaphore.buffer [[O1_E]], [[O1_AE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_store %{{.*}}, [[O1_BUF]] {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           nvws.semaphore.release [[O1_F]], [[O1_AE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %10 = arith.truncf %acc0_77 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %10, %4 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O0 local_load (partition 2): acquire O0 FULL, buffer and
      // load. Ownership remains live through its descriptor store below.
      // CHECK:           [[O0L_AF:%.*]] = nvws.semaphore.acquire [[O0_F]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O0L_BUF:%.*]] = nvws.semaphore.buffer [[O0_F]], [[O0L_AF]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_load [[O0L_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %13 = ttg.local_load %3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %14 = ttg.convert_layout %13 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      // Epilogue O1 local_load (partition 2) mirrors O0. Each empty release
      // follows the descriptor store that completes that channel's read.
      // CHECK:           [[O1L_AF:%.*]] = nvws.semaphore.acquire [[O1_F]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK:           [[O1L_BUF:%.*]] = nvws.semaphore.buffer [[O1_F]], [[O1L_AF]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK:           ttg.local_load [[O1L_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      // CHECK:           tt.descriptor_store
      // CHECK:           nvws.semaphore.release [[O0_E]], [[O0L_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK:           tt.descriptor_store
      // CHECK:           nvws.semaphore.release [[O1_E]], [[O1L_AF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %16 = ttg.local_load %4 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %17 = ttg.convert_layout %16 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_10[%qo_offset_y_23, %c0_i32], %14 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_11[%5, %c0_i32], %17 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      %tile_idx_87 = arith.addi %tile_idx_19, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      // Outer-loop end bridges R4 and R5 back to their point-of-use gates
      // (two arrivals each). Only the ACC0/ACC1 readout tokens ride the yield.
      // CHECK:           [[OX_R4:%.*]] = nvws.semaphore.acquire [[R4_E]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R4_IN]], [[OX_R4]] [#nvws.async_op<none>] {arrive_count = 2 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK:           [[OX_R5:%.*]] = nvws.semaphore.acquire [[R5_E]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[{{.*}}]> -> !ttg.async.token
      // CHECK:           nvws.semaphore.release [[R5_IN]], [[OX_R5]] [#nvws.async_op<none>] {arrive_count = 2 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK:           scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %{{.*}}, [[ACCF0_AE]], [[ACCF1_AE]] : i32, !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tile_idx_87 : i32
    // Outer-loop close: pinned warp-specialize attrs (stages, tag, types).
    // CHECK:           } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = {{.*}}, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4, 5>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
