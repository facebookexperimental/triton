// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s

// Test: inside->outside subtiled epilogue channel with a STRADDLE. This is the
// addmm EPILOGUE_SUBTILE=2, separate_epilogue_store=True,
// early_tma_store_lowering=False config (addmm_kernel_tma_persistent_ws).
//
// doGenerateSubtiledRegion folds only the producer chains (arith.truncf,
// task 0) into one ttng.subtiled_region, leaving the two flat consumers
// (tt.descriptor_store, task 2) outside. The two subtiles are DISTINCT
// single-copy staging buffers (not a reuse group). The tile-1 per-tile bias
// descriptor_load + offs_cn sit AFTER consumer0 (the offs_bn store), so the
// region is pinned BETWEEN the two consumers: consumer0 precedes the region,
// consumer1 (the offs_cn store) follows it.
//
// This guards two fixes in WSCodePartition (both required to compile AND run):
//
//  1. Dominance: the flat consumer's outer bufferIdx/phase are anchored at the
//     earliest endpoint (the pre-region consumer), not the region. Without it
//     the pass aborted in the verifier:
//       'arith.trunci' op operation destroyed but still has uses
//       use: nvws.consumer_wait(... <<UNKNOWN SSA VALUE>> ...)
//
//  2. Per-tile producer token: the numTiles sibling tokens are threaded as ONE
//     per-tile arg (ttng.subtiled_region addPerTilePosition) so each replicated
//     tile acquires/commits ONLY its own buffer's barrier. Threading them
//     shared made every tile arrive on every sibling's barrier (over-commit),
//     which compiled but DEADLOCKED at runtime.
//
// The RUN succeeding is the primary regression guard for fix 1 (pre-fix the pass
// aborts before any output). The barrier balance below is the guard for fix 2:
// producer (task 0) does exactly ONE acquire (wait_barrier) + ONE commit
// (arrive_barrier) per tile, each storing to its OWN buffer; and the
// epilogue-store partition (task 2) has a balanced ONE wait + ONE release
// (arrive_barrier) per consumer -- pre-fix the pre-region consumer's release was
// dropped (2 waits / 1 release), and the producer over-committed (2 arrives per
// tile).

// CHECK-LABEL: @addmm_kernel_tma_persistent_ws
// CHECK: ttg.warp_specialize
//
// Producer (task 0), tile 0: acquire its own buffer, store, commit (one each).
// CHECK:      arith.truncf {{.*}} {ttg.partition = array<i32: 0>}
// CHECK:      ttng.wait_barrier {{.*}}dstTask = 2{{.*}}ttg.partition = array<i32: 0>
// CHECK:      ttg.local_store {{.*}} {ttg.partition = array<i32: 0>}
// CHECK:      ttng.arrive_barrier {{.*}}dstTask = 2{{.*}}ttg.partition = array<i32: 0>
// tile 1: its own distinct buffer, one acquire + one commit (NOT both barriers).
// CHECK:      arith.truncf {{.*}} {ttg.partition = array<i32: 0>}
// CHECK:      ttng.wait_barrier {{.*}}dstTask = 2{{.*}}ttg.partition = array<i32: 0>
// CHECK:      ttg.local_store {{.*}} {ttg.partition = array<i32: 0>}
// CHECK:      ttng.arrive_barrier {{.*}}dstTask = 2{{.*}}ttg.partition = array<i32: 0>
//
// Epilogue-store partition (task 2): each flat consumer waits, loads, stores,
// then RELEASES its barrier (balanced 2 waits + 2 releases).
// CHECK:      ttng.wait_barrier {{.*}}ttg.partition = array<i32: 2>
// CHECK:      ttg.local_load {{.*}} {ttg.partition = array<i32: 2>}
// CHECK:      tt.descriptor_store {{.*}} {ttg.partition = array<i32: 2>}
// CHECK:      ttng.arrive_barrier {{.*}}ttg.partition = array<i32: 2>
// CHECK:      ttng.wait_barrier {{.*}}ttg.partition = array<i32: 2>
// CHECK:      ttg.local_load {{.*}} {ttg.partition = array<i32: 2>}
// CHECK:      tt.descriptor_store {{.*}} {ttg.partition = array<i32: 2>}
// CHECK:      ttng.arrive_barrier {{.*}}ttg.partition = array<i32: 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel_tma_persistent_ws(%a_desc: !tt.tensordesc<128x64xf16, #shared>, %b_desc: !tt.tensordesc<128x64xf16, #shared>, %c_desc: !tt.tensordesc<128x64xf16, #shared>, %bias_desc: !tt.tensordesc<128x64xf16, #shared>, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %k_tiles = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_0 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_1 = arith.divsi %num_pid_n, %c128_i32 : i32
    %k_tiles_2 = arith.addi %K, %k_tiles : i32
    %k_tiles_3 = arith.divsi %k_tiles_2, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m_0, %num_pid_n_1 : i32
    %num_pid_in_group = arith.muli %num_pid_n_1, %c8_i32 : i32
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_0, %first_pid_m : i32
      %group_size_m_4 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_4 : i32
      %pid_m_5 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_6 = arith.divsi %pid_n, %group_size_m_4 : i32
      %offs_am = arith.muli %pid_m_5, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_6, %c128_i32 : i32
      %accumulator, %accumulator_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_8 = ttng.tmem_store %cst, %accumulator[%accumulator_7], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_9:2 = scf.for %ki = %c0_i32 to %k_tiles_3 step %c1_i32 iter_args(%accumulator_23 = %false, %accumulator_24 = %accumulator_8) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_25 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %accumulator_26 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %accumulator_27 = ttg.memdesc_trans %accumulator_26 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %accumulator_28 = ttng.tc_gen5_mma %a_25, %accumulator_27, %accumulator[%accumulator_24], %accumulator_23, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_28 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %accumulator_10, %accumulator_11 = ttng.tmem_load %accumulator[%accumulator_9#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %acc_slices = tt.reshape %accumulator_10 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %acc_slices_12 = tt.trans %acc_slices {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %acc_slices_13, %acc_slices_14 = tt.split %acc_slices_12 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %acc_slices_15 = ttg.convert_layout %acc_slices_14 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x64xf32, #blocked>
      %acc_slices_16 = ttg.convert_layout %acc_slices_13 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x64xf32, #blocked>
      %bias = tt.descriptor_load %bias_desc[%offs_am, %offs_bn] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %bias_17 = arith.extf %bias {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
      %c = arith.addf %acc_slices_16, %bias_17 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked>
      %c_18 = arith.truncf %c {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      tt.descriptor_store %c_desc[%offs_am, %offs_bn], %c_18 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
      %offs_cn = arith.addi %offs_bn, %c64_i32 : i32
      %bias_19 = tt.descriptor_load %bias_desc[%offs_am, %offs_cn] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %bias_20 = arith.extf %bias_19 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> to tensor<128x64xf32, #blocked>
      %c_21 = arith.addf %acc_slices_15, %bias_20 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked>
      %c_22 = arith.truncf %c_21 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      tt.descriptor_store %c_desc[%offs_am, %offs_cn], %c_22 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.disallow_acc_multi_buffer, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
