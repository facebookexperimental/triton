// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=232448" | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=232448 reserve-auxiliary-smem=false" | FileCheck %s --check-prefix=NORESERVE

// Persistent 128x256x128 bf16 GEMM at num_stages=3 whose epilogue converts the
// 128x256 bf16 result from the TMEM-load layout to the store layout. That
// convert_layout needs 16 KB of scratch, and since it sits inside a partition it
// is live alongside the operand rings. A plan of A=3 / B=2 copies (229376 B)
// looks like it fits the budget, but with the scratch the kernel needs 246176 B
// and fails to launch (Inductor then silently recompiles it at num_stages=1,
// with a one-deep ring). The auxiliary reservation must include the scratch so
// the planner settles on A=2 / B=2. Without the reservation it picks A=3 / B=2.

// CHECK-LABEL: @triton_tem_fused_mm_0
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xbf16
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x256xbf16

// NORESERVE-LABEL: @triton_tem_fused_mm_0
// NORESERVE: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xbf16
// NORESERVE: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x256xbf16

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tem_fused_mm_0(%arg_A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg_B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ws_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c4096_i32 = arith.constant {async_task_id = array<i32: 1>} 4096 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 256 : i32
    %c8192_i32 = arith.constant {async_task_id = array<i32: 1>} 8192 : i32
    %c4096_i64 = arith.constant {async_task_id = array<i32: 1>} 4096 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
    %c256_i64 = arith.constant {async_task_id = array<i32: 1>} 256 : i64
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 128 : i32
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2>} 32 : i32
    %grid_m = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %cst = arith.constant {async_task_id = array<i32: 2>} dense<256> : tensor<128x1xi32, #blocked>
    %cst_0 = arith.constant {async_task_id = array<i32: 2>} dense<256> : tensor<1x256xi32, #blocked>
    %cst_1 = arith.constant {async_task_id = array<i32: 2>} dense<8192> : tensor<128x1xi32, #blocked>
    %false = arith.constant {async_task_id = array<i32: 0>} false
    %accumulator, %accumulator_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xbf16, #shared, #smem, mutable>
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %a_desc = tt.make_tensor_descriptor %arg_A, [%c8192_i32, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg_B, [%c4096_i32, %c256_i32], [%c256_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x256xbf16, #shared>
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 2>} : i32
    %_tmp_var0 = tt.make_range {async_task_id = array<i32: 2>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %_tmp_var3 = tt.make_range {async_task_id = array<i32: 2>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.splat %out_ptr0 {async_task_id = array<i32: 2>} : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>, #blocked>
    %tile_id_c_3 = scf.for %arg4 = %start_pid to %grid_m step %c148_i32 iter_args(%arg5 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %arg4, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %GROUP_M = arith.subi %grid_m, %first_pid_m {async_task_id = array<i32: 1>} : i32
      %GROUP_M_4 = arith.minsi %GROUP_M, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %pid_m = arith.remsi %arg4, %GROUP_M_4 {async_task_id = array<i32: 1>} : i32
      %pid_m_5 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 1>} : i32
      %pid_n = arith.remsi %arg4, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %pid_n_6 = arith.divsi %pid_n, %GROUP_M_4 {async_task_id = array<i32: 1>} : i32
      %offs_am = arith.muli %pid_m_5, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %offs_bn = arith.muli %pid_n_6, %c256_i32 {async_task_id = array<i32: 1>} : i32
      %accumulator_7:2 = scf.for %arg6 = %c0_i32 to %k_tiles step %c1_i32 iter_args(%arg7 = %false, %arg8 = %accumulator_2) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %arg6, %c128_i32 {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %a_desc[%offs_am, %offs_k] 32768 %a {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x128xbf16, #shared>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %b_desc[%offs_k, %offs_bn] 65536 %b {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x256xbf16, #shared>, i32, i32, !ttg.memdesc<128x256xbf16, #shared, #smem, mutable>
        %accumulator_27 = ttng.tc_gen5_mma %a, %b, %accumulator[%arg8], %arg7, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 2>} %true, %accumulator_27 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2>, tt.scheduled_max_stage = 2 : i32}
      %tile_id_c_8 = arith.addi %arg5, %c148_i32 {async_task_id = array<i32: 2>} : i32
      %group_id_9 = arith.divsi %tile_id_c_8, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %first_pid_m_10 = arith.muli %group_id_9, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %GROUP_M_11 = arith.subi %grid_m, %first_pid_m_10 {async_task_id = array<i32: 2>} : i32
      %GROUP_M_12 = arith.minsi %GROUP_M_11, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_m_13 = arith.remsi %tile_id_c_8, %GROUP_M_12 {async_task_id = array<i32: 2>} : i32
      %pid_m_14 = arith.addi %first_pid_m_10, %pid_m_13 {async_task_id = array<i32: 2>} : i32
      %pid_n_15 = arith.remsi %tile_id_c_8, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_n_16 = arith.divsi %pid_n_15, %GROUP_M_12 {async_task_id = array<i32: 2>} : i32
      %offs_cm = arith.muli %pid_m_14, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_cn = arith.muli %pid_n_16, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %_tmp_var0_17 = tt.splat %offs_cm {async_task_id = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %_tmp_var0_18 = arith.addi %_tmp_var0_17, %_tmp_var0 {async_task_id = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %_tmp_var0_19 = tt.expand_dims %_tmp_var0_18 {async_task_id = array<i32: 2>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %_tmp_var2 = arith.cmpi slt, %_tmp_var0_19, %cst_1 {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked>
      %_tmp_var3_20 = tt.splat %offs_cn {async_task_id = array<i32: 2>} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_21 = arith.addi %_tmp_var3_20, %_tmp_var3 {async_task_id = array<i32: 2>} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_22 = tt.expand_dims %_tmp_var3_21 {async_task_id = array<i32: 2>, axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %_tmp_var5 = arith.cmpi slt, %_tmp_var3_22, %cst_0 {async_task_id = array<i32: 2>} : tensor<1x256xi32, #blocked>
      %_tmp_var6 = tt.broadcast %_tmp_var2 {async_task_id = array<i32: 2>} : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
      %_tmp_var6_23 = tt.broadcast %_tmp_var5 {async_task_id = array<i32: 2>} : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
      %_tmp_var6_24 = arith.andi %_tmp_var6, %_tmp_var6_23 {async_task_id = array<i32: 2>} : tensor<128x256xi1, #blocked>
      %1 = arith.muli %_tmp_var0_19, %cst {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked>
      %2 = tt.broadcast %_tmp_var3_22 {async_task_id = array<i32: 2>} : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
      %3 = tt.broadcast %1 {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
      %4 = arith.addi %2, %3 {async_task_id = array<i32: 2>} : tensor<128x256xi32, #blocked>
      %5 = tt.addptr %0, %4 {async_task_id = array<i32: 2>} : tensor<128x256x!tt.ptr<bf16>, #blocked>, tensor<128x256xi32, #blocked>
      %accumulator_25, %accumulator_26 = ttng.tmem_load %accumulator[%accumulator_7#1] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %6 = arith.truncf %accumulator_25 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #linear> to tensor<128x256xbf16, #linear>
      %7 = ttg.convert_layout %6 {async_task_id = array<i32: 2>} : tensor<128x256xbf16, #linear> -> tensor<128x256xbf16, #blocked>
      tt.store %5, %7, %_tmp_var6_24 {async_task_id = array<i32: 2>} : tensor<128x256x!tt.ptr<bf16>, #blocked>
      scf.yield {async_task_id = array<i32: 2>} %tile_id_c_8 : i32
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
