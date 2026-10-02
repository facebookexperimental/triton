// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=4 smem-alloc-algo=1 smem-budget=208000" | FileCheck %s

// Persistent 128x256x64 bf16 GEMM, num_stages=4, EPILOGUE_SUBTILE=2. Before
// warp specialization the epilogue converts the whole 128x128x2 f32 subtile
// tensor (16 KB of scratch) ahead of the tt.split. After warp specialization,
// triton-nvidia-optimize-tmem-layouts turns the tmem_load -> reshape -> trans ->
// split chain into one TMEM load per subtile, and the layout cleanup sinks each
// subtile's conversion past the truncf. That leaves two 128x128 bf16
// conversions in the same region, 8 KB each, sharing one slot. The planner must
// reserve 8 KB, not 16 KB: the 4/4 operand rings (196608 B) fit this budget
// with an 8 KB reservation but not with a 16 KB one.

// CHECK-LABEL: @triton_tem_fused_mm_0
// CHECK: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<64x256xbf16

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tem_fused_mm_0(%arg_A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg_B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ws_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c4096_i32 = arith.constant {async_task_id = array<i32: 1>} 4096 : i32
    %c1024_i32 = arith.constant {async_task_id = array<i32: 1>} 1024 : i32
    %c8192_i32 = arith.constant {async_task_id = array<i32: 1>} 8192 : i32
    %c4096_i64 = arith.constant {async_task_id = array<i32: 1>} 4096 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
    %c1024_i64 = arith.constant {async_task_id = array<i32: 1>} 1024 : i64
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 256 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %num_pid_in_group = arith.constant {async_task_id = array<i32: 1, 2>} 32 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %cst = arith.constant {async_task_id = array<i32: 2>} dense<1024> : tensor<128x1xi32, #blocked>
    %cst_0 = arith.constant {async_task_id = array<i32: 2>} dense<1024> : tensor<1x128xi32, #blocked>
    %cst_1 = arith.constant {async_task_id = array<i32: 2>} dense<8192> : tensor<128x1xi32, #blocked>
    %false = arith.constant {async_task_id = array<i32: 0>} false
    %accumulator, %accumulator_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %a_desc = tt.make_tensor_descriptor %arg_A, [%c8192_i32, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg_B, [%c4096_i32, %c1024_i32], [%c1024_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<64x256xbf16, #shared>
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 2>} : i32
    %_tmp_var0 = tt.make_range {async_task_id = array<i32: 2>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %_tmp_var0_3 = tt.make_range {async_task_id = array<i32: 2>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.splat %out_ptr0 {async_task_id = array<i32: 2>} : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %tile_id_c_4 = scf.for %arg4 = %start_pid to %c256_i32 step %c148_i32 iter_args(%arg5 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %arg4, %num_pid_in_group {async_task_id = array<i32: 1>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %GROUP_M = arith.subi %c64_i32, %first_pid_m {async_task_id = array<i32: 1>} : i32
      %GROUP_M_5 = arith.minsi %GROUP_M, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %pid_m = arith.remsi %arg4, %GROUP_M_5 {async_task_id = array<i32: 1>} : i32
      %pid_m_6 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 1>} : i32
      %pid_n = arith.remsi %arg4, %num_pid_in_group {async_task_id = array<i32: 1>} : i32
      %pid_n_7 = arith.divsi %pid_n, %GROUP_M_5 {async_task_id = array<i32: 1>} : i32
      %offs_am = arith.muli %pid_m_6, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %offs_bn = arith.muli %pid_n_7, %c256_i32 {async_task_id = array<i32: 1>} : i32
      %accumulator_8:2 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %accumulator_2) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %arg6, %c64_i32 {async_task_id = array<i32: 1>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %a_desc[%offs_am, %offs_k] 16384 %a {async_task_id = array<i32: 1>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x64xbf16, #shared>, i32, i32, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %b_desc[%offs_k, %offs_bn] 32768 %b {async_task_id = array<i32: 1>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x256xbf16, #shared>, i32, i32, !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
        %accumulator_36 = ttng.tc_gen5_mma %a, %b, %accumulator[%arg8], %arg7, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 2>} %true, %accumulator_36 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2>, tt.scheduled_max_stage = 3 : i32}
      %tile_id_c_9 = arith.addi %arg5, %c148_i32 {async_task_id = array<i32: 2>} : i32
      %group_id_10 = arith.divsi %tile_id_c_9, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %first_pid_m_11 = arith.muli %group_id_10, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %GROUP_M_12 = arith.subi %c64_i32, %first_pid_m_11 {async_task_id = array<i32: 2>} : i32
      %GROUP_M_13 = arith.minsi %GROUP_M_12, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_m_14 = arith.remsi %tile_id_c_9, %GROUP_M_13 {async_task_id = array<i32: 2>} : i32
      %pid_m_15 = arith.addi %first_pid_m_11, %pid_m_14 {async_task_id = array<i32: 2>} : i32
      %pid_n_16 = arith.remsi %tile_id_c_9, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %pid_n_17 = arith.divsi %pid_n_16, %GROUP_M_13 {async_task_id = array<i32: 2>} : i32
      %offs_cm = arith.muli %pid_m_15, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_cn = arith.muli %pid_n_17, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %accumulator_18, %accumulator_19 = ttng.tmem_load %accumulator[%accumulator_8#1] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %acc = tt.reshape %accumulator_18 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #linear> -> tensor<128x2x128xf32, #linear1>
      %acc_20 = tt.trans %acc {async_task_id = array<i32: 2>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear1> -> tensor<128x128x2xf32, #linear2>
      %subtiles = ttg.convert_layout %acc_20 {async_task_id = array<i32: 2>} : tensor<128x128x2xf32, #linear2> -> tensor<128x128x2xf32, #blocked1>
      %subtiles_21, %subtiles_22 = tt.split %subtiles {async_task_id = array<i32: 2>} : tensor<128x128x2xf32, #blocked1> -> tensor<128x128xf32, #blocked>
      %1 = arith.truncf %subtiles_22 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %2 = arith.truncf %subtiles_21 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %_tmp_var0_23 = tt.splat %offs_cm {async_task_id = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %_tmp_var0_24 = arith.addi %_tmp_var0_23, %_tmp_var0 {async_task_id = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %_tmp_var0_25 = tt.expand_dims %_tmp_var0_24 {async_task_id = array<i32: 2>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %_tmp_var2 = arith.cmpi slt, %_tmp_var0_25, %cst_1 {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked>
      %_tmp_var3 = tt.splat %offs_cn {async_task_id = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_26 = arith.addi %_tmp_var3, %_tmp_var0_3 {async_task_id = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_27 = tt.expand_dims %_tmp_var3_26 {async_task_id = array<i32: 2>, axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %_tmp_var5 = arith.cmpi slt, %_tmp_var3_27, %cst_0 {async_task_id = array<i32: 2>} : tensor<1x128xi32, #blocked>
      %_tmp_var6 = tt.broadcast %_tmp_var2 {async_task_id = array<i32: 2>} : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %_tmp_var6_28 = tt.broadcast %_tmp_var5 {async_task_id = array<i32: 2>} : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %_tmp_var6_29 = arith.andi %_tmp_var6, %_tmp_var6_28 {async_task_id = array<i32: 2>} : tensor<128x128xi1, #blocked>
      %3 = arith.muli %_tmp_var0_25, %cst {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked>
      %4 = tt.broadcast %_tmp_var3_27 {async_task_id = array<i32: 2>} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %5 = tt.broadcast %3 {async_task_id = array<i32: 2>} : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %6 = arith.addi %4, %5 {async_task_id = array<i32: 2>} : tensor<128x128xi32, #blocked>
      %7 = tt.addptr %0, %6 {async_task_id = array<i32: 2>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      tt.store %7, %2, %_tmp_var6_29 {async_task_id = array<i32: 2>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %offs_cn_i = arith.addi %offs_cn, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %_tmp_var3_30 = tt.splat %offs_cn_i {async_task_id = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_31 = arith.addi %_tmp_var3_30, %_tmp_var0_3 {async_task_id = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %_tmp_var3_32 = tt.expand_dims %_tmp_var3_31 {async_task_id = array<i32: 2>, axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %_tmp_var5_33 = arith.cmpi slt, %_tmp_var3_32, %cst_0 {async_task_id = array<i32: 2>} : tensor<1x128xi32, #blocked>
      %_tmp_var6_34 = tt.broadcast %_tmp_var5_33 {async_task_id = array<i32: 2>} : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %_tmp_var6_35 = arith.andi %_tmp_var6, %_tmp_var6_34 {async_task_id = array<i32: 2>} : tensor<128x128xi1, #blocked>
      %8 = tt.broadcast %_tmp_var3_32 {async_task_id = array<i32: 2>} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %9 = arith.addi %8, %5 {async_task_id = array<i32: 2>} : tensor<128x128xi32, #blocked>
      %10 = tt.addptr %0, %9 {async_task_id = array<i32: 2>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      tt.store %10, %1, %_tmp_var6_35 {async_task_id = array<i32: 2>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      scf.yield {async_task_id = array<i32: 2>} %tile_id_c_9 : i32
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
