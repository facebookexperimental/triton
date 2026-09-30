// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -verify-diagnostics | FileCheck %s

// Code partitioning only synchronizes TMEM through cross-partition channels.
// When a TMEM producer and its consumer end up in the same partition there is
// no channel, so nothing orders them and nothing gives the buffer a live range.
// Meta autoWS must warn and fall back to a plain non-WS kernel with all WS
// metadata stripped for the two shapes below.

// Persistent GEMM, B via TMA, TMA-store epilogue: A is a pointer tl.load
// (f32 cast to bf16) promoted to TMEM (`tmem_alloc %src`) in the gemm
// partition, the same partition as the MMA. Each iteration's tmem_alloc
// rewrites A while the previous asynchronous MMA may still be reading it. This
// used to build Interval(SIZE_MAX, 0) in the TMEM memory planner
// (Allocation.h assert).
// CHECK-LABEL: @tmem_operand_a_same_task
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xbf16, #linear>)
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tmem_load
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_operand_a_same_task(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<5248> : tensor<128x1xi64, #blocked>
    %cst_0 = arith.constant dense<2048> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c5248_i32 = arith.constant 5248 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c41_i32 = arith.constant 41 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c5248_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared1>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg3 = %2 to %c128_i32 step %c148_i32  : i32 {
      %6 = arith.divsi %arg3, %c64_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %7 = arith.muli %6, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.subi %c16_i32, %7 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.minsi %8, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.remsi %arg3, %9 {ttg.partition = array<i32: 0>} : i32
      %11 = arith.addi %7, %10 {ttg.partition = array<i32: 0>} : i32
      %12 = arith.remsi %arg3, %c64_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.divsi %12, %9 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.muli %11, %c128_i32 {ttg.partition = array<i32: 0>} : i32
      %15 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.addi %16, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = arith.cmpi slt, %17, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %20 = tt.expand_dims %17 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %21 = arith.extsi %20 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
      %22 = arith.muli %21, %cst : tensor<128x1xi64, #blocked>
      %23 = tt.addptr %5, %22 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi64, #blocked>
      %24 = tt.broadcast %23 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %25 = tt.broadcast %19 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %26 = ttng.tmem_store %cst_2, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %27:2 = scf.for %arg4 = %c0_i32 to %c41_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %26) -> (i1, !ttg.async.token)  : i32 {
        %31 = arith.muli %arg4, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %32 = tt.descriptor_load %1[%31, %15] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared1> -> tensor<128x128xbf16, #blocked1>
        %33 = ttg.local_alloc %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %34 = tt.splat %31 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %35 = arith.addi %34, %4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %36 = tt.expand_dims %35 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
        %37 = tt.broadcast %36 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %38 = tt.addptr %24, %37 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
        %39 = tt.load %38, %25, %cst_1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
        %40 = arith.truncf %39 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %41 = ttg.convert_layout %40 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #linear>
        %result_5 = ttng.tmem_alloc %41 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>
        // expected-warning @below {{meta autoWS does not support an MMA whose A operand is written to TMEM in the MMA's own partition}}
        %42 = ttng.tc_gen5_mma %result_5, %33, %result[%arg6], %arg5, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %42 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_3, %token_4 = ttng.tmem_load %result[%27#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %28 = ttg.convert_layout %result_3 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      %29 = ttg.local_alloc %28 {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %30 = ttng.async_tma_copy_local_to_global %0[%14, %15] %29 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %30   {ttg.partition = array<i32: 0>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}


// -----

// A and B are pointer tl.loads staged through SMEM and the epilogue is a
// tl.store, so the partition scheduler creates neither a load nor an epilogue
// partition and every op propagates into the gemm task. handleOperandD then
// finds the post-loop tmem_load in the MMA's own task, which used to hit
// assert(false && "Unexpected Producer Found") -- and silently skip the
// channel on release builds, walking into further broken invariants.
// CHECK-LABEL: @post_loop_acc_load_same_task
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tmem_load
// CHECK: tt.store
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_acc_load_same_task(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_a = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %cst_b = arith.constant dense<8192> : tensor<64x128xi32, #blocked1>
    %cst_o = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %a_base = tt.splat %A : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
    %b_base = tt.splat %B : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked1>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_init = ttng.tmem_store %cst, %acc[%acc_tok], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %res:4 = scf.for %k = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_init, %a_ptrs = %a_base, %b_ptrs = %b_base) -> (i1, !ttg.async.token, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked1>)  : i32 {
      %a = tt.load %a_ptrs {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked>
      %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %b = tt.load %b_ptrs {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked1>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%tok], %use_acc, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %a_next = tt.addptr %a_ptrs, %cst_a {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
      %b_next = tt.addptr %b_ptrs, %cst_b {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked1>, tensor<64x128xi32, #blocked1>
      scf.yield %true, %mma_tok, %a_next, %b_next : i1, !ttg.async.token, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked1>
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32], ttg.partition.types = ["gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    // expected-warning @below {{meta autoWS does not support reading an MMA accumulator after its loop in the MMA's own partition}}
    %out, %out_tok = ttng.tmem_load %acc[%res#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %out_cvt = ttg.convert_layout %out {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
    %o_base = tt.splat %out_ptr {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
    %o_ptrs = tt.addptr %o_base, %cst_o {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>, tensor<128x128xi32, #blocked1>
    tt.store %o_ptrs, %out_cvt {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
