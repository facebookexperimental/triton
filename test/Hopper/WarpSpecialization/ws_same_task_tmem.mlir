// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" | FileCheck %s
// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK

// Code partitioning synchronizes TMEM through channels between a producer
// partition and a consumer partition. When the TMEM producer and consumer are
// in the same partition, the asynchronous MMA still needs ordering against the
// neighbouring TMEM accesses: the channel is built with the partition as its
// own consumer, and token lowering drops the same-partition full commit/wait
// but keeps the empty wait before the write and the MMA completion barrier.

// Persistent GEMM, B via TMA, TMA-store epilogue: A is a pointer tl.load
// (f32 cast to bf16) promoted to TMEM (`tmem_alloc %src`) in the gemm
// partition, the same partition as the MMA. The tmem_alloc is hoisted into a
// double-buffered TMEM slot written by a tmem_store. Before each write the
// gemm partition waits on the slot's empty barrier, which the MMA that read
// the slot two iterations earlier arrives on when it completes. Without the
// channel this used to build Interval(SIZE_MAX, 0) in the TMEM memory planner
// (Allocation.h assert).
// CHECK-LABEL: @tmem_operand_a_same_task
// CHECK: partition0(%{{.*}}: !tt.ptr<f32>, %[[A:arg[0-9]+]]: !ttg.memdesc<2x128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, %[[AEMPTY:arg[0-9]+]]: !ttg.memdesc<2x1xi64
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tt.load
// CHECK: %[[ASLOT:.*]] = ttg.memdesc_index %[[A]][%[[ASTOREIDX:[0-9]+]]]
// CHECK: %[[APHASE:.*]] = arith.xori
// CHECK: %[[APHASE32:.*]] = arith.extui %[[APHASE]]
// CHECK: ttng.wait_barrier %{{.*}}, %[[APHASE32]] {{.*}}direction = "backward"
// CHECK-NEXT: ttng.tmem_store %{{.*}}, %[[ASLOT]], %true
// CHECK: %[[AOP:.*]] = ttg.memdesc_index %[[A]][%[[AMMAIDX:[0-9]+]]]
// CHECK: %[[ADONE:.*]] = ttg.memdesc_index %[[AEMPTY]][
// CHECK-NEXT: ttng.tc_gen5_mma %[[AOP]], {{.*}}, %[[ADONE]][%true]
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
    scf.for %arg3 = %2 to %c128_i32 step %c148_i32  : i32 {
      %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
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

// A and B are TMA loads in the load partition, and the gemm partition both
// issues the MMA and reads the accumulator after the K loop. The accumulator
// is double-buffered across persistent tiles: the gemm partition waits on the
// tile's empty barrier before the first MMA, commits the MMAs to the full
// barrier after the loop, waits on it before the tmem_load, and releases the
// buffer after the read. This used to hit assert(false && "Unexpected
// Producer Found") in handleOperandD.
// CHECK-LABEL: @post_loop_acc_load_same_task_tma_persistent
// CHECK: default {
// CHECK: scf.for
// CHECK: %[[PH:.*]] = arith.trunci %{{.*}} : i64 to i1
// CHECK: %[[E0:.*]] = ttg.memdesc_index %[[EMPTY:.*]][%[[IDX:.*]]]
// CHECK: %[[NPH:.*]] = arith.xori %[[PH]], %true
// CHECK: %[[NPH32:.*]] = arith.extui %[[NPH]]
// CHECK: ttng.wait_barrier %[[E0]], %[[NPH32]]
// CHECK: scf.for
// CHECK: ttng.tc_gen5_mma
// CHECK: scf.yield
// CHECK: %[[F0:.*]] = ttg.memdesc_index %[[FULL:.*]][%[[IDX]]]
// CHECK-NEXT: ttng.tc_gen5_commit %[[F0]]
// CHECK: %[[F1:.*]] = ttg.memdesc_index %[[FULL]][%[[IDX]]]
// CHECK: %[[PH32:.*]] = arith.extui %[[PH]]
// CHECK-NEXT: ttng.wait_barrier %[[F1]], %[[PH32]]
// CHECK-NEXT: ttng.tmem_load
// CHECK: %[[E1:.*]] = ttg.memdesc_index %[[EMPTY]][%[[IDX]]]
// CHECK-NEXT: ttng.arrive_barrier %[[E1]], 1
// CHECK: partition0
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_acc_load_same_task_tma_persistent(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_o = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %descA = tt.make_tensor_descriptor %A, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %descB = tt.make_tensor_descriptor %B, [%c1024_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %c4_i32 = arith.constant 4 : i32
    %outer = scf.for %t = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%otok = %acc_tok) -> (!ttg.async.token) : i32 {
    %acc_init = ttng.tmem_store %cst, %acc[%otok], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %res:2 = scf.for %k = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_init) -> (i1, !ttg.async.token)  : i32 {
      %off = arith.muli %k, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %a = tt.descriptor_load %descA[%t, %off] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked2>
      %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xbf16, #blocked2>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %b = tt.descriptor_load %descB[%off, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked1>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%tok], %use_acc, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %mma_tok : i1, !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32], ttg.partition.types = ["gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    %out, %out_tok = ttng.tmem_load %acc[%res#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %out_cvt = ttg.convert_layout %out {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
    %o_base = tt.splat %out_ptr {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
    %o_ptrs = tt.addptr %o_base, %cst_o {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>, tensor<128x128xi32, #blocked1>
    tt.store %o_ptrs, %out_cvt {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>
    scf.yield %out_tok : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// A and B are pointer tl.loads staged through SMEM and the epilogue is a
// tl.store, so the partition scheduler creates neither a load nor an epilogue
// partition and every op propagates into the gemm task. With a single
// partition there is nothing to specialize, and code partitioning would not
// thread buffer counters through the loop, so the kernel is compiled without
// warp specialization and all WS metadata is stripped. The fallback is
// reported as a remark on the MMA, and only for this kernel; the specialized
// kernels in this file take no fallback.
// REMARK-NOT: does not support
// REMARK-NOT: compiling without warp specialization
// REMARK: remark: meta autoWS placed every op of this MMA kernel in one partition; compiling without warp specialization
// REMARK-NEXT: ttng.tc_gen5_mma
// REMARK-NOT: compiling without warp specialization
// CHECK-LABEL: @post_loop_acc_load_same_task(
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
    %out, %out_tok = ttng.tmem_load %acc[%res#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %out_cvt = ttg.convert_layout %out {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
    %o_base = tt.splat %out_ptr {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
    %o_ptrs = tt.addptr %o_base, %cst_o {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>, tensor<128x128xi32, #blocked1>
    tt.store %o_ptrs, %out_cvt {ttg.partition = array<i32: 0>} : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The GEMM of the first case with B as a pointer tl.load too. B is promoted to
// SMEM (`local_alloc %src`) in the gemm partition, the same partition as the
// MMA, so it gets the same treatment as A: the local_alloc is hoisted into a
// multi-buffered SMEM slot written by a local_store behind a wait on the
// slot's empty barrier, and the MMA arrives on both operands' empty barriers
// when it completes. Left in the loop without a channel, the local_alloc had
// no synchronization and the post-WS pipeliner failed to predicate it.

// CHECK-LABEL: @smem_operand_b_same_task
// CHECK: partition0(%{{.*}}: !tt.ptr<bf16>, %{{.*}}: !tt.ptr<f32>, %[[B:arg[0-9]+]]: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %[[BEMPTY:arg[0-9]+]]: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %[[A:arg[0-9]+]]: !ttg.memdesc<2x128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, %[[AEMPTY:arg[0-9]+]]: !ttg.memdesc<2x1xi64
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[BSLOT:.*]] = ttg.memdesc_index %[[B]][
// CHECK: %[[BBAR:.*]] = ttg.memdesc_index %[[BEMPTY]][%[[BIDX:.*]]]
// CHECK: ttng.wait_barrier %[[BBAR]], %{{.*}} {{.*}}direction = "backward"
// CHECK-NEXT: ttg.local_store %{{.*}}, %[[BSLOT]]
// CHECK: %[[ASLOT:.*]] = ttg.memdesc_index %[[A]][
// CHECK: %[[ABAR:.*]] = ttg.memdesc_index %[[AEMPTY]][%[[AIDX:.*]]]
// CHECK: ttng.wait_barrier %[[ABAR]], %{{.*}} {{.*}}direction = "backward"
// CHECK-NEXT: ttng.tmem_store %{{.*}}, %[[ASLOT]], %true
// CHECK: %[[BDONE:.*]] = ttg.memdesc_index %[[BEMPTY]][%[[BIDX]]]
// CHECK-NEXT: %[[ADONE:.*]] = ttg.memdesc_index %[[AEMPTY]][%[[AIDX]]]
// CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %{{.*}}[], %{{.*}}, %true, %[[BDONE]][%true], %[[ADONE]][%true]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @smem_operand_b_same_task(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<5248> : tensor<128x1xi64, #blocked>
    %cst_0 = arith.constant dense<2048> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<1024> : tensor<128x1xi64, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c41_i32 = arith.constant 41 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %1 = tt.get_program_id x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg3 = %1 to %c128_i32 step %c148_i32  : i32 {
      %6 = arith.divsi %arg3, %c64_i32 {ttg.partition = array<i32: 0>} : i32
      %7 = arith.muli %6, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %8 = arith.subi %c16_i32, %7 {ttg.partition = array<i32: 0>} : i32
      %9 = arith.minsi %8, %c8_i32 {ttg.partition = array<i32: 0>} : i32
      %10 = arith.remsi %arg3, %9 {ttg.partition = array<i32: 0>} : i32
      %11 = arith.addi %7, %10 {ttg.partition = array<i32: 0>} : i32
      %12 = arith.remsi %arg3, %c64_i32 {ttg.partition = array<i32: 0>} : i32
      %13 = arith.divsi %12, %9 {ttg.partition = array<i32: 0>} : i32
      %14 = arith.muli %11, %c128_i32 {ttg.partition = array<i32: 0>} : i32
      %15 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = arith.addi %15, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 0>} : i32
      %18 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %19 = arith.addi %18, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %22 = arith.cmpi slt, %16, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %24 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %25 = arith.extsi %24 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
      %26 = arith.muli %25, %cst : tensor<128x1xi64, #blocked>
      %27 = tt.addptr %5, %26 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi64, #blocked>
      %28 = tt.broadcast %27 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %29 = tt.broadcast %23 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %30 = ttng.tmem_store %cst_3, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %31:2 = scf.for %arg4 = %c0_i32 to %c41_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %30) -> (i1, !ttg.async.token)  : i32 {
        %35 = arith.muli %arg4, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %36 = tt.splat %35 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %37 = tt.splat %35 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %38 = arith.addi %36, %3 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %39 = arith.addi %37, %2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %40 = tt.expand_dims %38 {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
        %41 = arith.extsi %40 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
        %42 = arith.muli %41, %cst_1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x1xi64, #blocked>
        %43 = tt.addptr %4, %42 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x1x!tt.ptr<bf16>, #blocked>, tensor<128x1xi64, #blocked>
        %44 = tt.broadcast %43 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x1x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
        %45 = tt.addptr %44, %21 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
        %46 = tt.load %45 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<bf16>, #blocked>
        %47 = ttg.local_alloc %46 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %48 = tt.expand_dims %39 {axis = 0 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
        %49 = tt.broadcast %48 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %50 = tt.addptr %28, %49 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
        %51 = tt.load %50, %29, %cst_2 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
        %52 = arith.truncf %51 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %53 = ttg.convert_layout %52 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #linear>
        %result_6 = ttng.tmem_alloc %53 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>
        %54 = ttng.tc_gen5_mma %result_6, %47, %result[%arg6], %arg5, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %54 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %result_4, %token_5 = ttng.tmem_load %result[%31#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %32 = ttg.convert_layout %result_4 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      %33 = ttg.local_alloc %32 {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %34 = ttng.async_tma_copy_local_to_global %0[%14, %17] %33 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %34   {ttg.partition = array<i32: 0>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
