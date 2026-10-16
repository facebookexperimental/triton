// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s
//
// The loops carry tt.separate_epilogue_store (and tt.merge_epilogue), which
// drive the partition layout.
//
// Persistent WS GEMM with separate_epilogue_store whose epilogue has both a
// TMA store and a TMA reduce-add (c_desc.store(acc); d_desc.atomic_add(acc)).
//
// schedulePostLoopOps walks the K-loop results and overrides earlier
// assignments. It routed every TMAStoreTokenWaitOp to the epilogue-store
// partition, including the reduce's wait, while the reduce itself went to the
// epilogue partition. doCodePartition then cloned the reduce into the
// epilogue-store partition to satisfy the wait, so d came out 2x (bug #12's
// misrouting, reached through the post-loop walk instead of Phase 4).
//
// @nomma_store_and_reduce is the no-MMA (norm-style) form: its in-loop
// epilogue-store anchoring pulled the reduce's wait into epilogue_store the
// same way.
//
// Check the reduce and its wait share a partition, and the plain store's
// copy and wait stay in the distinct epilogue-store partition.

// CHECK-LABEL: @ses_store_and_reduce
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: [[STORE:[0-9]+]]>
// CHECK-NEXT: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[STORE]]>
// CHECK: ttng.async_tma_reduce add, {{.*}}ttg.partition = array<i32: [[RED:[0-9]+]]>
// CHECK-NEXT: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK-NOT: ttng.async_tma_reduce
// CHECK: ttg.partition.types = [{{.*}}"epilogue_store"

// CHECK-LABEL: @nomma_store_and_reduce
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: [[NSTORE:[0-9]+]]>
// CHECK-NEXT: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[NSTORE]]>
// CHECK: ttng.async_tma_reduce add, {{.*}}ttg.partition = array<i32: [[NRED:[0-9]+]]>
// CHECK-NEXT: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[NRED]]>
// CHECK: ttg.partition.types = [{{.*}}"epilogue_store"

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#nblocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ses_store_and_reduce(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.extsi %arg7 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg5, %arg7], [%0, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %2 = tt.make_tensor_descriptor %arg1, [%arg6, %arg7], [%0, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %3 = arith.extsi %arg6 : i32 to i64
    %4 = tt.make_tensor_descriptor %arg2, [%arg5, %arg6], [%3, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared1>
    %5 = tt.make_tensor_descriptor %arg3, [%arg5, %arg6], [%3, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared1>
    %6 = arith.addi %arg5, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.addi %arg6, %c127_i32 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = tt.get_program_id x : i32
    %11 = arith.muli %7, %9 : i32
    %12 = arith.addi %arg7, %c63_i32 : i32
    %13 = arith.divsi %12, %c64_i32 : i32
    scf.for %arg8 = %10 to %11 step %c148_i32  : i32 {
      %14 = arith.divsi %arg8, %9 : i32
      %15 = arith.muli %14, %c128_i32 : i32
      %16 = arith.remsi %arg8, %9 : i32
      %17 = arith.muli %16, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %18 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %19:2 = scf.for %arg9 = %c0_i32 to %13 step %c1_i32 iter_args(%arg10 = %false, %arg11 = %18) -> (i1, !ttg.async.token)  : i32 {
        %25 = arith.muli %arg9, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %26 = tt.descriptor_load %1[%15, %25] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %27 = ttg.local_alloc %26 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %28 = tt.descriptor_load %2[%17, %25] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %29 = ttg.local_alloc %28 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %30 = ttg.memdesc_trans %29 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
        %31 = ttng.tc_gen5_mma %27, %30, %result[%arg11], %arg10, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %31 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %20 = ttg.convert_layout %result_0 : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      %21 = ttg.local_alloc %20 : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %22 = ttng.async_tma_copy_local_to_global %4[%15, %17] %21 : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %22   : !ttg.async.token
      %23 = ttg.local_alloc %20 : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %24 = ttng.async_tma_reduce add, %5[%15, %17] %23 : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %24   : !ttg.async.token
    } {tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
  tt.func public @nomma_store_and_reduce(%x_desc: !tt.tensordesc<64x64xf32, #shared1>, %y_desc: !tt.tensordesc<64x64xf32, #shared1>, %z_desc: !tt.tensordesc<64x64xf32, #shared1>, %n_tiles: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
      %offs = arith.muli %i, %c64_i32 : i32
      %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<64x64xf32, #shared1> -> tensor<64x64xf32, #nblocked>
      %sq = arith.mulf %x, %x : tensor<64x64xf32, #nblocked>
      %r = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) : (tensor<64x64xf32, #nblocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #nblocked}>>
      %rinv = math.rsqrt %r : tensor<64xf32, #ttg.slice<{dim = 1, parent = #nblocked}>>
      %rinv_e = tt.expand_dims %rinv {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #nblocked}>> -> tensor<64x1xf32, #nblocked>
      %rinv_b = tt.broadcast %rinv_e : tensor<64x1xf32, #nblocked> -> tensor<64x64xf32, #nblocked>
      %norm = arith.mulf %x, %rinv_b : tensor<64x64xf32, #nblocked>
      %stage = ttg.local_alloc %norm : (tensor<64x64xf32, #nblocked>) -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<64x64xf32, #shared1>, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
      %stage2 = ttg.local_alloc %norm : (tensor<64x64xf32, #nblocked>) -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
      %rtok = ttng.async_tma_reduce add, %z_desc[%offs, %c0_i32] %stage2 : !tt.tensordesc<64x64xf32, #shared1>, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %rtok : !ttg.async.token
    } {tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}
