// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=1 smem-budget=232448" | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=1 smem-budget=232448" --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s --check-prefix=PART

// Real pre-planner IR of a persistent autoWS GEMM with two accumulators whose
// fp32 tiles are TMA-stored through the same descriptor under
// separate_epilogue_store (MRS e92, B_MODE=fused, BM64 BN128 BK128 ns1).
//
// The plan is over budget, so Phase 3.6 looks for reuse targets for the two
// TMA staging buffers (ids 4 and 5). It used to land staging 4 on staging 5 and
// staging 5 on operand 9. A staging host is never realized, so the planner
// under-counted SMEM, and the 4->5 pair's "load" task was the staging fill
// task, so code partitioning found the pairs inconsistent and dropped the
// cross-tile WAR token. Staging 5 then shared operand 9's SMEM with nothing
// stopping the next tile's load from overwriting it mid-store (silent wrong
// output).
//
// Check each staging buffer reuses a distinct operand host, and that code
// partitioning then emits the cross-tile WAR token (created at function entry,
// before any allocation) and realizes both reuses.

// CHECK-DAG: allocation.reuseTarget = 9 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32
// CHECK-DAG: allocation.reuseTarget = 12 : i32, allocation.shareGroup = 5 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.tmaStaging = 1 : i32

// PART-LABEL: tt.func public @e92_fused_kernel
// PART-NOT: ttg.local_alloc
// PART: nvws.create_token {loadType = 3 : i32, numBuffers = 1 : i32} : tensor<1x!nvws.token>
// PART-DAG: ttg.memdesc_reinterpret {{.*}}buffer.id = 4 : i32
// PART-DAG: ttg.memdesc_reinterpret {{.*}}buffer.id = 5 : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @e92_fused_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32) attributes {noinline = false} {
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 148 : i32
    %c2_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 2 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 63 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %13 = arith.addi %arg7, %c63_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %14 = arith.divsi %13, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %15 = arith.addi %arg8, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %16 = arith.divsi %15, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %17 = arith.muli %14, %16 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %18 = arith.addi %arg9, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %19 = arith.divsi %18, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %20 = tt.make_tensor_descriptor %arg2, [%arg9], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<128xf32, #shared2>
    %21 = arith.divsi %arg9, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %22 = arith.muli %arg7, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %23 = arith.extsi %22 {async_task_id = array<i32: 3>} : i32 to i64
    %24 = tt.make_tensor_descriptor %arg0, [%21, %22], [%23, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared1>
    %25 = tt.make_tensor_descriptor %arg1, [%21, %22], [%23, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared1>
    %26 = arith.muli %arg8, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %27 = arith.extsi %26 {async_task_id = array<i32: 3>} : i32 to i64
    %28 = tt.make_tensor_descriptor %arg3, [%21, %26], [%27, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %29 = arith.muli %arg12, %c2_i32 {async_task_id = array<i32: 2>} : i32
    %30 = arith.extsi %arg10 {async_task_id = array<i32: 2>} : i32 to i64
    %31 = tt.make_tensor_descriptor %arg5, [%29, %arg10], [%30, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %32 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %33 = arith.muli %16, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
    %34 = tt.splat %arg13 {async_task_id = array<i32: 0>} : f32 -> tensor<64x1xf32, #blocked>
    %35 = arith.subi %arg8, %c2_i32 {async_task_id = array<i32: 3>} : i32
    scf.for %arg14 = %32 to %17 step %c148_i32  : i32 {
      %36 = arith.divsi %arg14, %17 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %37 = arith.remsi %arg14, %17 {async_task_id = array<i32: 2, 3>} : i32
      %38 = arith.divsi %37, %33 {async_task_id = array<i32: 2, 3>} : i32
      %39 = arith.muli %38, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %40 = arith.subi %14, %39 {async_task_id = array<i32: 2, 3>} : i32
      %41 = arith.minsi %40, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %42 = arith.remsi %37, %41 {async_task_id = array<i32: 2, 3>} : i32
      %43 = arith.addi %39, %42 {async_task_id = array<i32: 2, 3>} : i32
      %44 = arith.remsi %37, %33 {async_task_id = array<i32: 2, 3>} : i32
      %45 = arith.divsi %44, %41 {async_task_id = array<i32: 2, 3>} : i32
      %46 = arith.muli %43, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %47 = arith.muli %45, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %48 = arith.muli %36, %19 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %49 = arith.addi %36, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %50 = arith.muli %49, %19 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %51 = arith.addi %arg7, %46 {async_task_id = array<i32: 3>} : i32
      %52 = arith.addi %35, %47 {async_task_id = array<i32: 3>} : i32
      %53:3 = scf.for %arg15 = %48 to %50 step %c1_i32 iter_args(%arg16 = %false, %arg17 = %token, %arg18 = %token_1) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %62 = arith.muli %arg15, %c64_i32 {async_task_id = array<i32: 3>} : i32
        %63 = arith.muli %arg15, %c128_i32 {async_task_id = array<i32: 3>} : i32
        nvws.descriptor_load %20[%63] 512 %6 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<128xf32, #shared2>, i32, !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %64 = ttg.local_load %6 {async_task_id = array<i32: 0>} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked1>
        %65 = tt.reshape %64 {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked1> -> tensor<64x2xf32, #blocked2>
        %66 = ttg.convert_layout %65 {async_task_id = array<i32: 0>} : tensor<64x2xf32, #blocked2> -> tensor<64x2xf32, #blocked>
        %outLHS, %outRHS = tt.split %66 {async_task_id = array<i32: 0>} : tensor<64x2xf32, #blocked> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        nvws.descriptor_load %24[%62, %46] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared1>, i32, i32, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
        %67 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %25[%62, %46] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared1>, i32, i32, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
        %68 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
        %69 = arith.addf %67, %68 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %70 = tt.expand_dims %outLHS {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
        %71 = arith.mulf %70, %34 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #blocked>
        %72 = tt.broadcast %71 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %73 = arith.addf %69, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %74 = arith.truncf %73 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
        nvws.descriptor_load %28[%62, %47] 32768 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x128xf32, #shared1>, i32, i32, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
        %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> tensor<64x128xf32, #blocked3>
        %76 = arith.truncf %75 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #blocked3> to tensor<64x128xbf16, #blocked3>
        ttg.local_store %76, %0 {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked3> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
        ttg.local_store %74, %1 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %77 = ttg.memdesc_trans %1 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared3, #smem, mutable>
        %78 = ttng.tc_gen5_mma %77, %0, %result[%arg17], %arg16, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x64xbf16, #shared3, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
        nvws.descriptor_load %24[%62, %51] 16384 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared1>, i32, i32, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
        %79 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %25[%62, %51] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared1>, i32, i32, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
        %80 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
        %81 = arith.addf %79, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %82 = tt.expand_dims %outRHS {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
        %83 = arith.mulf %82, %34 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #blocked>
        %84 = tt.broadcast %83 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %85 = arith.addf %81, %84 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %86 = arith.truncf %85 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
        nvws.descriptor_load %28[%62, %52] 32768 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x128xf32, #shared1>, i32, i32, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
        %87 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> tensor<64x128xf32, #blocked3>
        %88 = arith.truncf %87 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #blocked3> to tensor<64x128xbf16, #blocked3>
        ttg.local_store %88, %2 {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked3> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
        ttg.local_store %86, %3 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %89 = ttg.memdesc_trans %3 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared3, #smem, mutable>
        %90 = ttng.tc_gen5_mma %89, %2, %result_0[%arg18], %arg16, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x64xbf16, #shared3, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %78, %90 : i1, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>}
      %54 = arith.muli %36, %arg12 {async_task_id = array<i32: 2>} : i32
      %55 = arith.addi %54, %46 {async_task_id = array<i32: 2>} : i32
      %result_2, %token_3 = ttng.tmem_load %result[%53#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %56 = ttg.convert_layout %result_2 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> -> tensor<64x128xf32, #blocked3>
      ttg.local_store %56, %4 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #blocked3> -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
      %57 = ttng.async_tma_copy_local_to_global %31[%55, %47] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xf32, #shared1>, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %57   {async_task_id = array<i32: 2>} : !ttg.async.token
      %58 = arith.muli %49, %arg12 {async_task_id = array<i32: 2>} : i32
      %59 = arith.addi %58, %46 {async_task_id = array<i32: 2>} : i32
      %result_4, %token_5 = ttng.tmem_load %result_0[%53#2] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %60 = ttg.convert_layout %result_4 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> -> tensor<64x128xf32, #blocked3>
      ttg.local_store %60, %5 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #blocked3> -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
      %61 = ttng.async_tma_copy_local_to_global %31[%59, %47] %5 {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xf32, #shared1>, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %61   {async_task_id = array<i32: 2>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

