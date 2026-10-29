// RUN: triton-opt %s --split-input-file --nvgpu-test-ws-memory-planner="num-buffers=7 smem-alloc-algo=1 smem-budget=232448" | FileCheck %s

// 2-CTA persistent BF16 GEMM (decompose-K partial-BMM shape, 128x256x64, num_stages=6) whose epilogue splits the
// 128x256 accumulator into two 128x128 subtiles via tmem_load -> reshape -> trans -> convert_layout -> split and
// stages each subtile for a TMA store. The convert_layout on that chain needs 32 KB of scratch, but
// TMemSplitLoadPattern later rewrites the chain into two per-subtile TMEM loads whose conversions fold into the
// local_stores, so the conversion never exists at allocation time. Reserving its scratch costs the operand rings
// a copy (4 instead of 5, 197 KB used of 232 KB).
//
// @dk_partial: the subtiles only feed local_store, so no scratch is reserved and the rings get 5 copies.
// @dk_partial_subtile_math: a subtile feeds arithmetic before its local_store, so the per-subtile conversion may
// survive and the 32 KB reservation is kept. (Its staging buffers are not fused, so the rings get 2 copies; with
// reserve-auxiliary-smem=false they would get 3.)

// CHECK-LABEL: @dk_partial(
// CHECK: ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK: ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x128xbf16

// @dk_partial_subtile4 (second split-input-file section): EPILOGUE_SUBTILE=4, i.e. a split whose results are split
// again. TMemSplitLoadPattern rewrites both levels, so the subtiling conversion on the outer split (16 KB) is
// skipped too and the rings get 6 copies instead of 5.

// CHECK-LABEL: @dk_partial_subtile_math(
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x128xbf16

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @dk_partial(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 16 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c290_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 290 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
    %c18560_i32 = arith.constant {async_task_id = array<i32: 3>} 18560 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c37120_i64 = arith.constant {async_task_id = array<i32: 3>} 37120 : i64
    %c37120_i32 = arith.constant {async_task_id = array<i32: 3>} 37120 : i32
    %c1024_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 1024 : i32
    %c2048_i32 = arith.constant {async_task_id = array<i32: 2>} 2048 : i32
    %c296_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i32
    %c296_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c2_i32 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
    %4 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %5 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c37120_i32], [%c37120_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %6 = tt.make_tensor_descriptor %arg1, [%c37120_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared1>
    %7 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %8 = nvg.cluster_id {async_task_id = array<i32: 3>}
    %9 = arith.remsi %8, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %10 = arith.muli %9, %c128_i32 {async_task_id = array<i32: 3>} : i32
    scf.for %arg3 = %7 to %c32_i32 step %c32_i32  : i32 {
      %11 = arith.divsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %12 = arith.remsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %13 = arith.divsi %12, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %14 = arith.muli %13, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %15 = arith.subi %c8_i32, %14 {async_task_id = array<i32: 2, 3>} : i32
      %16 = arith.minsi %15, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %17 = arith.remsi %12, %16 {async_task_id = array<i32: 2, 3>} : i32
      %18 = arith.addi %14, %17 {async_task_id = array<i32: 2, 3>} : i32
      %19 = arith.remsi %12, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %20 = arith.divsi %19, %16 {async_task_id = array<i32: 2, 3>} : i32
      %21 = arith.muli %18, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %22 = arith.muli %20, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %23 = arith.muli %11, %c18560_i32 {async_task_id = array<i32: 3>} : i32
      %24 = arith.addi %22, %10 {async_task_id = array<i32: 3>} : i32
      %25:2 = scf.for %arg4 = %c0_i32 to %c290_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %token) -> (i1, !ttg.async.token)  : i32 {
        %34 = arith.muli %arg4, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        %35 = arith.addi %23, %34 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %5[%21, %35] 16384 %2 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_load} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %6[%35, %24] 16384 %3 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_b, two_cta_load} : !tt.tensordesc<64x128xbf16, #shared1>, i32, i32, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
        %36 = ttng.tc_gen5_mma %2, %3, %result[%arg6], %arg5, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 5 : i32, tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %36 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 5 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%25#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %26 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x256xf32, #linear> -> tensor<128x2x128xf32, #linear1>
      %27 = tt.trans %26 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear1> -> tensor<128x128x2xf32, #linear2>
      %28 = ttg.convert_layout %27 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear2> -> tensor<128x128x2xf32, #blocked>
      %outLHS, %outRHS = tt.split %28 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #blocked> -> tensor<128x128xf32, #blocked1>
      %29 = arith.muli %11, %c1024_i32 {async_task_id = array<i32: 2>} : i32
      %30 = arith.addi %29, %21 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outLHS, %0 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %31 = ttng.async_tma_copy_local_to_global %4[%30, %22] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %31   {async_task_id = array<i32: 2>} : !ttg.async.token
      %32 = arith.addi %22, %c128_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outRHS, %1 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %33 = ttng.async_tma_copy_local_to_global %4[%30, %32] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %33   {async_task_id = array<i32: 2>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func public @dk_partial_subtile_math(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 16 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c290_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 290 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
    %c18560_i32 = arith.constant {async_task_id = array<i32: 3>} 18560 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c37120_i64 = arith.constant {async_task_id = array<i32: 3>} 37120 : i64
    %c37120_i32 = arith.constant {async_task_id = array<i32: 3>} 37120 : i32
    %c1024_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 1024 : i32
    %c2048_i32 = arith.constant {async_task_id = array<i32: 2>} 2048 : i32
    %c296_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i32
    %c296_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c2_i32 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
    %4 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %5 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c37120_i32], [%c37120_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %6 = tt.make_tensor_descriptor %arg1, [%c37120_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared1>
    %7 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %8 = nvg.cluster_id {async_task_id = array<i32: 3>}
    %9 = arith.remsi %8, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %10 = arith.muli %9, %c128_i32 {async_task_id = array<i32: 3>} : i32
    scf.for %arg3 = %7 to %c32_i32 step %c32_i32  : i32 {
      %11 = arith.divsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %12 = arith.remsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %13 = arith.divsi %12, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %14 = arith.muli %13, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %15 = arith.subi %c8_i32, %14 {async_task_id = array<i32: 2, 3>} : i32
      %16 = arith.minsi %15, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %17 = arith.remsi %12, %16 {async_task_id = array<i32: 2, 3>} : i32
      %18 = arith.addi %14, %17 {async_task_id = array<i32: 2, 3>} : i32
      %19 = arith.remsi %12, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %20 = arith.divsi %19, %16 {async_task_id = array<i32: 2, 3>} : i32
      %21 = arith.muli %18, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %22 = arith.muli %20, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %23 = arith.muli %11, %c18560_i32 {async_task_id = array<i32: 3>} : i32
      %24 = arith.addi %22, %10 {async_task_id = array<i32: 3>} : i32
      %25:2 = scf.for %arg4 = %c0_i32 to %c290_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %token) -> (i1, !ttg.async.token)  : i32 {
        %34 = arith.muli %arg4, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        %35 = arith.addi %23, %34 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %5[%21, %35] 16384 %2 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_load} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %6[%35, %24] 16384 %3 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_b, two_cta_load} : !tt.tensordesc<64x128xbf16, #shared1>, i32, i32, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
        %36 = ttng.tc_gen5_mma %2, %3, %result[%arg6], %arg5, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 5 : i32, tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %36 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 5 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%25#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %26 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x256xf32, #linear> -> tensor<128x2x128xf32, #linear1>
      %27 = tt.trans %26 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear1> -> tensor<128x128x2xf32, #linear2>
      %28 = ttg.convert_layout %27 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear2> -> tensor<128x128x2xf32, #blocked>
      %outLHS, %outRHS = tt.split %28 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #blocked> -> tensor<128x128xf32, #blocked1>
      %29 = arith.muli %11, %c1024_i32 {async_task_id = array<i32: 2>} : i32
      %30 = arith.addi %29, %21 {async_task_id = array<i32: 2>} : i32
      %lhs_scaled = arith.addf %outLHS, %outLHS {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1>
      ttg.local_store %lhs_scaled, %0 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %31 = ttng.async_tma_copy_local_to_global %4[%30, %22] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %31   {async_task_id = array<i32: 2>} : !ttg.async.token
      %32 = arith.addi %22, %c128_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outRHS, %1 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %33 = ttng.async_tma_copy_local_to_global %4[%30, %32] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %33   {async_task_id = array<i32: 2>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// CHECK-LABEL: @dk_partial_subtile4(
// CHECK: ttg.local_alloc {buffer.copy = 6 : i32, buffer.id = 4 : i32{{.*}}} : () -> !ttg.memdesc<128x64xbf16
// CHECK: ttg.local_alloc {buffer.copy = 6 : i32, buffer.id = 5 : i32{{.*}}} : () -> !ttg.memdesc<64x128xbf16

#blocked = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0], [64, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @dk_partial_subtile4(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c192_i32 = arith.constant {async_task_id = array<i32: 2>} 192 : i32
    %c16_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 16 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c290_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 290 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 64 : i32
    %c18560_i32 = arith.constant {async_task_id = array<i32: 3>} 18560 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c37120_i64 = arith.constant {async_task_id = array<i32: 3>} 37120 : i64
    %c37120_i32 = arith.constant {async_task_id = array<i32: 3>} 37120 : i32
    %c1024_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 1024 : i32
    %c2048_i32 = arith.constant {async_task_id = array<i32: 2>} 2048 : i32
    %c296_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i32
    %c296_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 296 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c2_i32 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
    %6 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared>
    %7 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c37120_i32], [%c37120_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %8 = tt.make_tensor_descriptor %arg1, [%c37120_i32, %c296_i32], [%c296_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared1>
    %9 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %10 = nvg.cluster_id {async_task_id = array<i32: 3>}
    %11 = arith.remsi %10, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %12 = arith.muli %11, %c128_i32 {async_task_id = array<i32: 3>} : i32
    scf.for %arg3 = %9 to %c32_i32 step %c32_i32  : i32 {
      %13 = arith.divsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %14 = arith.remsi %arg3, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %15 = arith.divsi %14, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %16 = arith.muli %15, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %17 = arith.subi %c8_i32, %16 {async_task_id = array<i32: 2, 3>} : i32
      %18 = arith.minsi %17, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %19 = arith.remsi %14, %18 {async_task_id = array<i32: 2, 3>} : i32
      %20 = arith.addi %16, %19 {async_task_id = array<i32: 2, 3>} : i32
      %21 = arith.remsi %14, %c16_i32 {async_task_id = array<i32: 2, 3>} : i32
      %22 = arith.divsi %21, %18 {async_task_id = array<i32: 2, 3>} : i32
      %23 = arith.muli %20, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %24 = arith.muli %22, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %25 = arith.muli %13, %c18560_i32 {async_task_id = array<i32: 3>} : i32
      %26 = arith.addi %24, %12 {async_task_id = array<i32: 3>} : i32
      %27:2 = scf.for %arg4 = %c0_i32 to %c290_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %token) -> (i1, !ttg.async.token)  : i32 {
        %44 = arith.muli %arg4, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        %45 = arith.addi %25, %44 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %7[%23, %45] 16384 %4 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_load} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %8[%45, %26] 16384 %5 {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 0 : i32, multicast = false, two_cta_b, two_cta_load} : !tt.tensordesc<64x128xbf16, #shared1>, i32, i32, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
        %46 = ttng.tc_gen5_mma %4, %5, %result[%arg6], %arg5, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 5 : i32, tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %46 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 5 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%27#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %28 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x256xf32, #linear> -> tensor<128x2x128xf32, #linear1>
      %29 = tt.trans %28 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear1> -> tensor<128x128x2xf32, #linear2>
      %30 = ttg.convert_layout %29 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear2> -> tensor<128x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %30 {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear3> -> tensor<128x128xf32, #linear4>
      %31 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear4> -> tensor<128x2x64xf32, #blocked>
      %32 = tt.trans %31 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked1>
      %outLHS_2, %outRHS_3 = tt.split %32 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked1> -> tensor<128x64xf32, #blocked2>
      %33 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear4> -> tensor<128x2x64xf32, #blocked>
      %34 = tt.trans %33 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked1>
      %outLHS_4, %outRHS_5 = tt.split %34 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked1> -> tensor<128x64xf32, #blocked2>
      %35 = arith.muli %13, %c1024_i32 {async_task_id = array<i32: 2>} : i32
      %36 = arith.addi %35, %23 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outLHS_2, %0 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %37 = ttng.async_tma_copy_local_to_global %6[%36, %24] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %37   {async_task_id = array<i32: 2>} : !ttg.async.token
      %38 = arith.addi %24, %c64_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outRHS_3, %1 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %39 = ttng.async_tma_copy_local_to_global %6[%36, %38] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %39   {async_task_id = array<i32: 2>} : !ttg.async.token
      %40 = arith.addi %24, %c128_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outLHS_4, %2 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %41 = ttng.async_tma_copy_local_to_global %6[%36, %40] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %41   {async_task_id = array<i32: 2>} : !ttg.async.token
      %42 = arith.addi %24, %c192_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outRHS_5, %3 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %43 = ttng.async_tma_copy_local_to_global %6[%36, %42] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %43   {async_task_id = array<i32: 2>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
