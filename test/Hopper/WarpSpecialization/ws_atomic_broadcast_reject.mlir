// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" | FileCheck %s

// Case 3 (graceful reject): the persistent scf.while's tile counter is claimed
// by a NON-scalar (scatter) tt.atomic_rmw replicated across partitions. AutoWS
// cannot broadcast a data-dependent scatter, so it must bail out of warp
// specialization and leave a plain, compilable (non-WS) kernel: no
// ttg.warp_specialize op and no leftover partition ids / ttg.partitions.

// CHECK-LABEL: @reject_scatter_atomic
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: ttg.partition
// CHECK-NOT: ttg.partition
// The unsupported atomic itself is preserved (kernel still compiles).
// CHECK: tt.atomic_rmw

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#b1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_scatter_atomic(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg5: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg15: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg16, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg17, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg18, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    %9 = scf.while (%arg19 = %0) : (i32) -> i32 {
      %10 = arith.cmpi slt, %arg19, %7 : i32
      scf.condition(%10) %arg19 : i32
    } do {
    ^bb0(%arg19: i32):
      %10 = arith.divsi %arg19, %8 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %2, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      %14 = arith.remsi %arg19, %13 : i32
      %15 = arith.addi %11, %14 : i32
      %16 = arith.remsi %arg19, %8 : i32
      %17 = arith.divsi %16, %13 : i32
      %18 = arith.muli %15, %c128_i32 : i32
      %19 = arith.muli %17, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %20 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %21:2 = scf.for %arg20 = %c0_i32 to %6 step %c1_i32 iter_args(%arg21 = %false, %arg22 = %20) -> (i1, !ttg.async.token)  : i32 {
        %27 = arith.muli %arg20, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %28 = tt.descriptor_load %arg0[%18, %27] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %29 = ttg.local_alloc %28 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %30 = tt.descriptor_load %arg5[%19, %27] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %31 = ttg.local_alloc %30 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %32 = ttg.memdesc_trans %31 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %33 = ttng.tc_gen5_mma %29, %32, %result[%arg22], %arg21, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %33 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%21#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %22 = arith.truncf %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %23 = ttg.convert_layout %22 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      %24 = ttg.local_alloc %23 {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %25 = ttng.async_tma_copy_local_to_global %arg10[%18, %19] %24 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %25   {ttg.partition = array<i32: 0>} : !ttg.async.token
      // Scatter atomic: non-scalar (tensor of pointers) -> unsupported -> reject.
      %pt = tt.splat %arg15 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #b1>
      %vt = tt.splat %c1_i32 : i32 -> tensor<1xi32, #b1>
      %mt = tt.splat %true : i1 -> tensor<1xi1, #b1>
      %scatter = tt.atomic_rmw add, acq_rel, gpu, %pt, %vt, %mt : (tensor<1x!tt.ptr<i32>, #b1>, tensor<1xi32, #b1>, tensor<1xi1, #b1>) -> tensor<1xi32, #b1>
      %next_tile = tt.unsplat %scatter : tensor<1xi32, #b1>
      scf.yield %next_tile : i32
    }
    tt.return
  }
}
