// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" 2>/dev/null | FileCheck %s

// Case 1 keeps the zero tmem_store (as for a non-zero accumulator init); case 2
// is the IR the pipeline produces for a zero init (use_acc = false, no store).
// A CLC persistent GEMM whose K loop has a single iteration (K == BLOCK_K) and
// was folded away, so the MMA sits directly in the warp-specialized scf.while
// tile loop. Warp specialization used to fail here: handleOperandD required an
// enclosing scf.for, the TMEM planner only allocated TMEM inside innermost
// scf.for loops (the accumulator got no buffer.id), and getLastOpInBlock
// asserted on a non-scf.for parent.

// CHECK-LABEL: @clc_single_k_tile
// CHECK: ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = [[ACC:[0-9]+]] : i32}
// CHECK: ttg.warp_specialize
// Epilogue (default partition): wait for the MMA, read the accumulator, then
// release it back to the gemm partition (task 4).
// CHECK: default {
// CHECK: scf.while
// CHECK: ttng.wait_barrier {{.*}}direction = "forward", dstTask = 4 : i32, guardedBufferIds = array<i32: [[ACC]]>
// CHECK-NEXT: ttng.tmem_load
// CHECK-NEXT: ttg.memdesc_index
// CHECK-NEXT: ttng.arrive_barrier {{.*}}dstTask = 4 : i32, guardedBufferIds = array<i32: [[ACC]]>
// Gemm partition: wait (inverted phase) until the previous tile's accumulator
// was read before zeroing it and issuing the MMA.
// CHECK: ttng.clc_try_cancel_async
// CHECK: arith.xori
// CHECK: ttng.wait_barrier {{.*}}dstTask = 0 : i32, guardedBufferIds = array<i32: [[ACC]]>
// CHECK-NEXT: ttng.tmem_store
// CHECK: ttng.tc_gen5_mma {{.*}}is_async

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @clc_single_k_tile(%arg0: !tt.tensordesc<128x128xbf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x128xbf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf32, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.addi %arg15, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg16, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.muli %3, %c8_i32 : i32
    %5 = tt.get_program_id x : i32
    %6 = scf.while (%arg17 = %true, %arg18 = %5) : (i1, i32) -> i32 {
      scf.condition(%arg17) %arg18 : i32
    } do {
    ^bb0(%arg17: i32):
      %7 = ttng.clc_try_cancel_async : !ttg.async.token
      %8 = arith.divsi %arg17, %4 {ttg.partition = array<i32: 1, 3>} : i32
      %9 = arith.muli %8, %c8_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %10 = arith.subi %1, %9 {ttg.partition = array<i32: 1, 3>} : i32
      %11 = arith.minsi %10, %c8_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %12 = arith.remsi %arg17, %11 {ttg.partition = array<i32: 1, 3>} : i32
      %13 = arith.addi %9, %12 {ttg.partition = array<i32: 1, 3>} : i32
      %14 = arith.remsi %arg17, %4 {ttg.partition = array<i32: 1, 3>} : i32
      %15 = arith.divsi %14, %11 {ttg.partition = array<i32: 1, 3>} : i32
      %16 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %18 = tt.descriptor_load %arg0[%16, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %19 = ttg.local_alloc %18 {ttg.partition = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %20 = tt.descriptor_load %arg5[%17, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %21 = ttg.local_alloc %20 {ttg.partition = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %22 = ttg.memdesc_trans %21 {order = array<i32: 1, 0>, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %23 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %24 = ttng.tc_gen5_mma %19, %22, %result[%23], %true, %true {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_0, %token_1 = ttng.tmem_load %result[%24] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %25 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      %26 = ttg.local_alloc %25 {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %27 = ttng.async_tma_copy_local_to_global %arg10[%16, %17] %26 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %27   {ttg.partition = array<i32: 1>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %7 : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %isValid, %x : i1, i32
    } attributes {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.partition.types = ["computation", "epilogue", "epilogue_store", "load", "gemm"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// CHECK-LABEL: @clc_single_k_tile_use_acc_false
// CHECK: ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = [[ACC:[0-9]+]] : i32}
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: scf.while
// CHECK: ttng.wait_barrier {{.*}}direction = "forward", dstTask = 4 : i32, guardedBufferIds = array<i32: [[ACC]]>
// CHECK-NEXT: ttng.tmem_load
// CHECK-NEXT: ttg.memdesc_index
// CHECK-NEXT: ttng.arrive_barrier {{.*}}dstTask = 4 : i32, guardedBufferIds = array<i32: [[ACC]]>
// The gemm partition waits (inverted phase) for the epilogue's release
// before overwriting the accumulator with use_acc = false.
// CHECK: ttng.clc_try_cancel_async
// CHECK: arith.xori
// CHECK: ttng.wait_barrier {{.*}}dstTask = 0 : i32, guardedBufferIds = array<i32: [[ACC]]>
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, %false, %{{.*}}is_async

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @clc_single_k_tile_use_acc_false(%arg0: !tt.tensordesc<128x128xbf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x128xbf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf32, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.addi %arg15, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg16, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.muli %3, %c8_i32 : i32
    %5 = tt.get_program_id x : i32
    %6 = scf.while (%arg17 = %true, %arg18 = %5) : (i1, i32) -> i32 {
      scf.condition(%arg17) %arg18 : i32
    } do {
    ^bb0(%arg17: i32):
      %7 = ttng.clc_try_cancel_async : !ttg.async.token
      %8 = arith.divsi %arg17, %4 {ttg.partition = array<i32: 1, 3>} : i32
      %9 = arith.muli %8, %c8_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %10 = arith.subi %1, %9 {ttg.partition = array<i32: 1, 3>} : i32
      %11 = arith.minsi %10, %c8_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %12 = arith.remsi %arg17, %11 {ttg.partition = array<i32: 1, 3>} : i32
      %13 = arith.addi %9, %12 {ttg.partition = array<i32: 1, 3>} : i32
      %14 = arith.remsi %arg17, %4 {ttg.partition = array<i32: 1, 3>} : i32
      %15 = arith.divsi %14, %11 {ttg.partition = array<i32: 1, 3>} : i32
      %16 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1, 3>} : i32
      %18 = tt.descriptor_load %arg0[%16, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %19 = ttg.local_alloc %18 {ttg.partition = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %20 = tt.descriptor_load %arg5[%17, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
      %21 = ttg.local_alloc %20 {ttg.partition = array<i32: 3>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %22 = ttg.memdesc_trans %21 {order = array<i32: 1, 0>, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %23 = ttng.tc_gen5_mma %19, %22, %result[%token], %false, %true {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_0, %token_1 = ttng.tmem_load %result[%23] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %24 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      %25 = ttg.local_alloc %24 {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %26 = ttng.async_tma_copy_local_to_global %arg10[%16, %17] %25 {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %26   {ttg.partition = array<i32: 1>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %7 : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %isValid, %x : i1, i32
    } attributes {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.partition.types = ["computation", "epilogue", "epilogue_store", "load", "gemm"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
