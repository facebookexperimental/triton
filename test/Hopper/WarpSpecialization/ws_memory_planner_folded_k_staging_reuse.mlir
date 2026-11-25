// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=150000 reserve-auxiliary-smem=0 reject-over-device-limit=1" 2>&1 | FileCheck %s --check-prefix=OVER

// Persistent 2CTA fp32 GEMM (BLOCK 128x256x128, EPILOGUE_SUBTILE=4) at K == BLOCK_K:
// the K loop folded away, so the tile loop has no inner loop and is
// software-pipelined across tiles (operand loads at stage 0, epilogue at stage 1).
// Its operands take 64 KB + 128 KB and the fused TMA store staging 32 KB, more
// than the budget left after the auxiliary reservation. The staging must reuse
// the A operand's SMEM, as it does when the tile loop has an inner K loop, and
// the tile loop then runs at one stage. Without that, the plan exceeds the
// device limit and fails at launch (237736 > 232448 B on B200).

// CHECK-LABEL: @triton_blackwell_ws_persistent_tma
// CHECK-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 0 : i32{{.*}}128x64xf32
// CHECK-NOT: loop.stage = 1

// Buffers alone above the device limit are recorded on the module; the NVIDIA
// backend raises OutOfResources for it at compile time.
// OVER: warning: warp-specialized buffers need 196608 bytes of shared memory, more than the 150000 bytes available
// OVER: module attributes {{.*}}ttg.ws_shared_memory_overflow = 196608 : i32

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 4, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 1, 2], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @triton_blackwell_ws_persistent_tma(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i64
    %c8192_i64 = arith.constant {async_task_id = array<i32: 0, 1>} 8192 : i64
    %c8192_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 8192 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 128 : i32
    %c128_i64 = arith.constant {async_task_id = array<i32: 1>} 128 : i64
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 8 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 256 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 64 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 0>} 192 : i32
    %c2048_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 2048 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    %2 = tt.make_tensor_descriptor %arg2, [%c8192_i32, %c8192_i32], [%c8192_i64, %c1_i64] {async_task_id = array<i32: 0>} : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared>
    %3 = tt.get_program_id x {async_task_id = array<i32: 0, 1>} : i32
    %4 = tt.make_tensor_descriptor %arg0, [%c8192_i32, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %5 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c8192_i32], [%c8192_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<f32>, !tt.tensordesc<128x256xf32, #shared>
    %6 = arith.subi %3, %c148_i32 {async_task_id = array<i32: 0>} : i32
    %7 = scf.for %arg4 = %3 to %c2048_i32 step %c148_i32 iter_args(%arg5 = %6) -> (i32)  : i32 {
      %8 = arith.divsi %arg4, %c256_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %9 = arith.muli %8, %c8_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %10 = arith.subi %c64_i32, %9 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %11 = arith.minsi %10, %c8_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %12 = arith.remsi %arg4, %11 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %13 = arith.addi %9, %12 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %14 = arith.remsi %arg4, %c256_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %15 = arith.divsi %14, %11 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %16 = arith.muli %13, %c128_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %17 = arith.muli %15, %c256_i32 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      nvws.descriptor_load %4[%16, %c0_i32] 65536 %0 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x128xf32, #shared>, i32, i32, !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %18 = ttg.local_load %0 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked1>
      nvws.descriptor_load %5[%c0_i32, %17] 131072 %1 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x256xf32, #shared>, i32, i32, !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
      %19 = ttg.local_load %1 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x256xf32, #shared, #smem, mutable> -> tensor<128x256xf32, #blocked2>
      %20 = ttg.convert_layout %18 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %21 = ttg.convert_layout %19 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x256xf32, #blocked2> -> tensor<128x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %22 = tt.dot %20, %21, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, two_ctas} : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
      %23 = arith.addi %arg5, %c148_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %24 = arith.divsi %23, %c256_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %25 = arith.muli %24, %c8_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %26 = arith.subi %c64_i32, %25 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %27 = arith.minsi %26, %c8_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %28 = arith.remsi %23, %27 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %29 = arith.addi %25, %28 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %30 = arith.remsi %23, %c256_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %31 = arith.divsi %30, %27 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %32 = arith.muli %29, %c128_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %33 = arith.muli %31, %c256_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %34 = tt.reshape %22 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked3>
      %35 = tt.trans %34 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3> -> tensor<128x128x2xf32, #blocked4>
      %36 = ttg.convert_layout %35 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128x2xf32, #blocked4> -> tensor<128x128x2xf32, #linear>
      %outLHS, %outRHS = tt.split %36 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128x2xf32, #linear> -> tensor<128x128xf32, #linear1>
      %37 = tt.reshape %outLHS {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #blocked5>
      %38 = tt.trans %37 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %outLHS_0, %outRHS_1 = tt.split %38 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      %39 = tt.reshape %outRHS {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #blocked5>
      %40 = tt.trans %39 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %outLHS_2, %outRHS_3 = tt.split %40 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      %41 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      ttg.local_store %outLHS_0, %41 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked7> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %42 = ttng.async_tma_copy_local_to_global %2[%32, %33] %41 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %42   {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.async.token
      %43 = arith.addi %33, %c64_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %44 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      ttg.local_store %outRHS_1, %44 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked7> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %45 = ttng.async_tma_copy_local_to_global %2[%32, %43] %44 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %45   {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.async.token
      %46 = arith.addi %33, %c128_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %47 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      ttg.local_store %outLHS_2, %47 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked7> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %48 = ttng.async_tma_copy_local_to_global %2[%32, %46] %47 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %48   {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.async.token
      %49 = arith.addi %33, %c192_i32 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %50 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      ttg.local_store %outRHS_3, %50 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked7> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
      %51 = ttng.async_tma_copy_local_to_global %2[%32, %49] %50 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xf32, #shared>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %51   {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 0>} %23 : i32
    } {async_task_id = array<i32: 0, 1>, tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

