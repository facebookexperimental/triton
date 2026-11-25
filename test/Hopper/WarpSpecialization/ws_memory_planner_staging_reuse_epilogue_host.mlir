// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s
// RUN: sed -e '/%%38\[%%57\] 256 %%10 /s/ %%10 / %%22 /' -e '/%%76 = ttg.local_load %%10 /s/ %%10 / %%22 /' -e '/%%38\[%%137\] 256 %%22 /s/ %%22 / %%10 /' -e '/%%145 = ttg.local_load %%22 /s/ %%22 / %%10 /' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=LATE
// RUN: sed -e '/%%38\[%%57\] 256 %%10 /s/ %%10 / %%22 /' -e '/%%76 = ttg.local_load %%10 /s/ %%10 / %%22 /' -e '/%%38\[%%137\] 256 %%22 /s/ %%22 / %%10 /' -e '/%%145 = ttg.local_load %%22 /s/ %%22 / %%10 /' -e 's/^      %%145 = ttg.local_load %%10 \(.*\)$/      %%cst_adv = arith.constant {async_task_id = array<i32: 0>} dense<0.0> : tensor<64xf32, #blocked3>\n      %%145 = scf.for %%adv_i = %%c0_i32 to %%c1_i32 step %%c1_i32 iter_args(%%adv_a = %%cst_adv) -> (tensor<64xf32, #blocked3>) : i32 {\n        %%adv_v = ttg.local_load %%10 \1\n        scf.yield {async_task_id = array<i32: 0>} %%adv_v : tensor<64xf32, #blocked3>\n      } {async_task_id = array<i32: 0>}/' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=INNER
// RUN: sed -e '/%%38\[%%57\] 256 %%10 /s/ %%10 / %%22 /' -e '/%%76 = ttg.local_load %%10 /s/ %%10 / %%22 /' -e '/%%38\[%%137\] 256 %%22 /s/ %%22 / %%10 /' -e '/%%145 = ttg.local_load %%22 /s/ %%22 / %%10 /' -e 's/^      %%76 = ttg.local_load %%22 {async_task_id = array<i32: 0>} \(.*\)$/      %%76 = scf.if %%true -> (tensor<64xf32, #blocked3>) {\n        %%if_r = ttg.local_load %%22 {async_task_id = array<i32: 1>} \1\n        scf.yield {async_task_id = array<i32: 0, 1>} %%if_r : tensor<64xf32, #blocked3>\n      } else {\n        %%if_z = arith.constant {async_task_id = array<i32: 0, 1>} dense<0.000000e+00> : tensor<64xf32, #blocked3>\n        scf.yield {async_task_id = array<i32: 0, 1>} %%if_z : tensor<64xf32, #blocked3>\n      } {async_task_id = array<i32: 0, 1>}/' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=XPART
// RUN: sed -e '/%%38\[%%57\] 256 %%10 /i\      scf.for %%jj = %%c0_i32 to %%c1_i32 step %%c1_i32 : i32 {\n        scf.for %%kk = %%c0_i32 to %%c1_i32 step %%c1_i32 : i32 {\n        } {async_task_id = array<i32: 0, 1, 2, 3>}' -e '/ttng.async_tma_store_token_wait %%160 /a\      } {async_task_id = array<i32: 0, 1, 2, 3>}' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=NESTED
// RUN: sed -e '/ttg.local_store %%91, %%0 /s/async_task_id = array<i32: 0>/async_task_id = array<i32: 0, 2>/' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=WTASKS
// RUN: sed -e '/ttg.local_store %%[0-9]*, %%[0-3] {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3>/s/array<i32: 0>/array<i32: 0, 2>/' -e '/%%76 = ttg.local_load %%10 /s/array<i32: 0>/array<i32: 0, 2>/' %s | triton-opt - -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s --check-prefix=WEQ2

// TMA output staging must not reuse a buffer that is still read after a staging
// write in the same loop body. Real planner input: a persistent CLC GEMM with
// separate_epilogue_store=True and EPILOGUE_SUBTILE=4 whose epilogue TMA-loads
// a [64] fp32 per-column vector for each subtile (buffers 10, 14, 18, 22, read
// in subtile order) and TMA-stores a [64] fp32 column partial per subtile
// (staging group 0) plus a [64] row result per tile (staging 4). The column
// partial staging used to land on buffer 14, the second subtile's vector: the
// first subtile's partial overwrote it before it was read (a silent wrong
// answer in the row result). It must land on buffer 10, read before any
// partial is staged; the row result staging, written after every read, may
// take buffer 14.

// CHECK-LABEL: @fused_epilogue_host
// CHECK-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// CHECK: ttg.local_alloc {allocation.reuseTarget = 14 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

// The variants below are sed edits of this kernel (RUN lines above).
// Variant LATE.
// Same kernel with the first and last subtiles' vector buffers swapped: buffer
// 10, the planner's first choice, is now read by the last subtile, after three
// partials are staged. The partial staging must take buffer 22 (read first);
// the row staging, written after every read, may take buffer 10.
// LATE-LABEL: @fused_epilogue_host
// LATE-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 22 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// LATE: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

// Variant INNER (on top of LATE).
// The late-read variant with that read (of buffer 10) wrapped in a one-trip
// loop placed after the staged partials: an inner loop counts at its position,
// so it is still read after a staging write. The partial staging must take
// buffer 22, not 10.
// INNER-LABEL: @fused_epilogue_host
// INNER-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 22 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

// Variant XPART (on top of LATE).
// The late-read variant with the first-read vector (buffer 22) read under an
// scf.if by a different partition (task 1) than the staging writer (task 0).
// Program order does not order partitions, so buffer 22 is rejected; no other
// vector buffer is read before the first staged partial, so the partial
// staging stays standalone. The row staging may still take buffer 10.
// XPART-LABEL: @fused_epilogue_host
// XPART-NOT: allocation.reuseTarget = 22
// XPART-COUNT-4: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// XPART: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

// Variant NESTED.
// Same kernel with the epilogue inside a loop nested in the warp-specialized
// loop (an epilogue subtile loop). The program-order argument holds for one
// pass only; a later pass's read follows an earlier pass's staging write with
// nothing ordering them, so the staging must not take any vector buffer.
// NESTED-LABEL: @fused_epilogue_host
// NESTED-NOT: allocation.reuseTarget
// NESTED: tt.return

// Variant WTASKS. The first subtile's column-partial staging write is
// replicated into partitions 0 and 2, while buffer 10 is read only in partition
// 0. Program order orders the read only before partition 0's copy of the write;
// partition 2's copy could overwrite buffer 10 first. The read and the write
// must run in one and the same partition, so buffer 10 is rejected and the
// partial staging stays standalone. The row staging, written in partition 0
// only, may then take buffer 10.
// WTASKS-LABEL: @fused_epilogue_host
// WTASKS-NOT: allocation.reuseTarget = 10
// WTASKS-COUNT-4: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// WTASKS: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

// Variant WEQ2. Buffer 10's read and every column-partial staging write are
// both replicated into partitions 0 and 2. Equal partition sets are not enough:
// partition 2's copy of a write is not ordered after partition 0's read, so
// buffer 10 is rejected again and the row staging may still take buffer 14.
// WEQ2-LABEL: @fused_epilogue_host
// WEQ2-NOT: allocation.reuseTarget = 10
// WEQ2-COUNT-4: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// WEQ2: ttg.local_alloc {allocation.reuseTarget = 14 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @fused_epilogue_host(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%57] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %76 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
      %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
      %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
      %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
      %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
      %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
      %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
      %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
      %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%137] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %145 = ttg.local_load %22 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
      %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
      %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Real planner input from a fused GEMM (2-CTA, EPILOGUE_SUBTILE=1, `for`
// loop): the column-partial staging (buffers 1, 2) matches the TMA-loaded
// vector (buffer 8), which the epilogue reads after storing the partials, in
// the same partition with no inner loop. Buffer 8 is rejected and both partial
// stagings stay standalone; the row staging still lands on buffer 5 (read
// before it is written).

// CHECK-LABEL: @partial_before_vector_read
// CHECK-NOT: allocation.reuseTarget = 8
// CHECK: ttg.local_alloc {allocation.reuseTarget = 5 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64x256xf32
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<256xf32
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<256xf32
// CHECK-NOT: allocation.reuseTarget = 8
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 128], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @partial_before_vector_read(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<18> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x256xf32, #blocked>
    %cst_1 = arith.constant {async_task_id = array<i32: 0>} dense<18> : tensor<64xi32, #blocked1>
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 256 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 148 : i32
    %c36_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 36 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c18_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 18 : i32
    %c2_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 2 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 63 : i32
    %cst_2 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst_3 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x256xf32, #linear>
    %cst_4 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %c128_i32 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared2, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<32x128xbf16, #shared3, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    %9 = arith.addi %arg14, %c63_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %10 = arith.divsi %9, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %11 = arith.addi %10, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %12 = arith.divsi %11, %c2_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %13 = arith.muli %12, %c2_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %14 = arith.addi %arg15, %c255_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %15 = arith.divsi %14, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %16 = arith.muli %12, %c36_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %17 = arith.muli %16, %15 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %18 = arith.addi %arg16, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %19 = arith.divsi %18, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %20 = arith.muli %arg16, %c18_i32 {async_task_id = array<i32: 3>} : i32
    %21 = arith.extsi %20 {async_task_id = array<i32: 3>} : i32 to i64
    %22 = tt.make_tensor_descriptor %arg0, [%arg14, %20], [%21, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared2>
    %23 = arith.extsi %arg15 {async_task_id = array<i32: 3>} : i32 to i64
    %24 = tt.make_tensor_descriptor %arg1, [%20, %arg15], [%23, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x128xbf16, #shared3>
    %25 = arith.muli %arg15, %c18_i32 {async_task_id = array<i32: 2, 3>} : i32
    %26 = arith.extsi %25 {async_task_id = array<i32: 2, 3>} : i32 to i64
    %27 = tt.make_tensor_descriptor %arg4, [%arg14, %25], [%26, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x256xf32, #shared>
    %28 = tt.make_tensor_descriptor %arg5, [%arg14, %25], [%26, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x256xf32, #shared>
    %29 = tt.make_tensor_descriptor %arg6, [%arg14, %25], [%26, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x256xf32, #shared>
    %30 = tt.make_tensor_descriptor %arg8, [%arg14, %25], [%26, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64x256xf32, #shared>
    %31 = tt.make_tensor_descriptor %arg7, [%25], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<256xf32, #shared1>
    %32 = arith.muli %16, %arg15 {async_task_id = array<i32: 2>} : i32
    %33 = tt.make_tensor_descriptor %arg11, [%32], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<256xf32, #shared1>
    %34 = tt.make_tensor_descriptor %arg12, [%32], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<256xf32, #shared1>
    %35 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %36 = arith.muli %13, %15 {async_task_id = array<i32: 0, 2, 3>} : i32
    %37 = arith.muli %15, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %38 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %39 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %40 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %41 = tt.splat %arg14 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %42 = tt.splat %arg14 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %43 = tt.splat %arg14 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #blocked1>
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %45 = arith.sitofp %arg15 {async_task_id = array<i32: 0>} : i32 to f32
    %46 = tt.splat %45 {async_task_id = array<i32: 0>} : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %47 = tt.splat %arg3 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #linear}>>
    %48 = tt.splat %arg9 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %49 = tt.splat %arg10 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %50 = nvg.cluster_id {async_task_id = array<i32: 3>}
    %51 = arith.remsi %50, %c2_i32 {async_task_id = array<i32: 3>} : i32
    %52 = arith.muli %51, %c128_i32 {async_task_id = array<i32: 3>} : i32
    scf.for %arg17 = %35 to %17 step %c148_i32  : i32 {
      %53 = arith.divsi %arg17, %36 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg17, %36 {async_task_id = array<i32: 0, 2, 3>} : i32
      %55 = arith.divsi %54, %37 {async_task_id = array<i32: 0, 2, 3>} : i32
      %56 = arith.muli %55, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.subi %13, %56 {async_task_id = array<i32: 0, 2, 3>} : i32
      %58 = arith.minsi %57, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %59 = arith.remsi %54, %58 {async_task_id = array<i32: 0, 2, 3>} : i32
      %60 = arith.addi %56, %59 {async_task_id = array<i32: 0, 2, 3>} : i32
      %61 = arith.remsi %54, %37 {async_task_id = array<i32: 0, 2, 3>} : i32
      %62 = arith.divsi %61, %58 {async_task_id = array<i32: 0, 2, 3>} : i32
      %63 = arith.muli %60, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %64 = arith.muli %62, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %65 = arith.muli %53, %arg16 {async_task_id = array<i32: 3>} : i32
      %66 = arith.addi %64, %52 {async_task_id = array<i32: 3>} : i32
      %67:2 = scf.for %arg18 = %c0_i32 to %19 step %c1_i32 iter_args(%arg19 = %false, %arg20 = %token) -> (i1, !ttg.async.token)  : i32 {
        %155 = arith.muli %arg18, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        %156 = arith.addi %65, %155 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %22[%63, %156] 4096 %3 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false, two_cta_load} : !tt.tensordesc<64x32xbf16, #shared2>, i32, i32, !ttg.memdesc<64x32xbf16, #shared2, #smem, mutable>
        nvws.descriptor_load %24[%156, %66] 8192 %4 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false, two_cta_b, two_cta_load} : !tt.tensordesc<32x128xbf16, #shared3>, i32, i32, !ttg.memdesc<32x128xbf16, #shared3, #smem, mutable>
        %157 = ttng.tc_gen5_mma %3, %4, %result[%arg20], %arg19, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<64x32xbf16, #shared2, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared3, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %157 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 3 : i32}
      %68 = tt.splat %63 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = tt.splat %63 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %70 = tt.splat %63 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #blocked1>
      %71 = arith.addi %68, %38 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %72 = arith.addi %69, %39 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %73 = arith.addi %70, %40 {async_task_id = array<i32: 0>} : tensor<64xi32, #blocked1>
      %74 = arith.cmpi slt, %71, %41 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %75 = arith.cmpi slt, %72, %42 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %76 = arith.cmpi slt, %73, %43 {async_task_id = array<i32: 0>} : tensor<64xi32, #blocked1>
      %77 = tt.splat %64 {async_task_id = array<i32: 0>} : i32 -> tensor<256xi32, #blocked1>
      %78 = arith.addi %77, %44 {async_task_id = array<i32: 0>} : tensor<256xi32, #blocked1>
      %79 = arith.muli %53, %arg15 {async_task_id = array<i32: 0, 2, 3>} : i32
      %80 = tt.addptr %arg2, %79 {async_task_id = array<i32: 0>} : !tt.ptr<f32>, i32
      %81 = tt.splat %80 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
      %82 = tt.addptr %81, %78 {async_task_id = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
      %result_5, %token_6 = ttng.tmem_load %result[%67#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %83 = tt.load %82 {async_task_id = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked1>
      %84 = arith.truncf %83 {async_task_id = array<i32: 0>} : tensor<256xf32, #blocked1> to tensor<256xbf16, #blocked1>
      %85 = arith.extf %84 {async_task_id = array<i32: 0>} : tensor<256xbf16, #blocked1> to tensor<256xf32, #blocked1>
      %86 = ttg.convert_layout %85 {async_task_id = array<i32: 0>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %87 = tt.expand_dims %86 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x256xf32, #linear>
      %88 = tt.broadcast %87 {async_task_id = array<i32: 0>} : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
      %89 = arith.addf %result_5, %88 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %90 = arith.truncf %89 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> to tensor<64x256xbf16, #linear>
      %91 = arith.extf %90 {async_task_id = array<i32: 0>} : tensor<64x256xbf16, #linear> to tensor<64x256xf32, #linear>
      %92 = "tt.reduce"(%91) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg18: f32, %arg19: f32):
        %155 = arith.addf %arg18, %arg19 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %155 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x256xf32, #linear>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %93 = ttg.convert_layout %91 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x256xf32, #blocked>
      %94 = arith.addf %92, %cst_4 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %95 = arith.divf %94, %46 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %96 = tt.expand_dims %95 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xf32, #linear>
      %97 = arith.muli %72, %cst {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %98 = arith.muli %73, %cst_1 {async_task_id = array<i32: 0>} : tensor<64xi32, #blocked1>
      %99 = tt.addptr %47, %97 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %100 = tt.splat %53 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %101 = tt.splat %53 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #blocked1>
      %102 = tt.addptr %99, %100 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %103 = tt.broadcast %96 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #linear> -> tensor<64x256xf32, #linear>
      %104 = arith.subf %91, %103 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %105 = arith.mulf %104, %104 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %106 = "tt.reduce"(%105) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg18: f32, %arg19: f32):
        %155 = arith.addf %arg18, %arg19 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %155 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x256xf32, #linear>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %107 = arith.addf %106, %cst_4 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %108 = arith.divf %107, %46 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %109 = tt.load %102, %75, %cst_4 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #linear}>>
      %110 = tt.expand_dims %109 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xf32, #linear>
      %111 = tt.broadcast %110 {async_task_id = array<i32: 0>} : tensor<64x1xf32, #linear> -> tensor<64x256xf32, #linear>
      %112 = arith.mulf %104, %111 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %113 = arith.addi %79, %64 {async_task_id = array<i32: 2, 3>} : i32
      nvws.descriptor_load %27[%63, %113] 65536 %5 {async_task_id = array<i32: 3>, multicast = false, two_cta_load} : !tt.tensordesc<64x256xf32, #shared>, i32, i32, !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
      %114 = ttg.local_load %5 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #shared, #smem, mutable> -> tensor<64x256xf32, #blocked>
      %115 = arith.mulf %93, %114 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #blocked>
      ttg.local_store %115, %0 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #blocked> -> !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
      %116 = ttng.async_tma_copy_local_to_global %30[%63, %113] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64x256xf32, #shared>, !ttg.memdesc<64x256xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %116   {async_task_id = array<i32: 2>} : !ttg.async.token
      nvws.descriptor_load %28[%63, %113] 65536 %6 {async_task_id = array<i32: 3>, multicast = false, two_cta_load} : !tt.tensordesc<64x256xf32, #shared>, i32, i32, !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
      %117 = ttg.local_load %6 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #shared, #smem, mutable> -> tensor<64x256xf32, #blocked>
      %118 = arith.mulf %117, %93 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #blocked>
      nvws.descriptor_load %29[%63, %113] 65536 %7 {async_task_id = array<i32: 3>, multicast = false, two_cta_load} : !tt.tensordesc<64x256xf32, #shared>, i32, i32, !ttg.memdesc<64x256xf32, #shared, #smem, mutable>
      %119 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #shared, #smem, mutable> -> tensor<64x256xf32, #blocked>
      %120 = arith.mulf %118, %119 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #blocked>
      %121 = ttg.convert_layout %120 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #blocked> -> tensor<64x256xf32, #linear>
      %122 = arith.mulf %121, %112 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %123 = arith.muli %60, %c18_i32 {async_task_id = array<i32: 2>} : i32
      %124 = arith.addi %123, %53 {async_task_id = array<i32: 2>} : i32
      %125 = arith.muli %124, %arg15 {async_task_id = array<i32: 2>} : i32
      %126 = arith.addi %125, %64 {async_task_id = array<i32: 2>} : i32
      %127 = tt.expand_dims %74 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %128 = tt.expand_dims %75 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi1, #linear>
      %129 = tt.broadcast %127 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %130 = arith.select %129, %120, %cst_0 {async_task_id = array<i32: 0>} : tensor<64x256xi1, #blocked>, tensor<64x256xf32, #blocked>
      %131 = tt.broadcast %128 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #linear> -> tensor<64x256xi1, #linear>
      %132 = "tt.reduce"(%130) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg18: f32, %arg19: f32):
        %155 = arith.addf %arg18, %arg19 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %155 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %133 = ttg.convert_layout %132 {async_task_id = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
      ttg.local_store %133, %1 {async_task_id = array<i32: 0>} : tensor<256xf32, #blocked1> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %134 = ttng.async_tma_copy_local_to_global %33[%126] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<256xf32, #shared1>, !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %134   {async_task_id = array<i32: 2>} : !ttg.async.token
      %135 = arith.select %131, %122, %cst_3 {async_task_id = array<i32: 0>} : tensor<64x256xi1, #linear>, tensor<64x256xf32, #linear>
      %136 = ttg.convert_layout %135 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x256xf32, #blocked>
      %137 = "tt.reduce"(%136) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg18: f32, %arg19: f32):
        %155 = arith.addf %arg18, %arg19 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %155 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %138 = ttg.convert_layout %137 {async_task_id = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
      ttg.local_store %138, %2 {async_task_id = array<i32: 0>} : tensor<256xf32, #blocked1> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %139 = ttng.async_tma_copy_local_to_global %34[%126] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<256xf32, #shared1>, !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %139   {async_task_id = array<i32: 2>} : !ttg.async.token
      nvws.descriptor_load %31[%113] 1024 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<256xf32, #shared1>, i32, !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %140 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #blocked1>
      %141 = ttg.convert_layout %140 {async_task_id = array<i32: 0>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %142 = tt.expand_dims %141 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x256xf32, #linear>
      %143 = tt.broadcast %142 {async_task_id = array<i32: 0>} : tensor<1x256xf32, #linear> -> tensor<64x256xf32, #linear>
      %144 = arith.mulf %121, %143 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %145 = arith.mulf %112, %144 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear>
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x256xf32, #blocked>
      %147 = "tt.reduce"(%146) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg18: f32, %arg19: f32):
        %155 = arith.addf %arg18, %arg19 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %155 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x256xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %148 = arith.addf %147, %cst_2 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %149 = ttg.convert_layout %148 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
      %150 = tt.addptr %48, %98 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %151 = tt.addptr %150, %101 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %152 = ttg.convert_layout %108 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64xf32, #blocked1>
      tt.store %151, %152, %76 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>
      %153 = tt.addptr %49, %98 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %154 = tt.addptr %153, %101 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      tt.store %154, %149, %76 {async_task_id = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked1>
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
