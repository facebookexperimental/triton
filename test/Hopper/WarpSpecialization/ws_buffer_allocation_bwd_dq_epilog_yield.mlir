// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Regression test for the reorderEpilogOps terminator bug (bug #13 in
// .llms/rules/partition-scheduler-bugs.md).
//
// reorderEpilogOps (WSCodePartition.cpp, doBufferAllocation Step 2) collects
// "epilog blocks" as any block containing a tt.store / descriptor_store. The
// HSTU self-attn backward warp-specializes a loop body that BOTH carries values
// (an scf.yield with the loop-carried MMA accumulator tokens) AND contains an
// in-loop dq read-modify-write (tt.load + tt.dot + tt.store to global dq).
//
// The old code inserted the block terminator into epilogOps and then, while
// "streamlining" the channel forward slices, ran
// depOp->moveAfter(lastInBlockOperand(depOp)) on the scf.yield itself. That
// relocated the yield to just after its last operand-producer (the dq
// tmem_load), stranding the dq epilogue (mulf -> load -> add -> truncf ->
// store) AFTER the terminator. Ops after a terminator is invalid IR, which
// later crashed doMemoryPlanner in hasLoopCarriedAccToken via
// forOp.getBody()->getTerminator() (Block::getTerminator assert
// `mightHaveTerminator()`).
//
// Fix: never treat a block terminator as a reorderable epilog op. The in-loop
// dq tt.store must stay BEFORE the loop's value-carrying scf.yield.

// CHECK-LABEL: @_hstu_attn_bwd
// The in-loop dq read-modify-write store stays before the loop's scf.yield.
// CHECK: tt.store %{{.*}} evictionPolicy = evict_last
// CHECK-NEXT: scf.yield {{.*}}!ttg.async.token
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0], [0, 32]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_bwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x128xf32, #linear>
    %cst_0 = arith.constant {async_task_id = array<i32: 3>} dense<1.000000e+00> : tensor<64x64xf32, #linear1>
    %cst_1 = arith.constant {async_task_id = array<i32: 3>} dense<0.000000e+00> : tensor<64x64xf32, #linear1>
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %cst_2 = arith.constant {async_task_id = array<i32: 3>} dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c768_i32 = arith.constant {async_task_id = array<i32: 0, 2>} 768 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 2>} 256 : i32
    %c384_i32 = arith.constant {async_task_id = array<i32: 2>} 384 : i32
    %c512_i32 = arith.constant {async_task_id = array<i32: 0>} 512 : i32
    %c640_i32 = arith.constant {async_task_id = array<i32: 0>} 640 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %cst_3 = arith.constant {async_task_id = array<i32: 3>} dense<0> : tensor<64x64xi32, #linear1>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %1 = arith.divsi %0, %arg29 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %2 = arith.remsi %0, %arg29 {async_task_id = array<i32: 0, 2, 3>} : i32
    %3 = arith.extsi %2 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %4 = tt.addptr %arg4, %1 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>, i32
    %5 = tt.load %4 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
    %6 = tt.addptr %4, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>, i32
    %7 = tt.load %6 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
    %8 = arith.subi %7, %5 {async_task_id = array<i32: 0, 1, 2, 3>} : i64
    %9 = arith.trunci %8 {async_task_id = array<i32: 0, 1, 2, 3>} : i64 to i32
    %10 = tt.addptr %arg5, %1 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>, i32
    %11 = tt.load %10 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
    %12 = tt.addptr %10, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>, i32
    %13 = tt.load %12 {async_task_id = array<i32: 0, 1, 2, 3>} : !tt.ptr<i64>
    %14 = arith.subi %13, %11 {async_task_id = array<i32: 0, 1, 2, 3>} : i64
    %15 = arith.trunci %14 {async_task_id = array<i32: 0, 1, 2, 3>} : i64 to i32
    %16 = arith.extsi %arg11 : i32 to i64
    %17 = arith.muli %11, %16 : i64
    %18 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i64
    %19 = arith.extsi %arg13 : i32 to i64
    %20 = arith.muli %5, %19 : i64
    %21 = tt.addptr %arg1, %20 : !tt.ptr<bf16>, i64
    %22 = arith.extsi %arg15 : i32 to i64
    %23 = arith.muli %5, %22 : i64
    %24 = tt.addptr %arg2, %23 : !tt.ptr<bf16>, i64
    %25 = arith.extsi %arg17 : i32 to i64
    %26 = arith.muli %11, %25 : i64
    %27 = tt.addptr %arg6, %26 : !tt.ptr<bf16>, i64
    %28 = arith.extsi %arg19 {async_task_id = array<i32: 3>} : i32 to i64
    %29 = arith.muli %11, %28 {async_task_id = array<i32: 3>} : i64
    %30 = arith.extsi %arg20 {async_task_id = array<i32: 3>} : i32 to i64
    %31 = arith.muli %3, %30 {async_task_id = array<i32: 3>} : i64
    %32 = arith.addi %29, %31 {async_task_id = array<i32: 3>} : i64
    %33 = tt.addptr %arg7, %32 {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, i64
    %34 = arith.extsi %arg21 : i32 to i64
    %35 = arith.muli %5, %34 : i64
    %36 = tt.addptr %arg8, %35 : !tt.ptr<bf16>, i64
    %37 = arith.extsi %arg23 : i32 to i64
    %38 = arith.muli %5, %37 : i64
    %39 = tt.addptr %arg9, %38 : !tt.ptr<bf16>, i64
    %40 = tt.get_program_id y {async_task_id = array<i32: 0, 2>} : i32
    %41 = tt.get_num_programs y {async_task_id = array<i32: 0, 2>} : i32
    %42 = arith.muli %0, %41 {async_task_id = array<i32: 0, 2>} : i32
    %43 = arith.addi %40, %42 {async_task_id = array<i32: 0, 2>} : i32
    %44 = arith.muli %43, %c768_i32 {async_task_id = array<i32: 0, 2>} : i32
    %45 = tt.addptr %arg3, %44 {async_task_id = array<i32: 0, 2>} : !tt.ptr<i8>, i32
    %46 = tt.addptr %45, %c128_i32 {async_task_id = array<i32: 2>} : !tt.ptr<i8>, i32
    %47 = tt.addptr %45, %c256_i32 {async_task_id = array<i32: 2>} : !tt.ptr<i8>, i32
    %48 = tt.addptr %45, %c384_i32 {async_task_id = array<i32: 2>} : !tt.ptr<i8>, i32
    %49 = tt.addptr %45, %c512_i32 {async_task_id = array<i32: 0>} : !tt.ptr<i8>, i32
    %50 = tt.addptr %45, %c640_i32 {async_task_id = array<i32: 0>} : !tt.ptr<i8>, i32
    %51 = arith.muli %arg29, %arg32 : i32
    %52 = arith.extsi %51 : i32 to i64
    %53 = arith.muli %52, %c2_i64 : i64
    ttng.tensormap_create %45, %18, [%c64_i32, %c64_i32], [%51, %15], [%53], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %45 {async_task_id = array<i32: 0, 2>} : !tt.ptr<i8>
    %54 = arith.muli %arg29, %arg33 : i32
    %55 = arith.extsi %54 : i32 to i64
    %56 = arith.muli %55, %c2_i64 : i64
    ttng.tensormap_create %48, %27, [%c64_i32, %c64_i32], [%54, %15], [%56], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %48 {async_task_id = array<i32: 2>} : !tt.ptr<i8>
    ttng.tensormap_create %46, %21, [%c64_i32, %c64_i32], [%51, %9], [%53], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %46 {async_task_id = array<i32: 2>} : !tt.ptr<i8>
    ttng.tensormap_create %49, %36, [%c64_i32, %c64_i32], [%51, %9], [%53], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %49 {async_task_id = array<i32: 0>} : !tt.ptr<i8>
    ttng.tensormap_create %47, %24, [%c64_i32, %c64_i32], [%54, %9], [%56], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %47 {async_task_id = array<i32: 2>} : !tt.ptr<i8>
    ttng.tensormap_create %50, %39, [%c64_i32, %c64_i32], [%54, %9], [%56], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0, 1, 2, 3>, elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %50 {async_task_id = array<i32: 0>} : !tt.ptr<i8>
    %57 = tt.make_range {async_task_id = array<i32: 3>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %58 = tt.make_range {async_task_id = array<i32: 3>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %59 = tt.make_range {async_task_id = array<i32: 3>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
    %60 = tt.expand_dims %58 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %61 = tt.splat %arg19 : i32 -> tensor<1x64xi32, #blocked>
    %62 = arith.muli %60, %61 : tensor<1x64xi32, #blocked>
    %63 = tt.make_range {async_task_id = array<i32: 3>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %64 = tt.expand_dims %63 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %65 = tt.splat %arg19 {async_task_id = array<i32: 3>} : i32 -> tensor<1x64xi32, #blocked>
    %66 = arith.muli %64, %65 {async_task_id = array<i32: 3>} : tensor<1x64xi32, #blocked>
    %67 = tt.broadcast %66 {async_task_id = array<i32: 3>} : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %68 = tt.make_range {async_task_id = array<i32: 3>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %69 = tt.expand_dims %68 {async_task_id = array<i32: 3>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %70 = tt.broadcast %69 {async_task_id = array<i32: 3>} : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %71 = arith.addi %67, %70 {async_task_id = array<i32: 3>} : tensor<128x64xi32, #blocked>
    %72 = tt.splat %33 {async_task_id = array<i32: 3>} : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
    %73 = tt.addptr %72, %71 {async_task_id = array<i32: 3>} : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
    %74 = arith.extsi %arg14 {async_task_id = array<i32: 2>} : i32 to i64
    %75 = arith.muli %3, %74 {async_task_id = array<i32: 2>} : i64
    %76 = arith.trunci %75 {async_task_id = array<i32: 2>} : i64 to i32
    %77 = ttng.reinterpret_tensor_descriptor %46 {async_task_id = array<i32: 2>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %78 = arith.extsi %arg16 {async_task_id = array<i32: 2>} : i32 to i64
    %79 = arith.muli %3, %78 {async_task_id = array<i32: 2>} : i64
    %80 = arith.trunci %79 {async_task_id = array<i32: 2>} : i64 to i32
    %81 = ttng.reinterpret_tensor_descriptor %47 {async_task_id = array<i32: 2>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %82 = tt.splat %15 {async_task_id = array<i32: 3>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %83 = arith.extsi %arg12 {async_task_id = array<i32: 2>} : i32 to i64
    %84 = arith.muli %3, %83 {async_task_id = array<i32: 2>} : i64
    %85 = arith.trunci %84 {async_task_id = array<i32: 2>} : i64 to i32
    %86 = ttng.reinterpret_tensor_descriptor %45 {async_task_id = array<i32: 2>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %87 = tt.splat %arg25 {async_task_id = array<i32: 3>} : f32 -> tensor<64x64xf32, #linear1>
    %88 = arith.extsi %arg18 {async_task_id = array<i32: 2>} : i32 to i64
    %89 = arith.muli %3, %88 {async_task_id = array<i32: 2>} : i64
    %90 = arith.trunci %89 {async_task_id = array<i32: 2>} : i64 to i32
    %91 = ttng.reinterpret_tensor_descriptor %48 {async_task_id = array<i32: 2>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %92 = tt.splat %arg25 {async_task_id = array<i32: 3>} : f32 -> tensor<128x64xf32, #linear2>
    %93 = tt.splat %arg25 {async_task_id = array<i32: 0>} : f32 -> tensor<64x128xf32, #linear>
    %94 = arith.extsi %arg24 {async_task_id = array<i32: 0>} : i32 to i64
    %95 = arith.muli %3, %94 {async_task_id = array<i32: 0>} : i64
    %96 = arith.trunci %95 {async_task_id = array<i32: 0>} : i64 to i32
    %97 = ttng.reinterpret_tensor_descriptor %50 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %98 = arith.extsi %arg22 {async_task_id = array<i32: 0>} : i32 to i64
    %99 = arith.muli %3, %98 {async_task_id = array<i32: 0>} : i64
    %100 = arith.trunci %99 {async_task_id = array<i32: 0>} : i64 to i32
    %101 = ttng.reinterpret_tensor_descriptor %49 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x128xbf16, #shared>>
    scf.for %arg36 = %c0_i32 to %9 step %c64_i32  : i32 {
      %102 = tt.splat %arg36 {async_task_id = array<i32: 3>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %103 = arith.addi %102, %59 {async_task_id = array<i32: 3>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %104 = tt.descriptor_load %77[%arg36, %76] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked1>
      %105 = ttg.local_alloc %104 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %106 = tt.descriptor_load %81[%arg36, %80] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked1>
      %107 = ttg.local_alloc %106 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %108 = tt.expand_dims %103 {async_task_id = array<i32: 3>, axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<64x1xi32, #linear1>
      %109 = tt.broadcast %108 {async_task_id = array<i32: 3>} : tensor<64x1xi32, #linear1> -> tensor<64x64xi32, #linear1>
      %110 = ttg.memdesc_trans %105 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
      %result, %token = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_4, %token_5 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_6, %token_7 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_8, %token_9 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_10, %token_11 = ttng.tmem_alloc {async_task_id = array<i32: 1, 3>} : () -> (!ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %111 = ttng.tmem_store %cst, %result_8[%token_9], %true {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> -> !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %112 = ttng.tmem_store %cst, %result_4[%token_5], %true {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> -> !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %113:6 = scf.for %arg37 = %arg36 to %15 step %c64_i32 iter_args(%arg38 = %false, %arg39 = %token, %arg40 = %112, %arg41 = %token_7, %arg42 = %111, %arg43 = %token_11) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %123 = tt.splat %arg37 {async_task_id = array<i32: 3>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %124 = tt.splat %arg37 {async_task_id = array<i32: 3>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %125 = arith.addi %57, %123 {async_task_id = array<i32: 3>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %126 = arith.addi %58, %124 {async_task_id = array<i32: 3>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %127 = arith.cmpi slt, %126, %82 {async_task_id = array<i32: 3>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %128 = tt.load %arg26 {async_task_id = array<i32: 3>} : !tt.ptr<f32>
        %129 = tt.descriptor_load %86[%arg37, %85] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked1>
        %130 = ttg.local_alloc %129 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %131 = ttg.memdesc_trans %130 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
        %132 = ttng.tc_gen5_mma %105, %131, %result[%arg39], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %133 = tt.expand_dims %125 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xi32, #linear1>
        %134 = tt.broadcast %133 {async_task_id = array<i32: 3>} : tensor<1x64xi32, #linear1> -> tensor<64x64xi32, #linear1>
        %135 = arith.cmpi eq, %134, %109 {async_task_id = array<i32: 3>} : tensor<64x64xi32, #linear1>
        %136 = arith.subi %134, %109 {async_task_id = array<i32: 3>} : tensor<64x64xi32, #linear1>
        %137 = arith.cmpi sgt, %136, %cst_3 {async_task_id = array<i32: 3>} : tensor<64x64xi32, #linear1>
        %138 = arith.ori %135, %137 {async_task_id = array<i32: 3>} : tensor<64x64xi1, #linear1>
        %result_16, %token_17 = ttng.tmem_load %result[%132] {async_task_id = array<i32: 3>} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %139 = arith.mulf %result_16, %87 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %140 = arith.subf %cst_1, %139 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %141 = math.exp %140 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %142 = arith.addf %141, %cst_0 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %143 = tt.extern_elementwise %cst_0, %142 {async_task_id = array<i32: 3>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<64x64xf32, #linear1>, tensor<64x64xf32, #linear1>) -> tensor<64x64xf32, #linear1>
        %144 = arith.mulf %139, %143 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %145 = tt.splat %128 {async_task_id = array<i32: 3>} : f32 -> tensor<64x64xf32, #linear1>
        %146 = arith.mulf %144, %145 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %147 = arith.select %138, %146, %cst_1 {async_task_id = array<i32: 3>} : tensor<64x64xi1, #linear1>, tensor<64x64xf32, #linear1>
        %148 = arith.truncf %147 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %result_18 = ttng.tmem_alloc %148 {async_task_id = array<i32: 3>} : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>
        %149 = tt.descriptor_load %91[%arg37, %90] {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked1>
        %150 = ttg.local_alloc %149 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %151 = ttng.tc_gen5_mma %result_18, %150, %result_4[%arg40], %arg38, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %152 = ttg.memdesc_trans %150 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
        %153 = ttng.tc_gen5_mma %107, %152, %result_6[%arg41], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_19, %token_20 = ttng.tmem_load %result_6[%153] {async_task_id = array<i32: 3>} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %154 = arith.mulf %result_19, %143 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %155 = arith.subf %cst_0, %143 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %156 = arith.mulf %139, %155 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %157 = arith.addf %156, %cst_0 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %158 = arith.mulf %154, %157 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %159 = arith.mulf %158, %145 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1>
        %160 = arith.select %138, %159, %cst_1 {async_task_id = array<i32: 3>} : tensor<64x64xi1, #linear1>, tensor<64x64xf32, #linear1>
        %161 = arith.truncf %160 {async_task_id = array<i32: 3>} : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %162 = ttg.local_alloc %161 {async_task_id = array<i32: 3>} : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
        %result_21 = ttng.tmem_alloc %161 {async_task_id = array<i32: 3>} : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>
        %163 = ttng.tc_gen5_mma %result_21, %130, %result_8[%arg42], %arg38, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %164 = tt.expand_dims %127 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
        %165 = arith.muli %arg37, %arg19 {async_task_id = array<i32: 3>} : i32
        %166 = tt.splat %165 {async_task_id = array<i32: 3>} : i32 -> tensor<128x64xi32, #blocked>
        %167 = tt.addptr %73, %166 {async_task_id = array<i32: 3>} : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
        %168 = tt.broadcast %164 {async_task_id = array<i32: 3>} : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
        %169 = tt.load %167, %168, %cst_2 evictionPolicy = evict_last {async_task_id = array<i32: 3>} : tensor<128x64x!tt.ptr<bf16>, #blocked>
        %170 = ttng.tc_gen5_mma %110, %162, %result_10[%arg43], %false, %true {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #shared1, #smem>, !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable>
        %result_22, %token_23 = ttng.tmem_load %result_10[%170] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear2>
        %171 = arith.mulf %result_22, %92 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear2>
        %172 = ttg.convert_layout %169 {async_task_id = array<i32: 3>} : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #linear2>
        %173 = arith.extf %172 {async_task_id = array<i32: 3>} : tensor<128x64xbf16, #linear2> to tensor<128x64xf32, #linear2>
        %174 = arith.addf %173, %171 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear2>
        %175 = arith.truncf %174 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear2> to tensor<128x64xbf16, #linear2>
        %176 = ttg.convert_layout %175 {async_task_id = array<i32: 3>} : tensor<128x64xbf16, #linear2> -> tensor<128x64xbf16, #blocked>
        tt.store %167, %176, %168 evictionPolicy = evict_last {async_task_id = array<i32: 3>} : tensor<128x64x!tt.ptr<bf16>, #blocked>
        scf.yield {async_task_id = array<i32: 0>} %true, %token_17, %151, %token_20, %163, %token_23 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.loop_unroll_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      %result_12, %token_13 = ttng.tmem_load %result_8[%113#4] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %114 = arith.mulf %result_12, %93 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear>
      %result_14, %token_15 = ttng.tmem_load %result_4[%113#2] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %115 = arith.truncf %result_14 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> to tensor<64x128xbf16, #linear>
      %116 = ttg.convert_layout %115 {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #linear> -> tensor<64x128xbf16, #blocked1>
      %117 = ttg.local_alloc %116 {async_task_id = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
      %118 = ttng.async_tma_copy_local_to_global %97[%arg36, %96] %117 {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x128xbf16, #shared>>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %118   {async_task_id = array<i32: 0>} : !ttg.async.token
      %119 = arith.truncf %114 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> to tensor<64x128xbf16, #linear>
      %120 = ttg.convert_layout %119 {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #linear> -> tensor<64x128xbf16, #blocked1>
      %121 = ttg.local_alloc %120 {async_task_id = array<i32: 0>} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
      %122 = ttng.async_tma_copy_local_to_global %101[%arg36, %100] %121 {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x128xbf16, #shared>>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %122   {async_task_id = array<i32: 0>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    tt.return
  }
}

