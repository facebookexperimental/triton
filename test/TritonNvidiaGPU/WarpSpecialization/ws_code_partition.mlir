// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-code-partition=num-buffers=1 | FileCheck %s

// CHECK-LABEL: @matmul_kernel_one_consumer
// CHECK: %[[#TASKID:]] = triton_nvidia_gpu.get_async_task_id : i32
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[#WG0:]] = arith.cmpi eq, %[[#TASKID]], %c0_i32 : i32
// CHECK: scf.if %[[#WG0]]
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[#WG1:]] = arith.cmpi eq, %[[#TASKID]], %c1_i32 : i32
// CHECK: scf.if %[[#WG1]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_gpu.local_load
// CHECK: triton_gpu.local_load
// CHECK: tt.dot
// CHECK: triton_nvidia_gpu.consumer_release


#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_one_consumer(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c255_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 255 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 127 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
    %cst_0 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<256x128xf16, #blocked1>
    %cst_1 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf16, #blocked2>
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 256 : i32
    %cst_2 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<256> : tensor<128x256xi32, #blocked2>
    %0 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %1 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %2 = arith.divsi %1, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %3 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %4 = arith.divsi %3, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %5 = arith.muli %4, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %6 = arith.divsi %0, %5 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %7 = arith.muli %6, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %8 = arith.subi %2, %7 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %9 = arith.minsi %8, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %10 = arith.remsi %0, %5 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %11 = arith.remsi %10, %9 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %12 = arith.addi %7, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %13 = arith.divsi %10, %9 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %14 = arith.muli %12, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %15 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {async_task_id = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.splat %14 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.splat %14 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %20 = arith.addi %18, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %21 = arith.addi %19, %16 {async_task_id = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.splat %arg3 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %23 = arith.remsi %20, %22 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %24 = arith.muli %13, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %25 = tt.splat %24 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %17 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %23 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %30 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked2>
    %31 = arith.muli %29, %30 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
    %32 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = tt.expand_dims %32 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %34 = tt.broadcast %31 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %35 = tt.broadcast %33 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %36 = arith.addi %34, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256xi32, #blocked2>
    %37 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
    %38 = tt.addptr %37, %36 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
    %39 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %41 = tt.expand_dims %39 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %42 = tt.expand_dims %40 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %43 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked1>
    %44 = arith.muli %41, %43 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1>
    %45 = tt.expand_dims %28 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %46 = tt.broadcast %44 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %47 = tt.broadcast %45 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %48 = arith.addi %46, %47 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128xi32, #blocked1>
    %49 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %50 = tt.addptr %49, %48 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
    %51 = arith.addi %arg5, %c255_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %52 = arith.divsi %51, %c256_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %53 = arith.muli %arg7, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %54 = tt.splat %53 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x128xi32, #blocked1>
    %55:3 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %38, %arg12 = %50) -> (tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %74 = arith.muli %arg9, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %75 = arith.subi %arg5, %74 {async_task_id = dense<0> : vector<1xi32>} : i32
      %76 = tt.splat %75 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x256xi32, #blocked2>
      %77 = arith.cmpi slt, %33, %76 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked2>
      %78 = tt.broadcast %77 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %79 = tt.load %arg11, %78, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>
      %80 = tt.splat %75 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked1>
      %81 = arith.cmpi slt, %42, %80 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1>
      %82 = tt.broadcast %81 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi1, #blocked1> -> tensor<256x128xi1, #blocked1>
      %83 = tt.load %arg12, %82, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>
      %84 = triton_gpu.convert_layout %79 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %85 = triton_gpu.convert_layout %83 {async_task_id = dense<1> : vector<1xi32>} : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %86 = tt.dot %84, %85, %arg10, inputPrecision = tf32 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
      %87 = tt.addptr %arg11, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
      %88 = tt.addptr %arg12, %54 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
      scf.yield {async_task_id = dense<[0, 1]> : vector<2xi32>} %86, %87, %88 : tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>
    } {async_task_id = dense<[0, 1]> : vector<2xi32>}
    %56 = arith.truncf %55#0 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %57 = tt.expand_dims %21 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %58 = tt.splat %arg8 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %59 = arith.muli %58, %57 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    %60 = tt.splat %arg2 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %61 = tt.addptr %60, %59 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %62 = tt.expand_dims %26 {async_task_id = dense<1> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %63 = tt.broadcast %61 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %64 = tt.broadcast %62 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %65 = tt.addptr %63, %64 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %66 = tt.splat %arg3 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %67 = arith.cmpi slt, %57, %66 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    %68 = tt.splat %arg4 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<1x128xi32, #blocked1>
    %69 = arith.cmpi slt, %62, %68 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked1>
    %70 = tt.broadcast %67 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %71 = tt.broadcast %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %72 = arith.andi %70, %71 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked1>
    %73 = triton_gpu.convert_layout %56 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
    tt.store %65, %73, %72 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----


// CHECK-LABEL: @matmul_kernel_two_consumers
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: triton_nvidia_gpu.consumer_release

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_two_consumers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
    %cst_0 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_1 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x128xf16, #blocked1>
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
    %c63_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i32
    %cst_2 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.divsi %1, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = arith.divsi %3, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %5 = arith.muli %4, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %6 = arith.divsi %0, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %7 = arith.muli %6, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %8 = arith.subi %2, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %9 = arith.minsi %8, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %10 = arith.remsi %0, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %11 = arith.remsi %10, %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %12 = arith.addi %7, %11 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %13 = arith.divsi %10, %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %14 = arith.muli %12, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %15 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.make_range {async_task_id = dense<2> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.addi %17, %23 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.addi %18, %24 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.remsi %25, %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %28 = arith.muli %13, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %29 = tt.make_range {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %30 = tt.splat %28 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %31 = arith.addi %30, %29 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %32 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %33 = arith.remsi %31, %32 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %34 = tt.expand_dims %22 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %35 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
    %36 = arith.muli %34, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked>
    %37 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %38 = tt.expand_dims %37 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %39 = tt.broadcast %36 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %40 = tt.broadcast %38 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %41 = arith.addi %39, %40 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64xi32, #blocked>
    %42 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %43 = tt.addptr %42, %41 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %44 = tt.expand_dims %27 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %45 = arith.muli %44, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked>
    %46 = tt.broadcast %45 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %47 = arith.addi %46, %40 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64xi32, #blocked>
    %48 = tt.addptr %42, %47 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %49 = tt.expand_dims %16 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %50 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %51 = arith.muli %49, %50 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %52 = tt.expand_dims %33 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %53 = tt.broadcast %51 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %54 = tt.broadcast %52 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %55 = arith.addi %53, %54 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128xi32, #blocked1>
    %56 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %57 = tt.addptr %56, %55 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %58 = arith.addi %arg5, %c63_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %59 = arith.divsi %58, %c64_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %60 = tt.expand_dims %37 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %61 = tt.expand_dims %16 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %62 = arith.muli %arg7, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %63 = tt.splat %62 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x128xi32, #blocked1>
    %true = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
    %false = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
    %true_3 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
    %false_4 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
    %64:5 = scf.for %arg9 = %c0_i32 to %59 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %43, %arg13 = %57, %arg14 = %48) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %93 = arith.muli %arg9, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %94 = arith.subi %arg5, %93 {async_task_id = dense<0> : vector<1xi32>} : i32
      %95 = tt.splat %94 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x64xi32, #blocked>
      %96 = arith.cmpi slt, %60, %95 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked>
      %97 = tt.broadcast %96 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %98 = tt.load %arg12, %97, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %99 = triton_gpu.local_alloc %98 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x64xf16, #blocked>) -> !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>
      %100 = tt.splat %94 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
      %101 = arith.cmpi slt, %61, %100 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
      %102 = tt.broadcast %101 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
      %103 = tt.load %arg13, %102, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %104 = triton_gpu.local_alloc %103 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xf16, #blocked1>) -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory>
      %105 = tt.load %arg14, %97, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %106 = triton_gpu.local_alloc %105 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x64xf16, #blocked>) -> !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>
      %107 = triton_nvidia_gpu.warp_group_dot %99, %104, %arg10 {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma>
      %108 = triton_nvidia_gpu.warp_group_dot %106, %104, %arg11 {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma>
      %109 = tt.addptr %arg12, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %110 = tt.addptr %arg14, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %111 = tt.addptr %arg13, %63 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
      scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %107, %108, %109, %111, %110 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    %65 = arith.truncf %64#0 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %66 = arith.truncf %64#1 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %67 = tt.expand_dims %20 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %68 = tt.splat %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %69 = arith.muli %68, %67 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %70 = tt.splat %arg2 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %71 = tt.addptr %70, %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %72 = tt.expand_dims %31 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %73 = tt.broadcast %71 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %74 = tt.broadcast %72 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %75 = tt.addptr %73, %74 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %76 = tt.expand_dims %26 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %77 = arith.muli %68, %76 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %78 = tt.addptr %70, %77 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %79 = tt.broadcast %78 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %80 = tt.addptr %79, %74 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %81 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %82 = arith.cmpi slt, %67, %81 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %83 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<1x128xi32, #blocked1>
    %84 = arith.cmpi slt, %72, %83 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked1>
    %85 = tt.broadcast %82 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %86 = tt.broadcast %84 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %87 = arith.andi %85, %86 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xi1, #blocked1>
    %88 = arith.cmpi slt, %76, %81 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %89 = tt.broadcast %88 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %90 = arith.andi %89, %86 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xi1, #blocked1>
    %91 = triton_gpu.convert_layout %65 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %75, %91, %87 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    %92 = triton_gpu.convert_layout %66 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %80, %92, %90 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
