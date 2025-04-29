// RUN: triton-opt %s -split-input-file --triton-gpu-taskid-propagate=num-consumer-groups=1 | FileCheck %s

// CHECK-LABEL: @async_kernel
// CHECK: %0 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
// CHECK: %5 = tt.splat %arg2 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1024xi32>
// CHECK: %9 = tt.load %8, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
// CHECK: %10 = tt.splat %arg1 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK: tt.store %11, %9 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>

module {
  tt.func public @async_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %11, %9 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: @two_consumers
// CHECK: tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK: tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.splat %arg1 {async_task_id = dense<[1, 2]> : vector<2xi32>}
// CHECK: tt.store {{.*}} {async_task_id = dense<1> : vector<1xi32>}
// CHECK: tt.store {{.*}} {async_task_id = dense<2> : vector<1xi32>}

module {
  tt.func public @two_consumers(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.make_range {end = 2048 : i32, start = 1024 : i32} : tensor<1024xi32>
    %4 = tt.splat %1 : i32 -> tensor<1024xi32>
    %5 = arith.addi %4, %2 : tensor<1024xi32>
    %6 = arith.addi %4, %3 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %5 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.addptr %7, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %8 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %11 = tt.load %9 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %13 = tt.addptr %12, %5 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %14 = tt.addptr %12, %6 {async_task_id = dense<2> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %13, %10 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.store %14, %11 {async_task_id = dense<2> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: @_ragged_hstu_attn_bwd
// CHECK-COUNT-10: triton_nvidia_gpu.warp_group_dot %{{.*}}, %{{.*}}, %{{.*}} {async_task_id = dense<[1, 2]> : vector<2xi32>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_ragged_hstu_attn_bwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x64xf32, #mma1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma1>
    %c447_i32 = arith.constant 447 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_5 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} 1.000000e+00 : f32
    %cst_6 = arith.constant dense<-256> : tensor<128x64xi32, #mma1>
    %cst_7 = arith.constant dense<0> : tensor<128x64xi32, #mma1>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %arg28 : i32
    %2 = tt.addptr %arg3, %1 : !tt.ptr<i64>, i32
    %3 = tt.load %2 : !tt.ptr<i64>
    %4 = arith.remsi %0, %arg28 : i32
    %5 = arith.extsi %4 : i32 to i64
    %6 = tt.addptr %arg4, %3 : !tt.ptr<i64>, i64
    %7 = tt.load %6 : !tt.ptr<i64>
    %8 = tt.addptr %6, %c1_i32 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = arith.subi %9, %7 : i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = tt.addptr %arg5, %3 : !tt.ptr<i32>, i64
    %13 = tt.load %12 : !tt.ptr<i32>
    %14 = arith.extsi %arg10 : i32 to i64
    %15 = arith.muli %7, %14 : i64
    %16 = arith.extsi %arg11 : i32 to i64
    %17 = arith.muli %5, %16 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = tt.addptr %arg0, %18 : !tt.ptr<bf16>, i64
    %20 = arith.extsi %arg12 : i32 to i64
    %21 = arith.muli %7, %20 : i64
    %22 = arith.extsi %arg13 : i32 to i64
    %23 = arith.muli %5, %22 : i64
    %24 = arith.addi %21, %23 : i64
    %25 = tt.addptr %arg1, %24 : !tt.ptr<bf16>, i64
    %26 = arith.extsi %arg14 : i32 to i64
    %27 = arith.muli %7, %26 : i64
    %28 = arith.extsi %arg15 : i32 to i64
    %29 = arith.muli %5, %28 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = tt.addptr %arg2, %30 : !tt.ptr<bf16>, i64
    %32 = arith.extsi %arg16 : i32 to i64
    %33 = arith.muli %7, %32 : i64
    %34 = arith.extsi %arg17 : i32 to i64
    %35 = arith.muli %5, %34 : i64
    %36 = arith.addi %33, %35 : i64
    %37 = tt.addptr %arg6, %36 : !tt.ptr<bf16>, i64
    %38 = arith.extsi %arg18 : i32 to i64
    %39 = arith.muli %7, %38 : i64
    %40 = arith.extsi %arg19 : i32 to i64
    %41 = arith.muli %5, %40 : i64
    %42 = arith.addi %39, %41 : i64
    %43 = tt.addptr %arg7, %42 : !tt.ptr<bf16>, i64
    %44 = arith.extsi %arg20 : i32 to i64
    %45 = arith.muli %7, %44 : i64
    %46 = arith.extsi %arg21 : i32 to i64
    %47 = arith.muli %5, %46 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = tt.addptr %arg8, %48 : !tt.ptr<bf16>, i64
    %50 = arith.extsi %arg22 : i32 to i64
    %51 = arith.muli %7, %50 : i64
    %52 = arith.extsi %arg23 : i32 to i64
    %53 = arith.muli %5, %52 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = tt.addptr %arg9, %54 : !tt.ptr<bf16>, i64
    %56 = tt.get_program_id y : i32
    %57 = arith.muli %56, %c128_i32 : i32
    %58 = arith.cmpi slt, %57, %11 : i32
    scf.if %58 {
      %59 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %60 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %61 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %64 = tt.splat %57 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %65 = tt.splat %57 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %66 = arith.addi %64, %62 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %65, %63 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %68 = tt.expand_dims %59 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %69 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked1>
      %70 = arith.muli %68, %69 : tensor<1x64xi32, #blocked1>
      %71 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %73 = tt.broadcast %70 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %74 = tt.broadcast %72 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %75 = arith.addi %73, %74 : tensor<128x64xi32, #blocked1>
      %76 = tt.splat %19 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %77 = tt.addptr %76, %75 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %78 = tt.splat %arg18 : i32 -> tensor<1x64xi32, #blocked1>
      %79 = arith.muli %68, %78 : tensor<1x64xi32, #blocked1>
      %80 = tt.broadcast %79 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %81 = arith.addi %80, %74 : tensor<128x64xi32, #blocked1>
      %82 = tt.splat %43 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %83 = tt.addptr %82, %81 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %84 = tt.expand_dims %66 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %85 = tt.expand_dims %67 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1>
      %86 = tt.splat %arg12 : i32 -> tensor<128x1xi32, #blocked>
      %87 = arith.muli %84, %86 : tensor<128x1xi32, #blocked>
      %88 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %89 = tt.expand_dims %88 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %90 = tt.broadcast %87 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %91 = tt.broadcast %89 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %92 = arith.addi %90, %91 : tensor<128x128xi32, #blocked>
      %93 = tt.splat %25 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %94 = tt.addptr %93, %92 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %95 = tt.splat %arg14 : i32 -> tensor<128x1xi32, #blocked>
      %96 = arith.muli %84, %95 : tensor<128x1xi32, #blocked>
      %97 = tt.broadcast %96 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %98 = arith.addi %97, %91 : tensor<128x128xi32, #blocked>
      %99 = tt.splat %31 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %100 = tt.addptr %99, %98 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %101 = tt.splat %11 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %102 = arith.cmpi slt, %66, %101 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %103 = tt.expand_dims %61 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %104 = tt.splat %arg16 : i32 -> tensor<64x1xi32, #blocked>
      %105 = arith.muli %103, %104 : tensor<64x1xi32, #blocked>
      %106 = tt.broadcast %105 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %107 = tt.broadcast %89 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %108 = arith.addi %106, %107 : tensor<64x128xi32, #blocked>
      %109 = tt.splat %37 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %110 = tt.addptr %109, %108 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %111 = tt.expand_dims %102 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %112 = tt.broadcast %111 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %113 = tt.load %94, %112, %cst_4 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %114 = triton_gpu.local_alloc %113 : (tensor<128x128xbf16, #blocked>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %115 = tt.load %100, %112, %cst_4 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %116 = triton_gpu.local_alloc %115 : (tensor<128x128xbf16, #blocked>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %117 = arith.subi %11, %13 : i32
      %118 = tt.splat %117 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %119 = arith.cmpi slt, %67, %118 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %120 = arith.select %119, %67, %118 : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %121 = arith.addi %57, %c447_i32 : i32
      %122 = arith.divsi %121, %c64_i32 : i32
      %123 = arith.muli %122, %c64_i32 : i32
      %124 = tt.splat %11 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %125 = tt.splat %11 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %126 = tt.broadcast %85 : tensor<128x1xi32, #mma1> -> tensor<128x64xi32, #mma1>
      %127 = tt.splat %117 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %128 = tt.splat %arg24 : f32 -> tensor<128x64xf32, #blocked1>
      %129 = tt.splat %arg24 : f32 -> tensor<128x64xf32, #mma1>
      %130 = tt.expand_dims %120 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1>
      %131 = tt.broadcast %130 : tensor<128x1xi32, #mma1> -> tensor<128x64xi32, #mma1>
      %132 = arith.sitofp %arg29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 to f32
      %133 = arith.divf %cst_5, %132 {async_task_id = dense<[1, 2]> : vector<2xi32>} : f32
      %134 = tt.splat %133 : f32 -> tensor<128x64xf32, #mma1>
      %135 = arith.subi %117, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %136 = tt.splat %135 : i32 -> tensor<1x64xi32, #mma1>
      %137 = tt.trans %114 {order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      %true = arith.constant true
      %false = arith.constant false
      %true_8 = arith.constant true
      %false_9 = arith.constant false
      %true_10 = arith.constant true
      %false_11 = arith.constant false
      %true_12 = arith.constant true
      %false_13 = arith.constant false
      %true_14 = arith.constant true
      %false_15 = arith.constant false
      %138:2 = scf.for %arg33 = %57 to %123 step %c64_i32 iter_args(%arg34 = %cst, %arg35 = %cst) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>)  : i32 {
        %166 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %167 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %168 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %169 = arith.addi %59, %166 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %170 = arith.addi %60, %167 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %171 = arith.addi %61, %168 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %172 = arith.cmpi slt, %169, %124 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %173 = arith.cmpi slt, %171, %125 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %174 = tt.expand_dims %170 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %175 = tt.broadcast %174 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %176 = arith.cmpi eq, %175, %126 : tensor<128x64xi32, #mma1>
        %177 = arith.cmpi slt, %170, %127 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %178 = arith.select %177, %170, %127 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %179 = tt.expand_dims %172 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %180 = arith.muli %arg33, %arg10 {async_task_id = dense<0> : vector<1xi32>} : i32
        %181 = tt.splat %180 : i32 -> tensor<128x64xi32, #blocked1>
        %182 = tt.addptr %77, %181 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %183 = tt.broadcast %179 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %184 = tt.load %182, %183, %cst_3 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>
        %185 = triton_gpu.local_alloc %184 : (tensor<128x64xbf16, #blocked1>) -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %186 = triton_nvidia_gpu.warp_group_dot %114, %185, %cst_1, %false {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %187 = arith.mulf %186, %129 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %188 = tt.expand_dims %178 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %189 = tt.broadcast %188 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %190 = arith.subi %131, %189 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %191 = arith.subf %cst_1, %187 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %192 = math.exp %191 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %193 = arith.addf %192, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %194 = tt.extern_elementwise %cst_0, %193 {async_task_id = dense<[1, 2]> : vector<2xi32>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x64xf32, #mma1>, tensor<128x64xf32, #mma1>) -> tensor<128x64xf32, #mma1>
        %195 = arith.mulf %187, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %196 = arith.mulf %195, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %197 = arith.cmpi slt, %190, %cst_7 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %198 = arith.cmpi sge, %188, %136 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x64xi32, #mma1>
        %199 = tt.broadcast %198 : tensor<1x64xi1, #mma1> -> tensor<128x64xi1, #mma1>
        %200 = arith.andi %197, %199 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %201 = arith.ori %176, %200 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %202 = arith.cmpi sge, %190, %cst_6 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %203 = arith.andi %197, %202 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %204 = arith.ori %176, %203 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %205 = arith.ori %204, %201 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %206 = arith.select %205, %196, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %207 = arith.truncf %206 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %208 = tt.expand_dims %173 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %209 = arith.muli %arg33, %arg16 {async_task_id = dense<0> : vector<1xi32>} : i32
        %210 = tt.splat %209 : i32 -> tensor<64x128xi32, #blocked>
        %211 = tt.addptr %110, %210 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
        %212 = tt.broadcast %208 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
        %213 = tt.load %211, %212, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
        %214 = triton_gpu.local_alloc %213 : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %215 = triton_gpu.convert_layout %207 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %216 = triton_nvidia_gpu.warp_group_dot %215, %214, %arg35 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %217 = tt.trans %214 {order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %218 = triton_nvidia_gpu.warp_group_dot %116, %217, %cst_1, %false_11 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %219 = arith.mulf %218, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %220 = arith.subf %cst_0, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %221 = arith.mulf %187, %220 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %222 = arith.addf %221, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %223 = arith.mulf %219, %222 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %224 = arith.mulf %223, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %225 = arith.select %205, %224, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %226 = arith.truncf %225 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %227 = triton_gpu.local_alloc %226 : (tensor<128x64xbf16, #mma1>) -> !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory>
        %228 = tt.trans %185 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %229 = triton_gpu.convert_layout %226 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %230 = triton_nvidia_gpu.warp_group_dot %229, %228, %arg34 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %231 = tt.expand_dims %172 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %232 = arith.muli %arg33, %arg18 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
        %233 = tt.splat %232 : i32 -> tensor<128x64xi32, #blocked1>
        %234 = tt.addptr %83, %233 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %235 = triton_nvidia_gpu.warp_group_dot %137, %227, %cst_1, %false_15 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %236 = triton_gpu.convert_layout %235 : tensor<128x64xf32, #mma1> -> tensor<128x64xf32, #blocked1>
        %237 = arith.mulf %236, %128 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1>
        %238 = arith.truncf %237 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1> to tensor<128x64xbf16, #blocked1>
        %239 = tt.broadcast %231 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %240 = tt.atomic_rmw fadd, relaxed, gpu, %234, %238, %239 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xbf16, #blocked1>, tensor<128x64xi1, #blocked1>) -> tensor<128x64xbf16, #blocked1>
        scf.yield %230, %216 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>, tt.loop_unroll_factor = 1 : i32}
      %139 = arith.subi %117, %c256_i32 : i32
      %140 = arith.cmpi slt, %139, %123 : i32
      %141 = arith.select %140, %123, %139 : i32
      %true_16 = arith.constant true
      %false_17 = arith.constant false
      %true_18 = arith.constant true
      %false_19 = arith.constant false
      %true_20 = arith.constant true
      %false_21 = arith.constant false
      %true_22 = arith.constant true
      %false_23 = arith.constant false
      %true_24 = arith.constant true
      %false_25 = arith.constant false
      %142:2 = scf.for %arg33 = %141 to %11 step %c64_i32 iter_args(%arg34 = %138#0, %arg35 = %138#1) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>)  : i32 {
        %166 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %167 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %168 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %169 = arith.addi %59, %166 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %170 = arith.addi %60, %167 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %171 = arith.addi %61, %168 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %172 = arith.cmpi slt, %169, %124 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %173 = arith.cmpi slt, %171, %125 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %174 = tt.expand_dims %170 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %175 = tt.broadcast %174 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %176 = arith.cmpi eq, %175, %126 : tensor<128x64xi32, #mma1>
        %177 = arith.cmpi slt, %170, %127 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %178 = arith.select %177, %170, %127 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %179 = tt.expand_dims %172 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %180 = arith.muli %arg33, %arg10 {async_task_id = dense<0> : vector<1xi32>} : i32
        %181 = tt.splat %180 : i32 -> tensor<128x64xi32, #blocked1>
        %182 = tt.addptr %77, %181 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %183 = tt.broadcast %179 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %184 = tt.load %182, %183, %cst_3 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>
        %185 = triton_gpu.local_alloc %184 : (tensor<128x64xbf16, #blocked1>) -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %186 = triton_nvidia_gpu.warp_group_dot %114, %185, %cst_1, %false_17 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %187 = arith.mulf %186, %129 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %188 = tt.expand_dims %178 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %189 = tt.broadcast %188 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %190 = arith.subi %131, %189 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %191 = arith.subf %cst_1, %187 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %192 = math.exp %191 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %193 = arith.addf %192, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %194 = tt.extern_elementwise %cst_0, %193 {async_task_id = dense<[1, 2]> : vector<2xi32>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x64xf32, #mma1>, tensor<128x64xf32, #mma1>) -> tensor<128x64xf32, #mma1>
        %195 = arith.mulf %187, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %196 = arith.mulf %195, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %197 = arith.cmpi slt, %190, %cst_7 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %198 = arith.cmpi sge, %188, %136 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x64xi32, #mma1>
        %199 = tt.broadcast %198 : tensor<1x64xi1, #mma1> -> tensor<128x64xi1, #mma1>
        %200 = arith.andi %197, %199 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %201 = arith.ori %176, %200 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %202 = arith.cmpi sge, %190, %cst_6 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %203 = arith.andi %197, %202 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %204 = arith.ori %176, %203 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %205 = arith.ori %204, %201 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %206 = arith.select %205, %196, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %207 = arith.truncf %206 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %208 = tt.expand_dims %173 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %209 = arith.muli %arg33, %arg16 {async_task_id = dense<0> : vector<1xi32>} : i32
        %210 = tt.splat %209 : i32 -> tensor<64x128xi32, #blocked>
        %211 = tt.addptr %110, %210 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
        %212 = tt.broadcast %208 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
        %213 = tt.load %211, %212, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
        %214 = triton_gpu.local_alloc %213 : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %215 = triton_gpu.convert_layout %207 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %216 = triton_nvidia_gpu.warp_group_dot %215, %214, %arg35 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %217 = tt.trans %214 {order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %218 = triton_nvidia_gpu.warp_group_dot %116, %217, %cst_1, %false_21 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %219 = arith.mulf %218, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %220 = arith.subf %cst_0, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %221 = arith.mulf %187, %220 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %222 = arith.addf %221, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %223 = arith.mulf %219, %222 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %224 = arith.mulf %223, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %225 = arith.select %205, %224, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %226 = arith.truncf %225 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %227 = triton_gpu.local_alloc %226 : (tensor<128x64xbf16, #mma1>) -> !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory>
        %228 = tt.trans %185 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %229 = triton_gpu.convert_layout %226 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %230 = triton_nvidia_gpu.warp_group_dot %229, %228, %arg34 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %231 = tt.expand_dims %172 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %232 = arith.muli %arg33, %arg18 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
        %233 = tt.splat %232 : i32 -> tensor<128x64xi32, #blocked1>
        %234 = tt.addptr %83, %233 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %235 = triton_nvidia_gpu.warp_group_dot %137, %227, %cst_1, %false_25 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %236 = triton_gpu.convert_layout %235 : tensor<128x64xf32, #mma1> -> tensor<128x64xf32, #blocked1>
        %237 = arith.mulf %236, %128 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1>
        %238 = arith.truncf %237 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1> to tensor<128x64xbf16, #blocked1>
        %239 = tt.broadcast %231 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %240 = tt.atomic_rmw fadd, relaxed, gpu, %234, %238, %239 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xbf16, #blocked1>, tensor<128x64xi1, #blocked1>) -> tensor<128x64xbf16, #blocked1>
        scf.yield %230, %216 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %143 = tt.expand_dims %66 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %144 = tt.splat %arg22 : i32 -> tensor<128x1xi32, #blocked>
      %145 = arith.muli %143, %144 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked>
      %146 = tt.expand_dims %88 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %147 = tt.broadcast %145 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %148 = tt.broadcast %146 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %149 = arith.addi %147, %148 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xi32, #blocked>
      %150 = tt.splat %55 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %151 = tt.addptr %150, %149 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %152 = tt.splat %arg20 : i32 -> tensor<128x1xi32, #blocked>
      %153 = arith.muli %143, %152 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked>
      %154 = tt.broadcast %153 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %155 = arith.addi %154, %148 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xi32, #blocked>
      %156 = tt.splat %49 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %157 = tt.addptr %156, %155 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %158 = tt.splat %arg24 : f32 -> tensor<128x128xf32, #mma>
      %159 = arith.mulf %142#0, %158 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma>
      %160 = tt.expand_dims %102 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %161 = arith.truncf %142#1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      %162 = tt.broadcast %160 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %163 = triton_gpu.convert_layout %161 : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
      tt.store %151, %163, %162 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %164 = arith.truncf %159 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      %165 = triton_gpu.convert_layout %164 : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
      tt.store %157, %165, %162 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: @_kernel_matmul_fp8_row_tma_persistent_ws_cooperative
// CHECK:    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
// CHECK:    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
// CHECK:    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
// CHECK:    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
// CHECK:    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
// CHECK:    %c255_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 255 : i32
// CHECK:    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
// CHECK:    %cst = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
// CHECK:    %0 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %1 = arith.divsi %0, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %2 = arith.addi %arg4, %c255_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %3 = arith.divsi %2, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %4 = arith.muli %1, %3 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %5 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %6 = tt.get_num_programs x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %7 = arith.muli %3, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %8 = arith.addi %arg5, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %9 = arith.divsi %8, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %10 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
// CHECK:    %11 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
// CHECK:    %12 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #blocked>
// CHECK:    %13 = tt.splat %arg9 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
// CHECK:    %14 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<256xi32, #blocked1>
// CHECK:    %15 = tt.splat %arg10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
// CHECK:    scf.for %arg15 = %5 to %4 step %6  : i32 {
// CHECK:      %16 = arith.divsi %arg15, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %17 = arith.muli %16, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %18 = arith.subi %1, %17 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %19 = arith.minsi %18, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %20 = arith.remsi %arg15, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %21 = arith.remsi %20, %19 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %22 = arith.addi %17, %21 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %23 = arith.divsi %20, %19 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %24 = arith.muli %22, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %25 = arith.muli %23, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %true = arith.constant true
// CHECK:      %false = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} false
// CHECK:      %26:2 = scf.for %arg16 = %c0_i32 to %9 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
// CHECK:        %50 = tt.experimental_descriptor_load %arg0[%24, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<128x128xf8E4M3FN, #blocked2>
// CHECK:        %51 = triton_gpu.local_alloc %50 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
// CHECK:        %52 = tt.experimental_descriptor_load %arg1[%25, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256x128xf8E4M3FN, #blocked2>
// CHECK:        %53 = triton_gpu.local_alloc %52 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<256x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
// CHECK:        %54 = tt.trans %53 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory>
// CHECK:        %55 = triton_nvidia_gpu.warp_group_dot %51, %54, %cst, %false {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
// CHECK:        %56 = arith.addf %arg17, %55 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
// CHECK:        %57 = arith.addi %arg18, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
// CHECK:        scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %56, %57 : tensor<128x256xf32, #mma>, i32
// CHECK:      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
// CHECK:      %27 = arith.muli %22, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
// CHECK:      %28 = tt.splat %27 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #blocked>
// CHECK:      %29 = arith.addi %28, %10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
// CHECK:      %30 = arith.muli %23, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
// CHECK:      %31 = tt.splat %30 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<256xi32, #blocked1>
// CHECK:      %32 = arith.addi %31, %11 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
// CHECK:      %33 = arith.cmpi slt, %29, %12 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
// CHECK:      %34 = tt.addptr %13, %29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
// CHECK:      %35 = tt.load %34, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>
// CHECK:      %36 = arith.cmpi slt, %32, %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
// CHECK:      %37 = tt.addptr %15, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
// CHECK:      %38 = tt.load %37, %36 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>
// CHECK:      %39 = triton_gpu.convert_layout %35 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xf32, #blocked> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
// CHECK:      %40 = tt.expand_dims %39 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
// CHECK:      %41 = triton_gpu.convert_layout %38 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
// CHECK:      %42 = tt.expand_dims %41 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xf32, #blocked3>
// CHECK:      %43 = tt.broadcast %40 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
// CHECK:      %44 = tt.broadcast %42 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
// CHECK:      %45 = arith.mulf %43, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3>
// CHECK:      %46 = triton_gpu.convert_layout %45 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3> -> tensor<128x256xf32, #mma>
// CHECK:      %47 = arith.mulf %26#0, %46 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
// CHECK:      %48 = arith.truncf %47 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
// CHECK:      %49 = triton_gpu.convert_layout %48 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xbf16, #mma> -> tensor<128x256xbf16, #blocked2>
// CHECK:      tt.experimental_descriptor_store %arg2[%24, %25], %49 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<i8, 0>, tensor<128x256xbf16, #blocked2>
// CHECK:    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#loc = loc(unknown)
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel_matmul_fp8_row_tma_persistent_ws_cooperative(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg3: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg4: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg5: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg6: i32 loc(unknown), %arg7: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg8: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg11: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg12: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg13: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg14: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.addi %arg3, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg4, %c255_i32 : i32
    %3 = arith.divsi %2, %c256_i32 : i32
    %4 = arith.muli %1, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = tt.get_num_programs x : i32
    %7 = arith.muli %3, %c8_i32 : i32
    %8 = arith.addi %arg5, %c127_i32 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %11 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %12 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %13 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %14 = tt.splat %arg4 : i32 -> tensor<256xi32, #blocked1>
    %15 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
    scf.for %arg15 = %5 to %4 step %6  : i32 {
      %16 = arith.divsi %arg15, %7 : i32
      %17 = arith.muli %16, %c8_i32 : i32
      %18 = arith.subi %1, %17 : i32
      %19 = arith.minsi %18, %c8_i32 : i32
      %20 = arith.remsi %arg15, %7 : i32
      %21 = arith.remsi %20, %19 : i32
      %22 = arith.addi %17, %21 : i32
      %23 = arith.divsi %20, %19 : i32
      %24 = arith.muli %22, %c128_i32 : i32
      %25 = arith.muli %23, %c256_i32 : i32
      %true = arith.constant true
      %false = arith.constant false
      %26:2 = scf.for %arg16 = %c0_i32 to %9 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        %50 = tt.experimental_descriptor_load %arg0[%24, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<128x128xf8E4M3FN, #blocked2>
        %51 = triton_gpu.local_alloc %50 : (tensor<128x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
        %52 = tt.experimental_descriptor_load %arg1[%25, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256x128xf8E4M3FN, #blocked2>
        %53 = triton_gpu.local_alloc %52 : (tensor<256x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
        %54 = tt.trans %53 {order = array<i32: 1, 0>} : !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory>
        %55 = triton_nvidia_gpu.warp_group_dot %51, %54, %cst, %false {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        %56 = arith.addf %arg17, %55 : tensor<128x256xf32, #mma>
        %57 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %56, %57 : tensor<128x256xf32, #mma>, i32
      }
      %27 = arith.muli %22, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #blocked>
      %29 = arith.addi %28, %10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
      %30 = arith.muli %23, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %31 = tt.splat %30 : i32 -> tensor<256xi32, #blocked1>
      %32 = arith.addi %31, %11 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
      %33 = arith.cmpi slt, %29, %12 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
      %34 = tt.addptr %13, %29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %35 = tt.load %34, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>
      %36 = arith.cmpi slt, %32, %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
      %37 = tt.addptr %15, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
      %38 = tt.load %37, %36 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>
      %39 = triton_gpu.convert_layout %35 : tensor<128xf32, #blocked> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %40 = tt.expand_dims %39 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
      %41 = triton_gpu.convert_layout %38 : tensor<256xf32, #blocked1> -> tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %42 = tt.expand_dims %41 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xf32, #blocked3>
      %43 = tt.broadcast %40 : tensor<128x1xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
      %44 = tt.broadcast %42 : tensor<1x256xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
      %45 = arith.mulf %43, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3>
      %46 = triton_gpu.convert_layout %45 : tensor<128x256xf32, #blocked3> -> tensor<128x256xf32, #mma>
      %47 = arith.mulf %26#0, %46 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
      %48 = arith.truncf %47 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
      %49 = triton_gpu.convert_layout %48 : tensor<128x256xbf16, #mma> -> tensor<128x256xbf16, #blocked2>
      tt.experimental_descriptor_store %arg2[%24, %25], %49 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<i8, 0>, tensor<128x256xbf16, #blocked2>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: @_ragged_hstu_attn_bwd
// CHECK-COUNT-10: triton_nvidia_gpu.warp_group_dot %{{.*}}, %{{.*}}, %{{.*}} {async_task_id = dense<[1, 2]> : vector<2xi32>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_ragged_hstu_attn_bwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x64xf32, #mma1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma1>
    %c447_i32 = arith.constant 447 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_5 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} 1.000000e+00 : f32
    %cst_6 = arith.constant dense<-256> : tensor<128x64xi32, #mma1>
    %cst_7 = arith.constant dense<0> : tensor<128x64xi32, #mma1>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %arg28 : i32
    %2 = tt.addptr %arg3, %1 : !tt.ptr<i64>, i32
    %3 = tt.load %2 : !tt.ptr<i64>
    %4 = arith.remsi %0, %arg28 : i32
    %5 = arith.extsi %4 : i32 to i64
    %6 = tt.addptr %arg4, %3 : !tt.ptr<i64>, i64
    %7 = tt.load %6 : !tt.ptr<i64>
    %8 = tt.addptr %6, %c1_i32 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = arith.subi %9, %7 : i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = tt.addptr %arg5, %3 : !tt.ptr<i32>, i64
    %13 = tt.load %12 : !tt.ptr<i32>
    %14 = arith.extsi %arg10 : i32 to i64
    %15 = arith.muli %7, %14 : i64
    %16 = arith.extsi %arg11 : i32 to i64
    %17 = arith.muli %5, %16 : i64
    %18 = arith.addi %15, %17 : i64
    %19 = tt.addptr %arg0, %18 : !tt.ptr<bf16>, i64
    %20 = arith.extsi %arg12 : i32 to i64
    %21 = arith.muli %7, %20 : i64
    %22 = arith.extsi %arg13 : i32 to i64
    %23 = arith.muli %5, %22 : i64
    %24 = arith.addi %21, %23 : i64
    %25 = tt.addptr %arg1, %24 : !tt.ptr<bf16>, i64
    %26 = arith.extsi %arg14 : i32 to i64
    %27 = arith.muli %7, %26 : i64
    %28 = arith.extsi %arg15 : i32 to i64
    %29 = arith.muli %5, %28 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = tt.addptr %arg2, %30 : !tt.ptr<bf16>, i64
    %32 = arith.extsi %arg16 : i32 to i64
    %33 = arith.muli %7, %32 : i64
    %34 = arith.extsi %arg17 : i32 to i64
    %35 = arith.muli %5, %34 : i64
    %36 = arith.addi %33, %35 : i64
    %37 = tt.addptr %arg6, %36 : !tt.ptr<bf16>, i64
    %38 = arith.extsi %arg18 : i32 to i64
    %39 = arith.muli %7, %38 : i64
    %40 = arith.extsi %arg19 : i32 to i64
    %41 = arith.muli %5, %40 : i64
    %42 = arith.addi %39, %41 : i64
    %43 = tt.addptr %arg7, %42 : !tt.ptr<bf16>, i64
    %44 = arith.extsi %arg20 : i32 to i64
    %45 = arith.muli %7, %44 : i64
    %46 = arith.extsi %arg21 : i32 to i64
    %47 = arith.muli %5, %46 : i64
    %48 = arith.addi %45, %47 : i64
    %49 = tt.addptr %arg8, %48 : !tt.ptr<bf16>, i64
    %50 = arith.extsi %arg22 : i32 to i64
    %51 = arith.muli %7, %50 : i64
    %52 = arith.extsi %arg23 : i32 to i64
    %53 = arith.muli %5, %52 : i64
    %54 = arith.addi %51, %53 : i64
    %55 = tt.addptr %arg9, %54 : !tt.ptr<bf16>, i64
    %56 = tt.get_program_id y : i32
    %57 = arith.muli %56, %c128_i32 : i32
    %58 = arith.cmpi slt, %57, %11 : i32
    scf.if %58 {
      %59 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %60 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %61 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %64 = tt.splat %57 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %65 = tt.splat %57 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %66 = arith.addi %64, %62 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %65, %63 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %68 = tt.expand_dims %59 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %69 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked1>
      %70 = arith.muli %68, %69 : tensor<1x64xi32, #blocked1>
      %71 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %73 = tt.broadcast %70 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %74 = tt.broadcast %72 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %75 = arith.addi %73, %74 : tensor<128x64xi32, #blocked1>
      %76 = tt.splat %19 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %77 = tt.addptr %76, %75 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %78 = tt.splat %arg18 : i32 -> tensor<1x64xi32, #blocked1>
      %79 = arith.muli %68, %78 : tensor<1x64xi32, #blocked1>
      %80 = tt.broadcast %79 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %81 = arith.addi %80, %74 : tensor<128x64xi32, #blocked1>
      %82 = tt.splat %43 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %83 = tt.addptr %82, %81 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %84 = tt.expand_dims %66 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %85 = tt.expand_dims %67 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1>
      %86 = tt.splat %arg12 : i32 -> tensor<128x1xi32, #blocked>
      %87 = arith.muli %84, %86 : tensor<128x1xi32, #blocked>
      %88 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %89 = tt.expand_dims %88 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %90 = tt.broadcast %87 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %91 = tt.broadcast %89 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %92 = arith.addi %90, %91 : tensor<128x128xi32, #blocked>
      %93 = tt.splat %25 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %94 = tt.addptr %93, %92 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %95 = tt.splat %arg14 : i32 -> tensor<128x1xi32, #blocked>
      %96 = arith.muli %84, %95 : tensor<128x1xi32, #blocked>
      %97 = tt.broadcast %96 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %98 = arith.addi %97, %91 : tensor<128x128xi32, #blocked>
      %99 = tt.splat %31 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %100 = tt.addptr %99, %98 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %101 = tt.splat %11 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %102 = arith.cmpi slt, %66, %101 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %103 = tt.expand_dims %61 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %104 = tt.splat %arg16 : i32 -> tensor<64x1xi32, #blocked>
      %105 = arith.muli %103, %104 : tensor<64x1xi32, #blocked>
      %106 = tt.broadcast %105 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %107 = tt.broadcast %89 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %108 = arith.addi %106, %107 : tensor<64x128xi32, #blocked>
      %109 = tt.splat %37 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %110 = tt.addptr %109, %108 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %111 = tt.expand_dims %102 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %112 = tt.broadcast %111 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %113 = tt.load %94, %112, %cst_4 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %114 = triton_gpu.local_alloc %113 : (tensor<128x128xbf16, #blocked>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %115 = tt.load %100, %112, %cst_4 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %116 = triton_gpu.local_alloc %115 : (tensor<128x128xbf16, #blocked>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %117 = arith.subi %11, %13 : i32
      %118 = tt.splat %117 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %119 = arith.cmpi slt, %67, %118 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %120 = arith.select %119, %67, %118 : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
      %121 = arith.addi %57, %c447_i32 : i32
      %122 = arith.divsi %121, %c64_i32 : i32
      %123 = arith.muli %122, %c64_i32 : i32
      %124 = tt.splat %11 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %125 = tt.splat %11 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %126 = tt.broadcast %85 : tensor<128x1xi32, #mma1> -> tensor<128x64xi32, #mma1>
      %127 = tt.splat %117 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %128 = tt.splat %arg24 : f32 -> tensor<128x64xf32, #blocked1>
      %129 = tt.splat %arg24 : f32 -> tensor<128x64xf32, #mma1>
      %130 = tt.expand_dims %120 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1>
      %131 = tt.broadcast %130 : tensor<128x1xi32, #mma1> -> tensor<128x64xi32, #mma1>
      %132 = arith.sitofp %arg29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 to f32
      %133 = arith.divf %cst_5, %132 {async_task_id = dense<[1, 2]> : vector<2xi32>} : f32
      %134 = tt.splat %133 : f32 -> tensor<128x64xf32, #mma1>
      %135 = arith.subi %117, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %136 = tt.splat %135 : i32 -> tensor<1x64xi32, #mma1>
      %137 = tt.trans %114 {order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      %true = arith.constant true
      %false = arith.constant false
      %true_8 = arith.constant true
      %false_9 = arith.constant false
      %true_10 = arith.constant true
      %false_11 = arith.constant false
      %true_12 = arith.constant true
      %false_13 = arith.constant false
      %true_14 = arith.constant true
      %false_15 = arith.constant false
      %138:2 = scf.for %arg33 = %57 to %123 step %c64_i32 iter_args(%arg34 = %cst, %arg35 = %cst) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>)  : i32 {
        %166 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %167 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %168 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %169 = arith.addi %59, %166 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %170 = arith.addi %60, %167 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %171 = arith.addi %61, %168 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %172 = arith.cmpi slt, %169, %124 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %173 = arith.cmpi slt, %171, %125 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %174 = tt.expand_dims %170 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %175 = tt.broadcast %174 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %176 = arith.cmpi eq, %175, %126 : tensor<128x64xi32, #mma1>
        %177 = arith.cmpi slt, %170, %127 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %178 = arith.select %177, %170, %127 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %179 = tt.expand_dims %172 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %180 = arith.muli %arg33, %arg10 {async_task_id = dense<0> : vector<1xi32>} : i32
        %181 = tt.splat %180 : i32 -> tensor<128x64xi32, #blocked1>
        %182 = tt.addptr %77, %181 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %183 = tt.broadcast %179 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %184 = tt.load %182, %183, %cst_3 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>
        %185 = triton_gpu.local_alloc %184 : (tensor<128x64xbf16, #blocked1>) -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %186 = triton_nvidia_gpu.warp_group_dot %114, %185, %cst_1, %false {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %187 = arith.mulf %186, %129 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %188 = tt.expand_dims %178 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %189 = tt.broadcast %188 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %190 = arith.subi %131, %189 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %191 = arith.subf %cst_1, %187 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %192 = math.exp %191 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %193 = arith.addf %192, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %194 = tt.extern_elementwise %cst_0, %193 {async_task_id = dense<[1, 2]> : vector<2xi32>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x64xf32, #mma1>, tensor<128x64xf32, #mma1>) -> tensor<128x64xf32, #mma1>
        %195 = arith.mulf %187, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %196 = arith.mulf %195, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %197 = arith.cmpi slt, %190, %cst_7 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %198 = arith.cmpi sge, %188, %136 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x64xi32, #mma1>
        %199 = tt.broadcast %198 : tensor<1x64xi1, #mma1> -> tensor<128x64xi1, #mma1>
        %200 = arith.andi %197, %199 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %201 = arith.ori %176, %200 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %202 = arith.cmpi sge, %190, %cst_6 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %203 = arith.andi %197, %202 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %204 = arith.ori %176, %203 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %205 = arith.ori %204, %201 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %206 = arith.select %205, %196, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %207 = arith.truncf %206 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %208 = tt.expand_dims %173 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %209 = arith.muli %arg33, %arg16 {async_task_id = dense<0> : vector<1xi32>} : i32
        %210 = tt.splat %209 : i32 -> tensor<64x128xi32, #blocked>
        %211 = tt.addptr %110, %210 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
        %212 = tt.broadcast %208 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
        %213 = tt.load %211, %212, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
        %214 = triton_gpu.local_alloc %213 : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %215 = triton_gpu.convert_layout %207 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %216 = triton_nvidia_gpu.warp_group_dot %215, %214, %arg35 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %217 = tt.trans %214 {order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %218 = triton_nvidia_gpu.warp_group_dot %116, %217, %cst_1, %false_11 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %219 = arith.mulf %218, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %220 = arith.subf %cst_0, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %221 = arith.mulf %187, %220 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %222 = arith.addf %221, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %223 = arith.mulf %219, %222 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %224 = arith.mulf %223, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %225 = arith.select %205, %224, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %226 = arith.truncf %225 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %227 = triton_gpu.local_alloc %226 : (tensor<128x64xbf16, #mma1>) -> !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory>
        %228 = tt.trans %185 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %229 = triton_gpu.convert_layout %226 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %230 = triton_nvidia_gpu.warp_group_dot %229, %228, %arg34 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %231 = tt.expand_dims %172 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %232 = arith.muli %arg33, %arg18 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
        %233 = tt.splat %232 : i32 -> tensor<128x64xi32, #blocked1>
        %234 = tt.addptr %83, %233 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %235 = triton_nvidia_gpu.warp_group_dot %137, %227, %cst_1, %false_15 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %236 = triton_gpu.convert_layout %235 : tensor<128x64xf32, #mma1> -> tensor<128x64xf32, #blocked1>
        %237 = arith.mulf %236, %128 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1>
        %238 = arith.truncf %237 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1> to tensor<128x64xbf16, #blocked1>
        %239 = tt.broadcast %231 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %240 = tt.atomic_rmw fadd, relaxed, gpu, %234, %238, %239 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xbf16, #blocked1>, tensor<128x64xi1, #blocked1>) -> tensor<128x64xbf16, #blocked1>
        scf.yield %230, %216 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>, tt.loop_unroll_factor = 1 : i32}
      %139 = arith.subi %117, %c256_i32 : i32
      %140 = arith.cmpi slt, %139, %123 : i32
      %141 = arith.select %140, %123, %139 : i32
      %true_16 = arith.constant true
      %false_17 = arith.constant false
      %true_18 = arith.constant true
      %false_19 = arith.constant false
      %true_20 = arith.constant true
      %false_21 = arith.constant false
      %true_22 = arith.constant true
      %false_23 = arith.constant false
      %true_24 = arith.constant true
      %false_25 = arith.constant false
      %142:2 = scf.for %arg33 = %141 to %11 step %c64_i32 iter_args(%arg34 = %138#0, %arg35 = %138#1) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>)  : i32 {
        %166 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %167 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %168 = tt.splat %arg33 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %169 = arith.addi %59, %166 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %170 = arith.addi %60, %167 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %171 = arith.addi %61, %168 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %172 = arith.cmpi slt, %169, %124 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %173 = arith.cmpi slt, %171, %125 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %174 = tt.expand_dims %170 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %175 = tt.broadcast %174 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %176 = arith.cmpi eq, %175, %126 : tensor<128x64xi32, #mma1>
        %177 = arith.cmpi slt, %170, %127 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %178 = arith.select %177, %170, %127 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
        %179 = tt.expand_dims %172 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %180 = arith.muli %arg33, %arg10 {async_task_id = dense<0> : vector<1xi32>} : i32
        %181 = tt.splat %180 : i32 -> tensor<128x64xi32, #blocked1>
        %182 = tt.addptr %77, %181 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %183 = tt.broadcast %179 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %184 = tt.load %182, %183, %cst_3 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>
        %185 = triton_gpu.local_alloc %184 : (tensor<128x64xbf16, #blocked1>) -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %186 = triton_nvidia_gpu.warp_group_dot %114, %185, %cst_1, %false_17 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %187 = arith.mulf %186, %129 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %188 = tt.expand_dims %178 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1>
        %189 = tt.broadcast %188 : tensor<1x64xi32, #mma1> -> tensor<128x64xi32, #mma1>
        %190 = arith.subi %131, %189 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %191 = arith.subf %cst_1, %187 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %192 = math.exp %191 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %193 = arith.addf %192, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %194 = tt.extern_elementwise %cst_0, %193 {async_task_id = dense<[1, 2]> : vector<2xi32>, libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<128x64xf32, #mma1>, tensor<128x64xf32, #mma1>) -> tensor<128x64xf32, #mma1>
        %195 = arith.mulf %187, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %196 = arith.mulf %195, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %197 = arith.cmpi slt, %190, %cst_7 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %198 = arith.cmpi sge, %188, %136 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x64xi32, #mma1>
        %199 = tt.broadcast %198 : tensor<1x64xi1, #mma1> -> tensor<128x64xi1, #mma1>
        %200 = arith.andi %197, %199 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %201 = arith.ori %176, %200 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %202 = arith.cmpi sge, %190, %cst_6 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi32, #mma1>
        %203 = arith.andi %197, %202 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %204 = arith.ori %176, %203 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %205 = arith.ori %204, %201 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>
        %206 = arith.select %205, %196, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %207 = arith.truncf %206 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %208 = tt.expand_dims %173 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %209 = arith.muli %arg33, %arg16 {async_task_id = dense<0> : vector<1xi32>} : i32
        %210 = tt.splat %209 : i32 -> tensor<64x128xi32, #blocked>
        %211 = tt.addptr %110, %210 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
        %212 = tt.broadcast %208 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
        %213 = tt.load %211, %212, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
        %214 = triton_gpu.local_alloc %213 : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %215 = triton_gpu.convert_layout %207 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %216 = triton_nvidia_gpu.warp_group_dot %215, %214, %arg35 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %217 = tt.trans %214 {order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %218 = triton_nvidia_gpu.warp_group_dot %116, %217, %cst_1, %false_21 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %219 = arith.mulf %218, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %220 = arith.subf %cst_0, %194 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %221 = arith.mulf %187, %220 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %222 = arith.addf %221, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %223 = arith.mulf %219, %222 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %224 = arith.mulf %223, %134 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1>
        %225 = arith.select %205, %224, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xi1, #mma1>, tensor<128x64xf32, #mma1>
        %226 = arith.truncf %225 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #mma1> to tensor<128x64xbf16, #mma1>
        %227 = triton_gpu.local_alloc %226 : (tensor<128x64xbf16, #mma1>) -> !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory>
        %228 = tt.trans %185 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %229 = triton_gpu.convert_layout %226 : tensor<128x64xbf16, #mma1> -> tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
        %230 = triton_nvidia_gpu.warp_group_dot %229, %228, %arg34 {inputPrecision = 0 : i32} : tensor<128x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
        %231 = tt.expand_dims %172 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
        %232 = arith.muli %arg33, %arg18 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
        %233 = tt.splat %232 : i32 -> tensor<128x64xi32, #blocked1>
        %234 = tt.addptr %83, %233 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %235 = triton_nvidia_gpu.warp_group_dot %137, %227, %cst_1, %false_25 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma1>
        %236 = triton_gpu.convert_layout %235 : tensor<128x64xf32, #mma1> -> tensor<128x64xf32, #blocked1>
        %237 = arith.mulf %236, %128 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1>
        %238 = arith.truncf %237 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x64xf32, #blocked1> to tensor<128x64xbf16, #blocked1>
        %239 = tt.broadcast %231 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %240 = tt.atomic_rmw fadd, relaxed, gpu, %234, %238, %239 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xbf16, #blocked1>, tensor<128x64xi1, #blocked1>) -> tensor<128x64xbf16, #blocked1>
        scf.yield %230, %216 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %143 = tt.expand_dims %66 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %144 = tt.splat %arg22 : i32 -> tensor<128x1xi32, #blocked>
      %145 = arith.muli %143, %144 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked>
      %146 = tt.expand_dims %88 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %147 = tt.broadcast %145 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %148 = tt.broadcast %146 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %149 = arith.addi %147, %148 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xi32, #blocked>
      %150 = tt.splat %55 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %151 = tt.addptr %150, %149 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %152 = tt.splat %arg20 : i32 -> tensor<128x1xi32, #blocked>
      %153 = arith.muli %143, %152 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked>
      %154 = tt.broadcast %153 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %155 = arith.addi %154, %148 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xi32, #blocked>
      %156 = tt.splat %49 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %157 = tt.addptr %156, %155 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %158 = tt.splat %arg24 : f32 -> tensor<128x128xf32, #mma>
      %159 = arith.mulf %142#0, %158 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma>
      %160 = tt.expand_dims %102 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %161 = arith.truncf %142#1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      %162 = tt.broadcast %160 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %163 = triton_gpu.convert_layout %161 : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
      tt.store %151, %163, %162 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %164 = arith.truncf %159 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      %165 = triton_gpu.convert_layout %164 : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
      tt.store %157, %165, %162 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: @_kernel_matmul_fp8_row_tma_persistent_ws_cooperative
// CHECK:    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
// CHECK:    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
// CHECK:    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
// CHECK:    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
// CHECK:    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
// CHECK:    %c255_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 255 : i32
// CHECK:    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
// CHECK:    %cst = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
// CHECK:    %0 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %1 = arith.divsi %0, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %2 = arith.addi %arg4, %c255_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %3 = arith.divsi %2, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %4 = arith.muli %1, %3 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %5 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %6 = tt.get_num_programs x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %7 = arith.muli %3, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %8 = arith.addi %arg5, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %9 = arith.divsi %8, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:    %10 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
// CHECK:    %11 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
// CHECK:    %12 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #blocked>
// CHECK:    %13 = tt.splat %arg9 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
// CHECK:    %14 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<256xi32, #blocked1>
// CHECK:    %15 = tt.splat %arg10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
// CHECK:    scf.for %arg15 = %5 to %4 step %6  : i32 {
// CHECK:      %16 = arith.divsi %arg15, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %17 = arith.muli %16, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %18 = arith.subi %1, %17 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %19 = arith.minsi %18, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %20 = arith.remsi %arg15, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %21 = arith.remsi %20, %19 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %22 = arith.addi %17, %21 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %23 = arith.divsi %20, %19 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %24 = arith.muli %22, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %25 = arith.muli %23, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK:      %true = arith.constant true
// CHECK:      %false = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} false
// CHECK:      %26:2 = scf.for %arg16 = %c0_i32 to %9 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
// CHECK:        %50 = tt.experimental_descriptor_load %arg0[%24, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<128x128xf8E4M3FN, #blocked2>
// CHECK:        %51 = triton_gpu.local_alloc %50 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
// CHECK:        %52 = tt.experimental_descriptor_load %arg1[%25, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256x128xf8E4M3FN, #blocked2>
// CHECK:        %53 = triton_gpu.local_alloc %52 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<256x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
// CHECK:        %54 = tt.trans %53 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory>
// CHECK:        %55 = triton_nvidia_gpu.warp_group_dot %51, %54, %cst, %false {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
// CHECK:        %56 = arith.addf %arg17, %55 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
// CHECK:        %57 = arith.addi %arg18, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
// CHECK:        scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %56, %57 : tensor<128x256xf32, #mma>, i32
// CHECK:      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
// CHECK:      %27 = arith.muli %22, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
// CHECK:      %28 = tt.splat %27 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #blocked>
// CHECK:      %29 = arith.addi %28, %10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
// CHECK:      %30 = arith.muli %23, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
// CHECK:      %31 = tt.splat %30 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<256xi32, #blocked1>
// CHECK:      %32 = arith.addi %31, %11 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
// CHECK:      %33 = arith.cmpi slt, %29, %12 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
// CHECK:      %34 = tt.addptr %13, %29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
// CHECK:      %35 = tt.load %34, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>
// CHECK:      %36 = arith.cmpi slt, %32, %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
// CHECK:      %37 = tt.addptr %15, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
// CHECK:      %38 = tt.load %37, %36 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>
// CHECK:      %39 = triton_gpu.convert_layout %35 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xf32, #blocked> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
// CHECK:      %40 = tt.expand_dims %39 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
// CHECK:      %41 = triton_gpu.convert_layout %38 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xf32, #blocked1> -> tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
// CHECK:      %42 = tt.expand_dims %41 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xf32, #blocked3>
// CHECK:      %43 = tt.broadcast %40 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
// CHECK:      %44 = tt.broadcast %42 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
// CHECK:      %45 = arith.mulf %43, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3>
// CHECK:      %46 = triton_gpu.convert_layout %45 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3> -> tensor<128x256xf32, #mma>
// CHECK:      %47 = arith.mulf %26#0, %46 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
// CHECK:      %48 = arith.truncf %47 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
// CHECK:      %49 = triton_gpu.convert_layout %48 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xbf16, #mma> -> tensor<128x256xbf16, #blocked2>
// CHECK:      tt.experimental_descriptor_store %arg2[%24, %25], %49 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<i8, 0>, tensor<128x256xbf16, #blocked2>
// CHECK:    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#loc = loc(unknown)
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel_matmul_fp8_row_tma_persistent_ws_cooperative(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg3: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg4: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg5: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg6: i32 loc(unknown), %arg7: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg8: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg11: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32} loc(unknown), %arg12: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg13: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg14: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.addi %arg3, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg4, %c255_i32 : i32
    %3 = arith.divsi %2, %c256_i32 : i32
    %4 = arith.muli %1, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = tt.get_num_programs x : i32
    %7 = arith.muli %3, %c8_i32 : i32
    %8 = arith.addi %arg5, %c127_i32 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %11 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %12 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %13 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %14 = tt.splat %arg4 : i32 -> tensor<256xi32, #blocked1>
    %15 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
    scf.for %arg15 = %5 to %4 step %6  : i32 {
      %16 = arith.divsi %arg15, %7 : i32
      %17 = arith.muli %16, %c8_i32 : i32
      %18 = arith.subi %1, %17 : i32
      %19 = arith.minsi %18, %c8_i32 : i32
      %20 = arith.remsi %arg15, %7 : i32
      %21 = arith.remsi %20, %19 : i32
      %22 = arith.addi %17, %21 : i32
      %23 = arith.divsi %20, %19 : i32
      %24 = arith.muli %22, %c128_i32 : i32
      %25 = arith.muli %23, %c256_i32 : i32
      %true = arith.constant true
      %false = arith.constant false
      %26:2 = scf.for %arg16 = %c0_i32 to %9 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        %50 = tt.experimental_descriptor_load %arg0[%24, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<128x128xf8E4M3FN, #blocked2>
        %51 = triton_gpu.local_alloc %50 : (tensor<128x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
        %52 = tt.experimental_descriptor_load %arg1[%25, %arg18] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256x128xf8E4M3FN, #blocked2>
        %53 = triton_gpu.local_alloc %52 : (tensor<256x128xf8E4M3FN, #blocked2>) -> !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory>
        %54 = tt.trans %53 {order = array<i32: 1, 0>} : !tt.memdesc<256x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory>
        %55 = triton_nvidia_gpu.warp_group_dot %51, %54, %cst, %false {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !tt.memdesc<128x128xf8E4M3FN, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x256xf8E4M3FN, #shared1, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        %56 = arith.addf %arg17, %55 : tensor<128x256xf32, #mma>
        %57 = arith.addi %arg18, %c128_i32 : i32
        scf.yield %56, %57 : tensor<128x256xf32, #mma>, i32
      }
      %27 = arith.muli %22, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #blocked>
      %29 = arith.addi %28, %10 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
      %30 = arith.muli %23, %c256_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %31 = tt.splat %30 : i32 -> tensor<256xi32, #blocked1>
      %32 = arith.addi %31, %11 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
      %33 = arith.cmpi slt, %29, %12 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #blocked>
      %34 = tt.addptr %13, %29 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %35 = tt.load %34, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x!tt.ptr<f32>, #blocked>
      %36 = arith.cmpi slt, %32, %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256xi32, #blocked1>
      %37 = tt.addptr %15, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
      %38 = tt.load %37, %36 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<256x!tt.ptr<f32>, #blocked1>
      %39 = triton_gpu.convert_layout %35 : tensor<128xf32, #blocked> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %40 = tt.expand_dims %39 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
      %41 = triton_gpu.convert_layout %38 : tensor<256xf32, #blocked1> -> tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %42 = tt.expand_dims %41 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xf32, #blocked3>
      %43 = tt.broadcast %40 : tensor<128x1xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
      %44 = tt.broadcast %42 : tensor<1x256xf32, #blocked3> -> tensor<128x256xf32, #blocked3>
      %45 = arith.mulf %43, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #blocked3>
      %46 = triton_gpu.convert_layout %45 : tensor<128x256xf32, #blocked3> -> tensor<128x256xf32, #mma>
      %47 = arith.mulf %26#0, %46 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma>
      %48 = arith.truncf %47 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma>
      %49 = triton_gpu.convert_layout %48 : tensor<128x256xbf16, #mma> -> tensor<128x256xbf16, #blocked2>
      tt.experimental_descriptor_store %arg2[%24, %25], %49 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<i8, 0>, tensor<128x256xbf16, #blocked2>
    }
    tt.return
  }
}
