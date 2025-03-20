// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-data-partition=num-consumer-groups=2 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_ws_cooperative_kernel
// CHECK: %[[#GA1:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
// CHECK: %[[#GA2:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
// CHECK: %[[#LA1:]] = triton_gpu.local_alloc %[[#GA1]]
// CHECK: %[[#LA2:]] = triton_gpu.local_alloc %[[#GA2]]
// CHECK: %[[#GB:]] = tt.load {{.*}} : tensor<64x256x!tt.ptr<f16>
// CHECK: %[[#LB:]] = triton_gpu.local_alloc %[[#GB]]
// CHECK: %[[#C1:]] = triton_nvidia_gpu.warp_group_dot %[[#LA1]], %[[#LB]], {{.*}} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x256xf32, #mma>
// CHECK: %[[#C2:]] = triton_nvidia_gpu.warp_group_dot %[[#LA2]], %[[#LB]], {{.*}} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x256xf32, #mma>
// CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
// CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>



#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_ws_cooperative_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<64> : tensor<128x64xi32, #blocked>
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c255_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 255 : i32
    %c63_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i32
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
    %cst_0 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_1 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    %cst_2 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.divsi %0, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.addi %arg4, %c255_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.divsi %2, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = arith.muli %1, %3 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %5 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %6 = tt.get_num_programs x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %7 = arith.muli %3, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %8 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %10 = tt.splat %arg3 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.make_range {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %12 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %13 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked>
    %14 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.expand_dims %14 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %16 = tt.broadcast %15 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %17 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %18 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.expand_dims %18 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %20 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %21 = arith.muli %19, %20 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %22 = tt.broadcast %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %23 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %24 = arith.addi %arg5, %c63_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %25 = arith.divsi %24, %c64_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %26 = tt.expand_dims %14 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %27 = tt.expand_dims %18 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %28 = arith.muli %arg7, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %29 = tt.splat %28 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x256xi32, #blocked1>
    %30 = tt.splat %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %31 = tt.splat %arg2 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %32 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %33 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<1x256xi32, #blocked1>
    scf.for %arg9 = %5 to %4 step %6  : i32 {
      %34 = arith.divsi %arg9, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %35 = arith.muli %34, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %36 = arith.subi %1, %35 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %37 = arith.minsi %36, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %38 = arith.remsi %arg9, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %39 = arith.remsi %38, %37 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %40 = arith.addi %35, %39 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %41 = arith.divsi %38, %37 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %42 = arith.muli %40, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %43 = tt.splat %42 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %44 = tt.splat %42 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %45 = arith.addi %43, %8 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %46 = arith.addi %44, %9 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %47 = arith.remsi %45, %10 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %48 = arith.muli %41, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %49 = tt.splat %48 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %50 = arith.addi %49, %11 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %51 = arith.remsi %50, %12 {async_task_id = dense<0> : vector<1xi32>} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %52 = tt.expand_dims %47 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %53 = arith.muli %52, %13 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked>
      %54 = tt.broadcast %53 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %55 = arith.addi %54, %16 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64xi32, #blocked>
      %56 = tt.addptr %17, %55 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %57 = tt.expand_dims %51 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %58 = tt.broadcast %57 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
      %59 = arith.addi %22, %58 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256xi32, #blocked1>
      %60 = tt.addptr %23, %59 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %true = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
      %false = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
      %61:3 = scf.for %arg10 = %c0_i32 to %25 step %c1_i32 iter_args(%arg11 = %cst_2, %arg12 = %56, %arg13 = %60) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>)  : i32 {
        %76 = arith.muli %arg10, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
        %77 = arith.subi %arg5, %76 {async_task_id = dense<0> : vector<1xi32>} : i32
        %78 = tt.splat %77 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x64xi32, #blocked>
        %79 = arith.cmpi slt, %26, %78 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked>
        %80 = tt.broadcast %79 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
        %81 = tt.load %arg12, %80, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>
        %82 = triton_gpu.local_alloc %81 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
        %83 = tt.splat %77 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
        %84 = arith.cmpi slt, %27, %83 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
        %85 = tt.broadcast %84 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
        %86 = tt.load %arg13, %85, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        %87 = triton_gpu.local_alloc %86 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>
        %88 = triton_nvidia_gpu.warp_group_dot %82, %87, %arg11 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        %89 = tt.addptr %arg12, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
        %90 = tt.addptr %arg13, %29 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
        scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %88, %89, %90 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>
      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
      %62 = arith.truncf %61#0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %63 = tt.expand_dims %46 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %64 = arith.muli %30, %63 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked1>
      %65 = tt.addptr %31, %64 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
      %66 = tt.expand_dims %50 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %67 = tt.broadcast %65 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      %68 = tt.broadcast %66 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %69 = tt.addptr %67, %68 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
      %70 = arith.cmpi slt, %63, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked1>
      %71 = arith.cmpi slt, %66, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi32, #blocked1>
      %72 = tt.broadcast %70 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
      %73 = tt.broadcast %71 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
      %74 = arith.andi %72, %73 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xi1, #blocked1>
      %75 = triton_gpu.convert_layout %62 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.store %69, %75, %74 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @cross_dim_partition
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @cross_dim_partition(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} 0 : i32
    %c16384_i32 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} 16384 : i32
    %cst_0 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<true> : tensor<128x128xi1, #blocked>
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
    %c384_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 384 : i32
    %c2_i64 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 2 : i64
    %c64_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 64 : i32
    %c512_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 512 : i32
    %c4_i64 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 4 : i64
    %cst_1 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<128> : tensor<1x128xi32, #blocked>
    %0 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = tt.get_program_id y {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.divsi %1, %arg8 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.remsi %1, %arg8 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = tt.addptr %arg1, %2 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i32>, i32
    %5 = tt.load %4 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i32>
    %6 = tt.addptr %4, %c1_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i32>, i32
    %7 = tt.load %6 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i32>
    %8 = arith.subi %7, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %9 = arith.minsi %8, %arg10 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %10 = arith.muli %2, %c512_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %11 = tt.addptr %arg6, %10 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i8>, i32
    %12 = tt.addptr %11, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8>, i32
    %13 = tt.addptr %11, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8>, i32
    %14 = tt.addptr %11, %c384_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i8>, i32
    %15 = arith.muli %arg8, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %16 = arith.extsi %15 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
    %17 = arith.muli %16, %c2_i64 {async_task_id = dense<0> : vector<1xi32>} : i64
    %18 = arith.shrsi %17, %c4_i64 {async_task_id = dense<0> : vector<1xi32>} : i64
    tt.experimental_tensormap_create %11, %arg0, [%c64_i32, %c64_i32], [%15, %7], [%18], [%c1_i32, %c1_i32] {async_task_id = dense<0> : vector<1xi32>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %19 = arith.muli %arg9, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    tt.experimental_tensormap_create %12, %arg2, [%c64_i32, %c128_i32], [%15, %19], [%18], [%c1_i32, %c1_i32] {async_task_id = dense<0> : vector<1xi32>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.experimental_tensormap_create %13, %arg3, [%c64_i32, %c64_i32], [%15, %7], [%18], [%c1_i32, %c1_i32] {async_task_id = dense<0> : vector<1xi32>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.experimental_tensormap_create %14, %arg5, [%c64_i32, %c64_i32], [%15, %7], [%18], [%c1_i32, %c1_i32] {async_task_id = dense<0> : vector<1xi32>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %20 = arith.muli %0, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %21 = arith.cmpi slt, %20, %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    scf.if %21 {
      %22 = arith.muli %0, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %23 = arith.addi %5, %22 {async_task_id = dense<0> : vector<1xi32>} : i32
      %24 = arith.muli %3, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<64x128xbf16
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<64x128xbf16
      %25 = tt.experimental_descriptor_load %11[%23, %24] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8> -> tensor<128x128xbf16, #blocked1>
      %26 = triton_gpu.local_alloc %25 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #blocked1>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %27 = arith.muli %2, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<128x128xbf16
      %28 = tt.experimental_descriptor_load %12[%27, %24] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8> -> tensor<128x128xbf16, #blocked1>
      %29 = triton_gpu.local_alloc %28 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #blocked1>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      %30 = triton_nvidia_gpu.warp_group_dot %26, %29, %cst {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %31 = arith.truncf %30 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      %32 = triton_gpu.local_alloc %31 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #mma>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<64x128xbf16
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<64x128xbf16
      %33 = tt.experimental_descriptor_load %13[%23, %24] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8> -> tensor<128x128xbf16, #blocked1>
      %34 = triton_gpu.local_alloc %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #blocked1>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %35 = tt.trans %34 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<128x64xbf16, {{.*}} * !tt.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<128x64xbf16, {{.*}} * !tt.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
      %36 = triton_nvidia_gpu.warp_group_dot %35, %32, %cst {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %37 = triton_gpu.convert_layout %36 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
      %38 = arith.truncf %37 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %39 = arith.muli %27, %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %40 = arith.muli %39, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %41 = tt.addptr %arg4, %40 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16>, i32
      %42 = arith.muli %arg8, %c16384_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %43 = tt.addptr %41, %42 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16>, i32
      %44 = arith.muli %3, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %45 = tt.addptr %43, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16>, i32
      %46 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %47 = tt.expand_dims %46 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %48 = tt.splat %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<1x128xi32, #blocked>
      %49 = arith.muli %47, %48 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked>
      %50 = arith.muli %49, %cst_1 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked>
      %51 = tt.splat %45 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16> -> tensor<1x128x!tt.ptr<bf16>, #blocked>
      %52 = tt.addptr %51, %50 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128x!tt.ptr<bf16>, #blocked>, tensor<1x128xi32, #blocked>
      %53 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %54 = tt.expand_dims %53 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %55 = tt.broadcast %52 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %56 = tt.broadcast %54 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %57 = tt.addptr %55, %56 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %58 = tt.atomic_rmw fadd, relaxed, gpu, %57, %38, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xbf16, #blocked>, tensor<128x128xi1, #blocked>) -> tensor<128x128xbf16, #blocked>
      %59 = arith.addi %27, %c128_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<128x128xbf16
      %60 = tt.experimental_descriptor_load %12[%59, %24] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8> -> tensor<128x128xbf16, #blocked1>
      %61 = triton_gpu.local_alloc %60 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #blocked1>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %62 = tt.trans %61 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      %63 = triton_nvidia_gpu.warp_group_dot %34, %62, %cst {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %64 = arith.truncf %63 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
      // CHECK: tt.experimental_descriptor_load {{.*}} -> tensor<128x128xbf16
      %65 = tt.experimental_descriptor_load %12[%27, %24] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8> -> tensor<128x128xbf16, #blocked1>
      %66 = triton_gpu.local_alloc %65 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #blocked1>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %67 = tt.trans %66 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      %68 = triton_gpu.convert_layout %64 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : tensor<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : tensor<64x128xbf16, {{.*}} * !tt.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
      %69 = triton_nvidia_gpu.warp_group_dot %68, %67, %cst {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : tensor<128x128xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %70 = triton_gpu.convert_layout %69 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
      %71 = arith.truncf %70 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #blocked1> to tensor<128x128xbf16, #blocked1>
      %72 = arith.muli %0, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %73 = arith.addi %5, %72 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      %74 = arith.muli %3, %c128_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
      tt.experimental_descriptor_store %14[%73, %74], %71 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<i8>, tensor<128x128xbf16, #blocked1>
      %75 = triton_gpu.local_alloc %64 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128xbf16, #mma>) -> !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory>
      %76 = tt.trans %75 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory>
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<128x64xbf16, {{.*}} * !tt.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
      // CHECK: triton_nvidia_gpu.warp_group_dot {{.*}} : !tt.memdesc<128x64xbf16, {{.*}} * !tt.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
      %77 = triton_nvidia_gpu.warp_group_dot %76, %26, %cst {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x128xbf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<128x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %78 = triton_gpu.convert_layout %77 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
      %79 = arith.truncf %78 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %80 = tt.addptr %41, %c0_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16>, i32
      %81 = tt.addptr %80, %44 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16>, i32
      %82 = tt.splat %81 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<bf16> -> tensor<1x128x!tt.ptr<bf16>, #blocked>
      %83 = tt.addptr %82, %50 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128x!tt.ptr<bf16>, #blocked>, tensor<1x128xi32, #blocked>
      %84 = tt.broadcast %83 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
      %85 = tt.addptr %84, %56 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %86 = tt.atomic_rmw fadd, relaxed, gpu, %85, %79, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xbf16, #blocked>, tensor<128x128xi1, #blocked>) -> tensor<128x128xbf16, #blocked>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    tt.return
  }
}
