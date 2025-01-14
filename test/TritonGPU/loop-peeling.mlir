// RUN: triton-opt %s -split-input-file -tritongpu-matmul-loop-peeling | FileCheck %s

// CHECK-LABEL: @matmul_kernel
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.warp_group_dot


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c128_i32 : i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %13, %c256_i32 : i32
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %30 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked>
    %31 = arith.muli %29, %30 : tensor<128x1xi32, #blocked>
    %32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %36 = arith.addi %34, %35 : tensor<128x64xi32, #blocked>
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %38 = tt.addptr %37, %36 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %39 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %41 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked1>
    %42 = arith.muli %40, %41 : tensor<64x1xi32, #blocked1>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %45 = tt.broadcast %43 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %46 = arith.addi %44, %45 : tensor<64x256xi32, #blocked1>
    %47 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %48 = tt.addptr %47, %46 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
    %49 = arith.addi %arg5, %c63_i32 : i32
    %50 = arith.divsi %49, %c64_i32 : i32
    %51 = arith.cmpi sgt, %50, %c0_i32 : i32
    "llvm.intr.assume"(%51) : (i1) -> ()
    %52 = arith.muli %arg7, %c64_i32 : i32
    %53 = tt.splat %52 : i32 -> tensor<64x256xi32, #blocked1>
    %54 = triton_gpu.local_alloc  : () -> !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %55 = triton_gpu.local_alloc  : () -> !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %56 = arith.cmpi sgt, %50, %c0_i32 : i32
    %57 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %58 = arith.cmpi slt, %33, %57 : tensor<1x64xi32, #blocked>
    %59 = tt.broadcast %58 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %60 = triton_gpu.memdesc_subview %54[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %61 = tt.splat %56 : i1 -> tensor<128x64xi1, #blocked>
    %62 = arith.andi %61, %59 : tensor<128x64xi1, #blocked>
    %63 = triton_gpu.async_copy_global_to_local %38, %60 mask %62 other %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %64 = triton_gpu.async_commit_group %63
    %65 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked1>
    %66 = arith.cmpi slt, %40, %65 : tensor<64x1xi32, #blocked1>
    %67 = tt.broadcast %66 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
    %68 = triton_gpu.memdesc_subview %55[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %69 = tt.splat %56 : i1 -> tensor<64x256xi1, #blocked1>
    %70 = arith.andi %69, %67 : tensor<64x256xi1, #blocked1>
    %71 = triton_gpu.async_copy_global_to_local %48, %68 mask %70 other %cst_1 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %72 = triton_gpu.async_commit_group %71
    %73 = arith.cmpi sgt, %50, %c1_i32 : i32
    %74 = tt.addptr %38, %cst : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %75 = tt.addptr %48, %53 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
    %76 = arith.subi %arg5, %c64_i32 : i32
    %77 = tt.splat %76 : i32 -> tensor<1x64xi32, #blocked>
    %78 = arith.cmpi slt, %33, %77 : tensor<1x64xi32, #blocked>
    %79 = tt.broadcast %78 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %80 = triton_gpu.memdesc_subview %54[%c1_i32, %c0_i32, %c0_i32] : !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %81 = tt.splat %73 : i1 -> tensor<128x64xi1, #blocked>
    %82 = arith.andi %81, %79 : tensor<128x64xi1, #blocked>
    %83 = triton_gpu.async_copy_global_to_local %74, %80 mask %82 other %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %84 = triton_gpu.async_commit_group %83
    %85 = tt.splat %76 : i32 -> tensor<64x1xi32, #blocked1>
    %86 = arith.cmpi slt, %40, %85 : tensor<64x1xi32, #blocked1>
    %87 = tt.broadcast %86 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
    %88 = triton_gpu.memdesc_subview %55[%c1_i32, %c0_i32, %c0_i32] : !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %89 = tt.splat %73 : i1 -> tensor<64x256xi1, #blocked1>
    %90 = arith.andi %89, %87 : tensor<64x256xi1, #blocked1>
    %91 = triton_gpu.async_copy_global_to_local %75, %88 mask %90 other %cst_1 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %92 = triton_gpu.async_commit_group %91
    %93:7 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %74, %arg12 = %75, %arg13 = %c1_i32, %arg14 = %c-1_i32, %arg15 = %72, %arg16 = %92) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>, i32, i32, !triton_gpu.async.token, !triton_gpu.async.token)  : i32 {
      %114 = arith.subi %50, %c2_i32 : i32
      %115 = arith.cmpi slt, %arg9, %114 : i32
      %116 = arith.addi %arg14, %c1_i32 : i32
      %117 = arith.cmpi slt, %116, %c3_i32 : i32
      %118 = arith.select %117, %116, %c0_i32 : i32
      %119 = triton_gpu.memdesc_subview %54[%118, %c0_i32, %c0_i32] : !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      %120 = triton_gpu.async_wait %arg15 {num = 2 : i32}
      %121 = triton_gpu.memdesc_subview %55[%118, %c0_i32, %c0_i32] : !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
      %122 = triton_nvidia_gpu.warp_group_dot %119, %121, %arg10 {inputPrecision = 0 : i32, isAsync = true} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x256xf32, #mma>
      %123:3 = triton_nvidia_gpu.warp_group_dot_wait %122, %119, %121 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
      %124 = tt.addptr %arg11, %cst : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %125 = tt.addptr %arg12, %53 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %126 = arith.addi %arg13, %c1_i32 : i32
      %127 = arith.cmpi slt, %126, %c3_i32 : i32
      %128 = arith.select %127, %126, %c0_i32 : i32
      %129 = arith.addi %arg9, %c2_i32 : i32
      %130 = arith.muli %129, %c64_i32 : i32
      %131 = arith.subi %arg5, %130 : i32
      %132 = tt.splat %131 : i32 -> tensor<1x64xi32, #blocked>
      %133 = arith.cmpi slt, %33, %132 : tensor<1x64xi32, #blocked>
      %134 = tt.broadcast %133 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
      %135 = triton_gpu.memdesc_subview %54[%128, %c0_i32, %c0_i32] : !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      %136 = tt.splat %115 : i1 -> tensor<128x64xi1, #blocked>
      %137 = arith.andi %136, %134 : tensor<128x64xi1, #blocked>
      %138 = triton_gpu.async_copy_global_to_local %124, %135 mask %137 other %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      %139 = triton_gpu.async_commit_group %138
      %140 = tt.splat %131 : i32 -> tensor<64x1xi32, #blocked1>
      %141 = arith.cmpi slt, %40, %140 : tensor<64x1xi32, #blocked1>
      %142 = tt.broadcast %141 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
      %143 = triton_gpu.memdesc_subview %55[%128, %c0_i32, %c0_i32] : !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
      %144 = tt.splat %115 : i1 -> tensor<64x256xi1, #blocked1>
      %145 = arith.andi %144, %142 : tensor<64x256xi1, #blocked1>
      %146 = triton_gpu.async_copy_global_to_local %125, %143 mask %145 other %cst_1 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
      %147 = triton_gpu.async_commit_group %146
      scf.yield %123#0, %124, %125, %128, %118, %arg16, %147 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>, i32, i32, !triton_gpu.async.token, !triton_gpu.async.token
    }
    %94 = triton_nvidia_gpu.warp_group_dot_wait %93#0 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %95 = triton_gpu.async_wait  {num = 0 : i32}
    triton_gpu.local_dealloc %54 : !tt.memdesc<3x128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %55 : !tt.memdesc<3x64x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %96 = arith.truncf %94 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %97 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %98 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %99 = arith.muli %98, %97 : tensor<128x1xi32, #blocked1>
    %100 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %101 = tt.addptr %100, %99 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %102 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %103 = tt.broadcast %101 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %104 = tt.broadcast %102 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %105 = tt.addptr %103, %104 : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
    %106 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %107 = arith.cmpi slt, %97, %106 : tensor<128x1xi32, #blocked1>
    %108 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked1>
    %109 = arith.cmpi slt, %102, %108 : tensor<1x256xi32, #blocked1>
    %110 = tt.broadcast %107 : tensor<128x1xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
    %111 = tt.broadcast %109 : tensor<1x256xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
    %112 = arith.andi %110, %111 : tensor<128x256xi1, #blocked1>
    %113 = triton_gpu.convert_layout %96 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.store %105, %113, %112 : tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
