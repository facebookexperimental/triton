// RUN: triton-opt -split-input-file --tlx-propagate-layout %s| FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 8]}>
// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_kernel_tma_pipelined_hopper
  tt.func public @matmul_kernel_tma_pipelined_hopper(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked2>
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
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %20 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %21 = arith.addi %18, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %22 = arith.addi %19, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %23 = arith.addi %20, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %24 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %25 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %26 = arith.remsi %21, %24 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %27 = arith.remsi %22, %25 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %28 = arith.muli %13, %c256_i32 : i32
    %29 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %30 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.splat %28 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %32 = tt.splat %28 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %33 = arith.addi %31, %29 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %34 = arith.addi %32, %30 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %35 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %36 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %37 = arith.remsi %33, %35 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %38 = arith.remsi %34, %36 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %39 = tt.expand_dims %26 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %40 = tt.expand_dims %27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %41 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2>
    %42 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %43 = arith.muli %39, %41 : tensor<128x1xi32, #blocked2>
    %44 = arith.muli %40, %42 : tensor<128x1xi32, #blocked1>
    %45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %46 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %47 = tt.expand_dims %45 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %48 = tt.expand_dims %46 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %49 = tt.broadcast %43 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
    %50 = tt.broadcast %44 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %51 = tt.broadcast %47 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
    %52 = tt.broadcast %48 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %53 = arith.addi %49, %51 : tensor<128x64xi32, #blocked2>
    %54 = arith.addi %50, %52 : tensor<128x64xi32, #blocked1>
    %55 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
    %56 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %57 = tt.addptr %55, %53 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
    %58 = tt.addptr %56, %54 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %59 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %60 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %61 = tt.expand_dims %59 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %62 = tt.expand_dims %60 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %63 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked3>
    %64 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked>
    %65 = arith.muli %61, %63 : tensor<64x1xi32, #blocked3>
    %66 = arith.muli %62, %64 : tensor<64x1xi32, #blocked>
    %67 = tt.expand_dims %37 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %68 = tt.expand_dims %38 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %69 = tt.broadcast %65 : tensor<64x1xi32, #blocked3> -> tensor<64x256xi32, #blocked3>
    %70 = tt.broadcast %66 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %71 = tt.broadcast %67 : tensor<1x256xi32, #blocked3> -> tensor<64x256xi32, #blocked3>
    %72 = tt.broadcast %68 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %73 = arith.addi %69, %71 : tensor<64x256xi32, #blocked3>
    %74 = arith.addi %70, %72 : tensor<64x256xi32, #blocked>
    %75 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked3>
    %76 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %77 = tt.addptr %75, %73 : tensor<64x256x!tt.ptr<f16>, #blocked3>, tensor<64x256xi32, #blocked3>
    %78 = tt.addptr %76, %74 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #[[$SHARED]], #smem, mutable>
    %79 = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x256xf16, #[[$SHARED]], #smem, mutable>
    %80 = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable>
    %81 = arith.muli %arg7, %c64_i32 : i32
    %82 = tt.splat %81 : i32 -> tensor<64x256xi32, #blocked3>
    %83 = tt.splat %81 : i32 -> tensor<64x256xi32, #blocked>
    %84:4 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %58, %arg11 = %78, %arg12 = %57, %arg13 = %77) -> (tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x256x!tt.ptr<f16>, #blocked3>)  : i32 {
      %107 = ttg.memdesc_subview %79[%arg9, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %108 = ttg.memdesc_subview %80[%arg9, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %109 = arith.muli %arg9, %c64_i32 : i32
      %110 = arith.subi %arg5, %109 : i32
      %111 = tt.splat %110 : i32 -> tensor<1x64xi32, #blocked2>
      %112 = arith.cmpi slt, %47, %111 : tensor<1x64xi32, #blocked2>
      %113 = tt.broadcast %112 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
      %114 = ttg.async_copy_global_to_local %arg12, %107 mask %113 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> <128x64xf16, #shared, #smem, mutable>
      %115 = tt.splat %110 : i32 -> tensor<64x1xi32, #blocked3>
      %116 = arith.cmpi slt, %61, %115 : tensor<64x1xi32, #blocked3>
      %117 = tt.broadcast %116 : tensor<64x1xi1, #blocked3> -> tensor<64x256xi1, #blocked3>
      %118 = ttg.async_copy_global_to_local %arg13, %108 mask %117 : tensor<64x256x!tt.ptr<f16>, #blocked3> -> <64x256xf16, #shared, #smem, mutable>
      %119 = tt.addptr %arg12, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
      %120 = tt.addptr %arg10, %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %121 = tt.addptr %arg13, %82 : tensor<64x256x!tt.ptr<f16>, #blocked3>, tensor<64x256xi32, #blocked3>
      %122 = tt.addptr %arg11, %83 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %123 = ttg.async_commit_group
      scf.yield %120, %122, %119, %121 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x256x!tt.ptr<f16>, #blocked3>
    }
    %85 = arith.addi %arg5, %c63_i32 : i32
    %86 = arith.divsi %85, %c64_i32 : i32
    %87:3 = scf.for %arg9 = %c0_i32 to %86 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %84#0, %arg12 = %84#1) -> (tensor<128x256xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>)  : i32 {
      %107 = arith.remsi %arg9, %c2_i32 : i32
      %108 = ttg.memdesc_subview %79[%107, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %109 = ttg.memdesc_subview %80[%107, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      %110 = ttg.async_wait  {num = 0 : i32}
      // CHECK-NOT: tlx.require_layout
      %111 = tlx.require_layout %108 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      %112 = tlx.require_layout %109 : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      // CHECK: ttg.convert_layout %arg10 : tensor<128x256xf32, #blocked> -> tensor<128x256xf32, #mma>
      %113 = tlx.require_layout %arg10 : tensor<128x256xf32, #blocked> -> tensor<128x256xf32, #mma>
      ttng.fence_async_shared {bCluster = false}
      %114 = ttng.warp_group_dot %111, %112, %113 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %115:3 = ttng.warp_group_dot_wait %114, %111, %112 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %116 = arith.addi %arg9, %c2_i32 : i32
      %117 = arith.remsi %116, %c2_i32 : i32
      %118 = ttg.memdesc_subview %79[%117, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %119 = ttg.memdesc_subview %80[%117, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
      // CHECK: %[[WARP_GROUP_DOT_WAIT:.*]] = ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32} : tensor<128x256xf32, #mma>
      // CHECK: ttg.convert_layout %[[WARP_GROUP_DOT_WAIT]] : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>
      %120 = ttng.warp_group_dot_wait %115#0 {pendings = 1 : i32} : tensor<128x256xf32, #mma>
      %121 = tlx.release_layout %120 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>
      %122 = arith.muli %116, %c64_i32 : i32
      %123 = arith.subi %arg5, %122 : i32
      %124 = tt.splat %123 : i32 -> tensor<1x64xi32, #blocked2>
      %125 = arith.cmpi slt, %47, %124 : tensor<1x64xi32, #blocked2>
      %126 = tt.broadcast %125 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
      %127 = ttg.convert_layout %arg11 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
      %128 = ttg.async_copy_global_to_local %127, %118 mask %126 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> <128x64xf16, #shared, #smem, mutable>
      %129 = tt.splat %123 : i32 -> tensor<64x1xi32, #blocked3>
      %130 = arith.cmpi slt, %61, %129 : tensor<64x1xi32, #blocked3>
      %131 = tt.broadcast %130 : tensor<64x1xi1, #blocked3> -> tensor<64x256xi1, #blocked3>
      %132 = ttg.convert_layout %arg12 : tensor<64x256x!tt.ptr<f16>, #blocked> -> tensor<64x256x!tt.ptr<f16>, #blocked3>
      %133 = ttg.async_copy_global_to_local %132, %119 mask %131 : tensor<64x256x!tt.ptr<f16>, #blocked3> -> <64x256xf16, #shared, #smem, mutable>
      %134 = tt.addptr %arg11, %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %135 = tt.addptr %arg12, %83 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      scf.yield %121, %134, %135 : tensor<128x256xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>
    }
    %88 = ttng.warp_group_dot_wait %87#0 {pendings = 0 : i32} : tensor<128x256xf32, #blocked>
    %89 = arith.truncf %88 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>
    %90 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
    %91 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked3>
    %92 = arith.muli %91, %90 : tensor<128x1xi32, #blocked3>
    %93 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked3>
    %94 = tt.addptr %93, %92 : tensor<128x1x!tt.ptr<f16>, #blocked3>, tensor<128x1xi32, #blocked3>
    %95 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %96 = tt.broadcast %94 : tensor<128x1x!tt.ptr<f16>, #blocked3> -> tensor<128x256x!tt.ptr<f16>, #blocked3>
    %97 = tt.broadcast %95 : tensor<1x256xi32, #blocked3> -> tensor<128x256xi32, #blocked3>
    %98 = tt.addptr %96, %97 : tensor<128x256x!tt.ptr<f16>, #blocked3>, tensor<128x256xi32, #blocked3>
    %99 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked3>
    %100 = arith.cmpi slt, %90, %99 : tensor<128x1xi32, #blocked3>
    %101 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked3>
    %102 = arith.cmpi slt, %95, %101 : tensor<1x256xi32, #blocked3>
    %103 = tt.broadcast %100 : tensor<128x1xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %104 = tt.broadcast %102 : tensor<1x256xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
    %105 = arith.andi %103, %104 : tensor<128x256xi1, #blocked3>
    %106 = ttg.convert_layout %89 : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #blocked3>
    tt.store %98, %106, %105 : tensor<128x256x!tt.ptr<f16>, #blocked3>
    tt.return
  }
}

// -----
// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @ws_tma
  tt.func public @ws_tma(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.extsi %arg3 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%2, %c1_i64] : <i16>, <tensor<64x64xsi16>>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x64xi16, #[[$SHARED]], #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable>
    %5 = ttg.memdesc_subview %4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable
    %6 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %7 = ttg.memdesc_subview %6[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_subview %6[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %7, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %9 = arith.muli %0, %c64_i32 : i32
    %10 = arith.muli %1, %c64_i32 : i32
    ttg.warp_specialize(%7)
    default {
      ttng.wait_barrier %8, %c1_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      %11 = tlx.require_layout %5 : !ttg.memdesc<64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
      %12 = ttng.tensor_desc_to_tma_ptr %3 : !tt.tensordesc<tensor<64x64xsi16>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %12[%9, %10] %11, %7, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg4: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      %c0_i32_0 = arith.constant 0 : i32
      ttng.wait_barrier %arg4, %c0_i32_0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @require_layout_on_tensor
  tt.func public @require_layout_on_tensor(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #blocked> attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %2 = ttg.local_load %1 : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked1>
    // CHECK-NOT: tlx.require_layout
    // CHECK: ttg.convert_layout %{{.*}} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    %3 = tlx.require_layout %2 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    tt.return %3 : tensor<64x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @_attn_fwd
  tt.func public @_attn_fwd(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.tensordesc<tensor<128x128xf16>>, %arg5: i32, %arg6: i32, %arg7: i64, %arg8: i64, %arg9: !tt.tensordesc<tensor<64x128xf16>>, %arg10: i32, %arg11: i32, %arg12: i64, %arg13: i64, %arg14: !tt.tensordesc<tensor<64x128xf16>>, %arg15: i32, %arg16: i32, %arg17: i64, %arg18: i64, %arg19: !tt.tensordesc<tensor<128x128xf16>>, %arg20: i32, %arg21: i32, %arg22: i64, %arg23: i64, %arg24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #[[$SHARED]], #smem, mutable>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x64x128xf16, #[[$SHARED]], #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x64x128xf16, #[[$SHARED]], #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %4 = ttg.memdesc_subview %3[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %6 = ttg.memdesc_subview %5[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %7 = ttg.memdesc_subview %5[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_subview %5[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %10 = ttg.memdesc_subview %9[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %11 = ttg.memdesc_subview %9[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %12 = ttg.memdesc_subview %9[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %12, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %14 = ttg.memdesc_subview %13[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %15 = ttg.memdesc_subview %13[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %16 = ttg.memdesc_subview %13[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %18 = ttg.memdesc_subview %17[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %19 = ttg.memdesc_subview %17[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %20 = ttg.memdesc_subview %17[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%arg3, %arg1, %arg24, %cst_1, %arg19, %5, %9, %1, %cst, %cst_0, %3, %0, %arg0, %13, %17, %2)
    default {
      %21 = tt.get_program_id x : i32
      %22 = tt.get_program_id y : i32
      %23 = arith.divsi %22, %arg3 : i32
      %24 = arith.remsi %22, %arg3 : i32
      %25 = arith.muli %arg24, %arg3 : i32
      %26 = arith.muli %23, %25 : i32
      %27 = arith.muli %24, %arg24 : i32
      %28 = arith.addi %26, %27 : i32
      %29 = arith.muli %21, %c128_i32 : i32
      %30 = arith.addi %28, %29 : i32
      ttng.barrier_expect %4, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %31 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      %32 = tlx.require_layout %31 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      %33 = ttng.tensor_desc_to_tma_ptr %arg4 : !tt.tensordesc<tensor<128x128xf16>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %33[%30, %c0_i32] %32, %4, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      %34:2 = scf.for %arg25 = %c0_i32 to %arg24 step %c64_i32 iter_args(%arg26 = %28, %arg27 = %c0_i32) -> (i32, i32)  : i32 {
        %35 = arith.remsi %arg25, %c3_i32 : i32
        %36 = ttg.memdesc_subview %5[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %37 = arith.xori %arg27, %c1_i32 : i32
        ttng.wait_barrier %36, %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %38 = ttg.memdesc_subview %9[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %39 = ttg.memdesc_subview %1[%35, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.barrier_expect %38, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %40 = tlx.require_layout %39 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %41 = ttng.tensor_desc_to_tma_ptr %arg9 : !tt.tensordesc<tensor<64x128xf16>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %41[%arg26, %c0_i32] %40, %38, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %42 = ttg.memdesc_subview %13[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %42, %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %43 = ttg.memdesc_subview %17[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %44 = ttg.memdesc_subview %2[%35, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.barrier_expect %43, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %45 = tlx.require_layout %44 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %46 = ttng.tensor_desc_to_tma_ptr %arg14 : !tt.tensordesc<tensor<64x128xf16>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %46[%arg26, %c0_i32] %45, %43, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %47 = arith.addi %arg26, %c64_i32 : i32
        %48 = arith.cmpi eq, %35, %c2_i32 : i32
        %49 = scf.if %48 -> (i32) {
          scf.yield %37 : i32
        } else {
          scf.yield %arg27 : i32
        }
        scf.yield %47, %49 : i32, i32
      }
      ttg.warp_yield
    }
    partition0(%arg25: i32, %arg26: !tt.ptr<f32>, %arg27: i32, %arg28: tensor<128x128xf32, #blocked1>, %arg29: !tt.tensordesc<tensor<128x128xf16>>, %arg30: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg31: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>, %arg33: tensor<128xf32, #blocked>, %arg34: tensor<128xf32, #blocked>, %arg35: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, %arg37: f32, %arg38: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>) num_warps(4) {
      %c64_i32_2 = arith.constant 64 : i32
      %c128_i32_3 = arith.constant 128 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c2_i32_5 = arith.constant 2 : i32
      %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked2>
      %c3_i32_7 = arith.constant 3 : i32
      %c0_i32_8 = arith.constant 0 : i32
      %cst_9 = arith.constant 1.44269502 : f32
      %21 = arith.mulf %arg37, %cst_9 : f32
      %22 = ttg.memdesc_subview %arg35[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %22, %c0_i32_8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %23 = ttg.memdesc_subview %arg36[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %24:4 = scf.for %arg41 = %c0_i32_8 to %arg27 step %c64_i32_2 iter_args(%arg42 = %arg28, %arg43 = %arg33, %arg44 = %arg34, %arg45 = %c0_i32_8) -> (tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, i32)  : i32 {
        %53 = arith.remsi %arg41, %c3_i32_7 : i32
        %54 = ttg.memdesc_subview %arg31[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %54, %arg45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %55 = ttg.memdesc_subview %arg32[%53, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %56 = ttg.memdesc_trans %55 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %57 = tlx.require_layout %23 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %58 = tlx.require_layout %56 : !ttg.memdesc<128x64xf16, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        // CHECK: ttg.convert_layout %{{.+}}
        %59 = tlx.require_layout %cst_6 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #mma>
        %60 = ttng.warp_group_dot %57, %58, %59 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xf16, #shared2, #smem, mutable> * !ttg.memdesc<128x64xf16, #shared4, #smem, mutable> -> tensor<128x64xf32, #mma>
        %61 = ttng.warp_group_dot_wait %60 {pendings = 0 : i32} : tensor<128x64xf32, #mma>
        %62 = tlx.release_layout %61 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked2>
        %63 = ttg.memdesc_subview %arg30[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.arrive_barrier %63, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %64 = "tt.reduce"(%62) <{axis = 1 : i32}> ({
        ^bb0(%arg46: f32, %arg47: f32):
          %102 = arith.maxnumf %arg46, %arg47 : f32
          tt.reduce.return %102 : f32
        }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %65 = ttg.convert_layout %64 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128xf32, #blocked>
        %66 = tt.splat %21 : f32 -> tensor<128xf32, #blocked>
        %67 = arith.mulf %65, %66 : tensor<128xf32, #blocked>
        %68 = arith.maxnumf %arg44, %67 : tensor<128xf32, #blocked>
        %69 = tt.splat %21 : f32 -> tensor<128x64xf32, #blocked2>
        %70 = arith.mulf %62, %69 : tensor<128x64xf32, #blocked2>
        %71 = ttg.convert_layout %68 : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %73 = ttg.convert_layout %72 : tensor<128x1xf32, #blocked3> -> tensor<128x1xf32, #blocked4>
        %74 = tt.broadcast %73 : tensor<128x1xf32, #blocked4> -> tensor<128x64xf32, #blocked4>
        %75 = ttg.convert_layout %74 : tensor<128x64xf32, #blocked4> -> tensor<128x64xf32, #blocked2>
        %76 = arith.subf %70, %75 : tensor<128x64xf32, #blocked2>
        %77 = math.exp2 %76 : tensor<128x64xf32, #blocked2>
        %78 = arith.subf %arg44, %68 : tensor<128xf32, #blocked>
        %79 = math.exp2 %78 : tensor<128xf32, #blocked>
        %80 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
        ^bb0(%arg46: f32, %arg47: f32):
          %102 = arith.addf %arg46, %arg47 : f32
          tt.reduce.return %102 : f32
        }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %81 = ttg.convert_layout %80 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128xf32, #blocked>
        %82 = ttg.convert_layout %79 : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %83 = tt.expand_dims %82 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %84 = ttg.convert_layout %83 : tensor<128x1xf32, #blocked3> -> tensor<128x1xf32, #blocked4>
        %85 = tt.broadcast %84 : tensor<128x1xf32, #blocked4> -> tensor<128x128xf32, #blocked4>
        %86 = ttg.convert_layout %85 : tensor<128x128xf32, #blocked4> -> tensor<128x128xf32, #blocked1>
        %87 = arith.mulf %arg42, %86 : tensor<128x128xf32, #blocked1>
        %88 = arith.truncf %77 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
        %89 = ttg.memdesc_subview %arg39[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %89, %arg45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %90 = ttg.memdesc_subview %arg40[%53, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        // CHECK: ttg.convert_layout %{{.+}}
        %91 = tlx.require_layout %90 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %92 = tlx.require_layout %87 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #mma1>
        // CHECK-NOT: tlx.require_layout
        // CHECK: ttg.convert_layout %{{.+}}
        %93 = tlx.require_layout %88 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>>
        %94 = ttng.warp_group_dot %93, %91, %92 {inputPrecision = 0 : i32, isAsync = true} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 2}>> * !ttg.memdesc<64x128xf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma1>
        %95 = ttng.warp_group_dot_wait %94 {pendings = 0 : i32} : tensor<128x128xf32, #mma1>
        %96 = tlx.release_layout %95 : tensor<128x128xf32, #mma1> -> tensor<128x128xf32, #blocked1>
        %97 = ttg.memdesc_subview %arg38[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.arrive_barrier %97, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %98 = arith.mulf %arg43, %79 : tensor<128xf32, #blocked>
        %99 = arith.addf %98, %81 : tensor<128xf32, #blocked>
        %100 = arith.cmpi eq, %53, %c2_i32_5 : i32
        %101 = scf.if %100 -> (i32) {
          %102 = arith.xori %arg45, %c1_i32_4 : i32
          scf.yield %102 : i32
        } else {
          scf.yield %arg45 : i32
        }
        scf.yield %96, %99, %68, %101 : tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, i32
      }
      %25 = tt.get_program_id x : i32
      %26 = tt.get_program_id y : i32
      %27 = arith.divsi %26, %arg25 : i32
      %28 = arith.remsi %26, %arg25 : i32
      %29 = arith.muli %arg27, %arg25 : i32
      %30 = arith.muli %27, %29 : i32
      %31 = arith.muli %28, %arg27 : i32
      %32 = arith.addi %30, %31 : i32
      %33 = arith.muli %25, %c128_i32_3 : i32
      %34 = arith.addi %32, %33 : i32
      %35 = math.log2 %24#1 : tensor<128xf32, #blocked>
      %36 = arith.addf %24#2, %35 : tensor<128xf32, #blocked>
      %37 = ttg.convert_layout %24#1 : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %38 = tt.expand_dims %37 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
      %39 = ttg.convert_layout %38 : tensor<128x1xf32, #blocked3> -> tensor<128x1xf32, #blocked4>
      %40 = tt.broadcast %39 : tensor<128x1xf32, #blocked4> -> tensor<128x128xf32, #blocked4>
      %41 = ttg.convert_layout %40 : tensor<128x128xf32, #blocked4> -> tensor<128x128xf32, #blocked1>
      %42 = arith.divf %24#0, %41 : tensor<128x128xf32, #blocked1>
      %43 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
      %44 = tt.splat %33 : i32 -> tensor<128xi32, #blocked>
      %45 = arith.addi %44, %43 : tensor<128xi32, #blocked>
      %46 = arith.muli %26, %arg27 : i32
      %47 = tt.addptr %arg26, %46 : !tt.ptr<f32>, i32
      %48 = tt.splat %47 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %49 = tt.addptr %48, %45 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %50 = ttg.convert_layout %49 : tensor<128x!tt.ptr<f32>, #blocked> -> tensor<128x!tt.ptr<f32>, #blocked>
      %51 = ttg.convert_layout %36 : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
      tt.store %50, %51 : tensor<128x!tt.ptr<f32>, #blocked>
      %52 = arith.truncf %42 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
      tt.descriptor_store %arg29[%34, %c0_i32_8], %52 : !tt.tensordesc<tensor<128x128xf16>>, tensor<128x128xf16, #blocked1>
      ttg.warp_return
    } : (i32, !tt.ptr<f32>, i32, tensor<128x128xf32, #blocked1>, !tt.tensordesc<tensor<128x128xf16>>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, f32, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>
module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gdpa_kernel_tma_ws_blackwell
  tt.func public @gdpa_kernel_tma_ws_blackwell(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: f32, %arg20: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked>
    %cst_0 = arith.constant dense<4.471500e-02> : tensor<64x128xf32, #blocked>
    %cst_1 = arith.constant dense<0.797884583> : tensor<64x128xf32, #blocked>
    %cst_2 = arith.constant dense<5.000000e-01> : tensor<64x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.addi %arg17, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = tt.get_num_programs x : i32
    %4 = arith.muli %1, %arg15 : i32
    %5 = arith.muli %4, %arg16 : i32
    %6 = arith.divsi %5, %3 : i32
    %7 = arith.remsi %5, %3 : i32
    %8 = arith.cmpi slt, %2, %7 : i32
    %9 = scf.if %8 -> (i32) {
      %55 = arith.addi %6, %c1_i32 : i32
      scf.yield %55 : i32
    } else {
      scf.yield %6 : i32
    }
    %10 = arith.muli %arg18, %arg15 : i32
    %11 = arith.muli %arg16, %c128_i32 : i32
    %12 = arith.extsi %11 : i32 to i64
    %13 = tt.make_tensor_descriptor %arg2, [%10, %11], [%12, %c1_i64] : <bf16>, <tensor<128x128xbf16>>
    %14 = tt.make_tensor_descriptor %arg4, [%10, %11], [%12, %c1_i64] : <bf16>, <tensor<128x128xbf16>>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_4 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_5 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %20 = ttg.memdesc_subview %19[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %22 = ttg.memdesc_subview %21[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %22, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %24 = ttg.memdesc_subview %23[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %24, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %25 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %26 = ttg.memdesc_subview %25[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %27 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %28 = ttg.memdesc_subview %27[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %29 = ttg.memdesc_subview %27[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %30 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %31 = ttg.memdesc_subview %30[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %32 = ttg.memdesc_subview %30[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_subview %33[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %35 = ttg.memdesc_subview %33[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %36 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %37 = ttg.memdesc_subview %36[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %38 = ttg.memdesc_subview %36[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %39 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %40 = ttg.memdesc_subview %39[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.memdesc_subview %41[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %43 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.memdesc_subview %43[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %45 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %46 = ttg.memdesc_subview %45[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %46, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.memdesc_subview %47[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %48, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %49 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %50 = ttg.memdesc_subview %49[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %50, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %51 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %52 = ttg.memdesc_subview %51[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %52, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %54 = ttg.memdesc_subview %53[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%arg16, %arg3, %arg17, %arg5, %arg0, %arg1, %27, %19, %21, %33, %23, %25, %36, %30, %17, %13, %1, %result_4, %result_5, %49, %53, %41, %45, %47, %51, %39, %43, %15, %16, %result, %result_3, %arg19, %arg10, %arg14, %arg8, %2, %9, %18, %14)
    default {
      %55:2 = scf.for %arg21 = %c0_i32 to %9 step %c1_i32 iter_args(%arg22 = %c0_i32, %arg23 = %c0_i32) -> (i32, i32)  : i32 {
        %56 = arith.divsi %2, %1 : i32
        %57 = arith.divsi %56, %arg16 : i32
        %58 = tt.addptr %arg1, %57 : !tt.ptr<i32>, i32
        %59 = tt.load %58 : !tt.ptr<i32>
        %60 = tt.addptr %58, %c1_i32 : !tt.ptr<i32>, i32
        %61 = tt.load %60 : !tt.ptr<i32>
        %62 = arith.subi %61, %59 : i32
        %63 = arith.minsi %62, %arg17 : i32
        %64 = tt.addptr %arg3, %57 : !tt.ptr<i32>, i32
        %65 = tt.load %64 : !tt.ptr<i32>
        %66 = tt.addptr %64, %c1_i32 : !tt.ptr<i32>, i32
        %67 = tt.load %66 : !tt.ptr<i32>
        %68 = arith.subi %67, %65 : i32
        %69 = arith.remsi %2, %1 : i32
        %70 = arith.remsi %56, %arg16 : i32
        %71 = arith.extsi %70 : i32 to i64
        %72 = arith.extsi %arg14 : i32 to i64
        %73 = arith.muli %71, %72 : i64
        %74 = arith.muli %69, %c128_i32 : i32
        %75 = arith.cmpi slt, %74, %63 : i32
        %76:2 = scf.if %75 -> (i32, i32) {
          %77 = scf.for %arg24 = %c0_i32 to %68 step %c128_i32 iter_args(%arg25 = %arg22) -> (i32)  : i32 {
            %85 = arith.andi %arg25, %c1_i32 : i32
            %86 = ttg.memdesc_subview %result[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
            ttng.wait_barrier %42, %85 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %result_7 = ttng.tmem_load %86 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked1>
            %87 = tlx.release_layout %result_7 : tensor<64x128xf32, #blocked1> -> tensor<64x128xf32, #blocked>
            %88 = arith.mulf %87, %cst_2 : tensor<64x128xf32, #blocked>
            %89 = arith.mulf %87, %cst_1 : tensor<64x128xf32, #blocked>
            %90 = arith.mulf %87, %cst_0 : tensor<64x128xf32, #blocked>
            %91 = arith.mulf %90, %87 : tensor<64x128xf32, #blocked>
            %92 = arith.addf %91, %cst : tensor<64x128xf32, #blocked>
            %93 = arith.mulf %89, %92 : tensor<64x128xf32, #blocked>
            %94 = tt.elementwise_inline_asm "\0A            tanh.approx.f32 $0, $1;\0A            " {constraints = "=r,r", packed_element = 1 : i32, pure = true} %93 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
            %95 = arith.addf %94, %cst : tensor<64x128xf32, #blocked>
            %96 = arith.mulf %88, %95 : tensor<64x128xf32, #blocked>
            %97 = tt.splat %arg19 : f32 -> tensor<64x128xf32, #blocked>
            %98 = arith.mulf %96, %97 : tensor<64x128xf32, #blocked>
            %99 = arith.truncf %98 : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>
            %100 = ttg.memdesc_reinterpret %86 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
            // CHECK-NOT: tlx.require_layout
            %101 = tlx.require_layout %99 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked1>
            ttng.tmem_store %101, %100, %true : tensor<64x128xbf16, #blocked1> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
            ttng.arrive_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %50, %85 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %102 = arith.addi %arg25, %c1_i32 : i32
            scf.yield %102 : i32
          } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
          %78 = ttg.memdesc_subview %result_4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %result_6 = ttng.tmem_load %78 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked1>
          %79 = tlx.release_layout %result_6 : tensor<64x128xf32, #blocked1> -> tensor<64x128xf32, #blocked>
          ttng.arrive_barrier %48, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %80 = tt.make_tensor_descriptor %arg5, [%61, %11], [%12, %c1_i64] : <bf16>, <tensor<64x128xbf16>>
          %81 = arith.addi %59, %74 : i32
          %82 = arith.trunci %73 : i64 to i32
          %83 = arith.truncf %79 : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>
          tt.descriptor_store %80[%81, %82], %83 : !tt.tensordesc<tensor<64x128xbf16>>, tensor<64x128xbf16, #blocked>
          %84 = arith.addi %arg23, %c1_i32 : i32
          scf.yield %77, %84 : i32, i32
        } else {
          scf.yield %arg22, %arg23 : i32, i32
        }
        scf.yield %76#0, %76#1 : i32, i32
      }
      ttg.warp_yield
    }
    partition0(%arg21: i32, %arg22: !tt.ptr<i32>, %arg23: i32, %arg24: !tt.ptr<bf16>, %arg25: !tt.ptr<bf16>, %arg26: !tt.ptr<i32>, %arg27: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg28: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg36: !tt.tensordesc<tensor<128x128xbf16>>, %arg37: i32, %arg38: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg39: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg51: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg52: f32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg59: !tt.tensordesc<tensor<128x128xbf16>>) num_warps(4) {
      %c1_i64_6 = arith.constant 1 : i64
      %c64_i32 = arith.constant 64 : i32
      %true_7 = arith.constant true
      %cst_8 = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked>
      %cst_9 = arith.constant dense<4.471500e-02> : tensor<64x128xf32, #blocked>
      %cst_10 = arith.constant dense<0.797884583> : tensor<64x128xf32, #blocked>
      %cst_11 = arith.constant dense<5.000000e-01> : tensor<64x128xf32, #blocked>
      %c128_i32_12 = arith.constant 128 : i32
      %c0_i32_13 = arith.constant 0 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %55:2 = scf.for %arg60 = %c0_i32_13 to %arg57 step %c1_i32_14 iter_args(%arg61 = %c0_i32_13, %arg62 = %c0_i32_13) -> (i32, i32)  : i32 {
        %56 = arith.remsi %arg56, %arg37 : i32
        %57 = arith.divsi %arg56, %arg37 : i32
        %58 = arith.remsi %57, %arg21 : i32
        %59 = arith.extsi %58 : i32 to i64
        %60 = arith.extsi %arg54 : i32 to i64
        %61 = arith.muli %59, %60 : i64
        %62 = arith.divsi %57, %arg21 : i32
        %63 = tt.addptr %arg26, %62 : !tt.ptr<i32>, i32
        %64 = tt.load %63 : !tt.ptr<i32>
        %65 = tt.addptr %63, %c1_i32_14 : !tt.ptr<i32>, i32
        %66 = tt.load %65 : !tt.ptr<i32>
        %67 = arith.subi %66, %64 : i32
        %68 = arith.minsi %67, %arg23 : i32
        %69 = tt.addptr %arg22, %62 : !tt.ptr<i32>, i32
        %70 = tt.load %69 : !tt.ptr<i32>
        %71 = tt.addptr %69, %c1_i32_14 : !tt.ptr<i32>, i32
        %72 = tt.load %71 : !tt.ptr<i32>
        %73 = arith.subi %72, %70 : i32
        %74 = arith.muli %56, %c128_i32_12 : i32
        %75 = arith.cmpi slt, %74, %68 : i32
        %76:2 = scf.if %75 -> (i32, i32) {
          %77 = scf.for %arg63 = %c0_i32_13 to %73 step %c128_i32_12 iter_args(%arg64 = %arg61) -> (i32)  : i32 {
            %89 = arith.andi %arg64, %c1_i32_14 : i32
            %90 = ttg.memdesc_subview %arg51[%c0_i32_13, %c0_i32_13, %c0_i32_13] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
            %91 = ttg.memdesc_subview %arg43[%c0_i32_13] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %91, %89 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %result_16 = ttng.tmem_load %90 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked1>
            %92 = tlx.release_layout %result_16 : tensor<64x128xf32, #blocked1> -> tensor<64x128xf32, #blocked>
            %93 = arith.mulf %92, %cst_11 : tensor<64x128xf32, #blocked>
            %94 = arith.mulf %92, %cst_10 : tensor<64x128xf32, #blocked>
            %95 = arith.mulf %92, %cst_9 : tensor<64x128xf32, #blocked>
            %96 = arith.mulf %95, %92 : tensor<64x128xf32, #blocked>
            %97 = arith.addf %96, %cst_8 : tensor<64x128xf32, #blocked>
            %98 = arith.mulf %94, %97 : tensor<64x128xf32, #blocked>
            %99 = tt.elementwise_inline_asm "\0A            tanh.approx.f32 $0, $1;\0A            " {constraints = "=r,r", packed_element = 1 : i32, pure = true} %98 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
            %100 = arith.addf %99, %cst_8 : tensor<64x128xf32, #blocked>
            %101 = arith.mulf %93, %100 : tensor<64x128xf32, #blocked>
            %102 = tt.splat %arg52 : f32 -> tensor<64x128xf32, #blocked>
            %103 = arith.mulf %101, %102 : tensor<64x128xf32, #blocked>
            %104 = arith.truncf %103 : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>
            %105 = ttg.memdesc_reinterpret %90 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
            // CHECK-NOT: tlx.require_layout
            %106 = tlx.require_layout %104 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked1>
            ttng.tmem_store %106, %105, %true_7 : tensor<64x128xbf16, #blocked1> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
            %107 = ttg.memdesc_subview %arg47[%c0_i32_13] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.arrive_barrier %107, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %108 = ttg.memdesc_subview %arg41[%c0_i32_13] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %108, %89 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %109 = arith.addi %arg64, %c1_i32_14 : i32
            scf.yield %109 : i32
          } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
          %78 = arith.muli %arg21, %c128_i32_12 : i32
          %79 = arith.extsi %78 : i32 to i64
          %80 = tt.make_tensor_descriptor %arg24, [%66, %78], [%79, %c1_i64_6] : <bf16>, <tensor<64x128xbf16>>
          %81 = ttg.memdesc_subview %arg39[%c0_i32_13, %c0_i32_13, %c0_i32_13] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %result_15 = ttng.tmem_load %81 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked1>
          %82 = tlx.release_layout %result_15 : tensor<64x128xf32, #blocked1> -> tensor<64x128xf32, #blocked>
          %83 = ttg.memdesc_subview %arg45[%c0_i32_13] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.arrive_barrier %83, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %84 = arith.addi %64, %74 : i32
          %85 = arith.addi %84, %c64_i32 : i32
          %86 = arith.trunci %61 : i64 to i32
          %87 = arith.truncf %82 : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>
          tt.descriptor_store %80[%85, %86], %87 : !tt.tensordesc<tensor<64x128xbf16>>, tensor<64x128xbf16, #blocked>
          %88 = arith.addi %arg62, %c1_i32_14 : i32
          scf.yield %77, %88 : i32, i32
        } else {
          scf.yield %arg61, %arg62 : i32, i32
        }
        scf.yield %76#0, %76#1 : i32, i32
      }
      ttg.warp_return
    }
    partition1(%arg21: i32, %arg22: !tt.ptr<i32>, %arg23: i32, %arg24: !tt.ptr<bf16>, %arg25: !tt.ptr<bf16>, %arg26: !tt.ptr<i32>, %arg27: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg28: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg36: !tt.tensordesc<tensor<128x128xbf16>>, %arg37: i32, %arg38: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg39: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg51: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg52: f32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg59: !tt.tensordesc<tensor<128x128xbf16>>) num_warps(1) {
      %c2_i32 = arith.constant 2 : i32
      %true_6 = arith.constant true
      %false = arith.constant false
      %c128_i32_7 = arith.constant 128 : i32
      %c0_i32_8 = arith.constant 0 : i32
      %c1_i32_9 = arith.constant 1 : i32
      %55:4 = scf.for %arg60 = %c0_i32_8 to %arg57 step %c1_i32_9 iter_args(%arg61 = %c0_i32_8, %arg62 = %c0_i32_8, %arg63 = %c0_i32_8, %arg64 = %c0_i32_8) -> (i32, i32, i32, i32)  : i32 {
        %56 = arith.remsi %arg56, %arg37 : i32
        %57 = arith.divsi %arg56, %arg37 : i32
        %58 = arith.divsi %57, %arg21 : i32
        %59 = tt.addptr %arg26, %58 : !tt.ptr<i32>, i32
        %60 = tt.load %59 : !tt.ptr<i32>
        %61 = tt.addptr %59, %c1_i32_9 : !tt.ptr<i32>, i32
        %62 = tt.load %61 : !tt.ptr<i32>
        %63 = arith.subi %62, %60 : i32
        %64 = arith.minsi %63, %arg23 : i32
        %65 = tt.addptr %arg22, %58 : !tt.ptr<i32>, i32
        %66 = tt.load %65 : !tt.ptr<i32>
        %67 = tt.addptr %65, %c1_i32_9 : !tt.ptr<i32>, i32
        %68 = tt.load %67 : !tt.ptr<i32>
        %69 = arith.subi %68, %66 : i32
        %70 = arith.muli %56, %c128_i32_7 : i32
        %71 = arith.cmpi slt, %70, %64 : i32
        %72:4 = scf.if %71 -> (i32, i32, i32, i32) {
          %73 = arith.andi %arg61, %c1_i32_9 : i32
          %74 = arith.remsi %arg62, %c2_i32 : i32
          %75 = arith.divsi %arg62, %c2_i32 : i32
          %76 = arith.andi %75, %c1_i32_9 : i32
          %77 = ttg.memdesc_subview %arg28[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %78 = ttg.memdesc_subview %arg27[%74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %77, %73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %78, %76 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %79 = ttg.memdesc_subview %arg48[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          %80 = ttg.memdesc_subview %arg35[%74, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
          %81 = ttg.memdesc_subview %arg50[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %82 = ttg.memdesc_subview %arg42[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %83 = ttng.tc_gen5_mma %79, %80, %81[], %false, %true_6, %82[%true_6] : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %84 = ttg.memdesc_subview %arg29[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %84, %73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %85 = ttg.memdesc_subview %arg49[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          %86 = ttg.memdesc_subview %arg51[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %87 = ttg.memdesc_subview %arg30[%74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %88 = ttg.memdesc_subview %arg43[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %89 = ttng.tc_gen5_mma %85, %80, %86[], %false, %true_6, %87[%true_6], %88[%true_6] : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %90 = ttg.memdesc_subview %arg34[%74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %90, %76 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %91 = arith.andi %arg64, %c1_i32_9 : i32
          %92 = ttg.memdesc_subview %arg44[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %93 = ttg.memdesc_subview %arg45[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %92, %91 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %94 = arith.andi %arg63, %c1_i32_9 : i32
          %95 = ttg.memdesc_subview %arg46[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %95, %94 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %96 = ttg.memdesc_reinterpret %81 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
          %97 = ttg.memdesc_subview %arg40[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %98 = ttg.memdesc_subview %arg38[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %99 = ttg.memdesc_subview %arg58[%74, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
          // CHECK-NOT: tlx.require_layout
          %100 = tlx.require_layout %96 : !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>
          %101 = ttng.tc_gen5_mma %100, %99, %98[], %false, %true_6, %97[%true_6] : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %102 = arith.divsi %69, %c128_i32_7 : i32
          %103 = arith.addi %arg62, %c1_i32_9 : i32
          %104 = arith.addi %arg63, %c1_i32_9 : i32
          %105 = arith.subi %102, %c1_i32_9 : i32
          %106:6 = scf.for %arg65 = %c0_i32_8 to %105 step %c1_i32_9 iter_args(%arg66 = %99, %arg67 = %true_6, %arg68 = %103, %arg69 = %104, %arg70 = %arg63, %arg71 = %arg62) -> (!ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, i1, i32, i32, i32, i32)  : i32 {
            %119 = arith.remsi %arg68, %c2_i32 : i32
            %120 = arith.divsi %arg68, %c2_i32 : i32
            %121 = arith.andi %120, %c1_i32_9 : i32
            %122 = arith.andi %arg69, %c1_i32_9 : i32
            %123 = ttg.memdesc_subview %arg27[%119] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %123, %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %124 = ttg.memdesc_subview %arg35[%119, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
            %125 = ttng.tc_gen5_mma %79, %124, %81[], %false, %true_6, %82[%true_6] : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %126 = arith.andi %arg70, %c1_i32_9 : i32
            %127 = ttg.memdesc_subview %arg47[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %93, %91 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %127, %126 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %128 = ttg.memdesc_subview %arg39[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
            %129 = ttg.memdesc_subview %arg41[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %130 = arith.remsi %arg71, %c2_i32 : i32
            %131 = ttg.memdesc_subview %arg33[%130] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %132 = ttg.memdesc_reinterpret %86 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
            %133 = arith.xori %arg67, %true_6 : i1
            // CHECK-NOT: tlx.require_layout
            %134 = tlx.require_layout %132 : !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>
            %135 = ttng.tc_gen5_mma %134, %arg66, %128[], %133, %true_6, %129[%true_6], %131[%true_6] : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %136 = ttg.memdesc_subview %arg30[%119] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %137 = ttng.tc_gen5_mma %85, %124, %86[], %false, %true_6, %136[%true_6], %88[%true_6] : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %90, %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %95, %122 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %138 = ttg.memdesc_subview %arg58[%119, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
            %139 = ttng.tc_gen5_mma %100, %138, %98[], %true_6, %true_6, %97[%true_6] : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %140 = arith.addi %arg68, %c1_i32_9 : i32
            %141 = arith.addi %arg69, %c1_i32_9 : i32
            %142 = arith.addi %arg70, %c1_i32_9 : i32
            %143 = arith.addi %arg71, %c1_i32_9 : i32
            scf.yield %138, %false, %140, %141, %142, %143 : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, i1, i32, i32, i32, i32
          }
          ttng.wait_barrier %93, %91 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %107 = arith.andi %106#4, %c1_i32_9 : i32
          %108 = ttg.memdesc_subview %arg47[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %108, %107 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %109 = ttg.memdesc_reinterpret %86 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable>
          %110 = ttg.memdesc_subview %arg41[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %111 = arith.remsi %106#5, %c2_i32 : i32
          %112 = ttg.memdesc_subview %arg33[%111] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %113 = ttg.memdesc_subview %arg39[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %114 = arith.xori %106#1, %true_6 : i1
          // CHECK-NOT: tlx.require_layout
          %115 = tlx.require_layout %109 : !ttg.memdesc<64x128xbf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>
          %116 = ttng.tc_gen5_mma %115, %106#0, %113[], %114, %true_6, %110[%true_6], %112[%true_6] : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %117 = arith.addi %arg61, %c1_i32_9 : i32
          %118 = arith.addi %arg64, %c1_i32_9 : i32
          scf.yield %117, %106#2, %106#3, %118 : i32, i32, i32, i32
        } else {
          scf.yield %arg61, %arg62, %arg63, %arg64 : i32, i32, i32, i32
        }
        scf.yield %72#0, %72#1, %72#2, %72#3 : i32, i32, i32, i32
      }
      ttg.warp_return
    }
    partition2(%arg21: i32, %arg22: !tt.ptr<i32>, %arg23: i32, %arg24: !tt.ptr<bf16>, %arg25: !tt.ptr<bf16>, %arg26: !tt.ptr<i32>, %arg27: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg28: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg36: !tt.tensordesc<tensor<128x128xbf16>>, %arg37: i32, %arg38: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg39: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg51: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg52: f32, %arg53: i32, %arg54: i32, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, %arg59: !tt.tensordesc<tensor<128x128xbf16>>) num_warps(1) {
      %c1_i64_6 = arith.constant 1 : i64
      %c64_i32 = arith.constant 64 : i32
      %true_7 = arith.constant true
      %c128_i32_8 = arith.constant 128 : i32
      %c0_i32_9 = arith.constant 0 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %55 = scf.for %arg60 = %c0_i32_9 to %arg57 step %c1_i32_10 iter_args(%arg61 = %c0_i32_9) -> (i32)  : i32 {
        %56 = arith.remsi %arg56, %arg37 : i32
        %57 = arith.divsi %arg56, %arg37 : i32
        %58 = arith.remsi %57, %arg21 : i32
        %59 = arith.extsi %58 : i32 to i64
        %60 = arith.extsi %arg55 : i32 to i64
        %61 = arith.muli %59, %60 : i64
        %62 = arith.extsi %arg53 : i32 to i64
        %63 = arith.muli %59, %62 : i64
        %64 = arith.divsi %57, %arg21 : i32
        %65 = tt.addptr %arg26, %64 : !tt.ptr<i32>, i32
        %66 = tt.load %65 : !tt.ptr<i32>
        %67 = tt.addptr %65, %c1_i32_10 : !tt.ptr<i32>, i32
        %68 = tt.load %67 : !tt.ptr<i32>
        %69 = arith.subi %68, %66 : i32
        %70 = arith.minsi %69, %arg23 : i32
        %71 = tt.addptr %arg22, %64 : !tt.ptr<i32>, i32
        %72 = tt.load %71 : !tt.ptr<i32>
        %73 = tt.addptr %71, %c1_i32_10 : !tt.ptr<i32>, i32
        %74 = tt.load %73 : !tt.ptr<i32>
        %75 = arith.subi %74, %72 : i32
        %76 = arith.muli %56, %c128_i32_8 : i32
        %77 = arith.cmpi slt, %76, %70 : i32
        %78 = scf.if %77 -> (i32) {
          %79 = arith.muli %arg21, %c128_i32_8 : i32
          %80 = arith.extsi %79 : i32 to i64
          %81 = tt.make_tensor_descriptor %arg25, [%68, %79], [%80, %c1_i64_6] : <bf16>, <tensor<64x128xbf16>>
          %82 = arith.andi %arg61, %c1_i32_10 : i32
          %83 = ttg.memdesc_subview %arg31[%c0_i32_9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %84 = arith.xori %82, %c1_i32_10 : i32
          ttng.wait_barrier %83, %84 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %85 = ttg.memdesc_subview %arg28[%c0_i32_9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.barrier_expect %85, 32768, %true_7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %86 = ttg.memdesc_subview %arg48[%c0_i32_9, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          %87 = arith.addi %66, %76 : i32
          %88 = arith.trunci %61 : i64 to i32
          %89 = ttng.tensor_desc_to_tma_ptr %81 : !tt.tensordesc<tensor<64x128xbf16>> to !tt.ptr<i8>
          ttng.async_tma_copy_global_to_local %89[%87, %88] %86, %85, %true_7 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          %90 = ttg.memdesc_subview %arg32[%c0_i32_9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.wait_barrier %90, %84 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %91 = ttg.memdesc_subview %arg29[%c0_i32_9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          ttng.barrier_expect %91, 32768, %true_7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
          %92 = ttg.memdesc_subview %arg49[%c0_i32_9, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          %93 = arith.addi %87, %c64_i32 : i32
          ttng.async_tma_copy_global_to_local %89[%93, %88] %92, %91, %true_7 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
          scf.for %arg62 = %c0_i32_9 to %75 step %c128_i32_8  : i32 {
            %95 = ttg.memdesc_subview %arg30[%c0_i32_9] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %95, %c1_i32_10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %96 = ttg.memdesc_subview %arg27[%c0_i32_9] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.barrier_expect %96, 32768, %true_7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %97 = ttg.memdesc_subview %arg35[%c0_i32_9, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
            %98 = arith.addi %72, %arg62 : i32
            %99 = arith.trunci %63 : i64 to i32
            %100 = ttng.tensor_desc_to_tma_ptr %arg36 : !tt.tensordesc<tensor<128x128xbf16>> to !tt.ptr<i8>
            ttng.async_tma_copy_global_to_local %100[%98, %99] %97, %96, %true_7 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
            %101 = ttg.memdesc_subview %arg33[%c0_i32_9] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.wait_barrier %101, %c1_i32_10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %102 = ttg.memdesc_subview %arg34[%c0_i32_9] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            ttng.barrier_expect %102, 32768, %true_7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
            %103 = ttg.memdesc_subview %arg58[%c0_i32_9, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
            %104 = ttng.tensor_desc_to_tma_ptr %arg59 : !tt.tensordesc<tensor<128x128xbf16>> to !tt.ptr<i8>
            ttng.async_tma_copy_global_to_local %104[%98, %99] %103, %102, %true_7 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
          } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
          %94 = arith.addi %arg61, %c1_i32_10 : i32
          scf.yield %94 : i32
        } else {
          scf.yield %arg61 : i32
        }
        scf.yield %78 : i32
      }
      ttg.warp_return
    } : (i32, !tt.ptr<i32>, i32, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<i32>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, !tt.tensordesc<tensor<128x128xbf16>>, i32, !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, f32, i32, i32, i32, i32, i32, !ttg.memdesc<2x128x128xbf16, #shared, #smem, mutable>, !tt.tensordesc<tensor<128x128xbf16>>) -> ()
    tt.return
  }
}
