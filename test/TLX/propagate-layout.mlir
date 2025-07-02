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
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable>
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
