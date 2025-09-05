// RUN: triton-opt -split-input-file --tlx-propagate-layout %s| FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 8]}>
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
      %107 = ttg.memdesc_index %79[%arg9] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %108 = ttg.memdesc_index %80[%arg9] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
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
      %108 = ttg.memdesc_index %79[%107] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %109 = ttg.memdesc_index %80[%107] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
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
      %118 = ttg.memdesc_index %79[%117] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %119 = ttg.memdesc_index %80[%117] : !ttg.memdesc<2x64x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
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
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable
    %6 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %7 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_index %6[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %7, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %9 = arith.muli %0, %c64_i32 : i32
    %10 = arith.muli %1, %c64_i32 : i32
    ttg.warp_specialize(%7)
    default {
      ttng.wait_barrier %8, %c1_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      %11 = tlx.require_layout %5 : !ttg.memdesc<64x64xi16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
      ttng.async_tma_copy_global_to_local %3[%9, %10] %11, %7, %true : !tt.tensordesc<tensor<64x64xsi16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xi16, #shared2, #smem, mutable>
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
    %1 = ttg.memdesc_index %0 [%c0_i32] : !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
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
    %4 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %6 = ttg.memdesc_index %5[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %7 = ttg.memdesc_index %5[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_index %5[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %10 = ttg.memdesc_index %9[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %11 = ttg.memdesc_index %9[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %12 = ttg.memdesc_index %9[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %12, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %14 = ttg.memdesc_index %13[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %15 = ttg.memdesc_index %13[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %16 = ttg.memdesc_index %13[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %18 = ttg.memdesc_index %17[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %19 = ttg.memdesc_index %17[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %20 = ttg.memdesc_index %17[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
      %31 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NOT: tlx.require_layout
      %32 = tlx.require_layout %31 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      ttng.async_tma_copy_global_to_local %arg4[%30, %c0_i32] %32, %4, %true : !tt.tensordesc<tensor<128x128xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      %34:2 = scf.for %arg25 = %c0_i32 to %arg24 step %c64_i32 iter_args(%arg26 = %28, %arg27 = %c0_i32) -> (i32, i32)  : i32 {
        %35 = arith.remsi %arg25, %c3_i32 : i32
        %36 = ttg.memdesc_index %5[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %37 = arith.xori %arg27, %c1_i32 : i32
        ttng.wait_barrier %36, %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %38 = ttg.memdesc_index %9[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %39 = ttg.memdesc_index %1[%35] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.barrier_expect %38, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %40 = tlx.require_layout %39 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg9[%arg26, %c0_i32] %40, %38, %true : !tt.tensordesc<tensor<64x128xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %42 = ttg.memdesc_index %13[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %42, %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %43 = ttg.memdesc_index %17[%35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %44 = ttg.memdesc_index %2[%35] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.barrier_expect %43, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        // CHECK-NOT: tlx.require_layout
        %45 = tlx.require_layout %44 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg14[%arg26, %c0_i32] %45, %43, %true : !tt.tensordesc<tensor<64x128xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
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
      %22 = ttg.memdesc_index %arg35[%c0_i32_8] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %22, %c0_i32_8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %23 = ttg.memdesc_index %arg36[%c0_i32_8] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %24:4 = scf.for %arg41 = %c0_i32_8 to %arg27 step %c64_i32_2 iter_args(%arg42 = %arg28, %arg43 = %arg33, %arg44 = %arg34, %arg45 = %c0_i32_8) -> (tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, i32)  : i32 {
        %53 = arith.remsi %arg41, %c3_i32_7 : i32
        %54 = ttg.memdesc_index %arg31[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %54, %arg45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %55 = ttg.memdesc_index %arg32 [%53] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
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
        %63 = ttg.memdesc_index %arg30[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
        %89 = ttg.memdesc_index %arg39[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %89, %arg45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %90 = ttg.memdesc_index %arg40 [%53] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
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
        %97 = ttg.memdesc_index %arg38[%53] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK: #shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-NOT: #shared2
// CHECK-NOT: #shared3
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_store_local_load_dot(%arg0: !tt.ptr<f16>, %arg1: tensor<64x32x!tt.ptr<f16>, #blocked>, %arg2: tensor<32x64x!tt.ptr<f16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: %[[mem_desc1:.*]] = ttg.memdesc_index %{{.*}}
    %2 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    // CHECK: %[[mem_desc2:.*]] = ttg.memdesc_index %{{.*}}
    %3 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    %4 = tt.load %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked>
    %5 = tt.load %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked>
    ttg.local_store %4, %2 : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %5, %3 : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK-NOT tlx.require_layout %[[mem_desc1]]
    %6 = tlx.require_layout %2 : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared2, #smem, mutable>
    // CHECK: ttg.local_load %[[mem_desc1]] : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %7 = ttg.local_load %6 : !ttg.memdesc<64x32xf16, #shared2, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK-NOT tlx.require_layout %[[mem_desc2]]
    %8 = tlx.require_layout %3 : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared3, #smem, mutable>
    // CHECK: ttg.local_load %[[mem_desc2]] : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %9 = ttg.local_load %8 : !ttg.memdesc<32x64xf16, #shared3, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %10 = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    %11 = ttg.convert_layout %7 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %12 = ttg.convert_layout %9 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %13 = tt.dot %11, %12, %10, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %13 : tensor<64x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = false>
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: #[[$TMEM:.*]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = true>
// CHECK-DAG: #[[$TMEM1:.*]] = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = false>

module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tcgen5_fa_kernel
  tt.func public @tcgen5_fa_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%0, %result, %1, %2, %result_1, %result_0)
    default {
      ttg.warp_yield
    }
    partition0(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable>) num_warps(1) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg8[%c0_i32] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
      %4 = ttg.memdesc_index %arg10[%c0_i32] : !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable>
      %5 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %6 = ttng.tc_gen5_mma %3, %4, %5[], %false, %true : !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem, #ttng.tensor_memory, mutable>
      %8 = ttg.memdesc_index %arg11[%c0_i32] : !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
      %9 = ttg.memdesc_index %arg12[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NOT: tlx.require_layout
      %10 = tlx.require_layout %7 : !ttg.memdesc<64x32xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>
      %11 = ttng.tc_gen5_mma %10, %8, %9[], %false, %true : !ttg.memdesc<64x32xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg8: !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %true = arith.constant true
      %c0_i32 = arith.constant 0 : i32
      %3 = ttg.memdesc_index %arg9[%c0_i32] : !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_2 = ttng.tmem_load %3 : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #blocked>
      %4 = tlx.release_layout %result_2 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked1>
      %5 = arith.truncf %4 : tensor<64x32xf32, #blocked1> to tensor<64x32xf16, #blocked1>
      %6 = ttg.memdesc_index %arg13[%c0_i32] : !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf16, #tmem, #ttng.tensor_memory, mutable>
      %7 = tlx.require_layout %5 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #blocked>
      // CHECK: ttng.tmem_store {{.*}} : tensor<64x32xf16, #[[$BLOCKED]]> -> !ttg.memdesc<64x32xf16, #[[$TMEM1]]
      ttng.tmem_store %7, %6, %true : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #tmem, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x16x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<1x32x32xf16, #shared1, #smem, mutable>, !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x32xf16, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
