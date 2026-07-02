// RUN: triton-opt %s \
// RUN:   -pass-pipeline='builtin.module(convert-triton-to-tritongpu{target=cuda:100 num-warps=8 threads-per-warp=32 num-ctas=1},tritongpu-coalesce,tlx-propagate-layout,tlx-resolve-placeholder-layouts,tlx-rewrite-local-alias,tritongpu-F32DotTC,triton-nvidia-gpu-plan-cta,tritongpu-remove-layout-conversions,tritongpu-optimize-thread-locality,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands,triton-nvidia-optimize-descriptor-encoding,cse,tritongpu-fuse-nested-loops,canonicalize,triton-licm,tritongpu-optimize-accumulator-init,tritongpu-hoist-tmem-alloc,nvgpu-ws-data-partition{num-warp-groups=1})' \
// RUN:   --verify-each 2>&1 | FileCheck %s

// CHECK-NOT: warning: skipping M-dimension data partitioning
// CHECK-LABEL: @mxfp8_grouped_gemm_persistent_ws
// CHECK: ttng.tmem_alloc
// CHECK-SAME: !ttg.memdesc<128x256xf32
// CHECK: ttng.tmem_alloc
// CHECK-SAME: !ttg.memdesc<128x256xf32
// CHECK: tt.descriptor_load {{.*}}!tt.tensordesc<tensor<128x128xf8E4M3FN
// CHECK: tt.descriptor_load {{.*}}!tt.tensordesc<tensor<1x1x1x2x256xui8
// CHECK: ttng.tmem_alloc
// CHECK-SAME: tensor<128x4xi8
// CHECK-SAME: !ttg.memdesc<128x4xi8
// CHECK: ttng.tmem_alloc
// CHECK-SAME: tensor<128x4xi8
// CHECK-SAME: !ttg.memdesc<128x4xi8
// CHECK: ttng.tc_gen5_mma_scaled
// CHECK-SAME: !ttg.memdesc<128x128xf8E4M3FN
// CHECK-SAME: !ttg.memdesc<128x256xf32
// CHECK-SAME: !ttg.memdesc<128x4xi8
// CHECK: ttng.tc_gen5_mma_scaled
// CHECK-SAME: !ttg.memdesc<128x128xf8E4M3FN
// CHECK-SAME: !ttg.memdesc<128x256xf32
// CHECK-SAME: !ttg.memdesc<128x4xi8
// CHECK: } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 48 : i32} {
  tt.func public @mxfp8_grouped_gemm_persistent_ws(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c224_i32 = arith.constant 224 : i32
    %c192_i32 = arith.constant 192 : i32
    %c160_i32 = arith.constant 160 : i32
    %c96_i32 = arith.constant 96 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c151_i32 = arith.constant 151 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %c9437184_i64 = arith.constant 9437184 : i64
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c152_i32 = arith.constant 152 : i32
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c3072_i32 = arith.constant 3072 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1:4 = scf.for %arg9 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg10 = %0, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32) -> (i32, i32, i32, i32)  : i32 {
      %2 = tt.addptr %arg5, %arg9 : !tt.ptr<i32>, i32
      %3 = tt.load %2 : !tt.ptr<i32>
      %4 = arith.addi %3, %c127_i32 : i32
      %5 = arith.divsi %4, %c128_i32 : i32
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.divsi %6, %c2_i32 : i32
      %8 = arith.muli %7, %c2_i32 : i32
      %9 = arith.muli %7, %c24_i32 : i32
      %10 = arith.addi %arg11, %9 : i32
      %11 = arith.cmpi sge, %arg10, %arg11 : i32
      %12 = arith.cmpi slt, %arg10, %10 : i32
      %13 = arith.andi %11, %12 : i1
      %14 = scf.if %13 -> (i32) {
        %17 = arith.extsi %arg12 : i32 to i64
        %18 = arith.muli %17, %c3072_i64 : i64
        %19 = tt.addptr %arg0, %18 : !tt.ptr<f8E4M3FN>, i64
        %20 = tt.make_tensor_descriptor %19, [%3, %c3072_i32], [%c3072_i64, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<tensor<256x128xf8E4M3FN>>
        %21 = arith.extsi %arg9 : i32 to i64
        %22 = arith.muli %21, %c9437184_i64 : i64
        %23 = tt.addptr %arg1, %22 : !tt.ptr<f8E4M3FN>, i64
        %24 = tt.make_tensor_descriptor %23, [%c3072_i32, %c3072_i32], [%c3072_i64, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<tensor<256x128xf8E4M3FN>>
        %25 = tt.addptr %arg2, %18 : !tt.ptr<bf16>, i64
        %26 = tt.make_tensor_descriptor %25, [%3, %c3072_i32], [%c3072_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<256x32xbf16>>
        %27 = arith.muli %5, %arg7 : i32
        %28 = arith.muli %27, %c512_i32 : i32
        %29 = arith.muli %arg7, %c512_i32 : i32
        %30 = arith.extsi %arg13 : i32 to i64
        %31 = arith.extsi %arg7 : i32 to i64
        %32 = arith.muli %30, %31 : i64
        %33 = arith.muli %32, %c512_i64 : i64
        %34 = tt.addptr %arg3, %33 : !tt.ptr<i8>, i64
        %35 = arith.extsi %28 : i32 to i64
        %36 = arith.extsi %29 : i32 to i64
        %37 = tt.make_tensor_descriptor %34, [%c1_i32, %5, %arg7, %c4_i32, %c256_i32], [%35, %36, %c512_i64, %c256_i64, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<tensor<1x1x1x4x256xui8>>
        %38 = arith.muli %arg6, %arg7 : i32
        %39 = arith.muli %38, %c512_i32 : i32
        %40 = arith.extsi %arg8 : i32 to i64
        %41 = arith.muli %21, %40 : i64
        %42 = tt.addptr %arg4, %41 : !tt.ptr<i8>, i64
        %43 = arith.extsi %39 : i32 to i64
        %44 = tt.make_tensor_descriptor %42, [%c1_i32, %arg6, %arg7, %c2_i32, %c256_i32], [%43, %36, %c512_i64, %c256_i64, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<tensor<1x2x1x2x256xui8>>
        %45 = arith.subi %10, %arg10 : i32
        %46 = arith.addi %45, %c151_i32 : i32
        %47 = arith.divsi %46, %c152_i32 : i32
        %48 = scf.for %arg14 = %c0_i32 to %47 step %c1_i32 iter_args(%arg15 = %arg10) -> (i32)  : i32 {
          %49 = arith.subi %arg15, %arg11 : i32
          %50 = arith.remsi %49, %8 : i32
          %51 = arith.divsi %49, %8 : i32
          %52 = arith.muli %50, %c256_i32 : i32
          %53 = arith.muli %51, %c256_i32 : i32
          %54 = arith.muli %51, %c2_i32 : i32
          %55 = scf.for %arg16 = %c0_i32 to %c24_i32 step %c1_i32 iter_args(%arg17 = %cst) -> (tensor<256x256xf32>)  : i32 {
            %86 = arith.muli %arg16, %c128_i32 : i32
            %87 = tt.descriptor_load %20[%52, %86] : !tt.tensordesc<tensor<256x128xf8E4M3FN>> -> tensor<256x128xf8E4M3FN>
            %88 = tt.descriptor_load %24[%53, %86] : !tt.tensordesc<tensor<256x128xf8E4M3FN>> -> tensor<256x128xf8E4M3FN>
            %89 = tt.descriptor_load %37[%c0_i32, %50, %arg16, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<1x1x1x4x256xui8>> -> tensor<1x1x1x4x256xi8>
            %90 = tt.descriptor_load %44[%c0_i32, %54, %arg16, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<1x2x1x2x256xui8>> -> tensor<1x2x1x2x256xi8>
            %91 = tt.reshape %89 : tensor<1x1x1x4x256xi8> -> tensor<256x4xi8>
            %94 = tt.reshape %90 : tensor<1x2x1x2x256xi8> -> tensor<2x1x32x4x4xi8>
            %95 = tt.trans %94 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8> -> tensor<2x4x32x1x4xi8>
            %96 = tt.reshape %95 : tensor<2x4x32x1x4xi8> -> tensor<256x4xi8>
            %97 = tt.trans %88 {order = array<i32: 1, 0>} : tensor<256x128xf8E4M3FN> -> tensor<128x256xf8E4M3FN>
            %98 = tt.dot_scaled %87 scale %91, %97 scale %96, %arg17 lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<256x128xf8E4M3FN>, tensor<256x4xi8> * tensor<128x256xf8E4M3FN>, tensor<256x4xi8> -> tensor<256x256xf32>
            scf.yield %98 : tensor<256x256xf32>
          }
          %56 = tt.reshape %55 : tensor<256x256xf32> -> tensor<256x2x128xf32>
          %57 = tt.trans %56 {order = array<i32: 0, 2, 1>} : tensor<256x2x128xf32> -> tensor<256x128x2xf32>
          %outLHS, %outRHS = tt.split %57 : tensor<256x128x2xf32> -> tensor<256x128xf32>
          %58 = tt.reshape %outLHS : tensor<256x128xf32> -> tensor<256x2x64xf32>
          %59 = tt.trans %58 {order = array<i32: 0, 2, 1>} : tensor<256x2x64xf32> -> tensor<256x64x2xf32>
          %outLHS_0, %outRHS_1 = tt.split %59 : tensor<256x64x2xf32> -> tensor<256x64xf32>
          %60 = tt.reshape %outLHS_0 : tensor<256x64xf32> -> tensor<256x2x32xf32>
          %61 = tt.trans %60 {order = array<i32: 0, 2, 1>} : tensor<256x2x32xf32> -> tensor<256x32x2xf32>
          %outLHS_2, %outRHS_3 = tt.split %61 : tensor<256x32x2xf32> -> tensor<256x32xf32>
          %62 = tt.reshape %outRHS_1 : tensor<256x64xf32> -> tensor<256x2x32xf32>
          %63 = tt.trans %62 {order = array<i32: 0, 2, 1>} : tensor<256x2x32xf32> -> tensor<256x32x2xf32>
          %outLHS_4, %outRHS_5 = tt.split %63 : tensor<256x32x2xf32> -> tensor<256x32xf32>
          %64 = tt.reshape %outRHS : tensor<256x128xf32> -> tensor<256x2x64xf32>
          %65 = tt.trans %64 {order = array<i32: 0, 2, 1>} : tensor<256x2x64xf32> -> tensor<256x64x2xf32>
          %outLHS_6, %outRHS_7 = tt.split %65 : tensor<256x64x2xf32> -> tensor<256x64xf32>
          %66 = tt.reshape %outLHS_6 : tensor<256x64xf32> -> tensor<256x2x32xf32>
          %67 = tt.trans %66 {order = array<i32: 0, 2, 1>} : tensor<256x2x32xf32> -> tensor<256x32x2xf32>
          %outLHS_8, %outRHS_9 = tt.split %67 : tensor<256x32x2xf32> -> tensor<256x32xf32>
          %68 = tt.reshape %outRHS_7 : tensor<256x64xf32> -> tensor<256x2x32xf32>
          %69 = tt.trans %68 {order = array<i32: 0, 2, 1>} : tensor<256x2x32xf32> -> tensor<256x32x2xf32>
          %outLHS_10, %outRHS_11 = tt.split %69 : tensor<256x32x2xf32> -> tensor<256x32xf32>
          %70 = arith.truncf %outLHS_2 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %53], %70 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %71 = arith.addi %53, %c32_i32 : i32
          %72 = arith.truncf %outRHS_3 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %71], %72 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %73 = arith.addi %53, %c64_i32 : i32
          %74 = arith.truncf %outLHS_4 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %73], %74 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %75 = arith.addi %53, %c96_i32 : i32
          %76 = arith.truncf %outRHS_5 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %75], %76 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %77 = arith.addi %53, %c128_i32 : i32
          %78 = arith.truncf %outLHS_8 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %77], %78 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %79 = arith.addi %53, %c160_i32 : i32
          %80 = arith.truncf %outRHS_9 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %79], %80 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %81 = arith.addi %53, %c192_i32 : i32
          %82 = arith.truncf %outLHS_10 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %81], %82 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %83 = arith.addi %53, %c224_i32 : i32
          %84 = arith.truncf %outRHS_11 : tensor<256x32xf32> to tensor<256x32xbf16>
          tt.descriptor_store %26[%52, %83], %84 : !tt.tensordesc<tensor<256x32xbf16>>, tensor<256x32xbf16>
          %85 = arith.addi %arg15, %c152_i32 : i32
          scf.yield %85 : i32
        } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
        scf.yield %48 : i32
      } else {
        scf.yield %arg10 : i32
      }
      %15 = arith.addi %arg12, %3 : i32
      %16 = arith.addi %arg13, %5 : i32
      scf.yield %14, %10, %15, %16 : i32, i32, i32, i32
    }
    tt.return
  }
}
