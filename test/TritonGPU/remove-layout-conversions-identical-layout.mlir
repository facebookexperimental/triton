// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s

// The two blocked layouts below have identical linear layouts on a 64x1
// tensor: the order of the size-1 dimension is unobservable, so every element
// already sits in the same register, thread, and warp. The convert_layout
// ops move no data. Their producers feed only the converts, so removal
// moves the slice without duplicating any op.

// CHECK-LABEL: @identical_layout_convert_removed
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.store

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.target" = "cuda:90"} {
  tt.func @identical_layout_convert_removed(%ptr: !tt.ptr<f32>, %off: i32, %val: tensor<64x1xf32, #blocked_b>) {
    %splat_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked_a>
    %idx = tt.splat %off : i32 -> tensor<64x1xi32, #blocked_a>
    %cst = arith.constant dense<64> : tensor<64x1xi32, #blocked_a>
    %mask = arith.cmpi slt, %idx, %cst : tensor<64x1xi32, #blocked_a>
    %ptrs = tt.addptr %splat_ptr, %idx : tensor<64x1x!tt.ptr<f32>, #blocked_a>, tensor<64x1xi32, #blocked_a>
    %ptrs_cvt = ttg.convert_layout %ptrs : tensor<64x1x!tt.ptr<f32>, #blocked_a> -> tensor<64x1x!tt.ptr<f32>, #blocked_b>
    %mask_cvt = ttg.convert_layout %mask : tensor<64x1xi1, #blocked_a> -> tensor<64x1xi1, #blocked_b>
    tt.store %ptrs_cvt, %val, %mask_cvt : tensor<64x1x!tt.ptr<f32>, #blocked_b>
    tt.return
  }
}
