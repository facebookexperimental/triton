// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_adj = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0,1]}>

#ordered_strided = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
#ordered_narrow = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL: @test_1d_simple
tt.func private @test_1d_simple(%arg0: tensor<8xi32, #layout>) -> tensor<8xi32, #layout> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 7
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  tt.return %0 : tensor<8xi32, #layout>
}

// CHECK-LABEL: @test_1d_grouped
tt.func private @test_1d_grouped(%arg0: tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 3
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj>
  tt.return %0 : tensor<8xi32, #layout_adj>
}

// CHECK-LABEL: @test_2d_grouped
tt.func private @test_2d_grouped(%arg0: tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 7
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d>
  tt.return %0 : tensor<16x1xi32, #layout_2d>
}

// Keep register groups separate through the lane tree, then exchange warp
// totals once even when the scan spans multiple register groups.
// CHECK-LABEL: @ordered_groups
// CHECK: [[PAIR0:%.*]] = fadd float
// CHECK: [[PAIR1:%.*]] = fadd float
// CHECK: [[BITS:%.*]] = bitcast float [[PAIR0]] to i32
// CHECK: @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[BITS]],
// CHECK: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: @llvm.nvvm.shfl.sync.bfly
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: ret
tt.func private @ordered_groups(%arg0: tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false, reduction_ordering = "inner_tree"}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj>
  tt.return %0 : tensor<128xf32, #layout_adj>
}

// CHECK-LABEL: @ordered_reverse
// CHECK: shfl.sync.idx
// CHECK: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: @llvm.nvvm.shfl.sync.bfly
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: ret
tt.func private @ordered_reverse(%arg0: tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true, reduction_ordering = "inner_tree"}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj>
  tt.return %0 : tensor<128xf32, #layout_adj>
}

// CHECK-LABEL: @ordered_strided
// CHECK: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: @llvm.nvvm.shfl.sync.bfly
// CHECK-NOT: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: ret
tt.func private @ordered_strided(%arg0: tensor<4x32xf32, #ordered_strided>) -> tensor<4x32xf32, #ordered_strided> {
  %0 = "tt.scan"(%arg0) <{axis = 1 : i32, reverse = false, reduction_ordering = "inner_tree"}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<4x32xf32, #ordered_strided>) -> tensor<4x32xf32, #ordered_strided>
  tt.return %0 : tensor<4x32xf32, #ordered_strided>
}

// With fewer scan lanes than scan warps, retain shared-memory tree levels.
// CHECK-LABEL: @ordered_narrow
// CHECK: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: @llvm.nvvm.barrier.cta.sync.aligned.all
// CHECK: ret
tt.func private @ordered_narrow(%arg0: tensor<8x16xf32, #ordered_narrow>) -> tensor<8x16xf32, #ordered_narrow> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false, reduction_ordering = "inner_tree"}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<8x16xf32, #ordered_narrow>) -> tensor<8x16xf32, #ordered_narrow>
  tt.return %0 : tensor<8x16xf32, #ordered_narrow>
}

// This just prevents the test functions from being DCE'd.
tt.func public @anchor(%ptr: !llvm.ptr, %arg0: !llvm.struct<(i32)>, %arg1: !llvm.struct<(i32, i32)>, %arg2: !llvm.struct<(i32)>,
    %ordered0: !llvm.struct<(f32, f32, f32, f32)>,
    %ordered1: !llvm.struct<(f32, f32, f32, f32)>,
    %ordered2: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>,
    %ordered3: !llvm.struct<(f32, f32, f32, f32)>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i32)> to tensor<8xi32, #layout>
  %1 = tt.call @test_1d_simple(%0) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<8xi32, #layout> to !llvm.struct<(i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32)>, !llvm.ptr

  %3 = builtin.unrealized_conversion_cast %arg1 : !llvm.struct<(i32, i32)> to tensor<8xi32, #layout_adj>
  %4 = tt.call @test_1d_grouped(%3) : (tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj>
  %5 = builtin.unrealized_conversion_cast %4 : tensor<8xi32, #layout_adj> to !llvm.struct<(i32, i32)>
  llvm.store volatile %5, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr

  %6 = builtin.unrealized_conversion_cast %arg2 : !llvm.struct<(i32)> to tensor<16x1xi32, #layout_2d>
  %7 = tt.call @test_2d_grouped(%6) : (tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d>
  %8 = builtin.unrealized_conversion_cast %7 : tensor<16x1xi32, #layout_2d> to !llvm.struct<(i32)>
  llvm.store volatile %8, %ptr : !llvm.struct<(i32)>, !llvm.ptr


  %input0 = builtin.unrealized_conversion_cast %ordered0 : !llvm.struct<(f32, f32, f32, f32)> to tensor<128xf32, #layout_adj>
  %scan0 = tt.call @ordered_groups(%input0) : (tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj>
  %output0 = builtin.unrealized_conversion_cast %scan0 : tensor<128xf32, #layout_adj> to !llvm.struct<(f32, f32, f32, f32)>
  llvm.store volatile %output0, %ptr : !llvm.struct<(f32, f32, f32, f32)>, !llvm.ptr

  %input1 = builtin.unrealized_conversion_cast %ordered1 : !llvm.struct<(f32, f32, f32, f32)> to tensor<128xf32, #layout_adj>
  %scan1 = tt.call @ordered_reverse(%input1) : (tensor<128xf32, #layout_adj>) -> tensor<128xf32, #layout_adj>
  %output1 = builtin.unrealized_conversion_cast %scan1 : tensor<128xf32, #layout_adj> to !llvm.struct<(f32, f32, f32, f32)>
  llvm.store volatile %output1, %ptr : !llvm.struct<(f32, f32, f32, f32)>, !llvm.ptr

  %input2 = builtin.unrealized_conversion_cast %ordered2 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> to tensor<4x32xf32, #ordered_strided>
  %scan2 = tt.call @ordered_strided(%input2) : (tensor<4x32xf32, #ordered_strided>) -> tensor<4x32xf32, #ordered_strided>
  %output2 = builtin.unrealized_conversion_cast %scan2 : tensor<4x32xf32, #ordered_strided> to !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.store volatile %output2, %ptr : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>, !llvm.ptr

  %input3 = builtin.unrealized_conversion_cast %ordered3 : !llvm.struct<(f32, f32, f32, f32)> to tensor<8x16xf32, #ordered_narrow>
  %scan3 = tt.call @ordered_narrow(%input3) : (tensor<8x16xf32, #ordered_narrow>) -> tensor<8x16xf32, #ordered_narrow>
  %output3 = builtin.unrealized_conversion_cast %scan3 : tensor<8x16xf32, #ordered_narrow> to !llvm.struct<(f32, f32, f32, f32)>
  llvm.store volatile %output3, %ptr : !llvm.struct<(f32, f32, f32, f32)>, !llvm.ptr

  tt.return
}

}
