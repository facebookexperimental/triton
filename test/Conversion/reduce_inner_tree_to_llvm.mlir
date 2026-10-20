// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

// Test that the inner_tree reduction ordering produces count-up shuffle order
// (stride 2, 4, 8, 16) instead of the default count-down order (16, 8, 4, 2).
// For this layout, each register value belongs to a distinct contiguous group
// along the reduction axis. INNER_TREE keeps those groups separate through the
// warp reduction and combines the groups later when packing the result.

#linear = #ttg.linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @reduce_inner_tree
tt.func private @reduce_inner_tree(%arg0: tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> {
  // CHECK: [[SRC0:%.*]] = extractvalue {{.*}} %0, 0
  // CHECK: [[SRC1:%.*]] = extractvalue {{.*}} %0, 1
  // CHECK: [[SRC2:%.*]] = extractvalue {{.*}} %0, 2
  // CHECK: [[SRC3:%.*]] = extractvalue {{.*}} %0, 3

  // INNER_TREE count-up warp shuffle for each register group.
  // CHECK: [[S0_W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SRC0]], i32 2, i32 31)
  // CHECK: [[S0_A0:%.*]] = add i32 [[S0_W0]], [[SRC0]]
  // CHECK: [[S0_W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S0_A0]], i32 4, i32 31)
  // CHECK: [[S0_A1:%.*]] = add i32 [[S0_A0]], [[S0_W1]]
  // CHECK: [[S0_W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S0_A1]], i32 8, i32 31)
  // CHECK: [[S0_A2:%.*]] = add i32 [[S0_A1]], [[S0_W2]]
  // CHECK: [[S0_W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S0_A2]], i32 16, i32 31)

  // CHECK: [[S1_W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SRC1]], i32 2, i32 31)
  // CHECK: [[S1_A0:%.*]] = add i32 [[S1_W0]], [[SRC1]]
  // CHECK: [[S1_W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S1_A0]], i32 4, i32 31)
  // CHECK: [[S1_A1:%.*]] = add i32 [[S1_A0]], [[S1_W1]]
  // CHECK: [[S1_W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S1_A1]], i32 8, i32 31)
  // CHECK: [[S1_A2:%.*]] = add i32 [[S1_A1]], [[S1_W2]]
  // CHECK: [[S1_W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S1_A2]], i32 16, i32 31)

  // CHECK: [[S2_W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SRC2]], i32 2, i32 31)
  // CHECK: [[S2_A0:%.*]] = add i32 [[S2_W0]], [[SRC2]]
  // CHECK: [[S2_W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S2_A0]], i32 4, i32 31)
  // CHECK: [[S2_A1:%.*]] = add i32 [[S2_A0]], [[S2_W1]]
  // CHECK: [[S2_W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S2_A1]], i32 8, i32 31)
  // CHECK: [[S2_A2:%.*]] = add i32 [[S2_A1]], [[S2_W2]]
  // CHECK: [[S2_W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S2_A2]], i32 16, i32 31)

  // CHECK: [[S3_W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SRC3]], i32 2, i32 31)
  // CHECK: [[S3_A0:%.*]] = add i32 [[S3_W0]], [[SRC3]]
  // CHECK: [[S3_W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S3_A0]], i32 4, i32 31)
  // CHECK: [[S3_A1:%.*]] = add i32 [[S3_A0]], [[S3_W1]]
  // CHECK: [[S3_W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S3_A1]], i32 8, i32 31)
  // CHECK: [[S3_A2:%.*]] = add i32 [[S3_A1]], [[S3_W2]]
  // CHECK: [[S3_W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[S3_A2]], i32 16, i32 31)

  // CHECK: add i32 [[S0_A2]], [[S0_W3]]
  // CHECK: add i32 [[S1_A2]], [[S1_W3]]

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 0 : i32, reduction_ordering = "inner_tree"} : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>

  // CHECK: ret { i32, i32 }
  tt.return %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
}

tt.func @anchor(%ptr: !llvm.ptr, %arg0: tensor<32x16xi32, #linear>) {
  %0 = tt.call @reduce_inner_tree(%arg0) : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %1 = builtin.unrealized_conversion_cast %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> to !llvm.struct<(i32, i32)>
  llvm.store volatile %1, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr
  tt.return
}

}
