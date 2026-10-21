// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

// A blocked operand with an NPOT K (dim 0) has a Cyclic dim 0 and all radix-2
// bases. getOrderForMemory must read its per-dim order instead of failing on
// output flattening.

// CHECK: #[[$SHARED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-LABEL: @npot_k40_b_operand
// CHECK: %[[A:.*]] = ttg.local_alloc %arg0 : (tensor<40x64xf32, #{{.*}}>) -> !ttg.memdesc<40x64xf32, #[[$SHARED]], #smem>
// CHECK: ttg.local_load %[[A]] : !ttg.memdesc<40x64xf32, #[[$SHARED]], #smem> -> tensor<40x64xf32, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}}>>
// CHECK-NOT: ttg.convert_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @npot_k40_b_operand(%arg0: tensor<40x64xf32, #blocked>) -> tensor<40x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<40x64xf32, #blocked> -> tensor<40x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
    tt.return %0 : tensor<40x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
  }
}

// -----

// Power-of-two control: same layouts with K = 32.

// CHECK: #[[$SHARED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-LABEL: @pot_k32_b_operand
// CHECK: %[[A:.*]] = ttg.local_alloc %arg0 : (tensor<32x64xf32, #{{.*}}>) -> !ttg.memdesc<32x64xf32, #[[$SHARED]], #smem>
// CHECK: ttg.local_load %[[A]] : !ttg.memdesc<32x64xf32, #[[$SHARED]], #smem> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}}>>
// CHECK-NOT: ttg.convert_layout
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @pot_k32_b_operand(%arg0: tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
    tt.return %0 : tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
  }
}
