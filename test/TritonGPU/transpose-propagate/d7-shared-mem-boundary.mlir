// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D7: ttg.local_alloc consuming dot's result is classified as
// SharedMemBoundary, which behaves like BoundaryInsert (counted as
// boundary, not in ops).

// CHECK: plan roots=1
// CHECK-SAME: ops=1
// CHECK-SAME: boundary=1
// CHECK: tt.func @dot_into_local_alloc

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_into_local_alloc(
      %a: tensor<16x16xf16, #dotA>,
      %b: tensor<16x16xf16, #dotB>,
      %c: tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem> {
    %dot = tt.dot %a, %b, %c {tt.transpose_propagate_root}
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf16, #blocked>
    %alloc = ttg.local_alloc %dot : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    tt.return %alloc : !ttg.memdesc<16x16xf16, #shared, #smem>
  }
}
