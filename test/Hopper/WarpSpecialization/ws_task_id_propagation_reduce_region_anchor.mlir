// RUN: triton-opt %s --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression test for B-5-F1 / T273472464.
// A region-bodied anchor op should propagate its own task ID into its scalar
// region body, not a downstream consumer's task ID.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @reduce_body_keeps_anchor_task
  // CHECK:      %[[REDUCE:.*]] = "tt.reduce"(%{{.*}}) <{axis = 1 : i32}> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
  // CHECK-NEXT:   %[[BODY_ADD:.*]] = arith.addf %[[LHS]], %[[RHS]] {{.*}}ttg.partition = array<i32: 1>{{.*}} : f32
  // CHECK-NEXT:   tt.reduce.return %[[BODY_ADD]] {{.*}}ttg.partition = array<i32: 1>{{.*}} : f32
  // CHECK-NEXT: }) {{.*}} :
  // CHECK:      arith.addf %[[REDUCE]], %{{.*}} {{.*}}ttg.partition = array<i32: 0>{{.*}} : tensor<16xf32,
  tt.func public @reduce_body_keeps_anchor_task(%input: tensor<16x16xf32, #blocked>) {
    %bias = arith.constant dense<1.000000e+00> : tensor<16xf32, #slice>
    %sum = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {"ttg.partition" = array<i32: 1>} : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #slice>
    %out = arith.addf %sum, %bias {"ttg.partition" = array<i32: 0>} : tensor<16xf32, #slice>
    tt.return
  }
}
