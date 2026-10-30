// RUN: triton-opt %s -split-input-file \
// RUN:   --nvgpu-form-multi-cta-collectives=compute-capability=100 \
// RUN:   --nvgpu-lower-cluster-reduce-to-ttgir --verify-each | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-LABEL: @marked_sum
// CHECK: ttng.init_barrier
// CHECK: ttng.barrier_expect
// CHECK: ttng.cluster_arrive
// CHECK: ttng.cluster_wait
// CHECK-COUNT-2: ttg.async_remote_shmem_store
// CHECK: ttng.wait_barrier
// CHECK: arith.addf
// CHECK: ttg.barrier
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @marked_sum(%input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.addf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-LABEL: @single_cta_noop
// CHECK: "tt.reduce"
// CHECK-NOT: tt.multi_cta
// CHECK-NOT: ttng.cluster_reduce
// CHECK-NOT: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @single_cta_noop(%input: tensor<32xf32, #blocked>) -> f32 {
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.addf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}
