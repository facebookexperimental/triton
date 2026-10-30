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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK-LABEL: @canonical_ownership
// CHECK: ttg.async_remote_shmem_store
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 4 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @canonical_ownership(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) -> f32 {
    %pid = tt.get_program_id y : i32
    %c4 = arith.constant 4 : i32
    %c32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c4 : i32
    %lower = arith.muli %rank, %c32 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
    %partial = scf.for %off = %lower to %n step %c128 iter_args(%acc = %zero) -> (tensor<32xf32, #blocked>) : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %input : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      %tile = tt.load %ptrs : tensor<32x!tt.ptr<f32>, #blocked>
      %next = arith.addf %acc, %tile : tensor<32xf32, #blocked>
      scf.yield %next : tensor<32xf32, #blocked>
    }
    %sum = "tt.reduce"(%partial) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.addf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    scf.for %off = %lower to %n step %c128 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      tt.store %ptrs, %zero : tensor<32x!tt.ptr<f32>, #blocked>
    }
    tt.return %sum : f32
  }
}
