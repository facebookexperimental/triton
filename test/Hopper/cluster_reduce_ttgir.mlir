// RUN: triton-opt %s -split-input-file \
// RUN:   --nvgpu-form-multi-cta-collectives=compute-capability=90 \
// RUN:   --nvgpu-lower-cluster-reduce-to-ttgir --verify-each --verify-diagnostics | FileCheck %s
// RUN: triton-opt %s -split-input-file \
// RUN:   --nvgpu-form-multi-cta-collectives=compute-capability=100 \
// RUN:   --nvgpu-lower-cluster-reduce-to-ttgir --verify-each --verify-diagnostics | FileCheck %s
// RUN: not triton-opt %s -split-input-file \
// RUN:   --nvgpu-form-multi-cta-collectives='compute-capability=100 auto-warp-specialization=true' \
// RUN:   2>&1 | FileCheck %s --check-prefix=AUTOWS

// AUTOWS: error: multi_cta does not support automatic warp specialization in v1

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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_invariant_column_base_overlap(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %column_base = tt.addptr %ptr, %lane : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    scf.for %off = %lower to %n step %c64 : i32 {
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptrs = tt.addptr %column_base, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: @bounded_invariant_column_base
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @bounded_invariant_column_base(%output: !tt.ptr<f32>, %n: i32, %input: tensor<16xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %column_base = tt.addptr %ptr, %lane : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    scf.for %off = %lower to %n step %c64 : i32 {
      %base = tt.splat %off : i32 -> tensor<16xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<16xi32, #blocked>
      %ptrs = tt.addptr %column_base, %cols : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      tt.store %ptrs, %input : tensor<16x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<16xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_projected_row_base_overlap(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #slice>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice>
    %lane2d = tt.expand_dims %lane {axis = 1 : i32} : tensor<32xi32, #slice> -> tensor<32x1xi32, #parent>
    %values = tt.expand_dims %input {axis = 1 : i32} : tensor<32xf32, #slice> -> tensor<32x1xf32, #parent>
    %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #parent>
    %row_base = tt.addptr %ptr, %lane2d : tensor<32x1x!tt.ptr<f32>, #parent>, tensor<32x1xi32, #parent>
    scf.for %off = %lower to %n step %c64 : i32 {
      %iv = tt.splat %off : i32 -> tensor<32xi32, #slice>
      %offsets = arith.addi %iv, %lane : tensor<32xi32, #slice>
      %offsets2d = tt.expand_dims %offsets {axis = 1 : i32} : tensor<32xi32, #slice> -> tensor<32x1xi32, #parent>
      %ptrs = tt.addptr %row_base, %offsets2d : tensor<32x1x!tt.ptr<f32>, #parent>, tensor<32x1xi32, #parent>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %values : tensor<32x1x!tt.ptr<f32>, #parent>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #slice>) -> f32
    tt.return %sum : f32
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @bounded_projected_row_base
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @bounded_projected_row_base(%output: !tt.ptr<f32>, %n: i32, %input: tensor<16xf32, #slice>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice>
    %lane2d = tt.expand_dims %lane {axis = 1 : i32} : tensor<16xi32, #slice> -> tensor<16x1xi32, #parent>
    %values = tt.expand_dims %input {axis = 1 : i32} : tensor<16xf32, #slice> -> tensor<16x1xf32, #parent>
    %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #parent>
    %row_base = tt.addptr %ptr, %lane2d : tensor<16x1x!tt.ptr<f32>, #parent>, tensor<16x1xi32, #parent>
    scf.for %off = %lower to %n step %c64 : i32 {
      %iv = tt.splat %off : i32 -> tensor<16xi32, #slice>
      %offsets = arith.addi %iv, %lane : tensor<16xi32, #slice>
      %offsets2d = tt.expand_dims %offsets {axis = 1 : i32} : tensor<16xi32, #slice> -> tensor<16x1xi32, #parent>
      %ptrs = tt.addptr %row_base, %offsets2d : tensor<16x1x!tt.ptr<f32>, #parent>, tensor<16x1xi32, #parent>
      tt.store %ptrs, %values : tensor<16x1x!tt.ptr<f32>, #parent>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<16xf32, #slice>) -> f32
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

// -----

#tensor_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// Bind protocol operands so unrelated stores or barriers cannot satisfy these checks.
// CHECK-LABEL: @tensor_sum(
// CHECK-SAME: %[[PARTIAL:[a-zA-Z0-9_]+]]: tensor<128xf32, #blocked>
// CHECK: %[[RANK:[a-zA-Z0-9_]+]] = nvg.cluster_id
// CHECK: %[[SLOTS:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<2x128xf32
// CHECK: %[[LOCAL_SLOT:[a-zA-Z0-9_]+]] = ttg.memdesc_index %[[SLOTS]][%[[RANK]]]
// CHECK: %[[BARRIER:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
// CHECK: ttng.init_barrier %[[BARRIER]], 1
// CHECK: %[[ACTIVE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: ttng.barrier_expect %[[BARRIER]], 512, %[[ACTIVE]]
// CHECK: ttng.cluster_arrive
// CHECK: ttng.cluster_wait
// CHECK: ttg.local_store %[[PARTIAL]], %[[LOCAL_SLOT]]
// CHECK: %[[RANK0:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[SEND0:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[RANK]], %[[RANK0]] : i32
// CHECK: scf.if %[[SEND0]] {
// CHECK: ttg.async_remote_shmem_store %[[PARTIAL]], rank %[[RANK0]], %[[LOCAL_SLOT]] barrier %[[BARRIER]]
// CHECK: }
// CHECK-NOT: ttg.async_remote_shmem_store
// CHECK: %[[RANK1:[a-zA-Z0-9_]+]] = arith.constant 1 : i32
// CHECK: %[[SEND1:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[RANK]], %[[RANK1]] : i32
// CHECK: scf.if %[[SEND1]] {
// CHECK: ttg.async_remote_shmem_store %[[PARTIAL]], rank %[[RANK1]], %[[LOCAL_SLOT]] barrier %[[BARRIER]]
// CHECK: }
// CHECK-NOT: ttg.async_remote_shmem_store
// CHECK: %[[PHASE:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE]], %[[ACTIVE]]
// CHECK: %[[INDEX0:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[VIEW0:[a-zA-Z0-9_]+]] = ttg.memdesc_index %[[SLOTS]][%[[INDEX0]]]
// CHECK: %[[VALUE0:[a-zA-Z0-9_]+]] = ttg.local_load %[[VIEW0]]
// CHECK: %[[INDEX1:[a-zA-Z0-9_]+]] = arith.constant 1 : i32
// CHECK: %[[VIEW1:[a-zA-Z0-9_]+]] = ttg.memdesc_index %[[SLOTS]][%[[INDEX1]]]
// CHECK: %[[VALUE1:[a-zA-Z0-9_]+]] = ttg.local_load %[[VIEW1]]
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[VALUE0]], %[[VALUE1]] : tensor<128xf32, #blocked>
// CHECK: ttg.barrier local
// CHECK: tt.return %[[SUM]] : tensor<128xf32, #blocked>
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_sum(%partial: tensor<128xf32, #tensor_blocked>) -> tensor<128xf32, #tensor_blocked> {
    %result = "ttng.cluster_reduce"(%partial) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      "ttng.cluster_reduce.return"(%sum) : (f32) -> ()
    }) {cluster_axis = 0 : i32, group_size = 2 : i32} : (tensor<128xf32, #tensor_blocked>) -> tensor<128xf32, #tensor_blocked>
    tt.return %result : tensor<128xf32, #tensor_blocked>
  }
}

// -----

#loop_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#loop_slice = #ttg.slice<{dim = 1, parent = #loop_parent}>

// CHECK-LABEL: @loop_tensor_sum
// CHECK: scf.for
// CHECK-NOT: scf.yield
// CHECK: ttng.init_barrier
// CHECK-NOT: scf.yield
// CHECK: ttng.barrier_expect {{.*}}, 512,
// CHECK-NOT: scf.yield
// CHECK: ttng.cluster_arrive
// CHECK-NOT: scf.yield
// CHECK: ttng.cluster_wait
// CHECK-NOT: scf.yield
// CHECK: ttg.async_remote_shmem_store
// CHECK-NOT: scf.yield
// CHECK: ttg.async_remote_shmem_store
// CHECK-NOT: scf.yield
// CHECK: ttng.wait_barrier
// CHECK-NOT: scf.yield
// CHECK: ttg.barrier local
// CHECK: scf.yield
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 2 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_tensor_sum(%partial: tensor<128x32xf32, #loop_parent>) -> tensor<128xf32, #loop_slice> {
    %pid = tt.get_program_id y : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_index = arith.constant 2 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<128xf32, #loop_slice>
    %result = scf.for %i = %c0 to %c2_index step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #loop_slice> {
      %reduced = "tt.reduce"(%partial) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
      }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #loop_parent>) -> tensor<128xf32, #loop_slice>
      scf.yield %reduced : tensor<128xf32, #loop_slice>
    }
    tt.return %result : tensor<128xf32, #loop_slice>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_task_id(%input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    // expected-error @+1 {{multi_cta does not support warp specialization in v1}}
    %rank = arith.remsi %pid, %c2 {async_task_id = array<i32: 0>} : i32
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
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_partition(%input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    // expected-error @+1 {{multi_cta does not support warp specialization in v1}}
    %rank = arith.remsi %pid, %c2 {ttg.partition = array<i32: 0>} : i32
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
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_warp_specialize_marker(%input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    // expected-error @+1 {{multi_cta does not support warp specialization in v1}}
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.addf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) {axis = 0 : i32, tt.multi_cta, tt.warp_specialize} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_cumulative_lanes(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %cols2 = arith.addi %cols, %lane : tensor<32xi32, #blocked>
      %ptrs = tt.addptr %ptr, %cols2 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_chained_pointer_lanes(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs0 = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      %ptrs = tt.addptr %ptrs0, %lane : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: @bounded_lane_sum
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @bounded_lane_sum(%output: !tt.ptr<f32>, %n: i32, %input: tensor<16xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<16xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<16xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
      %cols2 = arith.addi %cols, %lane : tensor<16xi32, #blocked>
      %ptrs = tt.addptr %ptr, %cols2 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      tt.store %ptrs, %input : tensor<16x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<16xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_loop_carried_base(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    scf.for %off = %lower to %n step %c64 iter_args(%carried = %output) -> (!tt.ptr<f32>) : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %carried : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
      %minus = arith.constant -32 : i32
      %next = tt.addptr %carried, %minus : !tt.ptr<f32>, i32
      scf.yield %next : !tt.ptr<f32>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank_dependent_for_result(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %count = scf.for %i = %lower to %n step %c64 iter_args(%cnt = %c0) -> i32 : i32 {
      %next = arith.addi %cnt, %c1 : i32
      scf.yield %next : i32
    }
    %delta = arith.muli %count, %c32 : i32
    %shifted = tt.addptr %output, %delta : !tt.ptr<f32>, i32
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %shifted : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_captured_while_base(%output: !tt.ptr<f32>, %n: i32, %input: tensor<32xf32, #blocked>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %count = scf.while (%i = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %i, %rank : i32
      scf.condition(%cond) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1 : i32
      scf.yield %next : i32
    }
    %delta = arith.muli %count, %c32 : i32
    %shifted = tt.addptr %output, %delta : !tt.ptr<f32>, i32
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %shifted : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      // expected-error @+1 {{multi_cta store address must be a unit-stride projection}}
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_async_tma_copy(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %tile : !tt.tensordesc<32x32xf32, #tma>, !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_async_tma_reduce(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    ttng.async_tma_reduce add, %desc[%c0, %c0] %tile : !tt.tensordesc<32x32xf32, #tma>, !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_async_global_store(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    ttng.async_store %buf, %output, %c128 : !ttg.memdesc<32xf32, #shared, #smem, mutable>, !tt.ptr<f32>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_remote_store(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    ttg.async_remote_shmem_store %input, rank %rank, %buf barrier %barrier : tensor<32xf32, #blocked> -> !ttg.memdesc<32xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_remote_copy(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    ttg.async_remote_shmem_copy %buf, rank %rank, %buf barrier %barrier : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_impure_inline_asm(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // expected-error @+1 {{multi_cta memory writes require a proven ownership pattern}}
    %asm = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %rank : i32 -> i32
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @allow_local_scratch
// CHECK: ttg.local_store
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @allow_local_scratch(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %desc: !tt.tensordesc<32x32xf32, #tma>) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #tma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttg.local_store %input, %buf : tensor<32xf32, #blocked> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_same_buffer_side_store(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %statistics: !tt.ptr<f32>, %n: i32) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %is_zero = arith.cmpi eq, %rank, %c0 : i32
    %mask = tt.splat %is_zero : i1 -> tensor<32xi1, #blocked>
    %stat_base = tt.addptr %output, %c32 : !tt.ptr<f32>, i32
    %stat_ptr = tt.splat %stat_base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %stat_ptrs = tt.addptr %stat_ptr, %lane : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    // expected-error @+1 {{multi_cta rank-zero side stores must not share a buffer}}
    tt.store %stat_ptrs, %input, %mask : tensor<32x!tt.ptr<f32>, #blocked>
    scf.for %off = %lower to %n step %c64 : i32 {
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @separate_statistics_buffer
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @separate_statistics_buffer(%input: tensor<32xf32, #blocked>, %output: !tt.ptr<f32>, %statistics: !tt.ptr<f32>, %n: i32) -> f32 {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %is_zero = arith.cmpi eq, %rank, %c0 : i32
    %mask = tt.splat %is_zero : i1 -> tensor<32xi1, #blocked>
    %stat_base = tt.addptr %statistics, %c32 : !tt.ptr<f32>, i32
    %stat_ptr = tt.splat %stat_base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %stat_ptrs = tt.addptr %stat_ptr, %lane : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %stat_ptrs, %input, %mask : tensor<32x!tt.ptr<f32>, #blocked>
    scf.for %off = %lower to %n step %c64 : i32 {
      %base = tt.splat %off : i32 -> tensor<32xi32, #blocked>
      %cols = arith.addi %base, %lane : tensor<32xi32, #blocked>
      %ptr = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %ptr, %cols : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      tt.store %ptrs, %input : tensor<32x!tt.ptr<f32>, #blocked>
    }
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    tt.return %sum : f32
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @swapped_tensor_addf
// CHECK: ttg.async_remote_shmem_store
// CHECK: arith.addf {{.*}} : tensor<128xf32
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swapped_tensor_addf(%input: tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice> {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %result = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %rhs, %lhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice>
    tt.return %result : tensor<128xf32, #slice>
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @swapped_tensor_maxnumf
// CHECK: ttg.async_remote_shmem_store
// CHECK: arith.maxnumf {{.*}} : tensor<128xf32
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swapped_tensor_maxnumf(%input: tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice> {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %result = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.maxnumf %rhs, %lhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice>
    tt.return %result : tensor<128xf32, #slice>
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @swapped_tensor_maximumf
// CHECK: ttg.async_remote_shmem_store
// CHECK: arith.maximumf {{.*}} : tensor<128xf32
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swapped_tensor_maximumf(%input: tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice> {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %result = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.maximumf %rhs, %lhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice>
    tt.return %result : tensor<128xf32, #slice>
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @swapped_tensor_minnumf
// CHECK: ttg.async_remote_shmem_store
// CHECK: arith.minnumf {{.*}} : tensor<128xf32
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swapped_tensor_minnumf(%input: tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice> {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %result = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.minnumf %rhs, %lhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice>
    tt.return %result : tensor<128xf32, #slice>
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
// CHECK-LABEL: @swapped_tensor_minimumf
// CHECK: ttg.async_remote_shmem_store
// CHECK: arith.minimumf {{.*}} : tensor<128xf32
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @swapped_tensor_minimumf(%input: tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice> {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %result = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.minimumf %rhs, %lhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<128x32xf32, #parent>) -> tensor<128xf32, #slice>
    tt.return %result : tensor<128xf32, #slice>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_while_collective_bound(%input: tensor<32xf32, #blocked>) {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %bound = scf.while (%k = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %k, %rank : i32
      scf.condition(%cond) %k : i32
    } do {
    ^bb0(%k: i32):
      %next = arith.addi %k, %c1 : i32
      scf.yield %next : i32
    }
    scf.for %i = %c0 to %bound step %c1 : i32 {
      // expected-error @+1 {{multi_cta loop bounds and step must not depend on cluster rank}}
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_execute_region_collective_bound(%input: tensor<32xf32, #blocked>) {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %bound = scf.execute_region -> i32 {
      scf.yield %rank : i32
    }
    scf.for %i = %c0 to %bound step %c1 : i32 {
      // expected-error @+1 {{multi_cta loop bounds and step must not depend on cluster rank}}
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_index_switch_collective_bound(%input: tensor<32xf32, #blocked>) {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %selector = arith.constant 0 : index
    %bound = scf.index_switch %selector -> i32
    case 0 {
      scf.yield %rank : i32
    }
    default {
      scf.yield %c0 : i32
    }
    scf.for %i = %c0 to %bound step %c1 : i32 {
      // expected-error @+1 {{multi_cta loop bounds and step must not depend on cluster rank}}
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @reject_private_helper(%input: tensor<32xf32, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = nvg.cluster_id
    scf.for %i = %c0 to %n step %c1 : i32 {
      // expected-error @+1 {{multi_cta reduce must be contained in a kernel tt.func}}
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_cluster_rank_bound(%input: tensor<32xf32, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = nvg.cluster_id
    scf.for %i = %c0 to %rank step %c1 : i32 {
      // expected-error @+1 {{multi_cta loop bounds and step must not depend on cluster rank}}
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: @cluster_rank_uniform_kernel
// CHECK: ttg.async_remote_shmem_store
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cluster_rank_uniform_kernel(%input: tensor<32xf32, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %rank = nvg.cluster_id
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = "tt.reduce"(%input) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32, tt.multi_cta} : (tensor<32xf32, #blocked>) -> f32
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#cols = #ttg.slice<{dim = 0, parent = #parent}>
#rows = #ttg.slice<{dim = 1, parent = #parent}>
#linear = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: @converted_2d_store
// CHECK: ttg.async_remote_shmem_store
// CHECK: ttg.convert_layout
// CHECK: tt.store
// CHECK-NOT: ttng.cluster_reduce
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 2 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @converted_2d_store(%output: !tt.ptr<f32>, %n: i32, %input: tensor<4x32xf32, #parent>) {
    %pid = tt.get_program_id y : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %rank = arith.remsi %pid, %c2 : i32
    %lower = arith.muli %rank, %c32 : i32
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #rows>
    %strides = tt.splat %n : i32 -> tensor<4xi32, #rows>
    %row_offsets = arith.muli %r, %strides : tensor<4xi32, #rows>
    %row_expanded = tt.expand_dims %row_offsets {axis = 1 : i32} : tensor<4xi32, #rows> -> tensor<4x1xi32, #parent>
    %row_offsets2d = tt.broadcast %row_expanded : tensor<4x1xi32, #parent> -> tensor<4x32xi32, #parent>
    %base = tt.splat %output : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>, #parent>
    %row_ptrs = tt.addptr %base, %row_offsets2d : tensor<4x32x!tt.ptr<f32>, #parent>, tensor<4x32xi32, #parent>
    %sum = "tt.reduce"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) {axis = 1 : i32, tt.multi_cta} : (tensor<4x32xf32, #parent>) -> tensor<4xf32, #rows>
    scf.for %off = %lower to %n step %c64 : i32 {
      %lane = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #linear>
      %iv = tt.splat %off : i32 -> tensor<32xi32, #linear>
      %offsets = arith.addi %iv, %lane : tensor<32xi32, #linear>
      %converted = ttg.convert_layout %offsets : tensor<32xi32, #linear> -> tensor<32xi32, #cols>
      %expanded = tt.expand_dims %converted {axis = 0 : i32} : tensor<32xi32, #cols> -> tensor<1x32xi32, #parent>
      %offsets2d = tt.broadcast %expanded : tensor<1x32xi32, #parent> -> tensor<4x32xi32, #parent>
      %ptrs = tt.addptr %row_ptrs, %offsets2d : tensor<4x32x!tt.ptr<f32>, #parent>, tensor<4x32xi32, #parent>
      tt.store %ptrs, %input : tensor<4x32x!tt.ptr<f32>, #parent>
    }
    tt.return
  }
}
