// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1" -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIAG --implicit-check-not="stays ahead"

// Untagged ops after the warp-specialized loop read memory the loop wrote, so
// they must stay after the warp_specialize op.

// CHECK-LABEL: @post_loop_ops_after_ws
// CHECK-NOT: tt.load
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: tt.store
// CHECK: } : (
// CHECK-NEXT: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.store %{{.*}}, %[[V]]
// CHECK-NEXT: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_ops_after_ws(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// An untagged post-loop op that comes before a specialized post-loop op still
// moves after the warp_specialize op when no specialized op depends on it and
// the two cannot conflict in memory (here both only read).

// CHECK-LABEL: @interleaved_post_loop_ops_after_ws
// CHECK-NOT: tt.load %arg0
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: } : (
// CHECK-NEXT: %[[V:.*]] = tt.load
// CHECK-NEXT: tt.store %{{.*}}, %[[V]]
// CHECK-NEXT: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @interleaved_post_loop_ops_after_ws(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %out2: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    %w = tt.load %out {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out2, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// An untagged post-loop load that a specialized op depends on cannot move
// after the warp_specialize op, so it is reported instead of silently reading
// memory before the loop's writes land.

// CHECK-LABEL: @pinned_post_loop_load
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @pinned_post_loop_load(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    // DIAG: ws_specialize_post_loop_ops.mlir:[[@LINE+1]]:{{.*}} warning: {{.*}}stays ahead of the partitions
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out, %v {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// An untagged post-loop load of the buffer a later specialized op stores to
// cannot move past that store, so it is reported as well.

// CHECK-LABEL: @post_loop_load_before_specialized_store
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_load_before_specialized_store(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    // DIAG: ws_specialize_post_loop_ops.mlir:[[@LINE+1]]:{{.*}} warning: {{.*}}stays ahead of the partitions
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Different pointer arguments may alias (in-place kernels), so an untagged
// post-loop load cannot move past a specialized store to another argument
// either; it is reported.

// CHECK-LABEL: @post_loop_load_before_store_to_other_arg
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_loop_load_before_store_to_other_arg(%a: tensor<128x!tt.ptr<f32>, #blocked>, %b: tensor<128x!tt.ptr<f32>, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>, %out2: tensor<128x!tt.ptr<f32>, #blocked>, %n: index) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %one = arith.constant {async_task_id = array<i32: 0>} dense<1.0> : tensor<128xf32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 {
      tt.store %a, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      ttg.local_store %one, %alloc {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      tt.store %b, %loaded {async_task_id = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    // DIAG: ws_specialize_post_loop_ops.mlir:[[@LINE+1]]:{{.*}} warning: {{.*}}stays ahead of the partitions
    %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out, %one {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %out2, %v : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
