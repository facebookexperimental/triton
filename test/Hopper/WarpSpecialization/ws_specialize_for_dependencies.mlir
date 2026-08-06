// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s

// Follow each yielded operand independently, while retaining an update that
// feeds this task through an operation owned by another task.

// CHECK-LABEL: @independent_yield_operands
// CHECK: partition0
// CHECK: scf.for
// CHECK-NOT: i32
// CHECK: arith.addi {{.*}} : i64

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @independent_yield_operands(%src: tensor<16xf32, #blocked>, %dst: tensor<16x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %zero_i32 = arith.constant {async_task_id = array<i32: 0>} 0 : i32
    %one_i32 = arith.constant {async_task_id = array<i32: 0>} 1 : i32
    %zero_i64 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
    %one_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %zero_vec = arith.constant {async_task_id = array<i32: 1>} dense<0> : tensor<16xi64, #blocked>
    %buffer = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    %result:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%dead = %zero_i32, %live = %zero_i64) -> (i32, i64) {
      %next_dead = arith.addi %dead, %one_i32 {async_task_id = array<i32: 0>} : i32
      %next_live = arith.addi %live, %one_i64 {async_task_id = array<i32: 0>} : i64
      %live_vec = tt.splat %next_live {async_task_id = array<i32: 1>} : i64 -> tensor<16xi64, #blocked>
      %mask = arith.cmpi sge, %live_vec, %zero_vec {async_task_id = array<i32: 1>} : tensor<16xi64, #blocked>
      %doubled = arith.addf %src, %src {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked>
      ttg.local_store %doubled, %buffer {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %buffer {async_task_id = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
      tt.store %dst, %loaded, %mask {async_task_id = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked>
      scf.yield {async_task_id = array<i32: 0, 1>} %next_dead, %next_live : i32, i64
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    tt.return
  }
}

// -----

// Preserve a loop result that directly yields a value defined outside the
// loop.

// CHECK-LABEL: @yield_function_argument
// CHECK: ttg.warp_specialize
// CHECK: partition0({{.*}}%[[YIELDED:arg[0-9]+]]: index, %[[INITIAL:arg[0-9]+]]: index
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %[[INITIAL]], {{.*}}) -> (index, i64) {
// CHECK: scf.yield {{.*}}%[[YIELDED]], {{.*}} : index, i64
// CHECK: arith.index_cast %[[LOOP]]#0

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem1 = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @yield_function_argument(%initial: index, %yielded: index, %src: tensor<16xf32, #blocked1>, %dst: tensor<16x!tt.ptr<f32>, #blocked1>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %mask = arith.constant {async_task_id = array<i32: 1>} dense<true> : tensor<16xi1, #blocked1>
    %buffer = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared1, #smem1, mutable>
    %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%carry = %initial) -> (index) {
      ttg.local_store %src, %buffer {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked1> -> !ttg.memdesc<16xf32, #shared1, #smem1, mutable>
      %loaded = ttg.local_load %buffer {async_task_id = array<i32: 1>} : !ttg.memdesc<16xf32, #shared1, #smem1, mutable> -> tensor<16xf32, #blocked1>
      tt.store %dst, %loaded, %mask {async_task_id = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1>} %yielded : index
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    %result_i32 = arith.index_cast %result {async_task_id = array<i32: 1>} : index to i32
    %result_vec = tt.splat %result_i32 {async_task_id = array<i32: 1>} : i32 -> tensor<16xi32, #blocked1>
    %zero_vec = arith.constant {async_task_id = array<i32: 1>} dense<0> : tensor<16xi32, #blocked1>
    %result_mask = arith.cmpi sge, %result_vec, %zero_vec {async_task_id = array<i32: 1>} : tensor<16xi32, #blocked1>
    tt.store %dst, %src, %result_mask {async_task_id = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
