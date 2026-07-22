// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s

// Regression test for B-19-F2 / T273499458.
//
// The first task-1 loop result is consumed only as the init arg of a second
// task-1 loop. Specialization must keep that first loop result and feed it to
// the second cloned loop instead of substituting the first loop's original init
// value.

// CHECK-LABEL: @loop_result_feeds_second_loop
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: %[[FIRST:.*]] = scf.for {{.*}} -> (i32) {
// CHECK: arith.index_cast
// CHECK: scf.yield {{.*}} : i32
// CHECK: %[[SECOND:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[FIRST]]) -> (i32) {
// CHECK: arith.addi
// CHECK: scf.yield {{.*}} : i32
// CHECK: tt.splat %[[SECOND]]

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_result_feeds_second_loop(%src: tensor<16xf32, #blocked>, %dst: tensor<16x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : index
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
    %zero_vec = arith.constant {ttg.partition = array<i32: 1>} dense<0> : tensor<16xi32, #blocked>
    %alloc = ttg.local_alloc {ttg.partition = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 {
      %first = scf.for %i = %c0 to %c1 step %c1 iter_args(%ignored = %c0_i32) -> (i32) {
        %cast_i = arith.index_cast %i {ttg.partition = array<i32: 1>} : index to i32
        %next = arith.addi %cast_i, %c1_i32 {ttg.partition = array<i32: 1>} : i32
        scf.yield {ttg.partition = array<i32: 1>} %next : i32
      } {ttg.partition = array<i32: 1>}
      %second = scf.for %j = %c0 to %c1 step %c1 iter_args(%carry = %first) -> (i32) {
        %next_second = arith.addi %carry, %c1_i32 {ttg.partition = array<i32: 1>} : i32
        scf.yield {ttg.partition = array<i32: 1>} %next_second : i32
      } {ttg.partition = array<i32: 1>}
      %used_vec = tt.splat %second {ttg.partition = array<i32: 1>} : i32 -> tensor<16xi32, #blocked>
      %mask = arith.cmpi sge, %used_vec, %zero_vec {ttg.partition = array<i32: 1>} : tensor<16xi32, #blocked>
      %stored = arith.addf %src, %src {ttg.partition = array<i32: 0>} : tensor<16xf32, #blocked>
      ttg.local_store %stored, %alloc {ttg.partition = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
      tt.store %dst, %loaded, %mask {ttg.partition = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked>
    } {ttg.partition = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    tt.return
  }
}
