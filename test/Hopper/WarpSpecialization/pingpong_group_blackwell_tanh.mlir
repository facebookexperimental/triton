// RUN: triton-opt %s --nvgpu-test-ping-pong-prep="capability=100 num-stages=3" | FileCheck %s
// Ping-pong coverage for the blackwell-gdpa.py activation region: a pure
// SFU .approx inline asm (tanh.approx here, ex2.approx below) on a rank-2
// tensor is an expensive op on Blackwell. Two SFU ops per task separated by
// a store form two groups; non-SFU asm, rank-1 tanh, and SFU asm mixed with
// barrier ops (including bar.warp.sync / bar.cluster.wait spellings) get no
// pingpong_id. SFU math ops (rsqrt, sin) group the same way as the asm
// forms, and identifiers like bravo in comments don't disqualify.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @pingpong_group_blackwell_tanh
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST0:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[ID0:[0-9]+]] : i32
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST1:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[ID1:[0-9]+]] : i32
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST0]] : i32
// CHECK-SAME: pingpong_id = [[ID0]] : i32
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[FIRST1]] : i32
// CHECK-SAME: pingpong_id = [[ID1]] : i32
tt.func public @pingpong_group_blackwell_tanh(
    %arg0: tensor<64x128xf32, #blocked>,
    %arg1: tensor<64x128xf32, #blocked>,
    %out: tensor<64x128x!tt.ptr<f32>, #blocked>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %arg0, %acc1 = %arg1)
      -> (tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>) : i32 {
    %t0 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    tt.store %out, %t0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
    %t1 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %t0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %t2 = tt.elementwise_inline_asm "ex2.approx.f32 $0, $1;" {async_task_id = array<i32: 2>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc1 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    tt.store %out, %t2 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
    %t3 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {async_task_id = array<i32: 2>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %t2 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    scf.yield {async_task_id = array<i32: 1, 2>} %t1, %t3 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

// CHECK-LABEL: @no_pingpong_for_non_sfu_asm
// CHECK-NOT: pingpong_id
// CHECK: tt.return
tt.func public @no_pingpong_for_non_sfu_asm(
    %arg0: tensor<64x128xf32, #blocked>,
    %arg1: tensor<64x128xf32, #blocked>,
    %rarg0: tensor<128xf32, #blocked1>,
    %rarg1: tensor<128xf32, #blocked1>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:4 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %arg0, %acc1 = %arg1, %racc0 = %rarg0, %racc1 = %rarg1)
      -> (tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>) : i32 {
    %f0 = tt.elementwise_inline_asm "mul.f32x2 $0, $1, $2;" {async_task_id = array<i32: 1>, constraints = "=r,r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc0, %acc0 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %f1 = tt.elementwise_inline_asm "mul.f32x2 $0, $1, $2;" {async_task_id = array<i32: 2>, constraints = "=r,r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc1, %acc1 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %u0 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %racc0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked1>
    %u1 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {async_task_id = array<i32: 2>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %racc1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked1>
    scf.yield {async_task_id = array<i32: 1, 2>} %f0, %f1, %u0, %u1 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

// CHECK-LABEL: @no_pingpong_for_unsafe_asm
// CHECK-NOT: pingpong_id
// CHECK: tt.return
tt.func public @no_pingpong_for_unsafe_asm(
    %arg0: tensor<64x128xf32, #blocked>,
    %arg1: tensor<64x128xf32, #blocked>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %arg0, %acc1 = %arg1)
      -> (tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>) : i32 {
    %t0 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1; bar.sync 0;" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %t0b = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1; bar.cluster.wait 0;" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %t0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %t1 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1; bar.warp.sync 0;" {async_task_id = array<i32: 2>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc1 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    scf.yield {async_task_id = array<i32: 1, 2>} %t0b, %t1 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

// CHECK-LABEL: @pingpong_group_blackwell_math
// CHECK:      math.rsqrt
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[MFIRST0:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[MID0:[0-9]+]] : i32
// CHECK:      math.sin
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[MFIRST1:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[MID1:[0-9]+]] : i32
// CHECK:      math.rsqrt
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[MFIRST0]] : i32
// CHECK-SAME: pingpong_id = [[MID0]] : i32
// CHECK:      math.sin
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[MFIRST1]] : i32
// CHECK-SAME: pingpong_id = [[MID1]] : i32
tt.func public @pingpong_group_blackwell_math(
    %arg0: tensor<64x128xf32, #blocked>,
    %arg1: tensor<64x128xf32, #blocked>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %arg0, %acc1 = %arg1)
      -> (tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>) : i32 {
    %t0 = math.rsqrt %acc0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #blocked>
    %t1 = math.sin %t0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #blocked>
    %t2 = math.rsqrt %acc1 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #blocked>
    %t3 = math.sin %t2 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #blocked>
    scf.yield {async_task_id = array<i32: 1, 2>} %t1, %t3 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

// CHECK-LABEL: @pingpong_group_sfu_asm_comment
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 1>
// CHECK-SAME: pingpong_first_partition_id = [[CFIRST:[0-9]+]] : i32
// CHECK-SAME: pingpong_id = [[CID:[0-9]+]] : i32
// CHECK:      tt.elementwise_inline_asm
// CHECK-SAME: async_task_id = array<i32: 2>
// CHECK-SAME: pingpong_first_partition_id = [[CFIRST]] : i32
// CHECK-SAME: pingpong_id = [[CID]] : i32
tt.func public @pingpong_group_sfu_asm_comment(
    %arg0: tensor<64x128xf32, #blocked>,
    %arg1: tensor<64x128xf32, #blocked>,
    %num_tiles: i32
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.for %iv = %c0 to %num_tiles step %c1
      iter_args(%acc0 = %arg0, %acc1 = %arg1)
      -> (tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>) : i32 {
    %t0 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1; // bravo interpreter" {async_task_id = array<i32: 1>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %t1 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1; // bravo interpreter" {async_task_id = array<i32: 2>, constraints = "=r,r", loop.cluster = 0 : i32, loop.stage = 1 : i32, packed_element = 1 : i32, pure = true} %acc1 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked>
    scf.yield {async_task_id = array<i32: 1, 2>} %t0, %t1 : tensor<64x128xf32, #blocked>, tensor<64x128xf32, #blocked>
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}

} // module
