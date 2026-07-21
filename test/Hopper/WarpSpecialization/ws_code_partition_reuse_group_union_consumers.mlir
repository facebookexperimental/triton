// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3" | FileCheck %s

// Reuse groups share a single communication channel.  The channel must be
// built from the union of the logical buffers in the group, not just from the
// representative buffer.  Here %a and %b share buffer.id=0; %a is consumed only
// by task 1, while %b is consumed by both task 0 and task 1.
//
// The TMA producer (task 2) lives in the producer partition, while the two
// consumers are split into their own warp partitions:
//   - the `default` region is task 0, which reads only %b;
//   - the `partition0` region is task 1, which reads both %a and %b.
//
// The union fix guarantees that every consumer of the reused buffer gets a
// token and a barrier wait against the TMA producer.  Concretely:
//   - task 0 must wait on %b's TMA producer before its single local_load,
//     even though %a (the other buffer in the reuse group) is not consumed by
//     task 0 -- without the union it would silently drop this wait;
//   - task 1 must wait twice, once for each of %a and %b, before its two
//     local_loads;
//   - the producer acquires a token for both consumer tasks before each of
//     its two TMA copies, so neither consumer can observe the reused buffer
//     before its matching TMA load completes.

// CHECK-LABEL: @reuse_group_union_consumers
// CHECK: ttg.warp_specialize

// default region == task 0: waits on %b's TMA producer, then reads %b.
// CHECK:      default {
// CHECK:        ttg.partition = array<i32: 0>
// CHECK:        ttng.wait_barrier
// CHECK:        ttg.local_load

// partition0 == task 1: waits for both %a and %b before reading them.
// CHECK:      partition0(
// CHECK:        ttng.wait_barrier
// CHECK:        ttng.wait_barrier
// CHECK:        ttg.local_load
// CHECK:        ttg.local_load
// CHECK:        arith.addf

// partition1 == task 2 (producer): acquires the token for both consumer tasks
// before each TMA copy of the reused buffer.
// CHECK:      partition1(
// CHECK:        nvws.producer_acquire
// CHECK:        nvws.producer_acquire
// CHECK:        ttng.async_tma_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reuse_group_union_consumers(%a_desc: !tt.tensordesc<128x64xf16, #shared>, %b_desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1 : i32
    %c4 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 4 : i32
    %ptrs0 = tt.splat %out0 {ttg.partition = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {ttg.partition = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    scf.for %iv = %c0 to %c4 step %c1 : i32 {
      %a_tile = tt.descriptor_load %a_desc[%c0, %c0] {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %a_tile, %a {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b_tile = tt.descriptor_load %b_desc[%c0, %c0] {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %b_tile, %b {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b0 = ttg.local_load %b {ttg.partition = array<i32: 0>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      tt.store %ptrs0, %b0 {ttg.partition = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a1 = ttg.local_load %a {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %b1 = ttg.local_load %b {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %sum = arith.addf %a1, %b1 {ttg.partition = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      tt.store %ptrs1, %sum {ttg.partition = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    } {ttg.partition = array<i32: 0, 1, 2>, tt.warp_specialize}
    tt.return
  }
}
