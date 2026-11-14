// RUN: triton-opt %s --nvgpu-test-ws-atomic-broadcast | FileCheck %s

// Partial work-stealing under AutoWS: the CLC fetch replicated to every
// partition is moved to the owner (TMA-load) partition with its steal
// predicate intact, and the steal-masked isValid that the read produces is the
// value broadcast to all partitions. Every partition therefore leaves the
// persistent loop together when a non-stealing CTA finishes its tile.

// CHECK-LABEL: @steal_bcast
// CHECK:       %[[STEAL:.*]] = arith.cmpi slt
// CHECK:       scf.while
// CHECK:       ttng.clc_try_cancel_async %[[STEAL]] {async_task_id = array<i32: 1>}
// CHECK:       %[[VALID:.*]], %[[X:.*]], %{{.*}}, %{{.*}} = ttng.clc_read %{{.*}} {async_task_id = array<i32: 1>}
// CHECK:       arith.extui %[[VALID]] {async_task_id = array<i32: 1>}
// CHECK:       ttg.local_store {{.*}}async_task_id = array<i32: 1>
// CHECK:       ttg.local_load {{.*}}async_task_id = array<i32: 0, 1>
// CHECK:       %[[BVALID:.*]] = arith.trunci {{.*}}async_task_id = array<i32: 0, 1>
// CHECK:       scf.yield {async_task_id = array<i32: 0, 1>} %[[BVALID]],
// CHECK-NOT:   ttng.clc_try_cancel_async

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @steal_bcast(%desc: !tt.tensordesc<128x64xf16, #shared>, %n: i32) {
    %pid = tt.get_program_id x {async_task_id = array<i32: 0, 1>} : i32
    %steal = arith.cmpi slt, %pid, %n {async_task_id = array<i32: 0, 1>} : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %r:2 = scf.while (%v = %true, %x = %pid) : (i1, i32) -> (i1, i32) {
      scf.condition(%v) {async_task_id = array<i32: 0, 1>} %v, %x : i1, i32
    } do {
    ^bb0(%v: i1, %x: i32):
      %tok = ttng.clc_try_cancel_async %steal {async_task_id = array<i32: 0, 1>} : !ttg.async.token
      %a = tt.descriptor_load %desc[%x, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %valid, %nx, %ny, %nz = ttng.clc_read %tok {async_task_id = array<i32: 0, 1>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield {async_task_id = array<i32: 0, 1>} %valid, %nx : i1, i32
    } attributes {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
