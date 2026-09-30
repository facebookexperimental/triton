// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file -tritongpu-pipeline | FileCheck %s

// With epilogue peeling, a loop whose trip count is at most the pipeline depth
// runs the kernel loop zero times. The first peeled last-stage op then executes
// iteration 0, so an iter arg whose yielded value is defined outside the loop
// (here the MMA use_acc flag set to true after the first iteration) must still
// hold its init value there. Reading the yielded value instead made the first
// MMA accumulate onto stale TMEM.

#s = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bs = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tm = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tma = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peeled_mma_use_acc
  // CHECK: %[[KERNEL:.*]] = scf.for %{{.*}} = %c0_i32 to %c0_i32 {{.*}} iter_args(%[[USE_ACC:.*]] = %false)
  // CHECK:   ttng.tc_gen5_mma {{.*}}, %[[USE_ACC]], %true
  // CHECK:   scf.yield %true
  // CHECK: ttng.tc_gen5_mma {{.*}}, %[[KERNEL]], %true
  // CHECK: ttng.tc_gen5_mma {{.*}}, %true, %true
  tt.func public @peeled_mma_use_acc(%p: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %false = arith.constant false
    %true = arith.constant true
    %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xbf16, #tma, #ttng.tensor_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #s, #ttg.shared_memory, mutable>
    %c = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tm, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bs, #ttg.shared_memory, mutable>
    %r = scf.for %k = %c0 to %c2 step %c1 iter_args(%use_acc = %false) -> (i1) : i32 {
      tt.store %p, %k {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.ptr<i32>
      ttng.tc_gen5_mma %a, %b, %c, %use_acc, %true, %bar[%true] {is_async, loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xbf16, #tma, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xbf16, #s, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tm, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #bs, #ttg.shared_memory, mutable>
      ttng.wait_barrier %bar, %c0 deps %a, %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<1xi64, #bs, #ttg.shared_memory, mutable>, !ttg.memdesc<128x64xbf16, #tma, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xbf16, #s, #ttg.shared_memory, mutable>
      scf.yield %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return
  }
}

// -----

// With index bounds and a trip count equal to the pipeline depth, the expander
// takes its static path. The first peeled last-stage op still reads the kernel
// loop's value, and the loop result is the yielded value because the loop runs
// at least once, even though the kernel loop ran zero times.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peeled_static_result
  // CHECK: %[[KERNEL:.*]]:3 = scf.for %{{.*}} = %c0 to %c0
  // CHECK-NOT: arith.cmpi
  // CHECK: arith.select %[[KERNEL]]#0,
  // CHECK: tt.return %true
  tt.func @peeled_static_result(%p: !tt.ptr<i32>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %z = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %r = scf.for %iv = %c0 to %c2 step %c1 iter_args(%flag = %false) -> (i1) {
      %x = arith.index_cast %iv {loop.cluster = 1 : i32, loop.stage = 0 : i32} : index to i32
      %y = arith.select %flag, %x, %z {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      tt.store %p, %y {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.ptr<i32>
      scf.yield %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return %r : i1
  }
}

// -----

// Constant i32 bounds are not index constants, so the expander takes its
// dynamic path even though the trip count is known; the result select then
// folds to the yielded value.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peeled_i32_bounds_result
  // CHECK: %[[KERNEL:.*]]:3 = scf.for %{{.*}} = %c0_i32 to %c0_i32
  // CHECK: arith.select %[[KERNEL]]#0,
  // CHECK: tt.return %true
  tt.func @peeled_i32_bounds_result(%p: !tt.ptr<i32>) -> i1 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %false = arith.constant false
    %true = arith.constant true
    %r = scf.for %iv = %c0 to %c2 step %c1 iter_args(%flag = %false) -> (i1) : i32 {
      %x = arith.addi %iv, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %y = arith.select %flag, %x, %c0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      tt.store %p, %y {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.ptr<i32>
      scf.yield %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return %r : i1
  }
}

// -----

// A dynamic trip count may be at most the pipeline depth too. The first peeled
// last-stage op reads the kernel loop's value, and the loop result is selected
// on whether the loop ran at all.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peeled_dynamic
  // CHECK: %[[KERNEL:.*]]:3 = scf.for
  // CHECK: %[[RAN:.*]] = arith.cmpi sge, %{{.*}}, %c1_i32
  // CHECK: arith.select %[[KERNEL]]#0,
  // CHECK: %[[RESULT:.*]] = arith.select %[[RAN]], %true, %[[KERNEL]]#0 : i1
  // CHECK: tt.return %[[RESULT]]
  tt.func @peeled_dynamic(%p: !tt.ptr<i32>, %n: i32) -> i1 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%flag = %false) -> (i1) : i32 {
      %x = arith.addi %iv, %c1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %y = arith.select %flag, %x, %c0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      tt.store %p, %y {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.ptr<i32>
      scf.yield %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return %r : i1
  }
}
