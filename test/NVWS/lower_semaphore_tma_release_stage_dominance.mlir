// RUN: triton-opt %s --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --implicit-check-not=nvws.semaphore

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @release_stage_dominates_tma_load
  tt.func @release_stage_dominates_tma_load(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0], %tok {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
    nvws.descriptor_load %desc[%c0, %c0] 16384 %view {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>

    // The release has an authored +1 ring offset defined after the descriptor
    // load. AssignStagePhase turns it into (base + 1) % 3 next to the release.
    // The lowerer must reproduce that generated expression before indexing the
    // TMA completion barrier; using the unshifted acquire stage is incorrect.
    // CHECK: [[LOAD_VIEW:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {{.*}} -> !ttg.memdesc<128x64xf16,
    // CHECK-NEXT: [[ONE:%.*]] = arith.constant {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 1 : i32
    // CHECK-NEXT: [[RAW:%.*]] = arith.addi [[BASE:%.*]], [[ONE]]
    // CHECK-NEXT: [[DEPTH:%.*]] = arith.constant {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 3 : i32
    // CHECK-NEXT: [[REM:%.*]] = arith.remsi [[RAW]], [[DEPTH]]
    // CHECK-NEXT: [[ZERO:%.*]] = arith.constant {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
    // CHECK-NEXT: [[NEG:%.*]] = arith.cmpi slt, [[REM]], [[ZERO]]
    // CHECK-NEXT: [[WRAPPED:%.*]] = arith.addi [[REM]], [[DEPTH]]
    // CHECK-NEXT: [[RELEASE_STAGE:%.*]] = arith.select [[NEG]], [[WRAPPED]], [[REM]]
    // CHECK-NEXT: [[TMA_MBAR:%.*]] = ttg.memdesc_index %{{.*}}[[[RELEASE_STAGE]]]
    // CHECK: ttng.barrier_expect [[TMA_MBAR]], 16384
    // CHECK: ttng.async_tma_copy_global_to_local %arg0[%{{.*}}, %{{.*}}] %{{.*}}, [[TMA_MBAR]]
    %c1 = arith.constant 1 : i32
    nvws.semaphore.release %sem[%c1], %tok [#nvws.async_op<tma_load>] {loop.cluster = 9 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
